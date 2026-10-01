"""Bit-exact port of R's default random number generator.

R (>= 3.6) defaults to ``RNGkind("Mersenne-Twister", "Inversion",
"Rejection")``. The analysis pipeline draws every fold assignment and every
permutation with ``set.seed()`` followed by ``sample()``/``sample.int()``, so
reproducing those two calls exactly lets a GENBoostGPU run replay the folds
of an accepted R run without exporting anything from R.

Only what the pipeline uses is ported: ``set.seed``, ``unif_rand``,
``R_unif_index`` (rejection sampling) and ``sample.int`` without replacement.
The generator core is NumPy's MT19937, which is the same algorithm; only the
seeding (R's LCG scrambling) and the output transform differ, and both are
reproduced here.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["RRNG", "r_sample_int", "r_sample", "r_permutations"]

_MT_N = 624
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)
_TWO_NEG32 = 2.3283064365386963e-10  # 2^-32


class RRNG:
    """R's Mersenne-Twister stream after ``set.seed(seed)``."""

    def __init__(self, seed: int):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        if seed is None:
            raise ValueError("seed must be an integer")
        seed = int(seed) & 0xFFFFFFFF
        # RNG_Init: 50 rounds of initial scrambling ...
        for _ in range(50):
            seed = (69069 * seed + 1) & 0xFFFFFFFF
        # ... then 625 seeds: dummy[0] (mti) and mt[0..623].
        seeds = np.empty(_MT_N + 1, dtype=np.uint32)
        for j in range(_MT_N + 1):
            seed = (69069 * seed + 1) & 0xFFFFFFFF
            seeds[j] = seed
        # FixupSeeds(initial=TRUE) sets mti = 624, forcing a reload on first use.
        key = seeds[1:].copy()
        self._bitgen = np.random.MT19937()
        self._bitgen.state = {
            "bit_generator": "MT19937",
            "state": {"key": key, "pos": _MT_N},
        }

    def unif_rand(self, size: int | None = None):
        """``unif_rand()``: MT_genrand() passed through ``fixup``."""
        k = 1 if size is None else int(size)
        raw = self._bitgen.random_raw(k).astype(np.float64)
        x = raw * _TWO_NEG32
        x = np.where(x <= 0.0, 0.5 * _I2_32M1, x)
        x = np.where((1.0 - x) <= 0.0, 1.0 - 0.5 * _I2_32M1, x)
        return float(x[0]) if size is None else x

    def _rbits(self, bits: int) -> int:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        """``R_unif_index(dn)`` with the default "Rejection" sample kind."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv


def r_sample_int(rng: RRNG, n: int, size: int | None = None) -> np.ndarray:
    """``sample.int(n, size)`` without replacement (1-based, like R).

    ``sample.int``'s default ``useHash`` is ``n > 1e7 && size <= n/2``; only
    then does R use ``sample2``'s hash-based rejection. Everything else goes
    through ``do_sample``'s swap-remove loop. Both are ported.
    """
    n = int(n)
    size = n if size is None else int(size)
    if size > n:
        raise ValueError("cannot take a sample larger than the population")
    if n > 1e7 and size <= n / 2:
        # do_sample2: rejection against already-drawn values.
        out = np.empty(size, dtype=np.int64)
        seen = set()
        for i in range(size):
            while True:
                v = rng.unif_index(n) + 1
                if v not in seen:
                    break
            seen.add(v)
            out[i] = v
        return out
    pool = list(range(n))
    out = np.empty(size, dtype=np.int64)
    remaining = n
    for i in range(size):
        j = rng.unif_index(remaining)
        out[i] = pool[j] + 1
        remaining -= 1
        pool[j] = pool[remaining]
    return out


def r_sample(rng: RRNG, x, size: int | None = None) -> np.ndarray:
    """``sample(x, size)`` for a vector ``x`` of length > 1."""
    x = np.asarray(x)
    if x.ndim != 1 or x.size <= 1:
        raise ValueError("r_sample only supports vectors of length > 1")
    idx = r_sample_int(rng, x.size, size)
    return x[idx - 1]


_perm_kernel = None


def _get_perm_kernel():
    """numba loop for ``replicate(k, sample.int(n))`` over a raw MT32 buffer."""
    global _perm_kernel
    if _perm_kernel is None:
        import numba

        @numba.njit(cache=True)
        def fill(raw, n, k, out):
            pos = 0
            pool = np.empty(n, dtype=np.int64)
            for b in range(k):
                for i in range(n):
                    pool[i] = i
                remaining = n
                for i in range(n):
                    dn = remaining
                    bits = int(math.ceil(math.log2(dn))) if dn > 1 else 0
                    while True:
                        v = 0
                        nn = 0
                        while nn <= bits:
                            if pos >= raw.size:
                                return -1
                            u = raw[pos] * _TWO_NEG32
                            pos += 1
                            if u <= 0.0:
                                u = 0.5 * _I2_32M1
                            v = 65536 * v + int(math.floor(u * 65536))
                            nn += 16
                        dv = v & ((1 << bits) - 1)
                        if dn > dv:
                            break
                    j = dv
                    out[b, i] = pool[j] + 1
                    remaining -= 1
                    pool[j] = pool[remaining]
            return pos

        _perm_kernel = fill
    return _perm_kernel


def r_permutations(rng: RRNG, n: int, k: int) -> np.ndarray:
    """``replicate(k, sample.int(n))`` as a ``k x n`` matrix (1-based).

    Equivalent to ``k`` successive :func:`r_sample_int` calls (``n <= 1e7``),
    vectorized: raw generator output is drawn in bulk, consumed by a compiled
    loop, and the generator is then advanced by exactly what was consumed.
    """
    if n > 1e7:
        return np.stack([r_sample_int(rng, n) for _ in range(k)])
    fill = _get_perm_kernel()
    state = rng._bitgen.state
    size = 3 * n * k + 4096
    out = np.empty((k, n), dtype=np.int64)
    while True:
        rng._bitgen.state = state
        raw = rng._bitgen.random_raw(size).astype(np.float64)
        used = fill(raw, n, k, out)
        if used >= 0:
            break
        size *= 2
    rng._bitgen.state = state
    rng._bitgen.random_raw(int(used))
    return out
