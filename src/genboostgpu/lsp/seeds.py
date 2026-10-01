"""Module 03 seeds: ``00_shared/runid.R::seed_for`` (xxhash32 of the key)."""
from __future__ import annotations

__all__ = ["xxhash32", "seed_for"]

_P1, _P2, _P3, _P4, _P5 = 2654435761, 2246822519, 3266489917, 668265263, 374761393
_M = 0xFFFFFFFF


def _rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & _M


def xxhash32(data: bytes, seed: int = 0) -> int:
    """XXH32 (as ``digest::digest(x, "xxhash32", serialize = FALSE)``)."""
    n = len(data)
    i = 0
    if n >= 16:
        v1 = (seed + _P1 + _P2) & _M
        v2 = (seed + _P2) & _M
        v3 = seed & _M
        v4 = (seed - _P1) & _M
        while i + 16 <= n:
            vals = [int.from_bytes(data[i + 4 * k:i + 4 * k + 4], "little") for k in range(4)]
            v1 = (_rotl((v1 + vals[0] * _P2) & _M, 13) * _P1) & _M
            v2 = (_rotl((v2 + vals[1] * _P2) & _M, 13) * _P1) & _M
            v3 = (_rotl((v3 + vals[2] * _P2) & _M, 13) * _P1) & _M
            v4 = (_rotl((v4 + vals[3] * _P2) & _M, 13) * _P1) & _M
            i += 16
        h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & _M
    else:
        h = (seed + _P5) & _M
    h = (h + n) & _M
    while i + 4 <= n:
        w = int.from_bytes(data[i:i + 4], "little")
        h = (_rotl((h + w * _P3) & _M, 17) * _P4) & _M
        i += 4
    while i < n:
        h = (_rotl((h + data[i] * _P5) & _M, 11) * _P1) & _M
        i += 1
    h ^= h >> 15
    h = (h * _P2) & _M
    h ^= h >> 13
    h = (h * _P3) & _M
    h ^= h >> 16
    return h


def seed_for(run_id, region="", task="", repeat_i="", fold="") -> int:
    """``seed_for(run_id, region, task, repeat_i, fold)``: the first seven hex
    digits of the key's xxhash32, as an integer."""
    key = "|".join(str(v) for v in (run_id, region, task, repeat_i, fold))
    return int(f"{xxhash32(key.encode('utf-8')):08x}"[:7], 16)
