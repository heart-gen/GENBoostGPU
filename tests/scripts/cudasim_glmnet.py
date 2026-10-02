"""Run the warp-per-problem glmnet kernel in numba's CUDA simulator and compare
it with the CPU core. Invoked by tests/test_lgv_glmnet_gpu_sim.py in a
subprocess (NUMBA_ENABLE_CUDASIM must be set before numba.cuda is imported).

The simulator lacks warp shuffles and runs each lane as a Python thread, so
lanes are *not* in lockstep; that is what exposes global read-modify-write
races between lanes.
"""
import os
import sys
import threading

import numpy as np

assert os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
from numba.cuda.simulator.kernelapi import FakeCUDAModule  # noqa: E402


def _shfl_xor_sync(self, mask, value, offset):
    t = threading.current_thread()
    vals = t._manager.__dict__.setdefault("_gbg_vals", {})
    lane = t.threadIdx.x
    vals[lane] = value
    t.syncthreads()
    out = vals[lane ^ offset]
    t.syncthreads()
    return out


def _ballot_sync(self, mask, predicate):
    t = threading.current_thread()
    vals = t._manager.__dict__.setdefault("_gbg_ballot", {})
    vals[t.threadIdx.x] = bool(predicate)
    t.syncthreads()
    out = sum(1 << lane for lane, v in vals.items() if v)
    t.syncthreads()
    return out


FakeCUDAModule.shfl_xor_sync = _shfl_xor_sync
FakeCUDAModule.ballot_sync = _ballot_sync
FakeCUDAModule.syncwarp = lambda self, mask=0xFFFFFFFF: threading.current_thread().syncthreads()

from genboostgpu.lgv.glmnet import GlmnetProblem, fit_paths  # noqa: E402
from genboostgpu.lgv.glmnet_gpu import solve_batch_gpu  # noqa: E402

rng = np.random.default_rng(3)
probs = []
x = rng.binomial(2, 0.3, size=(14, 5)).astype(float)
x[:, 4] = 1.0                                  # constant column (ju = 0)
y = x @ np.array([0.8, -0.5, 0.3, 0.0, 0.0]) + rng.normal(size=14)
# both alphas share each x object (and so one device copy per standardize flag)
for a in (0.1, 1.0):
    for std in (True, False):
        probs.append(GlmnetProblem(x, y, a, standardize=std, nlambda=5))
x = rng.binomial(2, 0.4, size=(12, 6)).astype(float)
y = x[:, 0] - x[:, 3] + rng.normal(size=12)
for std in (True, False):
    probs.append(GlmnetProblem(x, y, 0.5, nlambda=5, type_gaussian="naive", standardize=std))
# n > 32: two register slots per lane (the n <= 14 problems leave slots empty)
x = rng.binomial(2, 0.3, size=(40, 6)).astype(float)
y = x[:, 1] - 0.5 * x[:, 2] + rng.normal(size=40)
probs.append(GlmnetProblem(x, y, 0.5, nlambda=5, type_gaussian="naive"))

cpu = fit_paths(probs)
sim = solve_batch_gpu(probs, warps_per_block=1, xp=np)
worst = 0.0
for i, (c, g) in enumerate(zip(cpu, sim)):
    if c.lambda_.size != g.lambda_.size or c.npasses != g.npasses:
        print(f"problem {i}: nlambda {c.lambda_.size}/{g.lambda_.size}, "
              f"npasses {c.npasses}/{g.npasses}")
        sys.exit(1)
    worst = max(worst, np.max(np.abs(c.beta - g.beta)), np.max(np.abs(c.a0 - g.a0)),
                np.max(np.abs(c.lambda_ - g.lambda_)))
print(f"max|diff| {worst:.3g}")
sys.exit(0 if worst < 1e-10 else 1)
