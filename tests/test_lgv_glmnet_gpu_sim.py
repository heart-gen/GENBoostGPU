"""GPU glmnet kernel vs the CPU core in numba's CUDA simulator (no GPU needed).

Slow (~8 min), so opt-in: ``GENBOOSTGPU_CUDASIM=1 pytest -m cudasim``.
"""
import os
import subprocess
import sys

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "scripts", "cudasim_glmnet.py")


@pytest.mark.cudasim
@pytest.mark.skipif(os.environ.get("GENBOOSTGPU_CUDASIM") != "1",
                    reason="set GENBOOSTGPU_CUDASIM=1 to run the CUDA-simulator test")
def test_gpu_kernel_matches_cpu_in_simulator():
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1")
    res = subprocess.run([sys.executable, SCRIPT], env=env, capture_output=True, text=True,
                         timeout=3600)
    assert res.returncode == 0, res.stdout + res.stderr
