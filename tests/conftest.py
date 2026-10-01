"""Test configuration.

The local-genetic-variance engine (``genboostgpu.lgv`` and friends) is tested
on the NumPy backend everywhere. Tests marked ``gpu`` need a CUDA device and
are skipped without one. The legacy RAPIDS tests (cuDF/cuML) are not
collected when RAPIDS is not installed.
"""
import os
import subprocess
import sys

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "r")

_LEGACY = [
    "test_cpg_orchestration.py", "test_cpg_runner.py", "test_cpg_tuning.py",
    "test_data_io.py", "test_enet_boosting.py", "test_snp_processing.py",
]


def _rapids_importable() -> bool:
    # Probe in a subprocess: a broken RAPIDS install can fail half-way through
    # import and leave this interpreter unable to import other GPU modules.
    try:
        res = subprocess.run([sys.executable, "-c", "import cudf, cuml"],
                             capture_output=True, timeout=120)
        return res.returncode == 0
    except Exception:
        return False


collect_ignore = [] if _rapids_importable() else list(_LEGACY)


def _gpu_ok():
    try:
        from genboostgpu.backend import gpu_available

        return gpu_available()
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device")


def pytest_collection_modifyitems(config, items):
    if _gpu_ok():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fx():
    """Path helper for committed R reference outputs."""
    def path(name):
        return os.path.join(FIXTURES, name)
    return path
