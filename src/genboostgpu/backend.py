"""Array-backend selection for the local-genetic-variance engine.

The engine runs the same code on CPU (NumPy) or GPU (CuPy). Callers ask for a
backend once and pass ``xp`` down; nothing below imports CuPy directly, so the
package imports and its tests run on machines without a GPU.
"""
from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "Backend",
    "get_backend",
    "gpu_available",
    "import_cupy",
    "is_environment_error",
    "to_host",
]

# CUDA libraries CuPy loads lazily (matmul -> cuBLAS, linalg -> cuSOLVER,
# elementwise kernels -> NVRTC, ...), in dependency order. pip
# ``nvidia-*-cu12`` wheels put them under site-packages/nvidia/<pkg>/lib,
# which CuPy 13 does not search.
_CUDA_WHEEL_LIBS = (
    ("nvjitlink", "libnvJitLink.so.12"),
    ("cublas", "libcublasLt.so.12"),
    ("cublas", "libcublas.so.12"),
    ("cusparse", "libcusparse.so.12"),
    ("cusolver", "libcusolver.so.11"),
    ("curand", "libcurand.so.10"),
    ("cufft", "libcufft.so.11"),
    ("cuda_nvrtc", "libnvrtc-builtins.so.12.*"),
    ("cuda_nvrtc", "libnvrtc.so.12"),
)


@lru_cache(maxsize=1)
def _preload_cuda_wheel_libs() -> None:
    """Load CUDA libraries from pip wheels when the loader cannot find them."""
    for pkg, lib in _CUDA_WHEEL_LIBS:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            continue
        except OSError:
            pass
        try:
            spec = importlib.util.find_spec(f"nvidia.{pkg}")
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        for root in spec.submodule_search_locations:
            paths = sorted(glob.glob(os.path.join(root, "lib", lib)))
            if paths:
                try:
                    ctypes.CDLL(paths[-1], mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


def import_cupy():
    """Import CuPy after making wheel-installed CUDA libraries loadable."""
    _preload_cuda_wheel_libs()
    import cupy

    return cupy


@dataclass(frozen=True)
class Backend:
    """Resolved array backend.

    ``name`` is ``"cpu"`` or ``"gpu"``; ``xp`` is the array module.
    """

    name: str
    xp: object
    device_id: int | None = None

    @property
    def is_gpu(self) -> bool:
        return self.name == "gpu"

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def to_host(self, a):
        return to_host(a)


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True when CuPy imports and at least one CUDA device is visible."""
    try:
        cp = import_cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_backend(device: str = "auto", device_id: int | None = None) -> Backend:
    """Resolve ``device`` (``"auto"``, ``"cpu"`` or ``"gpu"``) to a Backend.

    ``auto`` picks the GPU when one is usable. ``GENBOOSTGPU_DEVICE`` overrides
    ``auto`` so batch scripts can force the CPU path without code changes.
    """
    device = (device or "auto").lower()
    if device == "auto":
        device = os.environ.get("GENBOOSTGPU_DEVICE", "auto").lower()
    if device == "auto":
        device = "gpu" if gpu_available() else "cpu"
    if device == "cpu":
        return Backend("cpu", np, None)
    if device == "gpu":
        if not gpu_available():
            raise RuntimeError("device='gpu' requested but no CUDA device is usable")
        cp = import_cupy()

        if device_id is not None:
            cp.cuda.Device(int(device_id)).use()
        return Backend("gpu", cp, cp.cuda.Device().id)
    raise ValueError(f"Unknown device: {device!r}")


_ENV_ERROR_MARKERS = ("CuPy failed to load", "cannot open shared object file",
                      "CUDA_ERROR", "cudaError")


def is_environment_error(err: BaseException) -> bool:
    """True for errors of the runtime rather than of one locus.

    A missing library, host memory exhaustion or a CUDA/CuPy error would hit
    every locus alike. Runners re-raise these instead of recording a
    per-locus failure row, so the shard stops and a resubmission retries it.
    """
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if isinstance(err, (ImportError, MemoryError)):
            return True
        mod = type(err).__module__ or ""
        if mod.startswith(("cupy", "cupy_backends", "numba.cuda")):
            return True
        # CuPy reports a missing CUDA library as a plain RuntimeError/OSError.
        if any(m in str(err) for m in _ENV_ERROR_MARKERS):
            return True
        err = err.__cause__ or err.__context__
    return False


def to_host(a):
    """Return a NumPy array (or the input unchanged when it is not an array)."""
    if hasattr(a, "get") and type(a).__module__.startswith("cupy"):
        return a.get()
    return a
