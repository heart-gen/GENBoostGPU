"""Deprecation marker for the v0.3 boosting elastic-net entry points."""
import functools
import warnings

_MESSAGE = (
    "{name} is the legacy boosting elastic net. Its final_r2 is an in-sample "
    "fit and h2_val reuses the early-stopping split, so neither is an "
    "out-of-fold estimate. Use genboostgpu.lgv (Module 02 features) or "
    "genboostgpu.lsp (out-of-fold prediction) instead."
)


def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(_MESSAGE.format(name=func.__name__), DeprecationWarning,
                      stacklevel=2)
        return func(*args, **kwargs)
    return wrapper
