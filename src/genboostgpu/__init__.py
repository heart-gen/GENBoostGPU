"""GENBoostGPU.

Two families of entry points live here:

* the local-genetic-variance engine (``genboostgpu.lgv``, ``genboostgpu.lsp``,
  ``genboostgpu.sites``), which needs only NumPy and optionally CuPy; and
* the legacy boosting elastic net (``boosting_elastic_net``,
  ``run_windows_with_dask``, ``run_cpgs_*``), which needs RAPIDS (cuDF, cuML,
  dask-cuda). Its ``final_r2`` is an in-sample fit; see the docs.

Legacy names are imported lazily so ``import genboostgpu`` works without
RAPIDS installed.
"""
from __future__ import annotations

import importlib
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="pandera._pandas_deprecated"
)

_LEGACY_ATTRS = {
    # Core algorithms
    "boosting_elastic_net": "enet_boosting",
    # SNP processing
    "preprocess_genotypes": "snp_processing",
    "filter_zero_variance": "snp_processing",
    "filter_cis_window": "snp_processing",
    "run_ld_clumping": "snp_processing",
    "impute_snps": "snp_processing",
    # Data I/O
    "load_genotypes": "data_io",
    "load_phenotypes": "data_io",
    "save_results": "data_io",
    # VMR processing
    "run_windows_with_dask": "orchestration",
    "run_single_window": "vmr_runner",
    # CpG processing (million-scale)
    "run_single_cpg": "cpg_runner",
    "run_cpgs_with_dask": "cpg_orchestration",
    "run_cpgs_by_chromosome": "cpg_orchestration",
    # CpG tuning
    "select_tuning_cpgs": "cpg_tuning",
    "global_tune_cpg_params": "cpg_tuning",
    "leave_one_chromosome_out_tune": "cpg_tuning",
}

_LEGACY_MODULES = {
    "data_io", "vmr_runner", "orchestration", "enet_boosting",
    "snp_processing", "cpg_runner", "cpg_orchestration", "cpg_tuning",
    "hyperparams", "tuning",
}

__all__ = sorted(set(_LEGACY_ATTRS) | _LEGACY_MODULES | {"backend"})


def __getattr__(name):
    if name in _LEGACY_ATTRS:
        module = importlib.import_module(f".{_LEGACY_ATTRS[name]}", __name__)
        return getattr(module, name)
    if name in _LEGACY_MODULES or name == "backend":
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
