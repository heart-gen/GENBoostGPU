from . import data_io
from . import vmr_runner
from . import orchestration
from . import enet_boosting
from . import snp_processing
from . import cpg_runner
from . import cpg_orchestration
from . import cpg_tuning

from .vmr_runner import run_single_window
from .enet_boosting import boosting_elastic_net
from .orchestration import run_windows_with_dask
from .data_io import load_genotypes, load_phenotypes, save_results
from .snp_processing import (
    preprocess_genotypes,
    filter_zero_variance,
    filter_cis_window,
    run_ld_clumping,
    impute_snps
)

# CpG-specific modules for million-scale processing
from .cpg_runner import run_single_cpg
from .cpg_orchestration import (
    run_cpgs_with_dask,
    run_cpgs_by_chromosome,
)
from .cpg_tuning import (
    select_tuning_cpgs,
    global_tune_cpg_params,
    leave_one_chromosome_out_tune,
)

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="pandera._pandas_deprecated"
)

__all__ = [
    # Core algorithms
    "boosting_elastic_net",
    # SNP processing
    "preprocess_genotypes",
    "filter_zero_variance",
    "filter_cis_window",
    "run_ld_clumping",
    "impute_snps",
    # Data I/O
    "load_genotypes",
    "load_phenotypes",
    "save_results",
    # VMR processing
    "run_windows_with_dask",
    "run_single_window",
    # CpG processing (million-scale)
    "run_single_cpg",
    "run_cpgs_with_dask",
    "run_cpgs_by_chromosome",
    # CpG tuning
    "select_tuning_cpgs",
    "global_tune_cpg_params",
    "leave_one_chromosome_out_tune",
    # Modules
    "data_io",
    "vmr_runner",
    "orchestration",
    "enet_boosting",
    "snp_processing",
    "cpg_runner",
    "cpg_orchestration",
    "cpg_tuning",
]
