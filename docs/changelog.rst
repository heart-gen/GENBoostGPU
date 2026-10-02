Changelog
=========

Full release notes live on `GitHub <https://github.com/heart-gen/GENBoostGPU/releases>`_.
The highlights below summarise major updates.

Unreleased
   * **GEMMA OpenBLAS kernel pinned** (``bslmm.blas_coretype``, default
     ``SkylakeX``): GEMMA's OpenBLAS picks its kernel from the CPU and fell
     back to a generic one on quest13 nodes, moving ``bslmm_pve`` by up to
     1.3e-3 against sealed runs. A CPU without AVX-512 now stops the shard
     instead of silently changing the numbers.
   * ``fit_bslmm_pve`` accepts a relative ``work_dir``.
   * ``--gemma-blas-coretype`` on ``lgv init`` and ``sites init`` sets the
     kernel per run (``auto``, ``Haswell``, ...); unknown kernel names are
     rejected, since OpenBLAS would silently ignore them.

v0.4.0
   * **New engine for the manuscript endpoints** (``genboostgpu.lgv``,
     ``genboostgpu.lsp``, ``genboostgpu.sites``): Module 02 joint-model
     features (nested out-of-fold elastic net, Haseman-Elston, ``p_eff``, LD,
     GEMMA BSLMM), the frozen joint model, domain gate and within-cell relative
     local-SNP-contribution score, and Module 03 end-to-end out-of-fold
     prediction with a permutation-calibrated HE screen.
   * **R-faithful numerics**: a line-for-line port of glmnet 4.1-10's
     Gaussian path solver, including Eigen's summation order (CPU paths are
     bitwise identical to R; GPU via a warp-per-problem CUDA kernel agrees to
     rounding), and of R's Mersenne-Twister ``set.seed``/``sample()``. Replays
     of accepted runs reproduce folds, statuses, HE, ``p_eff``, LD and BSLMM
     exactly and the nested-EN features within R's own CPU-to-CPU variation
     (score Spearman 0.99999999 on a 500-task caudate replay).
   * **Deterministic SNP screen**: tied SNPs (identical genotypes) are ordered
     the same way on every CPU, GPU and BLAS thread count.
   * **Lower GPU cost**: one process per GPU with shard arrays, many loci per
     kernel launch, GEMMA on otherwise idle CPU cores or in a CPU-only array,
     and a full CPU fallback (``--device cpu``). The alphas of a fold share
     one device copy of their design matrix (standardized once by a
     per-matrix kernel), cutting GPU memory and host-to-device traffic 3x for
     Module 02 and 4x for Module 03.
   * **Inputs**: generic region phenotypes with PLINK1/PLINK2 genotypes
     (per-chromosome or genome-wide filesets, read one chromosome at a time); an
     adapter for the dna-methylation-heritability run layout;
     ``scripts/build_regions.R`` CpG/CpH region and tile builder (with
     ``--exclude-bed``, e.g. the ENCODE blacklist) and
     ``scripts/prepare_site_inputs.sh`` for million-scale site runs.
   * ``genboostgpu`` command-line interface and SLURM launchers.
   * **Packaging**: RAPIDS dependencies moved to the ``legacy`` extra; GPU
     solver in ``gpu``; PLINK2 reader in ``plink2``.
   * **Deprecated**: the boosting entry points (``boosting_elastic_net``,
     ``run_windows_with_dask``, ``run_single_window``, ``run_cpgs_*``) warn
     that ``final_r2`` is in-sample; ``scripts/prepare_cpg_inputs.R``.

v0.3.0
   * Introduces a CpG-centric pipeline built on the validated VMR workflow,
     enabling million-scale panels and curated signatures with checkpointable,
     restart-friendly execution.
   * Heritability (h²) reporting is more robust via null calibration and an
     unscaling fix applied to window metrics and summaries.
   * Documentation refresh: installation notes, tuned workflow reflected across
     the README and tutorials, plus a new CpG pipeline user-guide page.
   * **Breaking**: the legacy pipeline module and its documentation references
     were removed; migrate to the CpG pipeline and tuned VMR workflow.

v0.2.0
   * Multi-GPU orchestration now auto-tunes ``max_in_flight`` based on the number
     of detected devices and keeps Dask clusters alive until every future is
     drained. This prevents premature shutdowns on longer studies and improves
     throughput on 4+ GPU systems.
   * Added cohort-wide helpers in :mod:`genboostgpu.tuning`, including
     :func:`~genboostgpu.tuning.select_tuning_windows` for stratified sampling
     and :func:`~genboostgpu.tuning.global_tune_params` for Optuna-backed ridge
     refits derived from sparsity targets.
   * Documentation now covers the reproducibility checklist, deterministic
     Optuna configuration, and richer tutorials linked from ``examples/`` so new
     users can mirror the exact benchmarking pipelines.

v0.1.0
   * Initial public release with elastic net boosting, cis-window preprocessing,
     and PLINK/CuPy data loaders.
