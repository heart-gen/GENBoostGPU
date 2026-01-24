Changelog
=========

Full release notes live on `GitHub <https://github.com/heart-gen/GENBoostGPU/releases>`_.
The highlights below summarise major updates.

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
