Changelog
=========

Full release notes live on `GitHub <https://github.com/heart-gen/GENBoostGPU/releases>`_.
The highlights below summarise major updates.

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
