Reproducibility
===============

Re-running the same GENBoostGPU experiment should yield consistent SNP sets and
variance estimates. Use the checklist below to lock down sources of randomness
and capture metadata.

Random seeds
------------

* Set seeds in Python's ``random`` module, NumPy, and CuPy before invoking any
  pipelines:

  .. code-block:: python

     import random
     import numpy as np
     import cupy as cp

     random.seed(42)
     np.random.seed(42)
     cp.random.seed(42)

* Pass ``random_state`` explicitly to :func:`genboostgpu.enet_boosting.boosting_elastic_net`
  (default ``13``). The orchestrator propagates this via ``fixed_params`` when you
  reuse tuned hyperparameters.
* When performing global tuning, set the ``seed`` argument in
  :func:`genboostgpu.tuning.select_tuning_windows` so the sampling of windows is
  stable.
* Optuna supports deterministic execution through ``OPTUNA_SEED`` or by
  monkey-patching ``optuna.create_study`` as shown in :doc:`tuning`.

Deterministic settings
----------------------

* Fix hyperparameters via ``fixed_alpha``, ``fixed_l1_ratio``, and
  ``fixed_subsample`` when you want to avoid per-window Optuna searches.
* Keep the validation split deterministic by ensuring ``val_frac`` stays within
  ``(0, 0.9)`` so the same RNG path is followed.
* Disable working-set adaptation by passing ``adaptive_trials=False`` to
  :func:`genboostgpu.enet_boosting.boosting_elastic_net` if you need an identical
  number of trials per window.

Logging & artefacts
-------------------

* Each call to :func:`genboostgpu.vmr_runner.run_single_window` writes betas and
  ``h2`` trajectories through :func:`genboostgpu.data_io.save_results`. Archive
  these files alongside your downstream analyses.
* Append configuration dictionaries (hyperparameters, seeds, versions) to the
  saved TSV/Parquet outputs using the ``meta`` argument in ``save_results``.
* Capture software versions with ``session_info.show()`` (used in
  ``examples/simu_test_100n.py``) and store them next to summary tables.
* Use structured logging (e.g., ``logging.config.dictConfig``) in your wrapper
  scripts to mirror the information produced by Dask and Optuna.
