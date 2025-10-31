Hyperparameters & tuning
========================

The default settings in :func:`genboostgpu.enet_boosting.boosting_elastic_net`
work well for exploratory runs, but large cohorts benefit from carefully tuned
parameters. This page summarises the main knobs and shows how to automate the
search with Optuna.

Core boosting parameters
------------------------

``n_iter`` (default ``50``)
   Maximum boosting iterations. Early stopping typically halts before this
   limit; increase when signals are weak.
``batch_size`` (default ``500``)
   Size of the working set evaluated per iteration. Larger values improve model
   stability but consume more memory. For high-M windows, consider setting
   ``working_set={"K": 2048, "refresh": 5}``.
``n_trials`` (default ``20``)
   Number of Optuna trials used to tune ElasticNet hyperparameters per window.
   When ``fixed_alpha``/``fixed_l1_ratio`` are provided the tuning is skipped and
   ``n_trials`` is coerced to ``1``.
``alphas`` (default ``(0.1, 1.0)``)
   Range of ElasticNet ``alpha`` values searched by Optuna. Provide a tuple or a
   ``(low, high)`` pair to widen the space.
``l1_ratios`` (default ``(0.1, 0.9)``)
   Range of ElasticNet ``l1_ratio`` values.
``subsample_frac`` (default ``0.7``)
   Fraction of samples used within each Optuna trial. Reducing this speeds up
   tuning at the cost of more variance.
``ridge_grid`` (default ``(1e-3, ..., 10)``)
   Candidate ridge regression alphas evaluated during the final refit. Provide a
   tuple of floats or integers.
``val_frac`` (default ``0.2``)
   Fraction of samples kept aside for validation when early stopping monitors
   ``val_r2``.
``patience`` / ``min_delta`` / ``warmup`` (defaults ``5`` / ``1e-4`` / ``5``)
   Early stopping controls. ``warmup`` delays checks, ``min_delta`` is the minimum
   improvement, and ``patience`` counts how many stagnant iterations to allow.
``random_state`` (default ``13``)
   Seed used for validation splits and Optuna sampling.
``working_set``
   Dict with ``"K"`` (number of SNPs to evaluate) and ``"refresh"`` (how often to
   recompute correlations). Use it to stabilise runtimes on very large windows.

Global tuning workflow
----------------------

For cohort-wide defaults, use the helpers in :mod:`genboostgpu.tuning`:

1. :func:`genboostgpu.tuning.select_tuning_windows` stratifies a subset of
   windows based on SNP counts and chromosomes.
2. :func:`genboostgpu.tuning.global_tune_params` converts high-level targets
   (``c_lambda``, ``c_ridge``, ``subsample_frac``) into per-window ElasticNet
   parameters via :func:`genboostgpu.hyperparams.enet_from_targets`.
   The helper reuses the same Optuna ridge refit stack as the per-window runs
   and, when multiple GPUs are present, parallelises evaluation with the new
   ``max_in_flight`` defaults so global sweeps finish quickly.
3. Pass the resulting dictionary to the ``fixed_params`` callback in
   :func:`genboostgpu.orchestration.run_windows_with_dask`.

Optuna integration
------------------

The boosting core uses ``optuna.create_study`` with a median pruner and the
default sampler. To obtain deterministic behaviour, set Optuna's global seed
before launching any windows:

.. code-block:: python

   import functools
   import optuna
   from optuna.samplers import TPESampler

   optuna.study.create_study = functools.partial(
       optuna.create_study,
       sampler=TPESampler(seed=42, multivariate=True),
   )

   results = boosting_elastic_net(
       X, y, snp_ids,
       n_trials=30,
       alphas=(0.01, 1.0),
       l1_ratios=(0.05, 0.95),
       random_state=42,
   )

If you prefer not to monkey-patch, constrain ``alphas``/``l1_ratios`` and rely on
``fixed_alpha``/``fixed_l1_ratio`` to avoid stochastic searches. Per-window seeds
can be threaded through ``fixed_params`` and stored alongside outputs for later
replay.
