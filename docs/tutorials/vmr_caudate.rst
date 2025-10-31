VMR caudate tutorial
====================

``examples/vmr_test_caudate.py`` demonstrates running GENBoostGPU on variable
methylated regions (VMRs) derived from brain tissue. It mirrors the simulation
tutorial by selecting representative windows, tuning global hyperparameters, and
then dispatching the full workload with :func:`genboostgpu.orchestration.run_windows_with_dask`.

Setup
-----

1. Mirror the file layout under ``heritability/gcta/<region>/_m`` expected by the
   helper functions.
2. Export ``REGION=caudate`` (or another region present in your dataset).
3. Confirm GPU availability and seed all RNGs (the script seeds ``random``,
   ``numpy``, and ``cupy`` to ``42``).
4. Ensure the RAPIDS/CuPy versions installed in your environment satisfy the
   pins published in ``pyproject.toml``.

Tuning and orchestration
------------------------

* :func:`get_error_list` loads blacklisted windows so they can be skipped.
* :func:`build_windows` now reads each window's PLINK trio, assembles an
  aggregate ``bim`` table, and records per-window ``M_raw`` counts so the tuning
  heuristics can stratify by SNP density.
* :func:`genboostgpu.tuning.select_tuning_windows` draws a 5% sample with at
  least one window per chromosome. Those candidates feed into
  :func:`genboostgpu.tuning.global_tune_params`, which evaluates elastic net
  targets across GPUs using the same search grid described in the simulation
  tutorial.
* ``make_fixed_fn`` converts the chosen ``c_lambda``/``c_ridge`` pair into per-
  window ``fixed_alpha`` and ``fixed_l1_ratio`` values via
  :func:`genboostgpu.hyperparams.enet_from_targets`. The tuned subsample fraction
  and batch size are passed directly to
  :func:`genboostgpu.orchestration.run_windows_with_dask`, which fans the
  remaining windows across the cluster and writes summaries to ``results/`` with
  the ``vmr`` prefix.

Full script
-----------

.. literalinclude:: ../../examples/vmr_test_caudate.py
   :language: python
   :linenos:

Troubleshooting
---------------

* Missing imports or file paths will raise ``FileNotFoundError`` before any GPU
  work begins—double-check the directory layout.
* ``global_tune_params`` will read PLINK/phenotype files for the sampled windows
  during tuning. If I/O is a bottleneck, lower ``frac`` or ``n_max`` in
  ``select_tuning_windows``.
* Reduce ``n_trials`` passed to :func:`genboostgpu.orchestration.run_windows_with_dask`
  if VMR windows are extremely numerous; the tuned ``fixed_*`` parameters let you
  drop to one trial safely.
* Compare the saved results with the simulation outputs to validate that real
  phenotypes lead to the expected spread of ``final_r2`` values.
