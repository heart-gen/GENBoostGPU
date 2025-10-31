VMR caudate tutorial
====================

``examples/vmr_test_caudate.py`` demonstrates running GENBoostGPU on variable
methylated regions (VMRs) derived from brain tissue. It now mirrors the
simulation tutorial by tuning global hyperparameters before launching
:func:`genboostgpu.orchestration.run_windows_with_dask`, while still streaming
PLINK/phenotype files from disk.

Setup
-----

1. Mirror the file layout under ``heritability/gcta/<region>/_m`` expected by the
   helper functions.
2. Export ``REGION=caudate`` (or another region present in your dataset).
3. Confirm GPU availability and seed all RNGs (the script seeds ``random``,
   ``numpy``, and ``cupy`` to ``42``).

Tuning and orchestration flow
-----------------------------

* :func:`get_error_list` loads blacklisted windows so they can be skipped.
* :func:`build_windows` materialises window dictionaries and stitches together
  a combined BIM table so SNP counts (``M_raw``) are available for tuning.
* :func:`genboostgpu.tuning.select_tuning_windows` samples a stratified subset of
  windows based on SNP density, matching the approach used in the simulation
  tutorial.
* :func:`genboostgpu.tuning.global_tune_params` evaluates a modest grid of
  ``c_lambda``, ``c_ridge``, ``subsample_frac``, and ``batch_size`` values to
  derive fixed ElasticNet targets. The helper :func:`genboostgpu.hyperparams.enet_from_targets`
  converts the winning ``(c_lambda, c_ridge)`` pair into ``fixed_alpha`` and
  ``fixed_l1_ratio`` per window.
* :func:`genboostgpu.orchestration.run_windows_with_dask` distributes all windows
  across available GPUs, reusing the tuned hyperparameters and saving results in
  ``results/`` with a region-specific prefix.

The script prints the tuned configuration before orchestrating all VMRs so you
can decide whether to retune (e.g. by expanding the grid or changing
``WINDOW_SIZE``).

Full script
-----------

.. literalinclude:: ../../examples/vmr_test_caudate.py
   :language: python
   :linenos:

Troubleshooting
---------------

* Missing imports or file paths will raise ``FileNotFoundError`` before any GPU
  work begins—double-check the directory layout.
* Reduce ``n_trials`` if VMR windows are extremely numerous; the defaults are
  tuned for medium-sized cohorts.
* Compare the saved results with the simulation outputs to validate that real
  phenotypes lead to the expected spread of ``final_r2`` values.
