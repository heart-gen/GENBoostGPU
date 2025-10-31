Simulation tutorial
===================

This tutorial walks through ``examples/simu_test_100n.py``. It tunes global
hyperparameters on a synthetic dataset before orchestrating multi-window runs
with :func:`genboostgpu.orchestration.run_windows_with_dask`.

Setup
-----

1. Place the simulated PLINK/phenotype assets in ``inputs/simulated-data/_m`` as
   expected by the script.
2. Export ``NUM_SAMPLES`` to select the cohort size, e.g. ``NUM_SAMPLES=100``.
3. Activate the GPU environment from :doc:`installation` and ensure seeds are set
   globally (the script also seeds ``random``, ``numpy``, and ``cupy`` to ``42``).

Key stages
----------

* Load genotypes with :func:`genboostgpu.data_io.load_genotypes` and phenotypes
  from cuDF tables.
* Select representative windows via
  :func:`genboostgpu.tuning.select_tuning_windows`.
* Run :func:`genboostgpu.tuning.global_tune_params` to derive ElasticNet targets
  that map to ``fixed_alpha``/``fixed_l1_ratio``.
* Launch :func:`genboostgpu.orchestration.run_windows_with_dask` with the tuned
  parameters, saving Parquet summaries in ``results/``.

Full script
-----------

.. literalinclude:: ../../examples/simu_test_100n.py
   :language: python
   :linenos:

Next steps
----------

* Inspect the output Parquet files to confirm the tuned hyperparameters are being
  reused.
* Modify ``working_set`` or ``early_stop`` in the script to explore runtime
  trade-offs.
* Compare the per-window metrics with :doc:`tutorials/vmr_caudate` to see how real
  VMR data differs from simulations.
