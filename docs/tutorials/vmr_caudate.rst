VMR caudate tutorial
====================

``examples/vmr_test_caudate.py`` demonstrates running GENBoostGPU on variable
methylated regions (VMRs) derived from brain tissue. It uses the same
:func:`genboostgpu.orchestration.run_windows_with_dask` entry point as the
simulation tutorial but streams PLINK/phenotype files from disk.

Setup
-----

1. Mirror the file layout under ``heritability/gcta/<region>/_m`` expected by the
   helper functions.
2. Export ``REGION=caudate`` (or another region present in your dataset).
3. Confirm GPU availability and seed all RNGs (the script seeds ``random``,
   ``numpy``, and ``cupy`` to ``42``).

Workflow highlights
-------------------

* :func:`get_error_list` loads blacklisted windows so they can be skipped.
* :func:`get_vmr_list` enumerates VMR coordinates per region.
* ``build_windows`` prepares dictionaries that point to PLINK and phenotype
  files on disk; these map directly to the arguments accepted by
  :func:`genboostgpu.vmr_runner.run_single_window`.
* :func:`genboostgpu.orchestration.run_windows_with_dask` distributes the windows
  across available GPUs, saving results in ``results/`` with the ``vmr`` prefix.

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
