Troubleshooting
===============

Common deployment problems and their fixes.

CUDA / RAPIDS version mismatch
------------------------------

* Symptom: ``ImportError: libcudart.so`` or ``RuntimeError: CUDA error`` when
  importing ``cudf``/``cuml``.
* Fix: Verify that your installed RAPIDS packages (``cudf``, ``cuml``,
  ``dask-cuda``) match the CUDA driver version. GENBoostGPU targets the
  ``25.8`` release on CUDA 12.x. Recreate the environment with matching conda
  channels or upgrade the host driver.

Out of memory (OOM)
-------------------

* Lower ``batch_size`` and ``working_set['K']`` to reduce per-iteration memory.
* Enable RAPIDS Memory Manager by exporting ``RMM_POOL_SIZE=12GB`` before running
  scripts or rely on the default ``LocalCUDACluster`` settings.
* Chunk genotype loading or downsample windows. ``select_tuning_windows`` can
  help prioritise informative regions first.

Pandas / NumPy pinning conflicts
--------------------------------

* Symptom: ``ImportError`` complaining about binary incompatibilities between
  ``pandas`` and ``numpy`` or ``pandas-plink``.
* Fix: Use the version constraints shipped in ``pyproject.toml`` (``pandas>=2.3``
  and ``numpy<2.3``). If building docs on CPU, install those exact versions before
  ``pandas-plink`` to avoid ABI mismatches.

Dask connection issues
----------------------

* Symptom: Workers fail to connect or time out when launching multi-GPU runs.
* Fixes:

  - Ensure all nodes share the same CUDA/RAPIDS versions and NCCL libraries.
  - Disable the dashboard (default) or set ``--dashboard-address=:8787`` to a free
    port.
  - Export ``DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=60s`` for clusters with
    slow startup.
  - When running locally, try ``CUDA_VISIBLE_DEVICES=0`` to force single-GPU mode
    and confirm the pipeline works before scaling out.
