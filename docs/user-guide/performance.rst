Performance & scaling
=====================

GENBoostGPU is designed to scale from a single GPU workstation to multi-GPU
servers managed by Dask. This page collects best practices for keeping runs fast
and memory-efficient.

Execution modes
---------------

Single GPU
   If ``numba.cuda.gpus`` reports exactly one device, :func:`genboostgpu.orchestration.run_windows_with_dask`
   runs windows serially without spinning up Dask. This mode is ideal for
   debugging or laptop-scale experiments.
Multi GPU
   When more than one GPU is visible, the orchestrator launches a
   :class:`dask_cuda.LocalCUDACluster`. Windows are submitted asynchronously and
   throttled with ``max_in_flight`` to balance throughput and memory usage.

Memory management tips
----------------------

* Reduce ``batch_size`` and the ``working_set['K']`` parameter for massive
  windows. Smaller batches lower peak GPU memory at the cost of more iterations.
* Use chunked genotype loading if PLINK files do not fit in device memory. Wrap
  ``load_genotypes`` with your own CuPy ``memmap`` logic and pass ``geno_arr`` as
  needed windows are processed.
* Enable RAPIDS Memory Manager pools. Either set ``RMM_POOL_SIZE=12GB`` in the
  environment or rely on the default ``rmm_pool_size="12GB"`` that
  :func:`genboostgpu.orchestration.run_windows_with_dask` passes to Dask.
* Pre-allocate CuPy memory pools for repeated runs:

  .. code-block:: python

     import cupy as cp
     cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

* Prefer Parquet outputs via ``save=True``; they can be reloaded lazily for
  downstream aggregation without keeping everything in memory.

Distributed execution
---------------------

* Pin workers to GPUs using ``CUDA_VISIBLE_DEVICES`` or the ``--CUDA_VISIBLE_DEVICES``
  flag when launching via ``dask-scheduler``/``dask-cuda-worker``.
* Keep the dashboard disabled on headless clusters (default behaviour) to reduce
  port conflicts on shared systems.
* Set ``DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=True`` to let idle GPUs pull
  windows from busy workers when jobs vary in size.

Environment checklist
---------------------

* ``RAPIDS_VERSION`` should match the ``cudf``/``cuml`` wheels you installed.
* ``CUDA_VISIBLE_DEVICES`` controls which GPUs are used; set it to a comma-separated
  list or leave it unset to use all devices.
* ``OPTUNA_STORAGE`` (e.g., ``sqlite:///optuna.db``) turns on persistent study
  storage for large sweeps.
* ``GENBOOSTGPU_LOG_LEVEL=INFO`` (custom environment variable) can be exported to
  surface more orchestration logs if you hook it inside your scripts.
