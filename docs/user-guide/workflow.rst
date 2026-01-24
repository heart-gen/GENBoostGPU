Workflow
========

At a high level, GENBoostGPU moves data from disk or memory, filters and scores
SNPs, trains elastic net models in a boosting loop, and evaluates variance
explained. The diagram below highlights the major stages.

.. code-block:: text

   +-------------+      +------------------+      +-----------------+      +------------------+
   | Data input  | ---> | SNP preprocessing| ---> | Boosting elastic| ---> | Evaluation &     |
   | (PLINK, CuPy|      | (filtering, LD)  |      | net iterations  |      | persistence      |
   +-------------+      +------------------+      +-----------------+      +------------------+
           |                      |                        |                        |
           v                      v                        v                        v
   data_io.load_*      snp_processing.*         enet_boosting.boosting_*    data_io.save_results

Module responsibilities
-----------------------

:mod:`genboostgpu.data_io`
   Reads PLINK and phenotype files, emits CuPy/cuDF objects, and saves outputs to
   TSV/Parquet.
:mod:`genboostgpu.snp_processing`
   Applies zero-variance filtering, missing value imputation, cis-window selection,
   and LD clumping.
:mod:`genboostgpu.enet_boosting`
   Implements the boosting loop, Optuna-based ElasticNet tuning, and final ridge
   refit.
:mod:`genboostgpu.cpg_orchestration`
   CpG-centric orchestration utilities for scheduling boosting tasks across
   traits, chromosomes, or distributed Dask workers.
:mod:`genboostgpu.orchestration`
   High-level entry point. Launches :func:`genboostgpu.vmr_runner.run_single_window`
   across windows, optionally using :class:`dask_cuda.LocalCUDACluster` for
   multi-GPU execution.

Putting it together
-------------------

1. Build a list of windows (chromosome, start, end, phenotype ID/path).
2. Load or provide genotype/phenotype objects (:mod:`genboostgpu.data_io`).
3. Call :func:`genboostgpu.orchestration.run_windows_with_dask` to schedule work.
4. Inspect the resulting pandas DataFrame plus the saved parquet/TSV files.

For more detailed orchestration examples, see :doc:`tutorials/index`.
