Quick start
===========

The snippet below shows the fastest way to orchestrate a single window analysis
with :func:`genboostgpu.orchestration.run_windows_with_dask`. It generates
toy CuPy arrays, seeds all RNGs with ``42``, and saves results to ``results/``.

.. note::
   Ensure you have followed the :doc:`installation` guide and have a CUDA 12 GPU
   visible to the process (``CUDA_VISIBLE_DEVICES``).

.. code-block:: python
   :linenos:

   import cupy as cp
   import numpy as np
   import pandas as pd

   from genboostgpu.orchestration import run_windows_with_dask

   np.random.seed(42)
   cp.random.seed(42)

   n_samples, n_snps = 256, 512
   geno = cp.asarray(np.random.normal(size=(n_samples, n_snps)), dtype=cp.float32)
   bim = pd.DataFrame({
       "chrom": ["21"] * n_snps,
       "snp": [f"rs{i}" for i in range(n_snps)],
       "pos": np.arange(n_snps) * 100 + 150_000,
   })
   pheno = cp.asarray(np.random.normal(size=n_samples), dtype=cp.float32)

   windows = [{
       "chrom": 21,
       "start": 150_000,
       "end": 150_000,
       "pheno": pheno,
       "pheno_id": "trait_1",
   }]

   results = run_windows_with_dask(
       windows,
       geno_arr=geno,
       bim=bim,
       outdir="results",
       window_size=200_000,
       n_iter=30,
       n_trials=5,
       batch_size=512,
       prefix="quickstart",
   )

   results.to_csv("results/trait_1_summary.csv", index=False)
   print(results.head())

The call triggers :mod:`genboostgpu.vmr_runner` under the hood, which filters SNPs
in the cis-window, performs boosting iterations, and writes parquet/TSV files to
``results/``.

More to explore
----------------

* Inspect the saved parquet at ``results/quickstart.summary_windows.parquet`` for
  window-level metrics.
* Dive into :doc:`user-guide/workflow` to understand how each module contributes.
* Try the richer :doc:`tutorials/simulation` and :doc:`tutorials/vmr_caudate`
  walkthroughs that reuse the scripts shipped in ``examples/``.
