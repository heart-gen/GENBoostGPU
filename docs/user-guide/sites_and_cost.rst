Site-level runs and GPU cost
============================

Site-level features
-------------------

``genboostgpu sites`` computes Module 02 features for every unit of a
``build_regions.R --step qc`` output (CpG sites, CpH tiles or CpH sites)::

   genboostgpu sites init --run-dir runs/cph-sites --run-id cph-sites-v1 \
       --units-dir build/cph-caudate-sites/units \
       --genotypes '/path/LIBD.chr{chrom}.AA' --covariates covs.tsv \
       --numeric-covariates age --factor-covariates sex,diagnosis \
       --cohort AA --region caudate --features geometry,he,en --window-block-bp 10000
   genboostgpu sites run --run-dir runs/cph-sites --shard 0/32 --device gpu
   genboostgpu lgv combine --run-dir runs/cph-sites --no-score

``--window-block-bp B`` lets units in the same ``B``-bp bin share one cis
window ``[min(start) − 500 kb, max(end) + 500 kb]`` (recorded per row as
``window_start``/``window_end``); genotype QC, the GRM, ``p_eff`` and LD are
then computed once per block and HE is one batched quadratic form for all
units with complete data. ``B = 0`` keeps the exact per-unit window. Feature
tiers: ``geometry,he`` (very cheap), ``+en`` (nested cross-fit), ``--bslmm
inline`` (GEMMA). Scores are only emitted when all joint-model features are
present and the units fall inside the frozen model's characterized domain.

Where the GPU time goes
-----------------------

The nested cross-fit is the dominant cost: per region 180 glmnet paths
(10 outer splits × 3 alphas × [full fit + 5 inner folds]); Module 03 adds up to
600 per region. GENBoostGPU's GPU solver runs **one warp per glmnet problem**,
so problems from many regions are solved in a single kernel launch.

Guidelines for low GPU cost:

* **One process per GPU**, sharded with ``--shard i/N`` (SLURM array). No Dask
  cluster and no idle memory pools.
* **Batch enough problems.** ``--batch-loci`` (``lgv``) or ``--batch-units``
  (``sites``) sets how many loci share a launch; 16–64 keeps an A100 busy.
* **Keep MCMC off the GPU clock.** GEMMA runs on CPU cores: either inline on
  the GPU node's idle cores (``--gemma-workers``) or in a separate CPU array
  (``--bslmm separate``).
* **Size memory to the batch.** GPU memory is dominated by the path output
  buffer, which holds p × 100 coefficients per problem (glmnet's default
  ``pmax`` is p). At 1,500 screened SNPs and n ≈ 150, each batched locus needs
  about 0.3 GiB on the GPU, so a batch of 32 loci fits a 20 GB MIG slice but a
  batch of 120 does not. Only the coefficients a path actually uses (its
  largest active set, typically a few hundred SNPs) are copied back and kept
  on the host.
* **Keep the GPU fed** (``sites run``). After the GPU solves a batch, each
  unit's cross-validation summary and held-out predictions run on the CPU.
  ``sites run`` does this on a background thread while the GPU solves the
  next batch and the main thread prepares the one after, so a GPU job needs
  only about three busy cores plus any ``--gemma-workers``. Results are
  bitwise identical to a serial run. On 5,000 CpH units (A100, batch 64, 128
  donors) a run takes 961 s (5.2 units/s, about 53 GPU-hours per million
  units). Up to three batches are in host memory at once (one solving, one
  finishing, one being prepared); allow about 0.3 GiB of host RAM per batched
  unit (the batch-64 run peaked at 17 GiB).
* **Budget GEMMA separately.** BSLMM costs about 5–8 CPU-seconds per unit at
  the median and 11–20 on average (Module 02 cells, n = 55–153), roughly 90
  times the GPU time per unit. One GPU therefore outpaces dozens of inline
  GEMMA workers; at scale use ``--bslmm separate`` and size the CPU array
  from about 4,000 CPU-hours per million units.
* **No GPU at all** — ``--device cpu`` runs the same code with numba threads
  (``--cpu-threads``); the CPU solver is as fast per path as R's glmnet.
* **Measure.** ``examples/bench_lgv.py`` times loci/s for CPU and GPU on your
  data so the cheapest configuration can be chosen.

SLURM launchers
---------------

``scripts/slurm/lgv_submit.sh [gpu|cpu]`` submits the feature array
(``lgv_gpu.sh`` or ``lgv_cpu.sh``), the optional GEMMA array
(``lgv_bslmm_cpu.sh``) and ``lgv_combine.sh`` chained ``afterany``, and records
job IDs in ``submitted-jobs.tsv``. A cancelled shard is reconciled (and fails
the QC gate); resubmitting it resumes from the rows already written.
