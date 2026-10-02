Local genetic variance engine (Module 02)
=========================================

``genboostgpu lgv`` computes, for every region (VMR, CpH region, tile), the
features of the frozen Module 02 joint model of the
``dna-methylation-heritability`` analysis, applies that model, and derives the
within-cell **relative local SNP contribution score**. It replaces the R array
job of Module 02 Stage 01 and reproduces Stages 02–05, writing tables with the
R pipeline's file names and columns so a run can still be sealed by the R
Stage 06.

.. important::

   The score is a *relative* ranking within one cohort × region cell. Absolute
   locus-level PVE is not identifiable at these sample sizes
   (``PASS_RELATIVE_GENETIC_CONTROL / FAIL_ABSOLUTE_LOCUS_PVE``); every output
   row carries ``absolute_pve_interpretation_allowed = FALSE``.

What is computed per region
---------------------------

========================  =====================================================
Feature                   Definition (identical to the R pipeline)
========================  =====================================================
``rho2_oof``, ``r2_oof``  Nested out-of-fold elastic net: 5 outer folds × 2
                          repeats, inner 5-fold ``cv.glmnet``, alpha ∈ {0.1,
                          0.5, 1}, ``lambda.1se``, top-1500 marginal screen,
                          imputation/residualization/screening fit on training
                          donors only.
``he_h2``, ``he_se``      Haseman–Elston regression of pairwise residual
                          products on GRM relatedness.
``p_eff``                 Effective rank ``tr(K)^2 / tr(K^2)`` of the
                          standardized-genotype GRM.
``ld_metric``             Median adjacent-SNP r² over ≤200 evenly spaced pairs.
``bslmm_pve``             GEMMA 0.98.5 BSLMM (10k burn-in, 100k sampling,
                          rpace 10) on mean-imputed dosages and
                          covariate-residualized phenotype.
========================  =====================================================

Fidelity to R
-------------

* **glmnet** — the coordinate-descent path solver of glmnet 4.1-10 (covariance
  and naive modes, strong rules, KKT checks, ``fix.lam``) is ported line for
  line, including the summation order of Eigen's vectorized dot products
  (SSE2, as in the conda-forge/CRAN x86-64 builds). On the CPU, paths,
  intercepts and deviance ratios are **bitwise identical** to R with identical
  pass counts. ``cv.glmnet`` reproduces fold fits on their own lambda paths
  and ``lambda.interp``; ``cvm``/``cvsd`` agree to ~1e-15 (R's predictions go
  through its BLAS) and ``lambda.min``/``lambda.1se`` are R's.
* **GPU** — the warp-per-problem kernel runs the same algorithm but reduces
  across lanes and lets CUDA fuse multiply-adds, so it agrees with the CPU path
  to rounding (~1e-15 per step). Like any rounding difference, that can
  occasionally resolve a near-tied convergence or ``lambda.1se`` decision
  differently. Use ``--device cpu`` when a bit-exact replay is the goal.
* **Random numbers** — R's Mersenne-Twister seeding and ``sample()`` rejection
  sampling are ported bit for bit, so fold assignments (and Module 03
  permutations) equal R's without exporting anything.
* **Inputs** — the analysis-repo adapter reads per-VMR BEDs with bigsnpr's
  allele coding and orders donors exactly as R's ``merge()``; genotype
  matrices are bitwise R's.
* **SNP screen** — the top-1500 marginal-correlation screen sums in a fixed
  order, so SNPs with identical training genotypes (common: dozens of copies
  in strong-LD windows) get identical scores and keep their column order, on
  any CPU, GPU or thread count. R computes the same scores with its BLAS
  (OpenBLAS ``dgemv``), whose last-bit noise decides the order of such ties.
* **What R itself does not reproduce** — R's covariate residualization
  (``lm.fit``) and screen go through OpenBLAS, whose kernels depend on the CPU.
  Re-running R on identical inputs with ``OPENBLAS_CORETYPE=Haswell`` instead
  of ``SkylakeX`` changes ``rho2_oof`` of some VMRs by up to ~5e-4, because a
  1e-15 change in the adjusted phenotype flips a near-tied ``lambda.1se``.
  Replays therefore match sealed runs exactly for HE, ``p_eff``, LD and
  statuses, and match the nested-EN features within that CPU-to-CPU envelope
  (see below).
* **BSLMM** — GEMMA's ``h`` chain is reproduced exactly; its reported ``pve``
  additionally depends on the OpenBLAS kernel GEMMA ran with. The GEMMA binary
  links OpenBLAS 0.3.9 built with ``DYNAMIC_ARCH``, which picks a kernel from
  the CPU: ``SkylakeX`` on quest10 nodes, where the sealed R runs ran, but the
  generic ``Prescott`` fallback on quest13 (Xeon 8592+, a CPU 0.3.9 does not
  know). The two differ by up to 1.3e-3 in ``bslmm_pve``. GENBoostGPU pins the
  kernel with ``OPENBLAS_CORETYPE`` (``bslmm.blas_coretype`` in ``run.json``,
  default ``SkylakeX``) and refuses to start GEMMA on a CPU without AVX-512;
  set it to ``auto`` (or ``Haswell``) to run on such nodes, at the cost of
  comparability with sealed runs. With the pin, ``bslmm_pve`` matches the
  sealed ``lgv-all_individuals.EA-caudate-20260917`` values to 1e-16 on both
  node types.

Replay of an accepted run
-------------------------

500 random tasks of ``lgv-all_individuals.EA-caudate-20260917`` (n = 129;
489 completed, 11 ``qc_failed``), replayed on CPU nodes and compared row by row
with the sealed tables:

========================================  =====================================
Statuses, HE, ``p_eff``, LD               identical (≤ 1e-13)
``bslmm_pve`` (GEMMA, SkylakeX kernel)    identical (≤ 1e-16)
``rho2_oof``                              62 % bitwise; 3.5 % differ > 1e-4;
                                          max 4.6e-3; Spearman 0.999998
``pve_cis_joint_unbounded``               57 % bitwise; 2.7 % differ > 1e-4;
                                          max 1.6e-3; Spearman 0.9999999
``local_snp_contribution_score``          replayed rows inserted in the full
                                          cell (11,249 VMRs): max difference
                                          0.0033, Spearman 0.99999999, no
                                          quartile changes
========================================  =====================================

For the 106 VMRs where GENBoostGPU and the sealed run disagree, plus 20 that
agree, R itself was re-run on identical inputs with a different OpenBLAS
kernel: R vs sealed R differs > 1e-4 in 17 VMRs (max 2.0e-3), GENBoostGPU vs
sealed R in 17 VMRs (max 4.6e-3). The differences are the size of R's own
CPU-to-CPU variation; the slightly longer tail comes from the order of tied
SNPs in the screen.

Running on the analysis repository
----------------------------------

Export the frozen model once (R, no JSON package needed)::

   Rscript scripts/export_frozen_joint_model.R \
       --model <repo>/02_local_genetic_variance/_m/runs/lgv-joint-pve-train-20260820/combined/joint-pve-calibrator.rds \
       --sha256 9f26c3273746fda85d9bbf21e224857db9a1ad79a521582a12f241854c03223a \
       --out joint-pve-model.json

Replay an accepted Module 02 run (same tasks, donors, seeds and support)::

   genboostgpu lgv init --run-dir runs/replay-AA-caudate --run-id gbg-replay-AA-caudate \
       --replay <repo>/02_local_genetic_variance/_m/runs/lgv-AA-caudate-rescore-20260913 \
       --joint-model joint-pve-model.json
   GBG_RUN_DIR=runs/replay-AA-caudate GBG_N_SHARDS=8 GBG_ACCOUNT=p32505 \
       scripts/slurm/lgv_submit.sh gpu

``--task-ids`` or ``--limit`` make a smoke run (Stage 05 then returns
``PASS_SMOKE_ONLY_NOT_ACCEPTABLE``). A new run on an accepted Module 01 run or
01b estimation cell uses ``--dnam-upstream <run dir> --tasks <task table>
--cohort --region`` plus ``--support`` and ``--joint-model``.

Generic regions (CpG/CpH regions, tiles)
----------------------------------------

::

   genboostgpu lgv init --run-dir runs/cph-caudate --run-id cph-caudate-v1 \
       --regions build/regions.tsv --phenotypes build/phenotypes.parquet \
       --genotypes '/path/LIBD.chr{chrom}.AA' --covariates covs.tsv \
       --numeric-covariates age,snpPC1,snpPC2,snpPC3 --factor-covariates sex,diagnosis \
       --cohort AA --region caudate --bslmm inline \
       --joint-model joint-pve-model.json --support joint-pve-characterized-support.tsv

``--genotypes`` is either a per-chromosome pattern with ``{chrom}`` or one
genome-wide prefix (e.g. ``inputs/genotypes/TOPMed_LIBD.AA``); ``.bed`` and
``.pgen`` (``pip install genboostgpu[plink2]``) are detected. From a
genome-wide file only the current chromosome's variants are read and held as
int8 (chr21 of the TOPMed AA ``.pgen``: 212k variants × 526 donors, 20 s).
``--genotype-id-column`` picks the ``.fam``/``.psam`` column matched to
``sample_id`` (``FID`` when it holds BrNums). Donors are matched by ID across phenotype, covariate
and genotype tables; variant QC (MAF ≥ 0.05, missingness ≤ 0.05) is computed
on the analysis donors.

.. warning::

   The frozen model is characterized only for the cells and sample sizes in
   the support table (``allowed_n``). Regions of a new cohort or tissue are
   scored only where the domain gate passes; anything else is reported as
   ``outside_domain`` rather than ranked.

Run directory
-------------

``run.json`` (all settings, seeds policy, software versions), ``tasks.tsv``,
``task_rows/part-*.parquet`` (one terminal row per task, append-only, so a
killed shard resumes), and after ``combine``: ``results/combined/
observed-joint-features.tsv``, ``task-reconciliation.tsv``,
``observed-joint-estimates.tsv``, ``local-genetic-control-{cell}-{region}-vmrs.tsv``,
``observed-score-qc.tsv`` and ``results/run-manifest.json`` (decision, input
and output SHA-256s). ``--write-r-task-rows`` also writes
``results/task_rows/vmr-%07d.tsv`` for R Stage 02.

BSLMM placement
---------------

``--bslmm inline`` runs GEMMA chains in a CPU process pool on the GPU node
while the GPU solves the elastic net; ``--bslmm separate`` defers them to a
CPU-only array (``genboostgpu lgv bslmm``) so GPU allocations never wait on
MCMC; ``--bslmm off`` produces features only (``terminal_status =
features_only``) and no score.
