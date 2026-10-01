Out-of-fold prediction (Module 03)
==================================

``genboostgpu lsp`` is the secondary, translational endpoint: can a region be
predicted from local SNPs in held-out donors? It ports
``03_local_snp_prediction`` under its PI-locked ``config/prediction.yml``:

* run-level donor fold table (5 outer folds × 5 repeats, seeded by
  ``seed_for(run_id, region, repeat_i)``);
* inside each outer split, everything is fit on training donors and applied
  to held-out donors: covariate residualization and scaling, variant
  missingness/MAF filters, mean imputation, genotype scaling;
* a Haseman–Elston screen calibrated with 1000 within-fold permutations
  (one-sided, ``p = (1 + #{null ≥ observed}) / (1 + B)``, α = 0.05);
* inner 5-fold ``cv.glmnet`` (``standardize = FALSE``) over alpha
  {0.1, 0.5, 0.9, 1}, ``lambda.min``;
* the prespecified null prediction (training mean) for held-out donors when a
  fold's screen fails — failed folds are never dropped.

Metrics are computed from pooled OOF predictions: ``r2_pred_oof = 1 −
SSE/SST`` (negative values are kept), ``cor2_oof``, RMSE, MAE, calibration
intercept and slope (undefined when the prediction spread is degenerate),
screening pass frequency, and per-repeat spread. ``combine`` stops if the median
``r2_pred_oof`` exceeds the leakage tripwire (0.5).

Replaying an accepted run reproduces its donor folds and screening decisions
exactly. Per-donor predictions are not bit-for-bit. The in-fold residualized
phenotype already differs from R's at ~1e-14, from BLAS order and TSV
precision. glmnet's ``thresh=1e-7`` convergence then magnifies that difference
in lasso fits over SNPs in near-perfect LD, where the solution is not unique.

On 500 random loci of ``lsp-AA-caudate-20260925-a``, 92% of the 12,300 outer
folds agreed to 1e-6. Of the folds that differed, 90% had alpha = 1 in both
runs. ``r2_pred_oof`` had max \|Δ\| = 6e-4 and Spearman 1.000000, and the
median was identical.

::

   genboostgpu lsp init --run-dir runs/lsp-replay --run-id gbg-lsp-replay \
       --replay <repo>/03_local_snp_prediction/_m/runs/lsp-AA-caudate-20260925-a
   genboostgpu lsp run --run-dir runs/lsp-replay --shard 0/8 --device gpu
   genboostgpu lsp combine --run-dir runs/lsp-replay

``--from-lgv-run <GENBoostGPU lgv run>`` predicts the loci of a Module 02 run
(e.g. CpH regions) with a new run-level fold table.
