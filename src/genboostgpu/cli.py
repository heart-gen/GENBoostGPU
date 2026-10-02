"""Command-line interface: ``genboostgpu lgv ...``.

    genboostgpu lgv init     create a run directory (replay, analysis-repo, or generic inputs)
    genboostgpu lgv run      compute feature rows for one shard (GPU or CPU)
    genboostgpu lgv bslmm    GEMMA-only shard for bslmm_mode=separate (CPU job)
    genboostgpu lgv combine  reconcile shards, apply the frozen model, score, QC
    genboostgpu lsp init|run|combine   Module 03 out-of-fold prediction

See ``docs/user-guide/lgv_engine.rst``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

__all__ = ["main"]


def _parse_shard(text: str):
    if not text:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        n = os.environ.get("GENBOOSTGPU_N_SHARDS")
        if env is not None and n is not None:
            return int(env), int(n)
        return 0, 1
    i, n = text.split("/")
    return int(i), int(n)


def _bslmm_settings(a) -> dict:
    """GEMMA settings for run.json; only the OpenBLAS kernel is user-set."""
    from .lgv.bslmm import BslmmSettings

    kwargs = {}
    if a.gemma_blas_coretype is not None:
        kwargs["blas_coretype"] = a.gemma_blas_coretype
    try:
        return BslmmSettings(**kwargs).as_dict()
    except ValueError as err:
        sys.exit(f"--gemma-blas-coretype: {err}")


_CORETYPE_HELP = (
    "OpenBLAS kernel GEMMA runs with, set through OPENBLAS_CORETYPE "
    "(default SkylakeX; 'auto' lets OpenBLAS pick from the CPU). The kernel "
    "changes the rounding of GEMMA's BSLMM, so bslmm_pve moves by up to ~1e-3 "
    "between kernels; SkylakeX reproduces the sealed dna-methylation-heritability "
    "runs. SkylakeX needs an x86-64 CPU with AVX-512 (Intel Skylake-SP and later, "
    "AMD Zen 4 and later); on other CPUs runs stop before any task. Use 'auto' "
    "(or Haswell) there, accepting that bslmm_pve is then not identical to "
    "SkylakeX runs. Recorded in run.json as bslmm.blas_coretype.")


def _cmd_init(a):
    import pandas as pd

    from .lgv.runner import RunConfig, write_run
    from .lgv.tasks import DnamH2Source, read_regions_source
    from .io.phenotype import CovariateSpec

    if a.replay:
        from .adapters.dnam_h2 import read_run_manifest

        src = DnamH2Source.from_module02_run(a.replay, repo_root=a.repo_root)
        man = read_run_manifest(os.path.join(a.replay, "manifest.tsv"))
        tasks = src.tasks.copy()
        if a.task_ids:
            keep = {int(t) for t in a.task_ids.split(",")}
            tasks = tasks[tasks["task_id"].isin(keep)]
        if a.limit:
            tasks = tasks.head(int(a.limit))
        max_outside = 0.10
        thr = os.path.join(a.replay, "config", "thresholds.yml")
        for line in open(thr):
            if line.strip().startswith("max_outside_calibration_domain:"):
                max_outside = float(line.split(":", 1)[1].split("#")[0])
        cfg = RunConfig(
            run_id=a.run_id, cohort=src.cohort, region=src.region,
            source=src.describe(), seed_run_id=a.seed_run_id or man["run_id"],
            bslmm_mode=a.bslmm, bslmm=_bslmm_settings(a),
            joint_model_path=os.path.realpath(a.joint_model),
            joint_model_source_sha256=man.get("joint_model_sha256"),
            joint_model_run_id=man.get("joint_model_run_id", ""),
            support_path=os.path.realpath(
                os.path.join(a.replay, "config", "joint-pve-characterized-support.tsv")),
            support_cell=src.cohort, max_outside_calibration_domain=max_outside,
            smoke_run=bool(a.task_ids or a.limit),
            notes=f"replay of {os.path.realpath(a.replay)}")
        write_run(a.run_dir, cfg, tasks)
    elif a.regions:
        cov = CovariateSpec(path=a.covariates,
                            numeric=[c for c in (a.numeric_covariates or "").split(",") if c],
                            factors=[c for c in (a.factor_covariates or "").split(",") if c])
        src = read_regions_source(
            regions_path=a.regions, phenotype_path=a.phenotypes,
            genotype_pattern=a.genotypes, covariates=cov, cohort=a.cohort,
            region=a.region, region_set_id=a.region_set_id or "",
            min_cis_variants=a.min_cis_variants, window_bp=a.window_bp,
            maf_min=a.maf_min, missing_max=a.missing_max, expected_n=a.expected_n,
            genotype_id_column=a.genotype_id_column)
        cfg = RunConfig(
            run_id=a.run_id, cohort=a.cohort, region=a.region, source=src.describe(),
            seed_run_id=a.seed_run_id, bslmm_mode=a.bslmm, bslmm=_bslmm_settings(a),
            joint_model_path=os.path.realpath(a.joint_model) if a.joint_model else None,
            joint_model_source_sha256=a.joint_model_sha256,
            support_path=os.path.realpath(a.support) if a.support else None,
            support_cell=a.support_cell or a.cohort,
            max_outside_calibration_domain=a.max_outside, smoke_run=a.smoke)
        write_run(a.run_dir, cfg, src.tasks)
    elif a.dnam_upstream:
        tasks = pd.read_csv(a.tasks, sep="\t", dtype={"chrom": str})
        src = DnamH2Source(
            vmr_run_dir=a.dnam_upstream, cohort=a.cohort, region=a.region, tasks=tasks,
            min_cis_variants=a.min_cis_variants, expected_n=a.expected_n,
            estimation_group=a.estimation_group, catalog_cohort=a.catalog_cohort,
            covar_prefix=a.covar_prefix,
            upstream_vmr_run_id=os.path.basename(os.path.normpath(a.dnam_upstream)))
        cfg = RunConfig(
            run_id=a.run_id, cohort=a.cohort, region=a.region, source=src.describe(),
            seed_run_id=a.seed_run_id, bslmm_mode=a.bslmm, bslmm=_bslmm_settings(a),
            joint_model_path=os.path.realpath(a.joint_model) if a.joint_model else None,
            joint_model_source_sha256=a.joint_model_sha256,
            support_path=os.path.realpath(a.support) if a.support else None,
            support_cell=a.support_cell or a.cohort,
            max_outside_calibration_domain=a.max_outside, smoke_run=a.smoke)
        write_run(a.run_dir, cfg, tasks)
    else:
        sys.exit("init needs one of --replay, --dnam-upstream or --regions")
    print(f"Initialized {a.run_dir}")


def _cmd_run(a):
    from .lgv.runner import run_shard

    shard, n = _parse_shard(a.shard)
    run_shard(a.run_dir, shard, n, device=a.device, batch_loci=a.batch_loci,
              prep_threads=a.prep_threads, cpu_threads=a.cpu_threads,
              gemma_workers=a.gemma_workers, keep_bslmm_work=a.keep_bslmm_work)


def _cmd_bslmm(a):
    from .lgv.runner import run_bslmm_shard

    shard, n = _parse_shard(a.shard)
    run_bslmm_shard(a.run_dir, shard, n, gemma_workers=a.gemma_workers)


def _cmd_combine(a):
    from .lgv.combine import combine_run

    man = combine_run(a.run_dir, write_r_task_rows=a.write_r_task_rows,
                      score=not a.no_score)
    print(json.dumps({k: man[k] for k in ("run_id", "reconciliation", "decision")
                      if k in man}, indent=2, default=str))


def _cmd_lsp_init(a):
    import pandas as pd

    from .adapters.dnam_h2 import read_run_manifest
    from .lgv.tasks import DnamH2Source
    from .lsp.oof import LspSettings
    from .lsp.runner import init_lsp_run

    settings = LspSettings(n_permutations=a.n_permutations)
    if a.replay:
        man = read_run_manifest(os.path.join(a.replay, "manifest.tsv"))
        repo = a.repo_root or os.path.realpath(os.path.join(a.replay, "..", "..", "..", ".."))
        module = man.get("upstream_vmr_module") or "01_vmr_catalog"
        vmr_run_dir = os.path.join(repo, module, "_m", "runs", man["upstream_vmr_catalog_run_id"])
        tasks = pd.read_csv(os.path.join(a.replay, "task-manifest.tsv"), sep="\t",
                            dtype={"chrom": str})
        if a.task_ids:
            tasks = tasks[tasks["task_id"].isin({int(t) for t in a.task_ids.split(",")})]
        if a.limit:
            tasks = tasks.head(a.limit)
        tasks = tasks.assign(cohort=man["cohort"], region=man["region"])
        src = DnamH2Source(vmr_run_dir=vmr_run_dir, cohort=man["cohort"],
                           region=man["region"], tasks=tasks, min_cis_variants=100,
                           expected_n=int(man["n_donors"]),
                           estimation_group=man.get("estimation_group") or None,
                           catalog_cohort=man.get("catalog_cohort") or None,
                           covar_prefix=man.get("covar_prefix") or None,
                           upstream_vmr_run_id=man["upstream_vmr_catalog_run_id"],
                           apply_snp_qc=False)
        folds = pd.read_csv(os.path.join(a.replay, "donor-folds.tsv"), sep="\t",
                            dtype={"donor": str})
        init_lsp_run(a.run_dir, a.run_id, man["cohort"], man["region"], src, tasks,
                     settings, seed_run_id=a.seed_run_id or man["run_id"], folds=folds,
                     smoke_run=bool(a.task_ids or a.limit),
                     notes=f"replay of {os.path.realpath(a.replay)}")
    elif a.from_lgv_run:
        from .lgv.runner import load_run
        from .lgv.tasks import source_from_config

        cfg, tasks = load_run(a.from_lgv_run)
        src_cfg = dict(cfg.source, apply_snp_qc=False)
        src = source_from_config(src_cfg, tasks)
        init_lsp_run(a.run_dir, a.run_id, cfg.cohort, cfg.region, src, tasks, settings,
                     seed_run_id=a.seed_run_id, smoke_run=a.smoke,
                     notes=f"loci of GENBoostGPU run {os.path.realpath(a.from_lgv_run)}")
    else:
        sys.exit("lsp init needs --replay or --from-lgv-run")
    print(f"Initialized {a.run_dir}")


def _cmd_lsp_run(a):
    from .lsp.runner import run_lsp_shard

    shard, n = _parse_shard(a.shard)
    run_lsp_shard(a.run_dir, shard, n, device=a.device, batch_loci=a.batch_loci,
                  prep_threads=a.prep_threads, cpu_threads=a.cpu_threads)


def _cmd_lsp_combine(a):
    from .lsp.runner import combine_lsp_run

    print(json.dumps(combine_lsp_run(a.run_dir), indent=2, default=str))


def _cmd_sites_init(a):
    from .io.phenotype import CovariateSpec
    from .sites.runner import init_sites_run

    cov = CovariateSpec(path=a.covariates,
                        numeric=[c for c in (a.numeric_covariates or "").split(",") if c],
                        factors=[c for c in (a.factor_covariates or "").split(",") if c])
    init_sites_run(a.run_dir, a.run_id, a.cohort, a.region, a.units_dir, a.genotypes, cov,
                   features=tuple(a.features.split(",")), window_bp=a.window_bp,
                   window_block_bp=a.window_block_bp, maf_min=a.maf_min,
                   missing_max=a.missing_max, min_cis_variants=a.min_cis_variants,
                   genotype_id_column=a.genotype_id_column, unit_set_id=a.unit_set_id or "",
                   chroms=a.chroms.split(",") if a.chroms else None, bslmm_mode=a.bslmm,
                   seed_run_id=a.seed_run_id,
                   joint_model_path=os.path.realpath(a.joint_model) if a.joint_model else None,
                   joint_model_sha256=a.joint_model_sha256,
                   support_path=os.path.realpath(a.support) if a.support else None,
                   support_cell=a.support_cell, smoke_run=a.smoke,
                   bslmm_settings=_bslmm_settings(a))
    print(f"Initialized {a.run_dir}")


def _cmd_sites_run(a):
    from .sites.runner import run_sites_shard

    shard, n = _parse_shard(a.shard)
    run_sites_shard(a.run_dir, shard, n, device=a.device, batch_units=a.batch_units,
                    cpu_threads=a.cpu_threads, gemma_workers=a.gemma_workers)


def _cmd_regions_merge(a):
    from .io.regions_build import merge_build

    print(json.dumps(merge_build(a.build_dir, a.out_prefix), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="genboostgpu")
    sub = p.add_subparsers(dest="command", required=True)
    lgv = sub.add_parser("lgv", help="Module 02 local genetic variance engine")
    lsub = lgv.add_subparsers(dest="lgv_command", required=True)

    i = lsub.add_parser("init", help="create a run directory")
    i.add_argument("--run-dir", required=True)
    i.add_argument("--run-id", required=True)
    i.add_argument("--seed-run-id", help="run id hashed into seeds (default: run id; "
                   "replays default to the replayed run's id)")
    i.add_argument("--bslmm", choices=["inline", "separate", "off"], default="inline")
    i.add_argument("--gemma-blas-coretype", metavar="KERNEL", help=_CORETYPE_HELP)
    i.add_argument("--joint-model", help="frozen model JSON from "
                   "scripts/export_frozen_joint_model.R")
    i.add_argument("--joint-model-sha256", help="pin the source .rds sha256")
    i.add_argument("--support", help="joint-pve-characterized-support.tsv")
    i.add_argument("--support-cell")
    i.add_argument("--max-outside", type=float, default=0.10)
    i.add_argument("--smoke", action="store_true")
    g = i.add_argument_group("replay an analysis-repo Module 02 run")
    g.add_argument("--replay", help="Module 02 run directory (_m/runs/<run_id>)")
    g.add_argument("--repo-root")
    g.add_argument("--task-ids", help="comma-separated subset (marks a smoke run)")
    g.add_argument("--limit", type=int)
    g = i.add_argument_group("analysis-repo inputs (Module 01 run or 01b cell)")
    g.add_argument("--dnam-upstream")
    g.add_argument("--tasks", help="task table (task_id, chrom, start, end, vmr_id, ...)")
    g.add_argument("--cohort")
    g.add_argument("--region")
    g.add_argument("--estimation-group")
    g.add_argument("--catalog-cohort")
    g.add_argument("--covar-prefix")
    g = i.add_argument_group("generic region inputs")
    g.add_argument("--regions")
    g.add_argument("--phenotypes")
    g.add_argument("--genotypes", help="PLINK prefix pattern with {chrom}")
    g.add_argument("--covariates")
    g.add_argument("--numeric-covariates")
    g.add_argument("--factor-covariates")
    g.add_argument("--region-set-id")
    g.add_argument("--genotype-id-column", default="IID")
    g.add_argument("--window-bp", type=int, default=500_000)
    g.add_argument("--maf-min", type=float, default=0.05)
    g.add_argument("--missing-max", type=float, default=0.05)
    i.add_argument("--min-cis-variants", type=int, default=100)
    i.add_argument("--expected-n", type=int)
    i.set_defaults(func=_cmd_init)

    r = lsub.add_parser("run", help="compute one shard")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--shard", default="", help="i/N (default: SLURM_ARRAY_TASK_ID/"
                   "GENBOOSTGPU_N_SHARDS, else 0/1)")
    r.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    r.add_argument("--batch-loci", type=int, default=16)
    r.add_argument("--prep-threads", type=int, default=4)
    r.add_argument("--cpu-threads", type=int, default=1,
                   help="glmnet threads when --device cpu")
    r.add_argument("--gemma-workers", type=int)
    r.add_argument("--keep-bslmm-work", action="store_true")
    r.set_defaults(func=_cmd_run)

    b = lsub.add_parser("bslmm", help="GEMMA-only shard (bslmm_mode=separate)")
    b.add_argument("--run-dir", required=True)
    b.add_argument("--shard", default="")
    b.add_argument("--gemma-workers", type=int)
    b.set_defaults(func=_cmd_bslmm)

    c = lsub.add_parser("combine", help="reconcile, score and check a run")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--write-r-task-rows", action="store_true")
    c.add_argument("--no-score", action="store_true")
    c.set_defaults(func=_cmd_combine)

    lsp = sub.add_parser("lsp", help="Module 03 out-of-fold local SNP prediction")
    ssub = lsp.add_subparsers(dest="lsp_command", required=True)
    li = ssub.add_parser("init", help="create a Module 03 run directory")
    li.add_argument("--run-dir", required=True)
    li.add_argument("--run-id", required=True)
    li.add_argument("--seed-run-id")
    li.add_argument("--replay", help="03_local_snp_prediction run directory")
    li.add_argument("--from-lgv-run", help="GENBoostGPU lgv run whose loci to predict")
    li.add_argument("--repo-root")
    li.add_argument("--task-ids")
    li.add_argument("--limit", type=int)
    li.add_argument("--smoke", action="store_true")
    li.add_argument("--n-permutations", type=int, default=1000)
    li.set_defaults(func=_cmd_lsp_init)
    lr = ssub.add_parser("run", help="compute one shard")
    lr.add_argument("--run-dir", required=True)
    lr.add_argument("--shard", default="")
    lr.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    lr.add_argument("--batch-loci", type=int, default=8)
    lr.add_argument("--prep-threads", type=int, default=4)
    lr.add_argument("--cpu-threads", type=int, default=1)
    lr.set_defaults(func=_cmd_lsp_run)
    lc = ssub.add_parser("combine", help="per-locus metrics and run QC")
    lc.add_argument("--run-dir", required=True)
    lc.set_defaults(func=_cmd_lsp_combine)

    st = sub.add_parser("sites", help="site-level (million-scale) features")
    ssub2 = st.add_subparsers(dest="sites_command", required=True)
    si = ssub2.add_parser("init", help="create a site-level run directory")
    si.add_argument("--run-dir", required=True)
    si.add_argument("--run-id", required=True)
    si.add_argument("--seed-run-id")
    si.add_argument("--units-dir", required=True, help="build_regions.R <out>/units")
    si.add_argument("--genotypes", required=True, help="PLINK prefix pattern with {chrom}")
    si.add_argument("--covariates")
    si.add_argument("--numeric-covariates")
    si.add_argument("--factor-covariates")
    si.add_argument("--cohort", required=True)
    si.add_argument("--region", required=True)
    si.add_argument("--features", default="geometry,he,en")
    si.add_argument("--window-bp", type=int, default=500_000)
    si.add_argument("--window-block-bp", type=int, default=0)
    si.add_argument("--maf-min", type=float, default=0.05)
    si.add_argument("--missing-max", type=float, default=0.05)
    si.add_argument("--min-cis-variants", type=int, default=100)
    si.add_argument("--genotype-id-column", default="IID")
    si.add_argument("--unit-set-id")
    si.add_argument("--chroms")
    si.add_argument("--bslmm", choices=["inline", "off"], default="off")
    si.add_argument("--gemma-blas-coretype", metavar="KERNEL", help=_CORETYPE_HELP)
    si.add_argument("--joint-model")
    si.add_argument("--joint-model-sha256")
    si.add_argument("--support")
    si.add_argument("--support-cell")
    si.add_argument("--smoke", action="store_true")
    si.set_defaults(func=_cmd_sites_init)
    sr = ssub2.add_parser("run", help="compute one shard")
    sr.add_argument("--run-dir", required=True)
    sr.add_argument("--shard", default="")
    sr.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    sr.add_argument("--batch-units", type=int, default=64)
    sr.add_argument("--cpu-threads", type=int, default=1)
    sr.add_argument("--gemma-workers", type=int)
    sr.set_defaults(func=_cmd_sites_run)

    reg = sub.add_parser("regions", help="region builder utilities")
    rsub = reg.add_subparsers(dest="regions_command", required=True)
    rm = rsub.add_parser("merge", help="collect scripts/build_regions.R outputs")
    rm.add_argument("--build-dir", required=True)
    rm.add_argument("--out-prefix")
    rm.set_defaults(func=_cmd_regions_merge)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
