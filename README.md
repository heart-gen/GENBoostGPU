# GENBoostGPU
[![Read the Docs](https://readthedocs.org/projects/genboostgpu/badge/?version=latest)](https://genboostgpu.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/genboostgpu.svg)](https://pypi.org/project/genboostgpu/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/1055676922.svg)](https://doi.org/10.5281/zenodo.17238797)

**Genomic Elastic Net Boosting on GPU (GENBoostGPU)**

GENBoostGPU estimates **local genetic variance of DNA methylation** for regions
(CpG VMRs, CpH regions, tiles) and for millions of individual sites, on GPU or
CPU. Since v0.4 it is the engine behind Module 02 (relative local SNP
contribution score) and Module 03 (out-of-fold local SNP prediction) of the
`dna-methylation-heritability` analysis, and reproduces that pipeline's R
numerics: a line-for-line port of glmnet's elastic-net path solver, R's random
number generator, Haseman–Elston regression, GEMMA BSLMM orchestration, and
the frozen joint model.

---

## What it computes

| Command | Endpoint |
|---|---|
| `genboostgpu lgv` | Module 02 features per region — nested out-of-fold elastic net (`rho2_oof`, `r2_oof`), Haseman–Elston, genotype effective rank and LD, GEMMA BSLMM — then the frozen joint model, domain gate and **within-cell relative score** |
| `genboostgpu lsp` | Module 03 end-to-end out-of-fold prediction with a permutation-calibrated HE screen |
| `genboostgpu sites` | the same features for every CpG/CpH site or tile, with window blocks shared by nearby sites |
| `scripts/build_regions.R` | CpG and CpH (mCA/mCH) region and tile phenotypes from bsseq/HDF5 stores |

The score ranks loci within one cohort × region cell. Absolute locus-level PVE
is not identifiable at these sample sizes; every row carries
`absolute_pve_interpretation_allowed = FALSE`.

## Installation

```bash
pip install "genboostgpu[gpu,plink2]"   # GPU solver + PLINK 2 genotypes
pip install genboostgpu                 # CPU only
pip install "genboostgpu[legacy]"       # v0.3 boosting elastic net (RAPIDS)
```

Python ≥ 3.10. The GPU solver needs an NVIDIA GPU with CUDA 12 (CuPy and
numba-cuda). GEMMA 0.98.5 is required for BSLMM.

## Quick start

```bash
# frozen Module 02 joint model -> portable JSON (once)
Rscript scripts/export_frozen_joint_model.R --model joint-pve-calibrator.rds \
    --sha256 9f26c3273746fda85d9bbf21e224857db9a1ad79a521582a12f241854c03223a --out joint-pve-model.json

# region phenotypes + genome-wide genotypes
genboostgpu lgv init --run-dir runs/cph-caudate --run-id cph-caudate-v1 \
    --regions build/regions.tsv --phenotypes build/phenotypes.parquet \
    --genotypes '/path/LIBD.chr{chrom}.AA' --covariates covs.tsv \
    --numeric-covariates age --factor-covariates sex,diagnosis \
    --cohort AA --region caudate --joint-model joint-pve-model.json \
    --support joint-pve-characterized-support.tsv

GBG_RUN_DIR=runs/cph-caudate GBG_N_SHARDS=8 GBG_ACCOUNT=<account> \
    scripts/slurm/lgv_submit.sh gpu      # shards -> combine (score + QC gate)
```

`genboostgpu lgv init --replay <Module 02 run>` replays an accepted analysis
run with identical tasks, donors and seeds. See the
[user guide](https://genboostgpu.readthedocs.io/en/latest/user-guide/index.html)
for Module 03, the CpG/CpH builders, site-level runs and GPU-cost tuning.

## Lower GPU cost

* One warp per glmnet problem: the 180 paths of a region (and many regions)
  are solved in one CUDA launch; one process per GPU, sharded by SLURM array.
* GEMMA runs on the GPU node's idle CPU cores or in a separate CPU-only array,
  so GPU allocations never wait on MCMC.
* `--device cpu` runs everything with numba threads when GPUs are scarce.
* `examples/bench_lgv.py` measures loci/s per configuration.

---

## Legacy: boosting elastic net (v0.3)

> **Deprecated.** The boosting entry points remain importable and emit a
> `DeprecationWarning`. Their `final_r2` is an in-sample fit and `h2_val`
> reuses the early-stopping split, so neither is an out-of-fold estimate.

### Usage

GENBoostGPU can be used either for large-scale orchestration (many genomic windows across one or more GPUs) or for single-window testing/debugging.  

---

### Example 1: Run a Single Window

The simplest entry point is `run_single_window`, which takes either:
- **File paths** (PLINK genotypes + phenotype file + phenotype ID), or
- **Pre-loaded CuPy arrays** for genotypes and phenotypes.

```python
from genboostgpu.vmr_runner import run_single_window

result = run_single_window(
    chrom=21,
    start=10_000,
    end=510_000,
    geno_path="data/chr21_subset.bed",
    pheno_path="data/phenotypes.tsv",
    pheno_id="pheno_379",
    outdir="results",
    n_iter=50,
    n_trials=10
)

print(result)
````

Output is a Python dictionary, e.g.:

```python
{
  "chrom": 21,
  "start": 10000,
  "end": 510000,
  "num_snps": 742,
  "final_r2": 0.34,
  "h2_unscaled": 0.29,
  "n_iter": 37
}
```

This produces:

* Window-level summary (Python dict)
* Saved results (`.parquet`, betas, heritability estimates) in `results/`

---

### Example 2: Running on VMR Data

```bash
REGION=caudate python examples/vmr_test_caudate.py
```

Script outline (`examples/vmr_test_caudate.py`):

```python
from genboostgpu.orchestration import run_windows_with_dask

df = run_windows_with_dask(
    windows, error_regions=error_regions,
    outdir="results", window_size=500_000,
    n_iter=100, n_trials=20, use_window=True,
    save=True, prefix="vmr"
)
```

This runs boosting elastic net across all VMR-defined windows for the chosen region.

---

### Example 3: Running on Simulated Data

```bash
NUM_SAMPLES=100 python examples/simu_test_100n.py
```

Script outline (`examples/simu_test_100n.py`):

```python
from genboostgpu.orchestration import run_windows_with_dask

df = run_windows_with_dask(
    windows, outdir="results", window_size=500_000,
    n_iter=100, n_trials=10, use_window=False,
    save=True, prefix="simu_100"
)
```

This runs boosting elastic net across synthetic SNP–phenotype pairs for benchmarking.

---

### CpG pipeline (million-scale)

The million-scale CpG pipeline example lives in `examples/cpg_test_million.py`. It expects per-chromosome CpG manifests, per-chromosome phenotype tables, and a PLINK genotype prefix.

### Required directory layout

Match the default templates used by `examples/cpg_test_million.py`:

```text
data/
  cpg_manifests/
    cpg_manifest_chr{chrom}.parquet
  phenotypes/
    pheno_chr{chrom}.parquet
  genotypes/
    <plink_prefix>.bed
    <plink_prefix>.bim
    <plink_prefix>.fam
```

Concretely, the files should look like:

- `data/cpg_manifests/cpg_manifest_chr{chrom}.parquet`
- `data/phenotypes/pheno_chr{chrom}.parquet`
- `data/genotypes/<plink_prefix>.bed/.bim/.fam`

### Prepare CpG inputs from a BSseq object

If your `BSseq` object already exists in memory (for example, as `bs`), save it first:

```r
saveRDS(bs, "data/bsseq.rds")
```

If your sample identifiers live in `pData(bs)$sample_id`, remember that column name for the helper script via `--sample-id-col sample_id`.

Then run the repository helper script:

```bash
Rscript scripts/prepare_cpg_inputs.R --bsseq data/bsseq.rds --output data
```

Useful options:

- `--sample-id-col sample_id` when sample IDs are stored in a specific `pData(bs)` column.
- `--validate-fam data/genotypes/genotypes.fam` to ensure phenotype sample IDs match the PLINK `.fam` file.
- `--no-smooth` if the `BSseq` object is already smoothed or you do not want smoothing.
- `--min-cov 1` sets the median coverage filter (e.g., `1` keeps loci with median coverage ≥ 1).

### Outputs produced by the helper script

The script writes per-chromosome manifests and phenotypes that match the pipeline defaults:

- `data/cpg_manifests/cpg_manifest_chr1.parquet`, etc.
- `data/phenotypes/pheno_chr1.parquet`, etc.

### Run the CpG million-scale example

With the default output layout (`--output data`), you can run:

```bash
python examples/cpg_test_million.py --geno-path data/genotypes/genotypes
```

If you write to a different directory, override the templates:

```bash
python examples/cpg_test_million.py \
  --geno-path data/genotypes/genotypes \
  --cpg-manifest-template data/cpg_inputs/cpg_manifests/cpg_manifest_chr{chrom}.parquet \
  --pheno-template data/cpg_inputs/phenotypes/pheno_chr{chrom}.parquet
```

The defaults in `examples/cpg_test_million.py` assume `data/cpg_manifests/` and `data/phenotypes/`, so either use `--output data` or pass template overrides.

---

### GPU Scaling

* On a single GPU: runs without a Dask cluster.
* On multiple GPUs: `run_windows_with_dask` automatically launches a `LocalCUDACluster` and distributes windows across devices.

---

## Citation

If you use GENBoostGPU in your research, please cite:

> Alexis Bennett and Kynon J.M. Benjamin
> **GENBoostGPU: GPU-accelerated elastic net boosting for large-scale epigenomics**
> DOI: [10.5281/zenodo.17238798](https://doi.org/10.5281/zenodo.17238798)

---

## License

GENBoostGPU is licensed under the **GPL-3.0** license.
See the [LICENSE](LICENSE) file for details.

---


