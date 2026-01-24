# GENBoostGPU
[![Read the Docs](https://readthedocs.org/projects/genboostgpu/badge/?version=latest)](https://genboostgpu.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/genboostgpu.svg)](https://pypi.org/project/genboostgpu/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/1055676922.svg)](https://doi.org/10.5281/zenodo.17238797)

**Genomic Elastic Net Boosting on GPU (GENBoostGPU)**

GENBoostGPU provides a scalable framework for running elastic net regression with 
boosting across thousands of CpG sites, leveraging GPU acceleration with [RAPIDS cuML](https://rapids.ai), 
[CuPy](https://cupy.dev), and [cuDF](https://docs.rapids.ai/api/cudf/stable/).  
It supports SNP preprocessing, cis-window filtering, LD clumping, missing data 
imputation, and phenotype integration — all optimized for large-scale epigenomics.

---

## Features

- **Window-based orchestration**:
  - `run_windows_with_dask` coordinates execution across one or more GPUs using Dask.
  - Handles batch scheduling of thousands of genomic windows.
- **Single-window analysis**:
  - `run_single_window` executes boosting elastic net on one genomic region.
  - Accepts pre-loaded arrays (CuPy) or file paths (PLINK, phenotype tables).
- **GPU-accelerated boosting elastic net**:
  - Iterative boosting with cuML ElasticNet and final Ridge refit.
  - Early stopping based on stability of variance explained.
- **Automated SNP preprocessing**:
  - Zero-variance SNP filtering
  - Missing genotype imputation
  - LD clumping (PLINK-like) with CuPy
  - Cis-window SNP filtering
- **Hyperparameter optimization**:
  - Optuna-based tuning of ElasticNet (`alpha`, `l1_ratio`)
  - Ridge regression tuning with delayed evaluation
  - Optional manual cross-validation for custom grids
- **Scalability**:
  - Dask orchestration for multiple GPUs (`LocalCUDACluster`)
  - Single-GPU fallback for smaller jobs
- **Flexible outputs**:
  - SNP betas, heritability estimates, variance explained
  - Window-level summary tables (`.parquet`)
  - Intermediate ridge/elastic net models for reproducibility

---

## Installation

GENBoostGPU is available on [PyPI](https://pypi.org/project/genboostgpu/).  
It requires Python ≥3.10 and an NVIDIA GPU with CUDA 12.x.

```bash
pip install genboostgpu
````

For development (from source):

```bash
git clone https://github.com/heart-gen/GENBoostGPU.git
cd GENBoostGPU
poetry install
```

---

## Usage

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

## CpG pipeline (million-scale)

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


