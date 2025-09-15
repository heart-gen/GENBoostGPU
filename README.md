# GENBoostGPU

**Genomic Elastic Net Boosting on GPU (GENBoostGPU)**

GENBoostGPU provides a scalable framework for running elastic net regression with 
boosting across thousands of CpG sites, leveraging GPU acceleration with RAPIDS cuML, 
CuPy, and cuDF. It supports SNP preprocessing, cis-window filtering, LD clumping, 
missing data imputation, and phenotype integration — all optimized for large-scale 
epigenomics.

---

## Features
- GPU-accelerated **elastic net regression** with optional boosting
- SNP-level preprocessing:
  - Zero-variance SNP filtering
  - Missing genotype imputation
  - LD clumping (PLINK-like) on GPU
- Cis-window filtering for CpGs
- Integration of genotype (PLINK) and phenotype (CpG/VMR methylation) data
- Batch execution across **thousands of CpGs on a single GPU**
- Flexible output: betas, heritability estimates, cross-validation results

---

