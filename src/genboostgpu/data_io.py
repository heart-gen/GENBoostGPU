import cudf
import cupy as cp
import numpy as np
import pandas as pd
from pandas_plink import read_plink

__all__ = [
    "load_genotypes",
    "load_phenotypes",
    "save_results",
]

def load_phenotypes(pheno_file, header=True):
    """
    Reads phenotype (CpG methylation) data into cuDF.
    Rows = samples, columns = CpGs.
    """
    if header:
        return cudf.read_csv(pheno_file, sep="\t", header=0)
    else:
        return cudf.read_csv(pheno_file, sep="\t", header=None)


def load_genotypes(plink_prefix, dtype="float32"):
    """
    Reads PLINK genotype data and converts to CuPy.
    """
    (bim, fam, bed) = read_plink(plink_prefix)
    geno = bed.compute().astype(dtype)
    return cp.asarray(geno).T, bim, fam


def save_results(betas, h2_estimates, out_prefix, snp_ids=None, meta=None, 
                 zero_tol: float = 0.0):
    """
    Save only non-zero betas (|beta| > zero_tol) and all h2 estimates to disk.
    """
    betas_np = cp.asnumpy(betas).ravel()

    # Mask non-zero betas (with tolerance)
    nz_mask = (np.abs(betas_np) > float(zero_tol))
    nz_idx = np.nonzero(nz_mask)[0]

    # SNP IDs aligned to mask
    if snp_ids is None:
        snp_ids_arr = np.arange(betas_np.size)
    else:
        snp_ids_arr = np.asarray(snp_ids)

    betas_df = pd.DataFrame({
        "snp": snp_ids_arr[nz_mask],
        "beta": betas_np[nz_mask],
    })

    # h2 estimates
    try:
        h2_np = cp.asnumpy(h2_estimates)
    except Exception:
        h2_np = np.asarray(h2_estimates)

    h2_df = pd.DataFrame({
        "iteration": np.arange(h2_np.size),
        "h2": h2_np,
    })

    # Add metadata if provided
    if meta:
        for k, v in meta.items():
            betas_df[k] = v
            h2_df[k] = v

    # Write TSVs
    betas_df.to_csv(f"{out_prefix}_betas.tsv", sep="\t", index=False)
    h2_df.to_csv(f"{out_prefix}_h2.tsv", sep="\t", index=False)
