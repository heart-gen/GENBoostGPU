import cudf
import cupy as cp
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


def save_results(betas, h2_estimates, out_prefix, snp_ids=None, meta=None):
    """
    Save betas and h2 estimates to disk.
    """
    betas_np = cp.asnumpy(betas)

    betas_df = pd.DataFrame({
        "snp": snp_ids if snp_ids is not None else range(len(betas_np)),
        "beta": betas_np
    })

    h2_df = pd.DataFrame({
        "iteration": range(len(h2_estimates)),
        "h2": h2_estimates
    })

    if meta:
        for k, v in meta.items():
            betas_df[k] = v
            h2_df[k] = v

    # per-VMR outputs
    betas_df.to_csv(f"{out_prefix}_betas.tsv", sep="\t", index=False)
    h2_df.to_csv(f"{out_prefix}_h2.tsv", sep="\t", index=False)
