import cudf
import cupy as cp
import pandas as pd
from pandas_plink import read_plink

__all__ = [
    "load_genotypes",
    "load_phenotypes",
    "save_results",
]

def load_phenotypes(pheno_file):
    """
    Reads phenotype (CpG methylation) data into cuDF.
    Rows = samples, columns = CpGs.
    """
    return cudf.read_csv(pheno_file, sep="\t")


def load_genotypes(plink_prefix):
    """
    Reads PLINK genotype data and converts to cuDF DataFrame.
    """
    (bim, fam, bed) = read_plink(plink_prefix)
    geno_df = pd.DataFrame(bed.compute(), columns=bim.snp, index=fam.iid)
    return cudf.from_pandas(geno_df), bim, fam


def save_results(betas, h2_estimates, out_prefix):
    """
    Save betas and h2 estimates to disk.
    """
    betas_df = pd.DataFrame({"beta": cp.asnumpy(betas)})
    h2_df = pd.DataFrame({"iteration": range(len(h2_estimates)),
                          "h2": h2_estimates})

    betas_df.to_csv(f"{out_prefix}_betas.tsv", sep="\t", index=False)
    h2_df.to_csv(f"{out_prefix}_h2.tsv", sep="\t", index=False)
