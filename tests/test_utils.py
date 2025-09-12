import os
import tempfile
import cupy as cp
import pandas as pd
import cudf
import pytest
from genboostgpu import data_io

def test_load_phenotypes_reads_file(tmp_path):
    file = tmp_path / "pheno.tsv"
    pd.DataFrame({"sample":["a","b"],"cpg1":[0.1,0.2]}).to_csv(file, sep="\t", index=False)
    df = data_io.load_phenotypes(str(file))
    assert isinstance(df, cudf.DataFrame)
    assert "cpg1" in df.columns

def test_save_results_writes_files(tmp_path):
    betas = cp.array([0.1, 0.2, 0.3])
    h2 = [0.01, 0.02]
    out_prefix = str(tmp_path / "results")
    data_io.save_results(betas, h2, out_prefix)
    assert os.path.exists(f"{out_prefix}_betas.tsv")
    assert os.path.exists(f"{out_prefix}_h2.tsv")


