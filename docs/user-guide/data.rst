Data & formats
==============

GENBoostGPU accepts genomic inputs from a handful of common formats. Regardless
of the source, data are converted to GPU-friendly representations before hitting
:mod:`genboostgpu.enet_boosting`.

Supported inputs
----------------

PLINK (BED/BIM/FAM)
   Use :func:`genboostgpu.data_io.load_genotypes`, which wraps
   :mod:`pandas_plink`. Genotypes are returned as a CuPy array with samples on
   rows and SNPs on columns, plus the accompanying BIM/FAM tables (pandas or
   cuDF).
CuPy / NumPy arrays
   When you already hold genotypes in memory, pass the CuPy array directly via
   ``geno_arr``. NumPy arrays are accepted but will be copied to the GPU with
   :func:`cupy.asarray`.
Parquet/TSV phenotypes
   Phenotypes are typically stored as tab-separated files per CpG or VMR. Use
   :func:`genboostgpu.data_io.load_phenotypes` to read them into a cuDF DataFrame.
   Results written by :func:`genboostgpu.data_io.save_results` are parquet or TSV
   files with betas, variance explained, and metadata.

   If you are preparing large-scale CpG inputs, see :doc:`cpg_pipeline` for the
   recommended workflow. It documents the helper script
   ``scripts/prepare_cpg_inputs.R`` and the manifest/phenotype templates used by
   ``examples/cpg_test_million.py``.

Sample and variant alignment
----------------------------

* Samples must be in the same order across genotype and phenotype matrices.
  ``load_genotypes`` retains the FAM order and ``load_phenotypes`` preserves file
  order, so alignments are deterministic.
* The BIM table needs ``chrom``, ``pos``, and ``snp`` columns. It can be a pandas
  DataFrame or cuDF DataFrame. The :func:`genboostgpu.snp_processing.filter_cis_window`
  helper relies on those names to pull the right SNP indices.
* When providing preloaded arrays, supply the matching BIM metadata so cis-window
  filtering can locate positions.

Handling missingness and QC
---------------------------

:mod:`genboostgpu.snp_processing` standardises preprocessing so the models see a
clean matrix:

* ``filter_zero_variance`` removes monomorphic SNPs (default threshold ``1e-8``).
* ``impute_snps`` wraps ``cuml.preprocessing.SimpleImputer``. The default strategy
  (``most_frequent``) is well suited for hard-call genotypes; use ``mean`` for
  dosage-style data.
* ``run_ld_clumping`` performs phenotype-informed LD pruning using Pearson
  correlations from ``_corr_with_y_streaming``.

Missing CpG values are handled when reading the phenotype table—the values are
normalised in :func:`genboostgpu.vmr_runner.run_single_window`. If you need
custom logic (e.g., dropping low-quality samples), perform it before invoking the
pipeline and pass the cleaned CuPy vector via ``pheno``.
