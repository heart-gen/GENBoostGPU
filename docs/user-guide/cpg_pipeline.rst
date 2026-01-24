CpG pipeline
============

Overview
--------

This pipeline is designed for large-scale CpG heritability estimation using
GENBoostGPU's chromosome-wise CpG workflow and cross-chromosome hyperparameter
tuning entry points.

Inputs and expected files
-------------------------

The concrete templates and directory layout expected by the CpG pipeline are
defined by ``examples/cpg_test_million.py``:

* CpG manifest template:
  ``data/cpg_manifests/cpg_manifest_chr{chrom}.parquet``
* Phenotype template:
  ``data/phenotypes/pheno_chr{chrom}.parquet``
* Genotype PLINK prefix:
  ``data/genotypes/genotypes``

These templates are exposed directly as the ``--cpg-manifest-template`` and
``--pheno-template`` arguments in ``examples/cpg_test_million.py``.

Preparing inputs from BSseq
---------------------------

In-memory BSseq workflow
~~~~~~~~~~~~~~~~~~~~~~~~

If you already have a ``BSseq`` object in memory, first persist it to disk:

.. code-block:: r

   saveRDS(bs, "data/bsseq.rds")

Helper script command
~~~~~~~~~~~~~~~~~~~~~

Use the helper script to generate per-chromosome manifest and phenotype files:

.. code-block:: bash

   Rscript scripts/prepare_cpg_inputs.R --bsseq data/bsseq.rds --output data

Key flags
~~~~~~~~~

The most important options for large-scale runs are:

* ``--sample-id-col``: Column in ``colData(bs)`` that should be treated as the
  canonical sample identifier.
* ``--validate-fam``: Load the genotype ``.fam`` file and confirm its sample IDs
  match the BSseq sample IDs before writing outputs.
* ``--no-smooth``: Skip smoothing and use the raw methylation proportions.
* ``--min-cov``: Minimum coverage required for a CpG to be retained.
* ``--chromosomes``: Restrict processing to a subset of chromosomes.

Generated directories
~~~~~~~~~~~~~~~~~~~~~

By default, the helper script writes the following directories under the
``--output`` root:

* ``cpg_manifests/``: Per-chromosome CpG manifests.
* ``phenotypes/``: Per-chromosome phenotype matrices aligned to the manifests.

Running the pipeline
--------------------

A minimal end-to-end run looks like:

.. code-block:: bash

   python examples/cpg_test_million.py --geno-path data/genotypes/genotypes

Template overrides for non-default output roots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you wrote outputs somewhere other than ``data/``, override the templates so
that ``{chrom}`` points at the new location:

.. code-block:: bash

   python examples/cpg_test_million.py \
     --geno-path alt_data/genotypes/genotypes \
     --cpg-manifest-template alt_data/cpg_manifests/cpg_manifest_chr{chrom}.parquet \
     --pheno-template alt_data/phenotypes/pheno_chr{chrom}.parquet

Sample alignment guidance
-------------------------

Sample alignment is critical: genotype ``.fam`` IDs must match the BSseq sample
IDs used to build the phenotype matrices. Use ``--validate-fam`` in
``scripts/prepare_cpg_inputs.R`` to catch mismatches early in the preparation
step.

Entry points to search for
--------------------------

To understand or customize the pipeline internals, start with these entry
points:

* ``examples/cpg_test_million.py``
* ``genboostgpu.run_cpgs_by_chromosome``
* ``genboostgpu.global_tune_cpg_params``
