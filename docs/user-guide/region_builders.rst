CpG and CpH region builders
===========================

``scripts/build_regions.R`` turns bsseq / HDF5-backed SummarizedExperiment
stores into GENBoostGPU inputs. It reads ``.rds`` objects (e.g. the 308-sample
caudate ``nonCpGse.rds``), HDF5 SE directories, or per-chromosome ``.rda``
files (pattern with ``{chrom}``, e.g. ``bs_chr{chrom}_DLPFC_CpH.rda``), and
repoints stale HDF5 seed paths with ``--hdf5-dir``. Run it in an R environment
with bsseq, HDF5Array, DelayedArray, rhdf5 and data.table
(``/projects/p32505/opt/envs/epigenomics``).

Units
-----

* ``--unit site`` (default for ``--context CG``): one unit per cytosine;
  methylation ``M/Cov`` where ``Cov ≥ --min-cov``, kept if covered in ≥
  ``--min-covered-frac`` of samples.
* ``--unit tile`` (default for CpH contexts): fixed ``--tile-width`` windows,
  ``ΣM / ΣCov`` per sample, a sample's value counted when ``ΣCov ≥
  --min-tile-cov``. Per-site CpH methylation is too sparse to call regions on;
  tiles aggregate it first.

Contexts: ``CG``, ``CHG``, ``CHH`` (``c_context``) and ``CA``, ``CC``, ``CT``,
``CH`` (``trinucleotide_context``).

Excluded regions
----------------

``--exclude-bed a.bed[,b.bed.gz,...]`` drops every cytosine inside the listed
intervals before units are formed (BED: 0-based start, ``chr`` prefix
optional, gzip read directly, ``#``/``track``/``browser`` lines skipped). The
count and the files' md5 are recorded in ``manifest/qc-chr{c}.tsv``.

Use the ENCODE hg38 blacklist for CpH. On caudate chr21 (CA tiles, 128 AA
donors), 18 of the 28 regions called without it lay on the acrocentric short
arm (5–11 Mb), where apparent CpH variability is mapping artefact. The
blacklist covers 16 of them, and 17 with 97% coverage, but none of the 10
long-arm regions; those 18 also have no genotyped variants in their cis
window. Segmental-duplication tracks are too broad for this purpose: they
covered all 10 long-arm regions too.

Steps
-----

=========  ============================================================
``qc``     per chromosome → ``units/chr{c}.beta.h5`` (samples × units,
           ``sample_id``) and ``units/chr{c}.units.tsv.gz``.
           ``--min-sd`` / ``--top-per-chrom`` prefilter units for
           site-level runs.
``pca``    genome-wide: top ``--top-n`` variable units, optional
           regression on ``--snp-pcs`` (``--n-snp-pcs``), ``prcomp(scale
           = TRUE)`` → ``pca/meth_pcs.tsv``.
``call``   per chromosome: residualize units on ``--n-meth-pcs``
           methylation PCs, residual SD, per-chromosome ``--sd-quantile``
           cutoff, ``bsseq:::regionFinder3(maxGap = --max-gap)``, keep
           regions with more than ``--min-units`` units; phenotype = mean
           unit methylation per sample.
``tiles``  per chromosome: use the QC'd units directly as regions.
=========  ============================================================

``call`` mirrors Module 01's VMR definition (snpPC1–3 before PCA, meth
PC1–5, 99th percentile, ``maxGap`` 1000, ``n > 5``) with every constant a
parameter; for tiles ``--max-gap`` defaults to the tile width so adjacent
variable tiles merge.

Example: caudate CpH (mCA) regions::

   for c in $(seq 1 22); do      # one SLURM task per chromosome, large memory
     Rscript scripts/build_regions.R --step qc --chrom $c --context CA --unit tile \
         --tile-width 10000 --min-tile-cov 20 --min-covered-frac 0.8 \
         --input /projects/b1213/resources/libd_data/wgbs/raw-data/batch-3/bs_objs/batch3_combined/nonCpGse.rds \
         --hdf5-dir /projects/b1213/resources/libd_data/wgbs/raw-data/batch-3/bs_objs/batch3_combined \
         --samples caudate_samples.tsv --coldata-id brnum \
         --exclude-bed inputs/supportfiles/_m/hg38-blacklist.v2.bed.gz --out build/cph-caudate
   done
   Rscript scripts/build_regions.R --step pca --out build/cph-caudate \
       --snp-pcs snp_pcs.tsv --n-snp-pcs 3 --n-meth-pcs 5
   for c in $(seq 1 22); do
     Rscript scripts/build_regions.R --step call --chrom $c --unit tile --tile-width 10000 \
         --out build/cph-caudate
   done
   genboostgpu regions merge --build-dir build/cph-caudate

``caudate_samples.tsv`` has ``sample_id`` (the ID used in genotypes and
covariates, e.g. ``Br####``) and ``store_id`` (matched against
``--coldata-id``); apply donor filters (age, bisulfite conversion) when writing
it.

Site-level inputs
-----------------

``scripts/prepare_site_inputs.sh`` (``build_regions.R --step qc --unit site``)
writes the per-chromosome unit matrices that ``genboostgpu sites`` consumes;
it supersedes ``scripts/prepare_cpg_inputs.R``.
