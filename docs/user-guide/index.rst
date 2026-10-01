User guide
==========

The first four chapters cover the local-genetic-variance engine (v0.4+): the
Module 02 features and score, Module 03 out-of-fold prediction, building CpG
and CpH region inputs, and site-level runs with GPU-cost guidance. The
remaining chapters document the legacy boosting elastic net, whose
``final_r2`` is an in-sample fit.

.. toctree::
   :maxdepth: 1

   lgv_engine
   lsp_prediction
   region_builders
   sites_and_cost

Legacy boosting elastic net
---------------------------

.. toctree::
   :maxdepth: 1

   data
   cpg_pipeline
   workflow
   tuning
   performance
   reproducibility
