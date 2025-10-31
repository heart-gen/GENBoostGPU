Installation
============

GENBoostGPU relies on NVIDIA GPUs and RAPIDS for production workloads, but the
documentation and API reference can be built on CPU-only machines. The sections
below outline both workflows and highlight the CUDA/RAPIDS version constraints.

Prerequisites
-------------

* Python 3.10 or 3.11.
* CUDA 12.x drivers for GPU execution (tested with 12.4/12.6).
* Ability to install RAPIDS 25.8 packages (``cudf``, ``cuml``, ``dask-cuda``).

CPU-only workflow (docs, linting, type checks)
---------------------------------------------

Use this setup when you only need to build the documentation, run static
linters, or inspect the code base without touching GPU-backed functionality.
The heavy CUDA libraries are mocked automatically in the docs configuration.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -r docs/requirements.txt
   pip install -e . --no-deps
   pip install numpy pandas pytest session-info pyhere

This keeps the environment light-weight while ensuring the doc build and test
collection succeed. Avoid installing ``cudf`` or ``cuml`` on CPU-only hosts—they
are only published for CUDA-enabled platforms.

GPU / RAPIDS workflow (full pipeline execution)
-----------------------------------------------

For real analyses, install GENBoostGPU alongside RAPIDS components that match
your CUDA driver. The project currently targets RAPIDS ``25.8`` (``cudf``,
``cuml``) and ``cupy-cuda12x`` ``>=13.3``.

.. code-block:: bash

   mamba create -n genboostgpu -c rapidsai -c conda-forge -c nvidia \
       python=3.11 rapids=25.08 cupy-cuda12x=13.3 \
       dask-cuda=25.8 optuna scikit-learn pandas-plink
   mamba activate genboostgpu
   pip install genboostgpu

When working from a clone:

.. code-block:: bash

   git clone https://github.com/heart-gen/GENBoostGPU.git
   cd GENBoostGPU
   pip install -e .

Verify that the RAPIDS packages report the expected versions and CUDA runtime:

.. code-block:: bash

   python -c "import cudf, cupy; print(cudf.__version__, cupy.cuda.runtime.getVersion())"

If you upgrade drivers or move to a new CUDA minor version, rebuild the
environment so that ``cudf/cuML`` and ``cupy`` stay in sync.

Next steps
----------

* :doc:`quickstart` to run the minimal pipeline example.
* :doc:`tutorials/index` for end-to-end walkthroughs based on real scripts.
