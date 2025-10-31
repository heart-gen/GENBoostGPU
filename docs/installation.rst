Installation
============

GENBoostGPU relies on NVIDIA GPUs and RAPIDS for production workloads, but the
documentation and API reference can be built on CPU-only machines. The sections
below outline both workflows and highlight the CUDA/RAPIDS version constraints.
All runtime pins—including RAPIDS 25.8, ``cupy-cuda12x >=13.3``, ``optuna`` and
``scikit-learn``—are tracked in ``pyproject.toml``. Refer to that file whenever
you need the authoritative dependency matrix.

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
   pip install genboostgpu --no-deps
   pip install numpy pandas pytest session-info pyhere

This keeps the environment light-weight while ensuring the doc build and test
collection succeed. Avoid installing ``cudf-cu12`` or ``cuml-cu12`` on CPU-only
hosts—they are only published for CUDA-enabled platforms.

GPU / RAPIDS workflow (full pipeline execution)
-----------------------------------------------
For real analyses, install GENBoostGPU alongside RAPIDS components that match
your CUDA driver. The project currently targets the ``cudf-cu12``/``cuml-cu12``
packages at ``25.8.x`` and ``cupy-cuda12x >=13.3`` as defined in
``pyproject.toml``.

.. code-block:: bash

   mamba create -n genboostgpu -c rapidsai -c conda-forge -c nvidia \
       python=3.11 rapids=25.08 cupy-cuda12x=13.3 \
       dask-cuda=25.8 optuna scikit-learn pandas-plink
   mamba activate genboostgpu
   pip install genboostgpu

If you are working from a clone and need to install in editable mode, install
the published wheel first to ensure all GPU dependencies resolve correctly:

.. code-block:: bash

   git clone https://github.com/heart-gen/GENBoostGPU.git
   cd GENBoostGPU
   pip install genboostgpu
   pip install -e .

Verify that the RAPIDS packages report the expected versions and CUDA runtime:

.. code-block:: bash

   python -c "import cudf, cupy; print(cudf.__version__, cupy.cuda.runtime.getVersion())"

If you upgrade drivers or move to a new CUDA minor version, rebuild the
environment so that ``cudf-cu12``/``cuml-cu12`` and ``cupy-cuda12x`` stay in
sync.

Next steps
----------

* :doc:`quickstart` to run the minimal pipeline example.
* :doc:`tutorials/index` for end-to-end walkthroughs based on real scripts.
