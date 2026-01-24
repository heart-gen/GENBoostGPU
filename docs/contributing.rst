Contributing
============

We welcome bug reports, feature ideas, and pull requests. This guide covers the
basics for working on GENBoostGPU.

Development environment
-----------------------

1. Clone the repository and install the development extras:

   .. code-block:: bash

      git clone https://github.com/heart-gen/GENBoostGPU.git
      cd GENBoostGPU
      python -m venv .venv
      source .venv/bin/activate
      pip install -e ".[dev]"

2. Install the documentation tooling if you plan to build the docs locally:

   .. code-block:: bash

      pip install -r docs/requirements.txt

Running tests
-------------

* Unit tests: ``pytest -q``
* Coverage: ``pytest --cov=genboostgpu --cov-report=term-missing``
* Static analysis: ``ruff check .`` and ``mypy src``

Formatting & style
------------------

* Use `black <https://black.readthedocs.io/>`_ for formatting and `isort
  <https://pycqa.github.io/isort/>`_ for import ordering. ``ruff`` can run both
  via ``ruff format`` and ``ruff check --fix``.
* Docstrings follow Google or NumPy style (``napoleon`` is enabled in the docs).
* A ``.pre-commit-config.yaml`` is provided—run ``pre-commit install`` to enable
  hooks that enforce the style checks before every commit.

Issue & PR workflow
-------------------

* Search the `issue tracker <https://github.com/heart-gen/GENBoostGPU/issues>`_
  before filing a new bug to avoid duplicates.
* Include a minimal reproducible example (inputs, hyperparameters, expected vs
  actual behaviour). Compress large datasets or provide pointers to public
  sources if possible.
* Reference relevant modules (e.g., ``:mod:`genboostgpu.cpg_orchestration``) in your PR
  description to make review easier.

Adding tutorials
----------------

1. Place runnable scripts in ``examples/`` with seeded RNGs (``random_state=42``).
2. Document the narrative in ``docs/tutorials/<name>.rst`` and pull in the script
   with ``literalinclude`` so users can copy/paste it.
3. Cross-link to the relevant API pages using ``:mod:`` and ``:func:`` roles.
4. Update ``docs/tutorials/index.rst`` to include the new page.

Thank you for helping make GENBoostGPU better!
