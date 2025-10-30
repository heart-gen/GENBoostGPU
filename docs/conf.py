from __future__ import annotations

import os
import sys
from datetime import datetime

try:
    from importlib.metadata import PackageNotFoundError, version as get_version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    PackageNotFoundError = Exception  # type: ignore[assignment]

    def get_version(_: str) -> str:  # type: ignore[override]
        return "0.0.0"

# Ensure src directory is on sys.path so autodoc can import the package.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if os.path.isdir(SRC_DIR):
    sys.path.insert(0, SRC_DIR)

project = "GENBoostGPU"
author = "GENBoostGPU Developers"
current_year = datetime.utcnow().year
copyright = f"{current_year}, {author}"

# The full version, including alpha/beta/rc tags.
try:
    release = get_version("genboostgpu")
except PackageNotFoundError:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Mock heavy GPU dependencies so autodoc can run without them.
autodoc_mock_imports = [
    "cupy",
    "cudf",
    "cuml",
    "dask_cuda",
    "numba",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = ["_static"]
