from __future__ import annotations

import sys, pathlib
from datetime import datetime

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # local fallback if needed
    import tomli as tomllib 

PYPROJECT = {}
try:
    with open(ROOT_DIR / "pyproject.toml", "rb") as f:
        PYPROJECT = tomllib.load(f)
except Exception:
    pass

PROJECT_META = PYPROJECT.get("project", {})
project = PROJECT_META.get("name", "GENBoostGPU")
author = ", ".join(a.get("name", "") for a in PROJECT_META.get("authors", [])) or "GENBoostGPU Developers"
current_year = datetime.utcnow().year
copyright = f"{current_year}, {author}"

# Prefer installed distribution version; fall back to pyproject
try:
    from importlib.metadata import version as get_version
    release = get_version(project)
except Exception:
    release = PROJECT_META.get("version", "0.0.0")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# RST docs
source_suffix = [".rst"]

# Autodoc / Autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
python_use_unqualified_type_names = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

# Mock heavy GPU dependencies so autodoc can run without them.
autodoc_mock_imports = [
    "cupy", "cudf", "cuml", "dask_cuda", "numba",
    "torch", "pandas_plink", "pyarrow",
    "optuna", "sklearn", "dask", "distributed"
]

# Intersphinx cross-links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# HTML
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
html_title = f"{project} {release}"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Ensure _static exists to avoid warnings on fresh repos
_static = pathlib.Path(__file__).resolve().parent / "_static"
_static.mkdir(exist_ok=True)
html_static_path: list[str] = ["_static"]

# Quality gate
nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy.ndarray"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "pandas.Series"),
]
