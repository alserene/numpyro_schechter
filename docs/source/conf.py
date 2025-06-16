import os
import sys
from tomllib import load  # Python 3.11+

# Add your src folder to sys.path so autodoc can import your package
sys.path.insert(0, os.path.abspath("../../src"))

def format_authors(authors):
    if isinstance(authors, list):
        if len(authors) == 1:
            return authors[0]
        return ", ".join(authors[:-1]) + ", and " + authors[-1]
    return authors

# Load metadata from pyproject.toml
with open(os.path.abspath("../../pyproject.toml"), "rb") as f:
    pyproject = load(f)

project = pyproject["tool"]["poetry"]["name"]
author = format_authors(pyproject["tool"]["poetry"]["authors"])
release = pyproject["tool"]["poetry"]["version"]

# The short X.Y.Z version (optional, you can parse this if you want)
version = release.split("a")[0]  # strips off 'a1' alpha tag if present

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",     # auto-generate docs from docstrings
    "sphinx.ext.napoleon",    # support for Google/NumPy style docstrings
    "sphinx.ext.viewcode",    # add links to source code in docs
    "sphinx.ext.autosectionlabel",  # allows cross-referencing sections easily
    "myst_parser",            # support for Markdown files
]

autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
