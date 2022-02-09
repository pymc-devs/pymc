import os

import aeppl

# -- Project information

project = "aeppl"
author = "Aesara Developers"
copyright = f"2022, {author}"

version = aeppl.__version__
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if "." not in rtd_version and rtd_version.lower() != "stable":
        version = "dev"
else:
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "")
    if branch_name == "main":
        version = "dev"
release = version


# -- General configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_nb",
    "sphinx_copybutton",
]

templates_path = ["_templates"]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "code"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Options for extensions

jupyter_execute_notebooks = "auto"
# execution_excludepatterns = ["*.ipynb"]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

autosummary_generate = True
always_document_param_types = True

# -- Options for HTML output

html_theme = "furo"

intersphinx_mapping = {
    "aesara": ("https://aesara.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}
