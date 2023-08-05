# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "GRouNdGAN"
copyright = (
    "2023, Emad's COMBINE Lab: Yazdan Zinati, Abdulrahman Takiddeen, and Amin Emad"
)
author = "Yazdan Zinati"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',

]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__, __len__, __getitem__",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_context = {
  'display_github': True,
  'github_user': 'Emad-COMBINE-lab',
  'github_repo': 'GRouNdGAN',
  'github_version': 'master/',
  'conf_py_path': "docs/", # Path in the checkout to the docs root
}

pygments_style = "monokai"  # or xcode'
html_logo = "figs/logo.png"


html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'style_external_links': True,
    'style_nav_header_background': 'white',
}