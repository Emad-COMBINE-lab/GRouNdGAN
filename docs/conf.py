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
version = release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_favicon",
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

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_context = {
    "display_github": True,
    "github_user": "Emad-COMBINE-lab",
    "github_repo": "GRouNdGAN",
    "github_version": "master/",
    "conf_py_path": "docs/",  # Path in the checkout to the docs root
}

# pygments_style = "monokai"  # or xcode'
html_logo = "_static/logo.svg"


html_theme_options = {
    "repository_url": "https://github.com/Emad-COMBINE-lab/GRouNdGAN",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_source_button": True,
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_download_button" : True,
    "logo": {
        "alt_text": "GRouNdGAN - Home",
        # "text": "",
    },
    # "show_navbar_depth": 4,
    # "home_page_in_toc": True,
    "show_toc_level": 2,
    #   "announcement": "My announcement!",
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            "url": "https://github.com/Emad-COMBINE-lab/GRouNdGAN",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            # Label for this link
            "name": "Manuscript",
            "url": "https://www.biorxiv.org/content/10.1101/2023.07.25.550225v2",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
        {
            "name": "Build",
            "url": "https://github.com/Emad-COMBINE-lab/GRouNdGAN/actions",
            "icon": "https://github.com/Emad-COMBINE-lab/GRouNdGAN/actions/workflows/documentation.yaml/badge.svg?branch=master",
            "type": "url",
        },
        {
            "name": "Version",
            "url": "https://github.com/Emad-COMBINE-lab/GRouNdGAN",
            "icon": "https://img.shields.io/badge/Version-1.0-blue",
            "type": "url",
        },
    ],
}

favicons = [
    "logo.svg",
]

# html_sidebars = {
# }
