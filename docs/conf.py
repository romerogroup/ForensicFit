# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = 'Forensics Edge Matching'
copyright = '2020, Pedram Tavadze, Freddy Farah, Aldo Romero'
author = 'Pedram Tavadze, Freddy Farah, Aldo Romero'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     "sphinx.ext.autodoc",
#     "sphinx.ext.mathjax",
#     "sphinx.ext.githubpages",
#     "sphinx.ext.napoleon",
# ]



extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
]
source_suffix = '.rst'



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
master_doc = "index"
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
todo_include_todos = True
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
htmlhelp_basename = 'edgematchdoc'
man_pages = [
    (master_doc, 'edge_match', u'Edge Match Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'edge_match', u'Edge Match Documentation',
     author, 'edge_match', 'One line description of project.',
     'Miscellaneous'),
]

