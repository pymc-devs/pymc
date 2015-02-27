How to build PyMC3 docs
=======================

PyMC3 uses [mkdocs](http://www.mkdocs.org/) which allows us to write documentation in markdown.

We also compile IPython notebooks (found in docs/notebooks) to markdown which then get picked up
by mkdocs.

To build the docs you need to:

1. Convert the IPython Notebooks to markdown by running the convert_nbs_to_md.sh script from the
docs directory. If you added a new notebook you want included, you have to add it to mkdocs.yml.

2. Run `mkdocs build` from the root directory. This will create a subdirectory site with the static
html files you can view in your browser.

3. To deploy, run `mkdocs gh-deploy`. Note that this requires write access to the pymc3 repo.
