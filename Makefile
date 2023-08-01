# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXBUILD   = sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

rtd: export READTHEDOCS=true

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif

.PHONY: help clean html rtd view

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  rtd        to build the website without any cache"
	@echo "  clean      to clean cache and intermediate files"
	@echo "  view       to open the built html files"

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/api/generated
	rm -rf $(SOURCEDIR)/api/**/generated
	rm -rf $(SOURCEDIR)/api/**/classmethods
	rm -rf docs/jupyter_execute

html:
	$(SPHINXBUILD) $(SOURCEDIR) $(BUILDDIR) -b html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

rtd: clean
	$(SPHINXBUILD) $(SOURCEDIR) $(BUILDDIR) -b html -E
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

view:
	python -m webbrowser $(BUILDDIR)/index.html
