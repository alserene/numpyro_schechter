SPHINXBUILD   = poetry run sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  clean    to remove built files"
	@echo "  html     to build HTML documentation"

clean:
	rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
