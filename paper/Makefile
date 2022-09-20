BUILDDIR=build
FILENAME=paper

pdf:
	mkdir $(BUILDDIR) -p
	pandoc $(FILENAME).md \
    --citeproc \
    --from=markdown+tex_math_single_backslash+tex_math_dollars \
    --to=latex \
    --output=$(BUILDDIR)/$(FILENAME).pdf \
    --pdf-engine=xelatex

