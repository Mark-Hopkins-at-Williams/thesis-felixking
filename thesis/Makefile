# Makefile for compiling LaTeX thesis with BibTeX

# Main tex file
MAIN = thesis

# Compilation commands
PDFLATEX = pdflatex
BIBTEX = bibtex

# Compilation target
all: $(MAIN).pdf

# Compilation steps
$(MAIN).pdf: $(MAIN).tex
	$(PDFLATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(PDFLATEX) $(MAIN)
	$(PDFLATEX) $(MAIN)
	code thesis.pdf


# Clean up auxiliary files
clean:
	rm -f *.aux *.log *.bbl *.blg *.out $(MAIN).pdf

# Phony targets
.PHONY: all clean