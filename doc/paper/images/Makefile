OUTDIR?=latex.out
TEXMK?=latexmk -verbose -pdf -output-directory=$(OUTDIR)
PYTHON?=python -X dev

all: figures

help: 			## Show this help
	@echo -e "\nSpecify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

%.pdf: %.tex
	PYTHONWARNINGS=ignore $(TEXMK) $<
	@cp $(OUTDIR)/$@ .

%.png: %.pdf
	magick \
		-density 600 \
		$< \
		-quality 100 \
		-background white \
		-alpha remove \
		$@

quadrature_rules.png:
	magick \
		quadrature_rule_vr_order_11_2d.png \
		\( -size 900x`identify -format "%h" quadrature_rule_vr_order_11_2d.png` xc:white \) \
		quadrature_rule_wv_order_11_2d.png \
		+append \
		-background white \
		quadrature_rules.png

figures: simplices quadratures ## Compile all figures
	for img in simplices.png pkdo-2d.png quadrature_rules.png; do \
		magick "$${img}" -trim +repage "$${img}"; \
	done

simplices: simplices.png
quadratures: quadrature_rules.png
