# Paper Source

`L-CDAIDS.tex` and `references.bib` are the LaTeX source of the manuscript
behind this repository ("L-CDAIDS: Lightweight Concept-Drift-Aware
Intrusion Detection for Resource-Constrained IoT Networks"). `figures/`
holds the figures it includes.

## Compiling

Requires a standard TeX distribution (TeX Live or MiKTeX) with the
`longtable` and `placeins` packages, plus `bibtex`/`biber` for the
bibliography:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode L-CDAIDS.tex
```

or, without `latexmk`:

```bash
pdflatex L-CDAIDS.tex
bibtex L-CDAIDS
pdflatex L-CDAIDS.tex
pdflatex L-CDAIDS.tex
```

## Relationship to the rest of this repository

Every table and figure in the manuscript traces to a script, notebook, or
results folder under `experiments/` -- see `../docs/EXPERIMENT_MAP.md` for
the full mapping, which mirrors the manuscript's own Section 6.14
("Consolidated Experimental Settings").
