# Final clean scientific sources

`main.tex` is the journal manuscript; `supplementary_material.tex` is the
standalone supplement. The text, figures and tables are the final submission
version of 15 October 2026. All local dependencies used by these documents
are included. The highlighted review copy and reviewer correspondence stay
in the journal submission project.

To compile with a standard TeX installation from this directory:

```bash
latexmk -pdf -outdir=../results/generated/latex/main main.tex
latexmk -pdf -outdir=../results/generated/latex/supplement supplementary_material.tex
```

To rebuild and verify the figure/table assets from the repository root:

```bash
python scripts/reproduce_paper.py
```

`asset_manifest.json` distinguishes frozen source files, numerical inputs and
the 25 used publication assets. Unused figures and tables from earlier
manuscript revisions are excluded.
