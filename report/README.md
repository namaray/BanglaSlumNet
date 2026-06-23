# BanglaSlumNet LaTeX Report

This folder is ready to upload directly to Overleaf as a project.

## Main files

- `main.tex` - complete report
- `references.bib` - bibliography
- `diagram_instructions.md` - instructions for the designer
- `figures/` - current figures and placeholders for final diagrams

## Compile

Use the standard sequence:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Overleaf performs this automatically when the compiler is set to `pdfLaTeX`.

## Designer files expected

- `figures/system_architecture.pdf`
- `figures/compute_pipeline.pdf`

The report compiles without them by displaying framed placeholders. When the designer adds files with those exact names, the final diagrams appear automatically.

## Important scientific note

The numerical results currently discussed in the report are diagnostic. The initial result files showed prediction collapse, and the corrected visual-only run remained near random balanced accuracy. Do not rewrite these as final performance claims without completing corrected experiments and obtaining stronger evaluation labels.
