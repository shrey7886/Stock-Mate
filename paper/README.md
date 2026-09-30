# Foleo paper (IEEE conference format)

## Build on Overleaf
1. New Project → Blank Project → replace `main.tex` with this folder's `main.tex`.
2. Upload `symbol.jpg` (the NITK logo, in this folder) alongside it.
3. Compiler: pdfLaTeX (default). `IEEEtran` is built into Overleaf.

`main.tex` is self-contained: the benchmark values and the bibliography are embedded, so no other file is needed.
`main.pdf` here is a local preview build.

## Where the numbers come from
All benchmark figures sit between the `BEGIN/END GENERATED RESULTS` markers in `main.tex`. They were produced from
the completed run `reports/benchmark_runs/2026-09-27_1546` by:

```
python scripts/make_paper_tables.py reports/benchmark_runs/2026-09-27_1546 \
    --external reports/benchmark_runs/2026-09-27_1546/external_results_batch_codeexec.csv
```

which writes `generated_results.tex`. If the data ever changes, regenerate it and paste it over the marked block.
`references.bib` is the source of the embedded bibliography.
