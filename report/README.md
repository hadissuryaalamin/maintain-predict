# Baseline report

`baseline_report.tex` is the English LaTeX report of completed dataset analysis,
target-policy decisions, data splitting, and the executed baseline experiment.
All model results use the validation set; no final test evaluation is reported.

The `figures/` directory contains copies of the two existing validation
confusion-matrix PNGs. Keep it beside the LaTeX source when compiling.

From the project root, build the PDF with:

```powershell
Set-Location report
pdflatex -interaction=nonstopmode -halt-on-error baseline_report.tex
pdflatex -interaction=nonstopmode -halt-on-error baseline_report.tex
```

The output is `report/baseline_report.pdf`. Building this document does not train
models or change the saved data allocation.

Compilation requires a working LaTeX installation with the packages declared
in the source. MiKTeX was verified and the PDF compiled successfully. A restricted
agent session needs access to MiKTeX's existing user configuration and font caches
outside the workspace; without that access, MiKTeX can misleadingly report an
unfinished setup. No reinstall or system-wide setup change was required.
