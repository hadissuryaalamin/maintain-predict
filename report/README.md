# Baseline report

`baseline_report.tex` is the English LaTeX report of completed dataset analysis,
target-policy decisions, data splitting, and the executed dummy, logistic
regression, Random Forest, and XGBoost experiments. The 18 November 2026 update
includes tree-model searches, training-only CV, four feature experiments,
validation metrics, confusion matrices, and the precision/recall tradeoff.
It records the user's choice of seven-input XGBoost with torque × wear and
retention of threshold 0.5 without tuning. Training CV and validation results
are identified separately; no final test evaluation is reported.

The report now includes completed validation SHAP: grouped global importance,
the encoded-feature beeswarm, and TP/FN/FP/TN waterfall examples. It explains
log-odds, direction versus prediction correctness, and the different color
meanings in beeswarm and waterfall plots.

The `figures/` directory contains eight validation confusion matrices and six
SHAP plots copied from the saved runs. Keep it beside the LaTeX source when
compiling.

`verified_results.json` records the source summaries used for the update,
the selected model, threshold policy, and SHAP run metadata. The source is directly editable;
recompilation does not require the original model artifacts.

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
