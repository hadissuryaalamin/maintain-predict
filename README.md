# maintain-predict

Estimate machine failure risk from product type and operating measurements. The
first milestone is to compare a no-failure baseline, logistic regression, random
forest, and XGBoost on validation data, reserving a held-out test set for the
final model evaluation.

## Setup

On Ubuntu with Python 3.11 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,demo,dev]'
python -m pytest
```

Download `ai4i2020.csv` from the [AI4I 2020 Predictive Maintenance
Dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset)
(UCI, CC BY 4.0) and place it in `data/`. The dataset contains 10,000 synthetic
machine readings. Git ignores `data/`, `mlruns/`, and local model artifacts.

Start with `jupyter lab notebooks/01_eda.ipynb`. The feature allowlist includes
product type and the five operating
measurements. It excludes the failure-mode flags `TWF`, `HDF`, `PWF`, `OSF`,
and `RNF` to prevent target leakage and leaves out the identifiers `UDI` and
`Product ID`.

## Target definition

Our modeling target is **`Failure evidence`**, with policy identifier
`failure_evidence_or_v1`:

```text
Failure evidence = Machine failure OR TWF OR HDF OR PWF OR OSF OR RNF
```

A recorded positive in any outcome column is sufficient under this chosen
assumption. This retains all original positives and promotes flagged negatives;
it does not establish which source annotation is correct. The raw CSV and its
original `Machine failure` column remain unchanged. The flags define labels and
are excluded from predictor inputs.

On this file, the original label has 339 positives (3.39%). The derived target
has **357 positives (3.57%)**: 18 RNF-positive records are promoted and no
original positives are demoted. The nine original failures with no mode flag
stay positive. The increase in prevalence is a consequence of the policy, not
evidence for it. Random failure evidence may be unpredictable from the available
sensors. Results must identify this derived target and cannot be presented as
directly equivalent to the original-label benchmark.

## EDA

The [executed EDA notebook](notebooks/01_eda.ipynb) includes column roles and
types, missing-value and duplicate checks, class balance, sensor distributions
by outcome, correlations, a failure-label audit, and original-row-order analysis.
Its outputs are saved, and it ends with five observations in plain language.

The raw file contains 27 disagreements between the original overall label and
failure-mode flags. Section 8 audits those source annotations, documents the
chosen target policy, and compares the original and derived labels. The active
class-balance and outcome-grouped plots use `Failure evidence`. The notebook
explores the full file; this must be disclosed in later evaluation. Splitting
and baseline training have since been completed.

## Manual split exercise

The seven-step [split exercise](scripts/01_split_data.py) has been run manually.
The saved allocation was verified: 7,000 training rows (250 positives), 1,500
validation rows (54 positives), and 1,500 test rows (53 positives), with no
overlap. It uses a fixed 70/15/15 split stratified jointly by
`Type × Failure evidence`; tool wear is not an additional splitting group.

From the repository root in PowerShell, using the prepared project environment:

```powershell
.\.venv\Scripts\python.exe .\scripts\01_split_data.py
```

The script prints allocation and wear-balance tables, checks row overlap, and
with `SAVE_SPLIT=True` saves `data/splits/type_target_seed42.json`. Re-running it
with the same source, policy, and seed verifies that the manifest matches. It
does not fit preprocessing, train a model, or calculate test performance. The
baseline comparison has also been run manually.

## Manual baseline exercise

[scripts/02_baselines.py](scripts/02_baselines.py) has been run manually.
It restores the saved split, fits an always-negative classifier and balanced
logistic regression on training rows, and compares them on validation rows at
the fixed 0.5 threshold. Encoding and scaling are learned inside the training
pipeline. It does not generate a new split or evaluate final test performance.

```powershell
.\.venv\Scripts\python.exe .\scripts\02_baselines.py
```

Local metrics, plots, validation predictions, and skops models are saved under
`artifacts/baselines/`; MLflow runs use `mlflow.db` and `mlruns/`. This exercise
has been executed on the uploaded dataset. The numbered script cells
explain the fitting and prediction steps. See the guide in
`docs/baseline_guide.md` for metric definitions and expected sanity checks.

## Results

The completed baseline results below use validation average precision (AP), not final test
metrics. AP is the selected PR-curve summary, distinct from trapezoidal PR area.

| Model | Validation AP | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Always predict no failure evidence | 0.0360 | Undefined | 0.00% | 0.00% |
| Logistic regression | 0.5288 | 13.65% | 85.19% | 23.53% |
| Random forest | 0.7529 | 76.00% | 70.37% | 73.08% |
| XGBoost | 0.7875 | 69.84% | 81.48% | 75.21% |

## Manual Random Forest exercise

[scripts/03_random_forest.py](scripts/03_random_forest.py) has been run manually.
It restores the same split, selects among 12 Random Forest configurations
using five-fold training-only CV scored by AP, and evaluates the selected model
on validation at threshold 0.5. It also scores fresh logistic clones on the
same training folds as a CV reference. The saved baseline models are preserved.

```powershell
.\.venv\Scripts\python.exe .\scripts\03_random_forest.py
```

Results are saved separately under `artifacts/random_forest/` and tracked
in local MLflow. At 0.5, the model detected 38 of 54 positives, missed 16, and
raised 12 false alarms. Test performance remains reserved.
See `docs/random_forest_guide.md` for the numbered steps and search settings.

## Manual XGBoost exercise

[scripts/04_xgboost.py](scripts/04_xgboost.py) has been run manually.
It restores the same split, selects among 12 configurations using five-fold
training-only AP, and evaluates validation at threshold 0.5. Positive-class
weighting is recalculated from the labels in each fitting fold.

```powershell
.\.venv\Scripts\python.exe .\scripts\04_xgboost.py
```

XGBoost is already installed in the project environment. Outputs are saved
under `artifacts/xgboost/`, including the native `model.ubj` and separately
saved `preprocessor.skops`. At threshold 0.5 it detected 44 of 54 positives,
missed 10, and raised 19 false alarms. Final test evaluation remains reserved.
See `docs/xgboost_guide.md` for the steps, class weighting, and persistence format.

## Manual temperature-difference exercise

[scripts/05_temperature_difference.py](scripts/05_temperature_difference.py)
has been run manually. It adds process minus air temperature to the six
existing XGBoost inputs, preserving the previous hyperparameters. Paired
training-only five-fold CV compares six versus seven inputs before validation
evaluation at threshold 0.5.

```powershell
.\.venv\Scripts\python.exe .\scripts\05_temperature_difference.py
```

Outputs are saved to `artifacts/temperature_difference/`, separately from
earlier experiments. The exercise has been run manually. Training CV mean AP
rose from 0.7165 to 0.7514. At threshold 0.5, validation TP=41, FN=13, FP=12,
TN=1,434; precision=77.36%, recall=75.93%, F1=76.64%, AP=0.7805.
This variant remains a candidate, not an automatic replacement.
See `docs/temperature_difference_guide.md` for the comparison and inference steps.

## Manual power exercise

[scripts/06_power.py](scripts/06_power.py) has been run manually. It
compares six original inputs against six original inputs plus calculated
mechanical power (torque × rpm × 2π/60). Temperature difference is excluded.
Hyperparameters, CV folds, and threshold remain unchanged.

```powershell
.\.venv\Scripts\python.exe .\scripts\06_power.py
```

Outputs are saved under `artifacts/power/`. Mean training CV AP increased from
0.7165 to 0.7453, but validation AP fell to 0.7752. At 0.5, TP=43, FN=11,
FP=19, TN=1,427; precision=69.35%, recall=79.63%, F1=74.14%.
This variant has not replaced the original model.
See `docs/power_guide.md` for the comparison and units.

## Manual torque × wear exercise

[scripts/07_torque_wear.py](scripts/07_torque_wear.py) has been run manually.
It adds only torque × tool wear to the six original inputs, without
temperature difference or power. It uses the same fixed XGBoost settings,
paired training-only CV folds, and validation threshold 0.5.

```powershell
.\.venv\Scripts\python.exe .\scripts\07_torque_wear.py
```

Outputs are saved under `artifacts/torque_wear/`. Mean training CV AP increased
from 0.7165 to 0.7462. At 0.5, validation TP=46, FN=8, FP=16, TN=1,430;
precision=74.19%, recall=85.19%, F1=79.31%, AP=0.8483.
The user selected this seven-input variant and retained threshold 0.5 without
tuning. SHAP explanation and final test evaluation remain outstanding. See
`docs/model_selection.md` for the decision, exact artifacts, and limitations.
See `docs/torque_wear_guide.md` for details.

## Manual combined-feature exercise

[scripts/08_torque_wear_temperature_difference.py](scripts/08_torque_wear_temperature_difference.py)
has been run manually. It compares **seven inputs** (six raw + torque-wear)
against **eight inputs** (six raw + torque-wear + temperature difference).
Both use the saved torque-wear model's fixed XGBoost parameters, identical
training CV folds, and threshold 0.5. Power is excluded.

```powershell
.\.venv\Scripts\python.exe .\scripts\08_torque_wear_temperature_difference.py
```

Outputs are saved under `artifacts/torque_wear_temperature_difference/`.
Mean training CV AP increased from 0.7462 to 0.7847. At threshold 0.5,
validation TP=42, FN=12, FP=12, TN=1,434; precision=77.78%, recall=77.78%,
F1=77.78%, AP=0.8336. The user chose the seven-input torque-wear variant
for the next stage after reviewing this tradeoff.
See `docs/combined_features_guide.md` for the comparison and inference steps.

## Manual SHAP exercise

[scripts/09_shap.py](scripts/09_shap.py) has been run manually. It loads
the selected seven-input XGBoost model without refitting, explains validation
predictions with Tree SHAP, and keeps threshold 0.5 and the final test set reserved.
SHAP is installed in the project environment.

```powershell
.\.venv\Scripts\python.exe .\scripts\09_shap.py
```

Read `artifacts/shap/latest.json` after completion to find the run's output
folder. It contains global importance, an encoded-feature beeswarm, local
waterfalls, numerical SHAP values, and metadata. Contributions use log-odds,
not additive probability percentages. All 1,500 validation rows were explained
without refitting. Global mean absolute contributions are largest for rotational
speed, torque, and air temperature; torque-wear ranks fifth. Final test
evaluation remains reserved. See `docs/shap_guide.md` for interpretation and
limitations.

## Local API

The starter API exposes `GET /health`. Prediction depends on a trained model
and a validation threshold.

```bash
python -m uvicorn maintain_predict.serve:app
```

```bash
docker build -t maintain-predict .
docker run --rm -p 8000:8000 maintain-predict
curl http://localhost:8000/health
```
