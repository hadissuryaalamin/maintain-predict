# maintain-predict

Estimate machine failure risk from product type and operating measurements. The
first milestone is to compare a no-failure baseline, logistic regression, random
forest, and XGBoost on a held-out test set.

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
the data and training baselines are the next stage.

## Manual split exercise

Read [scripts/01_split_data.py](scripts/01_split_data.py) and run its seven
numbered steps yourself. The exercise is prepared and has not been executed on
the uploaded dataset. It uses a fixed 70/15/15 split stratified jointly by
`Type × Failure evidence`; tool wear is not an additional splitting group.

From the repository root in PowerShell, using the prepared project environment:

```powershell
.\.venv\Scripts\python.exe .\scripts\01_split_data.py
```

The script prints allocation and wear-balance tables, checks row overlap, and
with `SAVE_SPLIT=True` saves `data/splits/type_target_seed42.json`. It does not
fit preprocessing, train a model, or calculate test performance. The baseline
comparison remains a later exercise.

## Results

| Model | PR-AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Always predict no failure | Pending | Pending | Pending | Pending |
| Logistic regression | Pending | Pending | Pending | Pending |
| Random forest | Pending | Pending | Pending | Pending |
| XGBoost | Pending | Pending | Pending | Pending |

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
