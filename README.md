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

Start with `jupyter lab notebooks/01_eda.ipynb`. Use `Machine failure` as the
target. The feature allowlist includes product type and the five operating
measurements. It excludes the failure-mode flags `TWF`, `HDF`, `PWF`, `OSF`,
and `RNF` to prevent target leakage and leaves out the identifiers `UDI` and
`Product ID`.

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
