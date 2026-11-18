---
language: en
license: mit
library_name: xgboost
pipeline_tag: tabular-classification
tags:
  - predictive-maintenance
  - synthetic-data
  - binary-classification
  - skops
---

# maintain-predict: failure-evidence classification

A learning project using XGBoost to classify recorded failure evidence in the
synthetic AI4I 2020 dataset. The selected model uses six raw inputs plus the
calculated interaction **torque × tool wear**, with threshold **0.5**.
This is a row-level classifier, not a validated forecast of future breakdowns.

## Model and intended use

The selected estimator has 300 boosting rounds, maximum depth 5, learning rate
0.1, a binary logistic objective, and CPU histogram training. Positive-class
weighting addresses the rare positive labels; the final training weight is 27.
The fitted preprocessing one-hot encodes Type and passes the numeric features
through. There are seven features before encoding and nine encoded inputs.

Intended use is education, reproducible experiments, and demonstrations of a
tabular inference workflow. It is not validated for maintenance scheduling,
industrial control, or safety decisions. No real-machine deployment has been
evaluated. Scores have not been calibrated as physical failure probabilities.

## Data and target

Source: [AI4I 2020 Predictive Maintenance Dataset, UCI](https://doi.org/10.24432/C5HS5C),
S. Matzka (2020). The source dataset contains 10,000 synthetic observations,
not measurements from a deployed fleet. Its data licence is
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Our target is **Failure evidence**, policy `failure_evidence_or_v1`:

```text
Machine failure OR TWF OR HDF OR PWF OR OSF OR RNF
```

Any positive source outcome sets the target to 1. This raises the positive count
from 339 to 357 (3.57%) by promoting 18 rows. Nine rows with Machine failure=1
and no positive failure-mode flag remain positive. This is an explicit label
assumption, not a verified correction of the source dataset. The raw CSV is
unchanged. These metrics therefore describe the derived target, not the
original Machine failure target alone.

## Inputs and outputs

| Raw input | Meaning / unit |
| --- | --- |
| `Type` | Product category: L, M, or H |
| `Air temperature [K]` | Air temperature in kelvin |
| `Process temperature [K]` | Process temperature in kelvin |
| `Rotational speed [rpm]` | Revolutions per minute |
| `Torque [Nm]` | Torque in newton-metres |
| `Tool wear [min]` | Accumulated tool wear in minutes |

The loader calculates `Torque wear [Nm min]` as torque multiplied by tool wear.
Do not supply failure flags or identifiers as predictors. Extra columns are
ignored, and any supplied torque-wear column is recalculated. Numeric inputs
must be finite; Type must be one of the three known categories. These checks
do not establish that a reading is physically valid or within the model's
training distribution.

Outputs are `failure_score`, `predicted_label`, and `threshold`. Scores at least
0.5 produce label 1. Label 0 means no failure evidence predicted under our
policy; it does not certify that a real machine is healthy.

## Split and model selection

A saved split with seed 42 stratifies jointly by **Type and derived target**:

| Partition | Rows | Positive labels |
| --- | ---: | ---: |
| Train | 7,000 | 250 |
| Validation | 1,500 | 54 |
| Test | 1,500 | 53 |

Training-only cross-validation compared candidate models and engineered
features. The user selected the torque-wear variant after reviewing validation
tradeoffs; it was not the winner of every comparison. The threshold was retained
at 0.5 without tuning because no operating-cost or minimum-recall objective was
defined. The original fitted training model was evaluated once on the test set
after selection, without refitting. Packaging does not change that selection.

Leakage precautions: outcome flags, Machine failure, Product ID, and UDI are
excluded from predictors. Preprocessing is fitted inside training CV folds and
then on training data for the saved estimator. Torque-wear is a deterministic
row-wise calculation. Whole-file EDA occurred before modelling, and repeated
validation inspection informed selection; these limit claims of independence.

## Saved evaluation results

The following results are read from the completed project records, not produced
by rerunning evaluation during export. AP means average precision across score
thresholds; the other metrics use threshold 0.5.

<!-- SAVED_METRICS -->

The once-only test has 53 positive labels: 43 detected and 10 missed, with 14
false alarms. Accuracy alone is misleading because positives are rare.
There are no uncertainty intervals or verified performance estimates for real
equipment. Test results were not used to revise the model or threshold.

## Local usage

This repository uses a native XGBoost `.ubj` file and a fitted `.skops`
preprocessor, rather than pickling the custom training estimator. The inference
helper contains the feature calculation and loads both artifacts. MLflow and
the original project checkout are not required for prediction.

After obtaining the complete model repository, install `requirements.txt` in a
compatible environment. The export records the original Python version in
`export_manifest.json`; inference was checked in that environment only.
From the downloaded model directory:

```python
import pandas as pd
from predict import FailurePredictor

model = FailurePredictor()
rows = pd.DataFrame({
    "Type": ["L"],
    "Air temperature [K]": [300.0],
    "Process temperature [K]": [310.0],
    "Rotational speed [rpm]": [1500.0],
    "Torque [Nm]": [40.0],
    "Tool wear [min]": [100.0],
})
print(model.predict(rows))
```

This is a fabricated example, not a test observation. The model does not provide
a hosted inference endpoint merely by being stored on the Hub. A Gradio Space
is a later, separate step.

## Explainability and limitations

Tree SHAP explained all validation rows in raw log-odds. Rotational speed,
torque, and air temperature had the largest mean absolute contributions;
torque-wear ranked fifth. SHAP explains model predictions, not proven causes of
failure, correctness, or the effect of removing a feature.

The random row split is not a temporal holdout or a machine-group holdout.
Synthetic generation, row-order dependence, label assumptions, repeated
validation inspection, and the small positive test count restrict generalisation.
There is no defined future prediction horizon, external validation, calibrated
risk estimate, or monitoring system.

## Files and provenance

- `model.ubj`: unchanged selected XGBoost model.
- `preprocessor.skops`: unchanged fitted preprocessor, loaded with no additional
  trusted types.
- `predict.py`: standalone prediction helper.
- `model_contract.json`: input order, target policy, threshold, checksums,
  source MLflow run IDs, and saved validation/test metrics.
- `requirements.txt`: pinned inference dependencies from the registered release.
- `export_manifest.json`: file checksums and local version lineage.
- `LICENSE`: MIT licence for this project's code and trained model artifacts.

The raw dataset, row-level predictions, credentials, local database, absolute
workspace paths, and MLflow tracking files are not part of this package.
Checksums detect differences relative to the accompanying manifest; they do not
independently authenticate a publisher. Review code and artifact provenance
before executing a downloaded package.

## Licence and publication status

This project's code and trained model artifacts are released under the
[MIT License](LICENSE), copyright (c) 2026 Hadis Surya Al Amin.
The source dataset retains its CC BY 4.0 licence and attribution above.
Third-party dependencies retain their respective licences.
This is a locally prepared model card; no Hub upload or Space deployment is
performed by the preparation script.
