# %% Step 1: restore the saved allocation and prepare allowed inputs
r"""Manual exercise: always-negative and balanced logistic-regression baselines.

This exercise has been run manually; its saved validation results were verified.
Read the numbered # %% cells, then run them individually or run this whole file:

    .\.venv\Scripts\python.exe .\scripts\02_baselines.py

Both models fit training rows and are evaluated on validation rows only.
The script saves local MLflow runs, skops models, validation plots and scores.
It does not resplit the data, tune a threshold, or evaluate the final test set.
"""

from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd
import skops.io
from sklearn.dummy import DummyClassifier


candidate_roots = [Path.cwd(), *Path.cwd().parents]
if "__file__" in globals():
    candidate_roots.insert(0, Path(__file__).resolve().parents[1])
PROJECT_ROOT = next(
    (path for path in candidate_roots if (path / "src/maintain_predict").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("Run this exercise from the maintain-predict repository.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maintain_predict.data import load_saved_splits
from maintain_predict.evaluate import classification_metrics, save_validation_plots
from maintain_predict.features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY, select_features_and_target
from maintain_predict.train import make_logistic_pipeline

DATA_PATH = PROJECT_ROOT / "data/ai4i2020.csv"
MANIFEST_PATH = PROJECT_ROOT / "data/splits/type_target_seed42.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/baselines"
THRESHOLD = 0.5  # Fixed for this exercise; threshold selection is a later stage.

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
splits = load_saved_splits(DATA_PATH, MANIFEST_PATH)
X_train, y_train = select_features_and_target(splits.train)
X_validation, y_validation = select_features_and_target(splits.validation)
# Do not prepare test predictions or pass splits.test to either model.
print("Step 1: restored the existing split; no new random allocation")
print("Training input shape:", X_train.shape)
print("Validation input shape:", X_validation.shape)
print("Input columns:", list(X_train.columns))
print("Target:", TARGET_COLUMN, "with policy", TARGET_POLICY)


# %% Step 2: inspect the training/validation class balance before fitting
print("\nStep 2: training target counts")
print(y_train.value_counts().sort_index().to_string())
print("Validation positive rate:", f"{y_validation.mean():.2%}")
print("Keep the final test partition reserved for the end of the project.")


# %% Step 3: set up local tracking and define how each completed run is saved
# The tracking database and artifacts stay inside the repository's ignored paths.
tracking_uri = "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
mlflow.set_tracking_uri(tracking_uri)
experiment_name = "maintain-predict-baselines"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(
        experiment_name, artifact_location=(PROJECT_ROOT / "mlruns").as_uri()
    )
else:
    experiment_id = experiment.experiment_id

environment = {
    name: version(name) for name in ["numpy", "pandas", "scikit-learn", "mlflow", "skops"]
}
environment["python"] = platform.python_version()
common_parameters = {
    "source_sha256": manifest["source_sha256"],
    "target_policy": TARGET_POLICY,
    "target_name": TARGET_COLUMN,
    "random_state": manifest["random_state"],
    "stratification": "Type x Failure evidence",
    "feature_columns": json.dumps(list(FEATURE_COLUMNS)),
    "training_rows": len(X_train),
    "validation_rows": len(X_validation),
    "evaluation_partition": "validation",
    "threshold": THRESHOLD,
    "test_evaluated": False,
}
results = {}


def record_run(name, model, scores, metrics):
    """Save outputs inside the active MLflow run; never read the test partition."""
    model_dir = OUTPUT_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    model_path = model_dir / "model.skops"
    skops.io.dump(model, model_path)
    predictions_path = model_dir / "validation_predictions.csv"
    pd.DataFrame({
        "source_row_index": splits.validation.index,
        "UDI": splits.validation["UDI"].to_numpy(),
        TARGET_COLUMN: y_validation.to_numpy(),
        "failure_score": scores,
        "predicted_label": (scores >= THRESHOLD).astype(int),
    }).to_csv(predictions_path, index=False)

    plots = save_validation_plots(
        y_validation, scores, metrics, model_dir, title=name.replace("_", " ").title()
    )
    mlflow.log_params(common_parameters)
    mlflow.log_param("precision_defined", metrics["precision"] is not None)
    mlflow.log_metrics({
        "validation_" + key: metrics[key]
        for key in ["average_precision", "accuracy", "precision", "recall", "f1"]
        if metrics[key] is not None
    })
    mlflow.log_metrics({
        "validation_" + key: value for key, value in metrics["confusion_matrix"].items()
    })
    mlflow.log_dict(environment, "environment.json")
    mlflow.log_artifact(str(MANIFEST_PATH))
    for artifact in [metrics_path, model_path, predictions_path, *plots]:
        mlflow.log_artifact(str(artifact))
    results[name] = {
        "validation_metrics": metrics,
        "mlflow_run_id": mlflow.active_run().info.run_id,
        "model_path": str(model_path),
    }


# %% Step 4: baseline 0 always predicts no failure evidence
# fit establishes the two class labels; this classifier ignores sensor values.
# Think first: what should its accuracy and recall be on our validation set?
with mlflow.start_run(experiment_id=experiment_id, run_name="always_no_failure"):
    baseline_zero = DummyClassifier(strategy="constant", constant=0)
    baseline_zero.fit(X_train, y_train)
    print("\nStep 4: baseline class order:", baseline_zero.classes_)
    baseline_scores = baseline_zero.predict_proba(X_validation)[:, 1]
    baseline_metrics = classification_metrics(y_validation, baseline_scores, threshold=THRESHOLD)
    mlflow.log_params({"model_type": "DummyClassifier", "strategy": "constant", "constant": 0})
    record_run("always_no_failure", baseline_zero, baseline_scores, baseline_metrics)
print("Always-negative validation metrics:")
print(json.dumps(baseline_metrics, indent=2))


# %% Step 5: baseline 1 learns from the six inputs using logistic regression
# Open src/maintain_predict/train.py to inspect this short pipeline factory.
# Pipeline: one-hot Type + scale numeric sensors -> balanced logistic regression.
# Only this .fit call learns preprocessing and classifier parameters.
logistic = make_logistic_pipeline(random_state=manifest["random_state"])
print("\nStep 5: unfitted logistic pipeline")
print(logistic)
with mlflow.start_run(experiment_id=experiment_id, run_name="balanced_logistic_regression"):
    logistic.fit(X_train, y_train)
    print("Learned transformed input names:")
    print(logistic.named_steps["preprocess"].get_feature_names_out())
    logistic_scores = logistic.predict_proba(X_validation)[:, 1]
    logistic_metrics = classification_metrics(y_validation, logistic_scores, threshold=THRESHOLD)
    mlflow.log_params({
        "model_type": "LogisticRegression", "class_weight": "balanced",
        "solver": "lbfgs", "max_iter": 2000,
    })
    record_run("logistic_regression", logistic, logistic_scores, logistic_metrics)
print("Logistic-regression validation metrics:")
print(json.dumps(logistic_metrics, indent=2))


# %% Step 6: compare validation results and save a summary
# AP = average precision. Precision is undefined when there are no positive predictions.
# The 0.5 decision threshold is still the fixed default, not a selected optimum.
comparison = pd.DataFrame({
    name: {
        key: result["validation_metrics"][key]
        for key in ["average_precision", "accuracy", "precision", "recall", "f1"]
    }
    for name, result in results.items()
}).T
print("\nStep 6: validation-only comparison")
print(comparison.round(4).to_string(na_rep="undefined"))
comparison.to_csv(OUTPUT_DIR / "comparison.csv", index_label="model")
summary = {
    "target_policy": TARGET_POLICY,
    "source_sha256": manifest["source_sha256"],
    "split_manifest": str(MANIFEST_PATH),
    "random_state": manifest["random_state"],
    "threshold": THRESHOLD,
    "evaluation_partition": "validation",
    "test_evaluated": False,
    "environment": environment,
    "models": results,
}
(OUTPUT_DIR / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("\nSaved outputs:", OUTPUT_DIR)
print("Local MLflow tracking URI:", tracking_uri)
print("No test predictions or test performance metrics were calculated.")
