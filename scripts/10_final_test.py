r"""Evaluate the selected saved model on the final test set once.

    .\.venv\Scripts\python.exe .\scripts\10_final_test.py

Seven inputs, torque-wear, threshold 0.5. No fitting or model comparisons.
If results already exist, print them without predicting again. An incomplete
evaluation folder blocks rerunning inference; recover using saved predictions.
"""

from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd
import skops.io
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maintain_predict.data import load_saved_splits
from maintain_predict.evaluate import classification_metrics, save_evaluation_plots
from maintain_predict.explain import SELECTED_FEATURE_COLUMNS
from maintain_predict.features import TARGET_COLUMN, TARGET_POLICY, add_torque_wear, select_features_and_target


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_selection(project_root, manifest):
    """Require exactly the same seven-input artifacts used in validation SHAP."""
    folder = project_root / "artifacts/torque_wear"
    selection = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    pointer = json.loads((project_root / "artifacts/shap/latest.json").read_text(encoding="utf-8"))
    explanation = json.loads(Path(pointer["summary_file"]).read_text(encoding="utf-8"))
    fingerprints = {
        "model_sha256": sha256(folder / "model.ubj"),
        "preprocessor_sha256": sha256(folder / "preprocessor.skops"),
    }
    if (
        selection["source_sha256"] != manifest["source_sha256"]
        or selection["target_policy"] != TARGET_POLICY
        or selection["random_state"] != manifest["random_state"]
        or selection["feature_columns"] != list(SELECTED_FEATURE_COLUMNS)
        or selection["threshold"] != 0.5
        or selection["evaluation_partition"] != "validation"
        or explanation["selected_model_run_id"] != selection["mlflow_run_id"]
        or explanation["source_sha256"] != manifest["source_sha256"]
        or explanation["feature_columns"] != list(SELECTED_FEATURE_COLUMNS)
        or explanation["threshold"] != 0.5
        or any(explanation[key] != value for key, value in fingerprints.items())
    ):
        raise ValueError("Model, features, threshold, or SHAP checksums differ from the frozen selection.")
    preprocessing = skops.io.load(folder / "preprocessor.skops", trusted=[])
    if preprocessing.feature_names_in_.tolist() != list(SELECTED_FEATURE_COLUMNS):
        raise ValueError("Selected preprocessor input schema differs.")
    model = XGBClassifier()
    model.load_model(folder / "model.ubj")
    if model.classes_.tolist() != [0, 1]:
        raise ValueError("Expected binary class order [0, 1].")
    return selection, explanation, fingerprints, preprocessing, model


def main(project_root=PROJECT_ROOT):
    project_root = Path(project_root)
    output = project_root / "artifacts/final_test"
    # Step 1: an existing final evaluation is read, never recomputed.
    if output.exists():
        if (output / "results.json").is_file():
            result = json.loads((output / "results.json").read_text(encoding="utf-8"))
            print("Final test was already evaluated. Reading saved results; no new predictions.")
            print(json.dumps(result["test_metrics"], indent=2))
            return result
        raise RuntimeError(
            "Final-test folder exists but is incomplete. Do not rerun inference. "
            "Recover metrics/plots from test_predictions.csv if it was saved."
        )

    # Step 2: restore membership and load the already selected saved model.
    manifest_path = project_root / "data/splits/type_target_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection, explanation, fingerprints, preprocessing, model = load_frozen_selection(
        project_root, manifest,
    )
    splits = load_saved_splits(project_root / "data/ai4i2020.csv", manifest_path)
    X_test, y_test = select_features_and_target(splits.test)
    X_test = add_torque_wear(X_test)
    encoded = preprocessing.transform(X_test)  # Transform only; never fit.
    environment = {
        name: version(name)
        for name in ["numpy", "pandas", "scikit-learn", "xgboost", "skops", "mlflow"]
    }
    environment["python"] = platform.python_version()
    mlflow.set_tracking_uri("sqlite:///" + (project_root / "mlflow.db").as_posix())
    experiment_name = "maintain-predict-final-test"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment else mlflow.create_experiment(
            experiment_name, artifact_location=(project_root / "mlruns").as_uri(),
        )
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name="selected_model_final_test"):
        # Step 3: freeze the full configuration BEFORE looking at predictions.
        # Exclusive directory creation prevents two callers from evaluating twice.
        output.mkdir(parents=True, exist_ok=False)
        configuration = {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_model_run_id": selection["mlflow_run_id"],
            "shap_run_id": explanation["mlflow_run_id"], **fingerprints,
            "source_sha256": manifest["source_sha256"],
            "split_manifest_sha256": sha256(manifest_path),
            "target_name": TARGET_COLUMN, "target_policy": TARGET_POLICY,
            "random_state": manifest["random_state"], "evaluation_partition": "test",
            "feature_columns": list(SELECTED_FEATURE_COLUMNS),
            "fixed_params": selection["fixed_params"], "threshold": 0.5,
            "threshold_tuned": False, "model_refitted": False,
            "selected_before_test": True, "whole_file_eda_before_split": True,
            "test_source_row_indices": splits.test.index.tolist(),
            "test_udi": splits.test["UDI"].tolist(),
            "environment": environment, "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        (output / "evaluation_config.json").write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8",
        )
        mlflow.log_params({
            "selected_model_run_id": selection["mlflow_run_id"],
            "shap_run_id": explanation["mlflow_run_id"], **fingerprints,
            "source_sha256": manifest["source_sha256"], "target_policy": TARGET_POLICY,
            "feature_columns": json.dumps(list(SELECTED_FEATURE_COLUMNS)),
            "threshold": 0.5, "threshold_tuned": False,
            "evaluation_partition": "test", "model_refitted": False,
            "observations": len(X_test),
        })

        # Step 4: the ONE test inference call. Labels are not model inputs.
        scores = model.predict_proba(encoded)[:, 1]
        predictions = pd.DataFrame({
            "source_row_index": splits.test.index,
            "UDI": splits.test["UDI"].to_numpy(),
            TARGET_COLUMN: y_test.to_numpy(),
            "failure_score": scores,
            "predicted_label": (scores >= 0.5).astype(int),
        })
        # Save predictions first so failures later can be recovered without inference.
        predictions.to_csv(output / "test_predictions.csv", index=False)

        # Step 5: compare scores against the test labels and save final metrics.
        metrics = classification_metrics(y_test, scores, threshold=0.5)
        (output / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
        )
        save_evaluation_plots(
            y_test, scores, metrics, output, title="Selected XGBoost + torque x wear",
            partition="test",
        )
        comparison = {}
        for partition, values in {
            "validation": selection["validation_metrics"], "test": metrics,
        }.items():
            comparison[partition] = {
                key: values[key] for key in
                ["average_precision", "accuracy", "precision", "recall", "f1"]
            }
            comparison[partition].update(values["confusion_matrix"])
        pd.DataFrame.from_dict(comparison, orient="index").to_csv(
            output / "validation_test_comparison.csv", index_label="partition",
        )
        folder = project_root / "artifacts/torque_wear"
        if fingerprints != {
            "model_sha256": sha256(folder / "model.ubj"),
            "preprocessor_sha256": sha256(folder / "preprocessor.skops"),
        }:
            raise RuntimeError("Selected artifact checksums changed during evaluation.")
        result = {
            **configuration, "test_evaluated": True, "test_metrics": metrics,
            "validation_metrics_at_selection": selection["validation_metrics"],
        }
        mlflow.log_metrics({
            **{
                "test_" + key: metrics[key]
                for key in ["average_precision", "accuracy", "precision", "recall", "f1"]
                if metrics[key] is not None
            },
            **{"test_" + key: value for key, value in metrics["confusion_matrix"].items()},
        })
        mlflow.log_dict(result, "results.json")
        for path in output.iterdir():
            if path.is_file():
                mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(manifest_path))
    # A completed receipt is written only after tracking and file creation succeed.
    (output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    print("Final test complete; saved under", output)
    print(json.dumps(metrics, indent=2))
    return result


if __name__ == "__main__":
    main()
