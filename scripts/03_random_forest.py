r"""Manual exercise: train and compare Random Forest.

Read the numbered steps, then run this script yourself from the project root:
    .\.venv\Scripts\python.exe .\scripts\03_random_forest.py

Search uses training rows only. Validation uses the fixed 0.5 threshold.
The script never predicts on test rows or overwrites the original baselines.
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
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maintain_predict.data import load_saved_splits
from maintain_predict.evaluate import classification_metrics, save_validation_plots
from maintain_predict.features import (
    FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY, select_features_and_target,
)
from maintain_predict.train import make_logistic_pipeline, make_random_forest_search


def load_baseline_comparison(project_root, manifest, validation_indices):
    """Reject a stale comparison before spending time on training."""
    baseline_dir = project_root / "artifacts/baselines"
    summary = json.loads((baseline_dir / "results.json").read_text(encoding="utf-8"))
    if (
        summary["source_sha256"] != manifest["source_sha256"]
        or summary["target_policy"] != TARGET_POLICY
        or summary["random_state"] != manifest["random_state"]
        or summary["evaluation_partition"] != "validation"
        or summary["threshold"] != 0.5
        or summary["test_evaluated"] is not False
    ):
        raise ValueError("Saved baselines do not match this source, policy, or evaluation.")
    metrics = {}
    for name, item in summary["models"].items():
        predictions = pd.read_csv(baseline_dir / name / "validation_predictions.csv")
        if predictions["source_row_index"].tolist() != list(validation_indices):
            raise ValueError("Saved baseline validation membership does not match.")
        metrics[name] = item["validation_metrics"]
    return metrics


def main(project_root=PROJECT_ROOT):
    # Step 1: restore the EXISTING split; do not make a new allocation.
    project_root = Path(project_root)
    manifest_path = project_root / "data/splits/type_target_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = load_saved_splits(project_root / "data/ai4i2020.csv", manifest_path)
    X_train, y_train = select_features_and_target(splits.train)
    X_validation, y_validation = select_features_and_target(splits.validation)
    baseline_metrics = load_baseline_comparison(
        project_root, manifest, splits.validation.index,
    )
    print("Step 1: training", X_train.shape, "; validation", X_validation.shape)
    print("Target:", TARGET_COLUMN, "; policy:", TARGET_POLICY)
    print("The final test partition is reserved.")

    # Step 2: inspect the UNFITTED pipeline and small search.
    search = make_random_forest_search(random_state=manifest["random_state"])
    print("\nStep 2: Random Forest pipeline")
    print(search.estimator)
    print("Parameter choices:", search.param_grid)
    print("One-hot Type + unchanged sensors -> balanced Random Forest")

    # Step 3: configure tracking. The upcoming .fit() performs actual training.
    output_dir = project_root / "artifacts/random_forest"
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (project_root / "mlflow.db").as_posix())
    experiment_name = "maintain-predict-random-forest"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment else mlflow.create_experiment(
            experiment_name, artifact_location=(project_root / "mlruns").as_uri(),
        )
    )
    environment = {
        name: version(name)
        for name in ["numpy", "pandas", "scikit-learn", "mlflow", "skops"]
    }
    environment["python"] = platform.python_version()

    with mlflow.start_run(experiment_id=experiment_id, run_name="balanced_random_forest"):
        mlflow.log_params({
            "source_sha256": manifest["source_sha256"],
            "target_policy": TARGET_POLICY,
            "random_state": manifest["random_state"],
            "feature_columns": json.dumps(list(FEATURE_COLUMNS)),
            "training_rows": len(X_train),
            "validation_rows": len(X_validation),
            "class_weight": "balanced", "threshold": 0.5,
            "evaluation_partition": "validation", "test_evaluated": False,
            "selection_metric": "average_precision",
            "cv_stratification": "target", "cv_folds": search.cv.n_splits,
        })

        # Step 4: train candidates ONLY inside training folds.
        # GridSearchCV fits each pipeline separately in each fold.
        # refit=True then trains the selected pipeline on all training rows.
        print("\nStep 4: training-only search, then refit on all training rows")
        search.fit(X_train, y_train)
        forest = search.best_estimator_
        best_index = search.best_index_
        forest_cv_mean = float(search.cv_results_["mean_test_score"][best_index])
        forest_cv_std = float(search.cv_results_["std_test_score"][best_index])
        print("Selected parameters:", search.best_params_)
        print(f"Selected Random Forest CV AP: {forest_cv_mean:.4f} +/- {forest_cv_std:.4f}")

        # Fresh logistic clones use the same training folds for a CV reference.
        # This does not overwrite or refit the saved baseline model artifact.
        logistic_cv = cross_val_score(
            make_logistic_pipeline(random_state=manifest["random_state"]),
            X_train, y_train, cv=search.cv, scoring="average_precision",
            n_jobs=1, error_score="raise",
        )
        cv_comparison = {
            "random_forest_selected": {
                "mean_ap": forest_cv_mean, "std_ap": forest_cv_std,
                "fold_ap": [
                    float(search.cv_results_[f"split{i}_test_score"][best_index])
                    for i in range(search.cv.n_splits)
                ],
            },
            "logistic_regression_reference": {
                "mean_ap": float(logistic_cv.mean()),
                "std_ap": float(logistic_cv.std()),
                "fold_ap": logistic_cv.tolist(),
            },
        }
        print("Training-only CV comparison:", json.dumps(cv_comparison, indent=2))

        # Step 5: predict validation ONCE at the fixed threshold.
        print("\nStep 5: validation evaluation at threshold 0.5")
        scores = forest.predict_proba(X_validation)[:, 1]
        metrics = classification_metrics(y_validation, scores, threshold=0.5)
        print(json.dumps(metrics, indent=2))

        # Step 6: save this experiment separately from the original baselines.
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
        )
        pd.DataFrame({
            "source_row_index": splits.validation.index,
            "UDI": splits.validation["UDI"].to_numpy(),
            TARGET_COLUMN: y_validation.to_numpy(),
            "failure_score": scores,
            "predicted_label": (scores >= 0.5).astype(int),
        }).to_csv(output_dir / "validation_predictions.csv", index=False)
        skops.io.dump(forest, output_dir / "model.skops")
        pd.DataFrame(search.cv_results_).to_csv(output_dir / "cv_results.csv", index=False)
        (output_dir / "cv_comparison.json").write_text(
            json.dumps(cv_comparison, indent=2) + "\n", encoding="utf-8",
        )
        save_validation_plots(y_validation, scores, metrics, output_dir, title="Random Forest")
        comparison = {}
        for name, values in {**baseline_metrics, "random_forest": metrics}.items():
            comparison[name] = {
                key: values[key]
                for key in ["average_precision", "accuracy", "precision", "recall", "f1"]
            }
            comparison[name].update(values["confusion_matrix"])
        comparison = pd.DataFrame.from_dict(comparison, orient="index")
        comparison.to_csv(output_dir / "comparison.csv", index_label="model")
        print("\nValidation comparison (no test evaluation)")
        print(comparison.round(4).to_string(na_rep="undefined"))
        summary = {
            "source_sha256": manifest["source_sha256"],
            "target_policy": TARGET_POLICY, "random_state": manifest["random_state"],
            "split_manifest": str(manifest_path),
            "threshold": 0.5, "evaluation_partition": "validation",
            "test_evaluated": False, "best_params": search.best_params_,
            "cv": cv_comparison, "validation_metrics": metrics,
            "environment": environment, "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        (output_dir / "results.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({
            "forest_cv_ap_mean": forest_cv_mean, "forest_cv_ap_std": forest_cv_std,
            "logistic_cv_ap_mean": float(logistic_cv.mean()),
            "logistic_cv_ap_std": float(logistic_cv.std()),
            **{
                "validation_" + key: metrics[key]
                for key in ["average_precision", "accuracy", "precision", "recall", "f1"]
                if metrics[key] is not None
            },
            **{"validation_" + key: value for key, value in metrics["confusion_matrix"].items()},
        })
        mlflow.log_dict(environment, "environment.json")
        mlflow.log_artifact(str(manifest_path))
        for artifact in output_dir.iterdir():
            if artifact.is_file():
                mlflow.log_artifact(str(artifact))
    print("\nSaved:", output_dir)
    print("No test predictions, threshold tuning, or baseline overwrites.")


if __name__ == "__main__":
    main()
