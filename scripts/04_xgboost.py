r"""Manual exercise: XGBoost on the frozen split.

Run yourself from the repository root:
    .\.venv\Scripts\python.exe .\scripts\04_xgboost.py

Training starts at search.fit(X_train, y_train). The five-fold search never
uses validation labels. There is no early stopping or test-set evaluation.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maintain_predict.data import load_saved_splits
from maintain_predict.evaluate import classification_metrics, save_validation_plots
from maintain_predict.features import (
    FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY, select_features_and_target,
)
from maintain_predict.train import make_xgboost_search


def load_previous_results(project_root, manifest, validation):
    """Check earlier experiments against current validation membership and labels."""
    artifacts = project_root / "artifacts"
    baseline_dir = artifacts / "baselines"
    baseline = json.loads((baseline_dir / "results.json").read_text(encoding="utf-8"))
    forest_dir = artifacts / "random_forest"
    forest = json.loads((forest_dir / "results.json").read_text(encoding="utf-8"))
    _, target = select_features_and_target(validation)
    for summary in [baseline, forest]:
        if (
            summary["source_sha256"] != manifest["source_sha256"]
            or summary["target_policy"] != TARGET_POLICY
            or summary["random_state"] != manifest["random_state"]
            or summary["evaluation_partition"] != "validation"
            or summary["threshold"] != 0.5
            or summary["test_evaluated"] is not False
        ):
            raise ValueError("Saved experiments do not match the current evaluation.")
    metrics = {}
    for name, directory, values in [
        *[
            (name, baseline_dir / name, item["validation_metrics"])
            for name, item in baseline["models"].items()
        ],
        ("random_forest", forest_dir, forest["validation_metrics"]),
    ]:
        predictions = pd.read_csv(directory / "validation_predictions.csv")
        if (
            predictions["source_row_index"].tolist() != validation.index.tolist()
            or predictions[TARGET_COLUMN].tolist() != target.tolist()
            or predictions["UDI"].tolist() != validation["UDI"].tolist()
        ):
            raise ValueError("Saved validation rows or labels do not match.")
        recalculated = classification_metrics(target, predictions["failure_score"])
        if (
            recalculated["confusion_matrix"] != values["confusion_matrix"]
            or ((predictions["failure_score"] >= 0.5).astype(int)
                != predictions["predicted_label"]).any()
        ):
            raise ValueError("Saved comparison metrics or predictions are inconsistent.")
        metrics[name] = values
    return metrics


def main(project_root=PROJECT_ROOT):
    # Step 1: restore the existing split and check the previous results.
    project_root = Path(project_root)
    manifest_path = project_root / "data/splits/type_target_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = load_saved_splits(project_root / "data/ai4i2020.csv", manifest_path)
    X_train, y_train = select_features_and_target(splits.train)
    X_validation, y_validation = select_features_and_target(splits.validation)
    previous = load_previous_results(project_root, manifest, splits.validation)
    print("Step 1: training", X_train.shape, "; validation", X_validation.shape)
    print("Target:", TARGET_COLUMN, "; policy:", TARGET_POLICY)
    print("No test predictions will be generated.")

    # Step 2: inspect the unfitted pipeline and search.
    # FoldWeightedXGBClassifier.fit calculates negative/positive from each fold.
    search = make_xgboost_search(random_state=manifest["random_state"])
    print("\nStep 2: one-hot Type + unchanged sensors -> weighted XGBoost")
    print(search.estimator)
    print("Parameter choices:", search.param_grid)

    # Step 3: track this exercise in its own experiment and output folder.
    output_dir = project_root / "artifacts/xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (project_root / "mlflow.db").as_posix())
    experiment_name = "maintain-predict-xgboost"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment else mlflow.create_experiment(
            experiment_name, artifact_location=(project_root / "mlruns").as_uri(),
        )
    )
    environment = {
        name: version(name)
        for name in ["numpy", "pandas", "scikit-learn", "mlflow", "skops", "xgboost"]
    }
    environment["python"] = platform.python_version()

    with mlflow.start_run(experiment_id=experiment_id, run_name="weighted_xgboost"):
        mlflow.log_params({
            "source_sha256": manifest["source_sha256"],
            "target_policy": TARGET_POLICY,
            "random_state": manifest["random_state"],
            "feature_columns": json.dumps(list(FEATURE_COLUMNS)),
            "training_rows": len(X_train), "validation_rows": len(X_validation),
            "class_weight_policy": "negative/positive recomputed on each fit",
            "threshold": 0.5, "evaluation_partition": "validation",
            "test_evaluated": False, "selection_metric": "average_precision",
            "cv_stratification": "target", "cv_folds": search.cv.n_splits,
            "objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
            "early_stopping": False,
        })

        # Step 4: actual training -- candidates see training folds only.
        # refit=True then fits the winner using all training rows.
        print("\nStep 4: training-only CV search, then refit")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        classifier = model.named_steps["model"]
        index = search.best_index_
        cv = {
            "mean_ap": float(search.cv_results_["mean_test_score"][index]),
            "std_ap": float(search.cv_results_["std_test_score"][index]),
            "fold_ap": [
                float(search.cv_results_[f"split{i}_test_score"][index])
                for i in range(search.cv.n_splits)
            ],
        }
        print("Selected parameters:", search.best_params_)
        print("Training-only CV:", json.dumps(cv, indent=2))
        print("Final training scale_pos_weight:", classifier.get_params()["scale_pos_weight"])

        # Step 5: validation inference with the untuned 0.5 threshold.
        print("\nStep 5: validation at threshold 0.5")
        scores = model.predict_proba(X_validation)[:, 1]
        metrics = classification_metrics(y_validation, scores, threshold=0.5)
        print(json.dumps(metrics, indent=2))

        # Step 6: save outputs without overwriting any earlier model.
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

        # Use XGBoost's native format for the classifier and skops for preprocessing.
        # The saved model can be loaded into an ordinary XGBClassifier.
        classifier.save_model(output_dir / "model.ubj")
        skops.io.dump(model.named_steps["preprocess"], output_dir / "preprocessor.skops")
        pd.DataFrame(search.cv_results_).to_csv(output_dir / "cv_results.csv", index=False)
        save_validation_plots(y_validation, scores, metrics, output_dir, title="XGBoost")
        comparison = {}
        for name, values in {**previous, "xgboost": metrics}.items():
            comparison[name] = {
                key: values[key]
                for key in ["average_precision", "accuracy", "precision", "recall", "f1"]
            }
            comparison[name].update(values["confusion_matrix"])
        comparison = pd.DataFrame.from_dict(comparison, orient="index")
        comparison.to_csv(output_dir / "comparison.csv", index_label="model")
        print("\nValidation comparison")
        print(comparison.round(4).to_string(na_rep="undefined"))
        summary = {
            "source_sha256": manifest["source_sha256"],
            "target_policy": TARGET_POLICY, "random_state": manifest["random_state"],
            "split_manifest": str(manifest_path),
            "threshold": 0.5, "evaluation_partition": "validation",
            "test_evaluated": False, "best_params": search.best_params_,
            "scale_pos_weight": float(classifier.get_params()["scale_pos_weight"]),
            "cv": cv, "validation_metrics": metrics, "environment": environment,
            "model_file": "model.ubj", "preprocessor_file": "preprocessor.skops",
            "feature_columns": list(FEATURE_COLUMNS),
            "encoded_feature_names": model.named_steps["preprocess"].get_feature_names_out().tolist(),
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        (output_dir / "results.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        mlflow.log_params({
            **search.best_params_, "final_scale_pos_weight": summary["scale_pos_weight"],
        })
        mlflow.log_metrics({
            "cv_ap_mean": cv["mean_ap"], "cv_ap_std": cv["std_ap"],
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
    print("No test evaluation, threshold tuning, or earlier-model overwrites.")


if __name__ == "__main__":
    main()
