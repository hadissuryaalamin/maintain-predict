r"""Manual exercise: torque-wear alone versus torque-wear plus temperature difference.

Run yourself:
    .\.venv\Scripts\python.exe .\scripts\08_torque_wear_temperature_difference.py

Compare seven versus eight inputs using identical fixed XGBoost parameters and
training-only CV folds. Power is excluded. No search, threshold tuning, or test prediction.
"""

from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import pandas as pd
import skops.io
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maintain_predict.data import load_saved_splits
from maintain_predict.evaluate import classification_metrics, save_validation_plots
from maintain_predict.features import (
    FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY, TORQUE_WEAR_COLUMN,
    TEMPERATURE_DIFFERENCE_COLUMN,
    add_torque_wear, add_torque_wear_and_temperature_difference, select_features_and_target,
)
from maintain_predict.train import make_xgboost_pipeline


def load_reference(project_root, manifest, validation):
    """Check the seven-input torque-wear reference before fitting new models."""
    folder = project_root / "artifacts/torque_wear"
    summary = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    if (
        summary["source_sha256"] != manifest["source_sha256"]
        or summary["target_policy"] != TARGET_POLICY
        or summary["random_state"] != manifest["random_state"]
        or summary["threshold"] != 0.5
        or summary["evaluation_partition"] != "validation"
        or summary["test_evaluated"] is not False
        or summary["feature_columns"] != [*FEATURE_COLUMNS, TORQUE_WEAR_COLUMN]
        or summary["raw_feature_columns"] != list(FEATURE_COLUMNS)
        or set(summary["fixed_params"]) != {
            "model__n_estimators", "model__max_depth", "model__learning_rate",
        }
    ):
        raise ValueError("Saved XGBoost reference does not match this experiment.")
    predictions = pd.read_csv(folder / "validation_predictions.csv")
    _, target = select_features_and_target(validation)
    if (
        predictions["source_row_index"].tolist() != validation.index.tolist()
        or predictions[TARGET_COLUMN].tolist() != target.tolist()
        or predictions["UDI"].tolist() != validation["UDI"].tolist()
        or predictions["predicted_label"].tolist()
        != (predictions["failure_score"] >= 0.5).astype(int).tolist()
    ):
        raise ValueError("Saved XGBoost validation rows or labels do not match.")
    metrics = classification_metrics(target, predictions["failure_score"])
    if metrics["confusion_matrix"] != summary["validation_metrics"]["confusion_matrix"]:
        raise ValueError("Saved XGBoost confusion matrix does not match predictions.")
    return summary, metrics


def main(project_root=PROJECT_ROOT):
    # Step 1: restore the SAME allocation and read the prior XGBoost settings.
    project_root = Path(project_root)
    manifest_path = project_root / "data/splits/type_target_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = load_saved_splits(project_root / "data/ai4i2020.csv", manifest_path)
    X_train, y_train = select_features_and_target(splits.train)
    X_validation, y_validation = select_features_and_target(splits.validation)
    reference, reference_metrics = load_reference(project_root, manifest, splits.validation)

    # Step 2: reference has seven inputs; the combined variant has eight.
    X_train = add_torque_wear(X_train)
    X_validation = add_torque_wear(X_validation)
    X_train_combined = add_torque_wear_and_temperature_difference(X_train)
    X_validation_combined = add_torque_wear_and_temperature_difference(X_validation)
    print("Step 2: torque-wear reference training inputs:", X_train.shape)
    print("With temperature difference:", X_train_combined.shape)
    print(X_train_combined[[
        TORQUE_WEAR_COLUMN, TEMPERATURE_DIFFERENCE_COLUMN,
    ]].head().to_string())

    # Step 3: freeze the PREVIOUSLY selected hyperparameters for both versions.
    parameters = reference["fixed_params"]
    original = make_xgboost_pipeline(
        random_state=manifest["random_state"], include_torque_wear=True,
    )
    original.set_params(**parameters)
    augmented = make_xgboost_pipeline(
        random_state=manifest["random_state"], include_torque_wear=True,
        include_temperature_difference=True,
    )
    augmented.set_params(**parameters)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=manifest["random_state"])
    print("\nStep 3: identical fixed parameters:", parameters)

    output_dir = project_root / "artifacts/torque_wear_temperature_difference"
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (project_root / "mlflow.db").as_posix())
    experiment_name = "maintain-predict-torque-wear-temperature-difference"
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
    with mlflow.start_run(experiment_id=experiment_id, run_name="xgboost_torque_wear_temperature_difference"):
        mlflow.log_params({
            **parameters, "source_sha256": manifest["source_sha256"],
            "target_policy": TARGET_POLICY, "random_state": manifest["random_state"],
            "added_feature": TEMPERATURE_DIFFERENCE_COLUMN,
            "formula": "process temperature - air temperature, retaining torque * tool wear",
            "cv_folds": 5, "selection_metric": "average_precision",
            "threshold": 0.5, "evaluation_partition": "validation",
            "training_rows": len(X_train), "validation_rows": len(X_validation),
            "test_evaluated": False, "new_hyperparameter_search": False,
            "reference_run_id": reference["mlflow_run_id"],
        })

        # Step 4: actual fitting starts here, using training folds only.
        # Both versions get exactly the same fold membership and y_train.
        original_cv = cross_val_score(
            original, X_train, y_train, cv=cv,
            scoring="average_precision", n_jobs=1, error_score="raise",
        )
        augmented_cv = cross_val_score(
            augmented, X_train_combined, y_train, cv=cv,
            scoring="average_precision", n_jobs=1, error_score="raise",
        )
        paired_delta = augmented_cv - original_cv
        cv_results = pd.DataFrame({
            "fold": np.arange(1, cv.n_splits + 1),
            "torque_wear_ap": original_cv, "combined_ap": augmented_cv,
            "ap_difference": paired_delta,
        })
        print("\nStep 4: paired training-only CV comparison")
        print(cv_results.to_string(index=False))
        cv_gain = float(paired_delta.mean())
        print("Mean AP difference:", cv_gain)
        print("Positive CV gain is exploratory evidence, not statistical proof.")

        # Step 5: fit the eight-input version on all training rows.
        augmented.fit(X_train_combined, y_train)
        scores = augmented.predict_proba(X_validation_combined)[:, 1]
        metrics = classification_metrics(y_validation, scores, threshold=0.5)
        print("\nStep 5: validation at the unchanged threshold 0.5")
        print(json.dumps(metrics, indent=2))

        # Step 6: save this feature experiment separately.
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
        )
        pd.DataFrame({
            "source_row_index": splits.validation.index,
            "UDI": splits.validation["UDI"].to_numpy(), TARGET_COLUMN: y_validation.to_numpy(),
            "failure_score": scores, "predicted_label": (scores >= 0.5).astype(int),
        }).to_csv(output_dir / "validation_predictions.csv", index=False)
        cv_results.to_csv(output_dir / "cv_results.csv", index=False)
        augmented.named_steps["model"].save_model(output_dir / "model.ubj")
        skops.io.dump(augmented.named_steps["preprocess"], output_dir / "preprocessor.skops")
        save_validation_plots(
            y_validation, scores, metrics, output_dir, title="XGBoost + torque x wear + temperature difference",
        )
        comparison = {}
        for name, values in {
            "xgboost_torque_wear": reference_metrics,
            "xgboost_torque_wear_temperature_difference": metrics,
        }.items():
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
            "source_sha256": manifest["source_sha256"], "target_policy": TARGET_POLICY,
            "random_state": manifest["random_state"], "split_manifest": str(manifest_path),
            "threshold": 0.5, "evaluation_partition": "validation", "test_evaluated": False,
            "reference_run_id": reference["mlflow_run_id"], "fixed_params": parameters,
            "new_hyperparameter_search": False,
            "raw_feature_columns": list(FEATURE_COLUMNS),
            "reference_feature_columns": [*FEATURE_COLUMNS, TORQUE_WEAR_COLUMN],
            "feature_columns": X_train_combined.columns.tolist(),
            "added_feature": TEMPERATURE_DIFFERENCE_COLUMN,
            "formula": "Process temperature [K] - Air temperature [K]; retain Torque * Tool wear",
            "cv": {
                "reference_mean_ap": float(original_cv.mean()),
                "reference_std_ap": float(original_cv.std()),
                "augmented_mean_ap": float(augmented_cv.mean()),
                "augmented_std_ap": float(augmented_cv.std()),
                "paired_mean_ap_difference": cv_gain,
                "paired_std_ap_difference": float(paired_delta.std()),
                "exploratory_cv_gain_positive": cv_gain > 0,
            },
            "validation_metrics": metrics, "reference_validation_metrics": reference_metrics,
            "scale_pos_weight": float(
                augmented.named_steps["model"].get_params()["scale_pos_weight"],
            ),
            "environment": environment, "model_file": "model.ubj",
            "preprocessor_file": "preprocessor.skops",
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        (output_dir / "results.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        mlflow.log_metrics({
            **{key: value for key, value in summary["cv"].items() if type(value) is float},
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
    print("Review CV and validation tradeoffs before deciding whether to retain the feature.")


if __name__ == "__main__":
    main()
