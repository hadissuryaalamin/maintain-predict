r"""Manual feature exercise: add power to XGBoost.

Run yourself:
    .\.venv\Scripts\python.exe .\scripts\06_power.py

Compare six inputs with seven inputs using identical fixed hyperparameters and
training-only CV folds. Only power is added; temperature difference is excluded.
No search, threshold tuning, or test prediction.
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
    FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY, POWER_COLUMN,
    add_power, select_features_and_target,
)
from maintain_predict.train import make_xgboost_pipeline


def load_reference(project_root, manifest, validation):
    """Check the six-input XGBoost reference before fitting new models."""
    folder = project_root / "artifacts/xgboost"
    summary = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    if (
        summary["source_sha256"] != manifest["source_sha256"]
        or summary["target_policy"] != TARGET_POLICY
        or summary["random_state"] != manifest["random_state"]
        or summary["threshold"] != 0.5
        or summary["evaluation_partition"] != "validation"
        or summary["test_evaluated"] is not False
        or summary["feature_columns"] != list(FEATURE_COLUMNS)
        or set(summary["best_params"]) != {
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

    # Step 2: add one new feature; the original six remain.
    # Power [W] = Torque [Nm] * Rotational speed [rpm] * 2*pi/60
    X_train_power = add_power(X_train)
    X_validation_power = add_power(X_validation)
    print("Step 2: original training inputs:", X_train.shape)
    print("With power:", X_train_power.shape)
    print(X_train_power[[
        "Torque [Nm]", "Rotational speed [rpm]", POWER_COLUMN,
    ]].head().to_string())

    # Step 3: freeze the PREVIOUSLY selected hyperparameters for both versions.
    parameters = reference["best_params"]
    original = make_xgboost_pipeline(random_state=manifest["random_state"])
    original.set_params(**parameters)
    augmented = make_xgboost_pipeline(
        random_state=manifest["random_state"], include_power=True,
    )
    augmented.set_params(**parameters)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=manifest["random_state"])
    print("\nStep 3: identical fixed parameters:", parameters)

    output_dir = project_root / "artifacts/power"
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (project_root / "mlflow.db").as_posix())
    experiment_name = "maintain-predict-power"
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
    with mlflow.start_run(experiment_id=experiment_id, run_name="xgboost_power"):
        mlflow.log_params({
            **parameters, "source_sha256": manifest["source_sha256"],
            "target_policy": TARGET_POLICY, "random_state": manifest["random_state"],
            "added_feature": POWER_COLUMN,
            "formula": "torque * rotational speed * 2*pi/60",
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
            augmented, X_train_power, y_train, cv=cv,
            scoring="average_precision", n_jobs=1, error_score="raise",
        )
        paired_delta = augmented_cv - original_cv
        cv_results = pd.DataFrame({
            "fold": np.arange(1, cv.n_splits + 1),
            "original_ap": original_cv, "power_ap": augmented_cv,
            "ap_difference": paired_delta,
        })
        print("\nStep 4: paired training-only CV comparison")
        print(cv_results.to_string(index=False))
        cv_gain = float(paired_delta.mean())
        print("Mean AP difference:", cv_gain)
        print("Positive CV gain is exploratory evidence, not statistical proof.")

        # Step 5: fit the seven-input version on all training rows.
        augmented.fit(X_train_power, y_train)
        scores = augmented.predict_proba(X_validation_power)[:, 1]
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
            y_validation, scores, metrics, output_dir, title="XGBoost + power",
        )
        comparison = {}
        for name, values in {
            "xgboost_original": reference_metrics,
            "xgboost_power": metrics,
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
            "feature_columns": X_train_power.columns.tolist(),
            "added_feature": POWER_COLUMN,
            "formula": "Torque [Nm] * Rotational speed [rpm] * 2*pi/60",
            "cv": {
                "original_mean_ap": float(original_cv.mean()),
                "original_std_ap": float(original_cv.std()),
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
