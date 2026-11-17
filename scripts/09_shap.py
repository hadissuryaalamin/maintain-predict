r"""Manual exercise: explain the saved seven-input model, without retraining.

Run yourself:
    .\.venv\Scripts\python.exe .\scripts\09_shap.py

SHAP is computed on validation only. Keep the retained threshold 0.5 and final
test set reserved. The numbered steps load, check, explain, plot, and record.
"""

from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import skops.io
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maintain_predict.data import load_saved_splits
from maintain_predict.evaluate import classification_metrics
from maintain_predict.explain import (
    SELECTED_FEATURE_COLUMNS, group_encoded_contributions,
    reconstruction_errors, representative_positions,
)
from maintain_predict.features import (
    TARGET_COLUMN, TARGET_POLICY, add_torque_wear, select_features_and_target,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_selected_model(project_root, manifest, validation):
    """Validate the selected artifact against the saved validation predictions."""
    directory = project_root / "artifacts/torque_wear"
    summary = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if (
        summary["source_sha256"] != manifest["source_sha256"]
        or summary["target_policy"] != TARGET_POLICY
        or summary["random_state"] != manifest["random_state"]
        or summary["feature_columns"] != list(SELECTED_FEATURE_COLUMNS)
        or summary["threshold"] != 0.5
        or summary["evaluation_partition"] != "validation"
        or summary["test_evaluated"] is not False
    ):
        raise ValueError("Saved model does not match the selected seven-input experiment.")
    X, target = select_features_and_target(validation)
    X = add_torque_wear(X)
    previous = pd.read_csv(directory / "validation_predictions.csv")
    if (
        previous["source_row_index"].tolist() != validation.index.tolist()
        or previous["UDI"].tolist() != validation["UDI"].tolist()
        or previous[TARGET_COLUMN].tolist() != target.tolist()
    ):
        raise ValueError("Saved validation membership or labels differ from the frozen split.")
    # Only the selected sklearn preprocessor is loaded; no custom code is trusted.
    preprocessing = skops.io.load(directory / "preprocessor.skops", trusted=[])
    if preprocessing.feature_names_in_.tolist() != list(SELECTED_FEATURE_COLUMNS):
        raise ValueError("Saved preprocessor has a different input feature schema.")
    classifier = XGBClassifier()
    classifier.load_model(directory / "model.ubj")
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Expected class order [0, 1] for failure-evidence scores.")
    encoded = preprocessing.transform(X)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded_names = preprocessing.get_feature_names_out().tolist()
    encoded = pd.DataFrame(encoded, columns=encoded_names, index=X.index)
    scores = classifier.predict_proba(encoded.to_numpy())[:, 1]
    if (
        not np.allclose(scores, previous["failure_score"], atol=1e-7, rtol=1e-6)
        or previous["predicted_label"].tolist() != (scores >= 0.5).astype(int).tolist()
        or classification_metrics(target, scores)["confusion_matrix"]
        != summary["validation_metrics"]["confusion_matrix"]
    ):
        raise ValueError("Loaded model does not reproduce the saved validation predictions.")
    return summary, classifier, X, target, encoded, scores


def save_plots(directory, explanation, grouped, X, rows, choices, scores, target, predicted):
    """Plot global groups, encoded contributions, and deterministic local examples."""
    importance = pd.Series(
        np.abs(grouped).mean(axis=0), index=SELECTED_FEATURE_COLUMNS,
    ).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    bars = ax.barh(importance.index, importance.to_numpy(), color="#275b75")
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.margins(x=0.18)
    ax.set(
        xlabel="Mean absolute grouped SHAP value (log-odds)",
        title="Selected XGBoost: validation feature contributions",
    )
    fig.savefig(directory / "global_importance.png", dpi=150)
    plt.close(fig)

    shap.plots.beeswarm(explanation, max_display=len(explanation.feature_names), show=False)
    plt.gca().set_xlabel("SHAP value (log-odds; positive pushes toward failure evidence)")
    plt.title("Validation contributions: encoded features")
    plt.gcf().savefig(directory / "beeswarm_encoded.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    examples = []
    for outcome, position in choices.items():
        udi = int(rows["UDI"].iloc[position])
        local = shap.Explanation(
            values=grouped[position],
            base_values=float(explanation.base_values[position]),
            data=X.iloc[position].to_numpy(),
            feature_names=list(SELECTED_FEATURE_COLUMNS),
        )
        shap.plots.waterfall(local, max_display=len(SELECTED_FEATURE_COLUMNS), show=False)
        plt.gcf().suptitle(
            f"{outcome} | UDI {udi} | score {scores[position]:.3f}"
            f" | actual {int(target.iloc[position])} | predicted {int(predicted[position])}",
            fontsize=10, y=1.02,
        )
        filename = f"waterfall_{outcome}_UDI{udi}.png"
        plt.gcf().savefig(directory / filename, dpi=150, bbox_inches="tight")
        plt.close("all")
        examples.append({
            "outcome": outcome, "source_row_index": int(rows.index[position]),
            "UDI": udi, "score": float(scores[position]),
            "actual": int(target.iloc[position]), "predicted": int(predicted[position]),
            "file": filename,
        })
    return examples


def main(project_root=PROJECT_ROOT):
    # Step 1: restore validation membership; never use test rows.
    project_root = Path(project_root)
    manifest_path = project_root / "data/splits/type_target_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = load_saved_splits(project_root / "data/ai4i2020.csv", manifest_path)
    summary, classifier, X, target, encoded, scores = load_selected_model(
        project_root, manifest, splits.validation,
    )
    predicted = (scores >= 0.5).astype(int)
    model_dir = project_root / "artifacts/torque_wear"
    fingerprints = {
        "model_sha256": sha256(model_dir / "model.ubj"),
        "preprocessor_sha256": sha256(model_dir / "preprocessor.skops"),
    }
    print("Step 1: loaded the saved model; predictions match its existing validation results.")
    print("Seven input columns:", list(X.columns))

    # Step 2: configure Tree SHAP in raw/log-odds space.
    # Path counts already stored in the trained model provide its reference.
    explainer = shap.TreeExplainer(
        classifier, model_output="raw",
        feature_perturbation="tree_path_dependent",
        feature_names=encoded.columns.tolist(),
    )
    print("\nStep 2: no refitting; baseline and contributions are in log-odds.")

    # Step 3: calculate explanations for ALL validation rows, in saved order.
    # The saved classifier was fitted on a numeric matrix. Keep display names
    # in TreeExplainer, but do not pass bracketed unit labels into XGBoost DMatrix.
    explanation = explainer(encoded.to_numpy(), check_additivity=True, approximate=False)
    if np.asarray(explanation.values).ndim != 2:
        raise ValueError("Expected one binary log-odds output per encoded feature.")
    explanation.base_values = np.broadcast_to(
        np.asarray(explanation.base_values), (len(encoded),),
    ).copy()
    grouped = group_encoded_contributions(explanation.values, encoded.columns.tolist())
    margins = classifier.predict(encoded.to_numpy(), output_margin=True)
    checks = reconstruction_errors(explanation.base_values, grouped, margins, scores)
    print("Step 3: baseline + SHAP contributions reproduces the original model output.")
    print(json.dumps(checks, indent=2))

    # Step 4: aggregate Type indicators, rank global contributions, pick examples.
    # Global rank is mean(abs(signed sum)) per group, not an ablation test.
    importance = pd.DataFrame({
        "feature": list(SELECTED_FEATURE_COLUMNS),
        "mean_abs_shap_log_odds": np.abs(grouped).mean(axis=0),
    }).sort_values("mean_abs_shap_log_odds", ascending=False)
    choices = representative_positions(target, predicted, X.index)
    print("\nStep 4: global feature contributions")
    print(importance.to_string(index=False))

    # Step 5: save each explanation run in its own folder and track it.
    mlflow.set_tracking_uri("sqlite:///" + (project_root / "mlflow.db").as_posix())
    experiment_name = "maintain-predict-shap"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment else mlflow.create_experiment(
            experiment_name, artifact_location=(project_root / "mlruns").as_uri(),
        )
    )
    environment = {
        name: version(name)
        for name in ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "mlflow", "skops"]
    }
    environment["python"] = platform.python_version()
    with mlflow.start_run(experiment_id=experiment_id, run_name="selected_model_validation_shap"):
        run_id = mlflow.active_run().info.run_id
        directory = project_root / "artifacts/shap" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        importance.to_csv(directory / "global_importance.csv", index=False)
        pd.DataFrame({
            "encoded_feature": encoded.columns,
            "mean_abs_shap_log_odds": np.abs(explanation.values).mean(axis=0),
        }).sort_values("mean_abs_shap_log_odds", ascending=False).to_csv(
            directory / "encoded_global_importance.csv", index=False,
        )
        identity = pd.DataFrame({
            "source_row_index": X.index, "UDI": splits.validation["UDI"].to_numpy(),
        })
        pd.concat([
            identity,
            pd.DataFrame(grouped, columns=SELECTED_FEATURE_COLUMNS),
        ], axis=1).to_csv(directory / "shap_values_grouped.csv", index=False)
        pd.concat([
            identity, pd.DataFrame(explanation.values, columns=encoded.columns),
        ], axis=1).to_csv(directory / "shap_values_encoded.csv", index=False)
        outcomes = [
            ("TP" if p else "FN") if y else ("FP" if p else "TN")
            for y, p in zip(target, predicted)
        ]
        pd.concat([
            identity, X.reset_index(drop=True),
            pd.DataFrame({
                TARGET_COLUMN: target.to_numpy(), "score": scores,
                "predicted_label": predicted, "outcome": outcomes,
                "base_log_odds": explanation.base_values, "model_log_odds": margins,
            }),
        ], axis=1).to_csv(directory / "row_explanations.csv", index=False)
        examples = save_plots(
            directory, explanation, grouped, X, splits.validation,
            choices, scores, target, predicted,
        )
        result = {
            "source_sha256": manifest["source_sha256"], "target_policy": TARGET_POLICY,
            "random_state": manifest["random_state"],
            "selected_model_run_id": summary["mlflow_run_id"], **fingerprints,
            "feature_columns": list(SELECTED_FEATURE_COLUMNS),
            "encoded_feature_names": encoded.columns.tolist(),
            "explanation_partition": "validation", "observations": len(X),
            "threshold": 0.5, "test_evaluated": False, "model_refitted": False,
            "method": "TreeExplainer", "feature_perturbation": "tree_path_dependent",
            "model_output": "raw", "units": "log-odds", "approximate": False,
            "background": "training path counts stored in the fitted trees; no external background",
            "type_grouping": "post-hoc signed sum of Type indicator contributions",
            "example_rule": "smallest source-row index per available TP/FN/FP/TN category",
            "examples": examples, "reconstruction_checks": checks,
            "global_importance": importance.to_dict(orient="records"),
            "environment": environment, "mlflow_run_id": run_id,
        }
        (directory / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8",
        )
        mlflow.log_params({
            "selected_model_run_id": summary["mlflow_run_id"],
            "source_sha256": manifest["source_sha256"], "target_policy": TARGET_POLICY,
            **fingerprints, "partition": "validation", "threshold": 0.5,
            "observations": len(X), "method": "TreeExplainer",
            "feature_perturbation": "tree_path_dependent", "units": "log-odds",
            "test_evaluated": False, "model_refitted": False,
        })
        mlflow.log_metrics(checks)
        mlflow.log_artifacts(str(directory))
        mlflow.log_artifact(str(manifest_path))
        if fingerprints != {
            "model_sha256": sha256(model_dir / "model.ubj"),
            "preprocessor_sha256": sha256(model_dir / "preprocessor.skops"),
        }:
            raise RuntimeError("Selected model artifacts changed during explanation.")
    # Step 6: publish a pointer only after the explanation run completed.
    (project_root / "artifacts/shap/latest.json").write_text(
        json.dumps({
            "output_dir": str(directory), "summary_file": str(directory / "summary.json"),
            "mlflow_run_id": run_id,
        }, indent=2) + "\n", encoding="utf-8",
    )
    print("\nStep 6: outputs:", directory)
    print("Open global_importance.png first, then the local waterfall examples.")
    print("SHAP explains this model, not physical causation. No test evaluation was performed.")


if __name__ == "__main__":
    main()
