r"""Manual exercise: package and register maintain-predict version 1 locally.

    .\.venv\Scripts\python.exe .\scripts\11_register_model.py

No training or dataset evaluation. Smoke checks use fabricated input rows.
Existing matching v1 is reused; conflicting registry contents are not overwritten.
"""

from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.pyfunc
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import ColSpec, Schema
import numpy as np
import pandas as pd
import skops.io
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = PROJECT_ROOT / "src/maintain_predict"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maintain_predict.features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY, TORQUE_WEAR_COLUMN, add_torque_wear

MODEL_NAME = "maintain-predict"
MODEL_ALIAS = "selected"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_inputs():
    """Fabricated rows, not observations from training/validation/test."""
    return pd.DataFrame({
        "Type": ["L", "M", "H"],
        "Air temperature [K]": [300.0, 300.5, 299.5],
        "Process temperature [K]": [310.0, 311.0, 309.5],
        "Rotational speed [rpm]": [1500.0, 1600.0, 1400.0],
        "Torque [Nm]": [40.0, 35.0, 45.0],
        "Tool wear [min]": [100.0, 150.0, 50.0],
    })


def release_contract(project_root):
    """Read saved evidence and verify frozen artifacts; never read dataset rows."""
    model_dir = project_root / "artifacts/torque_wear"
    selection = json.loads((model_dir / "results.json").read_text(encoding="utf-8"))
    final = json.loads((project_root / "artifacts/final_test/results.json").read_text(encoding="utf-8"))
    pointer = json.loads((project_root / "artifacts/shap/latest.json").read_text(encoding="utf-8"))
    shap = json.loads(Path(pointer["summary_file"]).read_text(encoding="utf-8"))
    fingerprints = {
        "model_sha256": sha256(model_dir / "model.ubj"),
        "preprocessor_sha256": sha256(model_dir / "preprocessor.skops"),
    }
    expected_features = [*FEATURE_COLUMNS, TORQUE_WEAR_COLUMN]
    if (
        final["test_evaluated"] is not True or final["model_refitted"] is not False
        or final["selected_before_test"] is not True
        or final["evaluation_partition"] != "test"
        or final["threshold"] != 0.5 or selection["threshold"] != 0.5
        or final["target_policy"] != TARGET_POLICY or selection["target_policy"] != TARGET_POLICY
        or final["feature_columns"] != expected_features
        or selection["feature_columns"] != expected_features
        or final["selected_model_run_id"] != selection["mlflow_run_id"]
        or shap["selected_model_run_id"] != selection["mlflow_run_id"]
        or final["source_sha256"] != selection["source_sha256"]
        or shap["source_sha256"] != selection["source_sha256"]
        or any(final[key] != value or shap[key] != value for key, value in fingerprints.items())
    ):
        raise ValueError("Model, SHAP, or final-test evidence differs from the selected frozen configuration.")
    return {
        "raw_feature_columns": list(FEATURE_COLUMNS),
        "model_feature_columns": expected_features,
        "derived_feature_formula": "Torque [Nm] * Tool wear [min]",
        "target_name": TARGET_COLUMN, "target_policy": TARGET_POLICY,
        "threshold": 0.5, "threshold_tuned": False, **fingerprints,
        "source_sha256": selection["source_sha256"],
        "source_model_run_id": selection["mlflow_run_id"],
        "shap_run_id": shap["mlflow_run_id"], "final_test_run_id": final["mlflow_run_id"],
        "fixed_params": selection["fixed_params"],
        "validation_metrics": selection["validation_metrics"], "test_metrics": final["test_metrics"],
        "limitations": [
            "Synthetic dataset; assumed failure-evidence label policy.",
            "Whole-file EDA and repeated validation inspection.",
            "Scores are not automatically calibrated physical failure risks.",
            "No defined forecast horizon; registration is not deployment.",
        ],
    }


def check_predictions(actual, expected_scores):
    if actual.columns.tolist() != ["failure_score", "predicted_label", "threshold"]:
        raise ValueError("Packaged model returned an unexpected output schema.")
    if (
        not np.allclose(actual["failure_score"], expected_scores, atol=1e-7, rtol=1e-6)
        or actual["predicted_label"].tolist() != (expected_scores >= 0.5).astype(int).tolist()
        or not (actual["threshold"] == 0.5).all()
    ):
        raise ValueError("Packaged predictions differ from the selected native model.")


def main(project_root=PROJECT_ROOT):
    project_root = Path(project_root)
    # Step 1: verify the previously evaluated model; do not open any dataset CSV.
    contract = release_contract(project_root)
    requirements = [
        f"{name}=={version(name)}"
        for name in ["mlflow", "numpy", "pandas", "scipy", "scikit-learn", "skops", "xgboost"]
    ]
    code_hashes = {
        name: sha256(CODE_DIRECTORY / name)
        for name in ["__init__.py", "features.py", "registry_model.py", "mlflow_entrypoint.py"]
    }
    fingerprint = hashlib.sha256(json.dumps({
        "contract": contract, "code_hashes": code_hashes, "pip_requirements": requirements,
    }, sort_keys=True).encode()).hexdigest()
    tracking_uri = "sqlite:///" + (project_root / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)  # Explicitly local, even if an external URI is configured.
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    if versions and (
        len(versions) != 1 or str(versions[0].version) != "1"
        or versions[0].tags.get("package_fingerprint") != fingerprint
    ):
        raise RuntimeError("Registry name already has different contents. No new version or alias was written.")
    if versions:
        aliases = client.get_registered_model(MODEL_NAME).aliases
        if MODEL_ALIAS in aliases and str(aliases[MODEL_ALIAS]) != "1":
            raise RuntimeError("Selected alias points elsewhere; it will not be overwritten.")

    # Step 2: prepare fabricated input examples and a native prediction reference.
    examples = synthetic_inputs()
    model_dir = project_root / "artifacts/torque_wear"
    preprocessor = skops.io.load(model_dir / "preprocessor.skops", trusted=[])
    native = XGBClassifier()
    native.load_model(model_dir / "model.ubj")
    expected_scores = native.predict_proba(preprocessor.transform(add_torque_wear(examples)))[:, 1]
    signature = ModelSignature(
        inputs=Schema([
            ColSpec("string" if name == "Type" else "double", name)
            for name in FEATURE_COLUMNS
        ]),
        outputs=Schema([
            ColSpec("double", "failure_score"), ColSpec("long", "predicted_label"),
            ColSpec("double", "threshold"),
        ]),
    )
    output = project_root / "artifacts/registry"
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "model_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    if versions:
        # A matching completed/partially completed v1 can be verified and reused.
        registered_version = versions[0]
        package_run_id = registered_version.run_id
        print("Matching version 1 already exists; no new model version will be created.")
    else:
        experiment_name = "maintain-predict-registry"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id if experiment else mlflow.create_experiment(
                experiment_name, artifact_location=(project_root / "mlruns").as_uri(),
            )
        )
        with mlflow.start_run(experiment_id=experiment_id, run_name="package_selected_model_v1"):
            package_run_id = mlflow.active_run().info.run_id
            mlflow.log_params({
                "registered_model_name": MODEL_NAME, "threshold": 0.5,
                "raw_inputs": 6, "model_inputs": 7,
                "source_model_run_id": contract["source_model_run_id"],
                "final_test_run_id": contract["final_test_run_id"],
                "target_policy": TARGET_POLICY, "package_fingerprint": fingerprint,
                "model_refitted": False, "dataset_evaluated": False,
            })
            # Step 3: bundle source code, model, preprocessor, and contract.
            info = mlflow.pyfunc.log_model(
                name="selected_model",
                python_model=str(CODE_DIRECTORY / "mlflow_entrypoint.py"),
                code_paths=[str(CODE_DIRECTORY)],
                artifacts={
                    "classifier": str(model_dir / "model.ubj"),
                    "preprocessor": str(model_dir / "preprocessor.skops"),
                    "contract": str(contract_path),
                },
                signature=signature, input_example=examples,
                pip_requirements=requirements,
                metadata={
                    "package_fingerprint": fingerprint, "target_policy": TARGET_POLICY,
                    "source_model_run_id": contract["source_model_run_id"],
                    "threshold": 0.5,
                },
            )
            # Step 4: load the package and check fabricated rows before registration.
            packaged = mlflow.pyfunc.load_model(info.model_uri)
            check_predictions(packaged.predict(examples), expected_scores)
            registered_version = mlflow.register_model(
                info.model_uri, MODEL_NAME,
                tags={
                    "package_fingerprint": fingerprint, "target_policy": TARGET_POLICY,
                    "source_model_run_id": contract["source_model_run_id"],
                    "final_test_run_id": contract["final_test_run_id"],
                    "threshold": "0.5", "deployment_status": "not_deployed",
                },
            )
            if str(registered_version.version) != "1":
                raise RuntimeError("Another registry writer created a version concurrently; inspect before continuing.")
            for partition in ["validation", "test"]:
                mlflow.log_metrics({
                    f"{partition}_{key}": contract[f"{partition}_metrics"][key]
                    for key in ["average_precision", "accuracy", "precision", "recall", "f1"]
                    if contract[f"{partition}_metrics"][key] is not None
                })
            mlflow.log_artifact(str(contract_path))

    # Step 5: check the registered URI, then assign a non-production alias.
    model_uri = f"models:/{MODEL_NAME}/1"
    registered = mlflow.pyfunc.load_model(model_uri)
    predictions = registered.predict(examples)
    check_predictions(predictions, expected_scores)
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, "1")
    client.update_model_version(
        name=MODEL_NAME, version="1",
        description=(
            "User-selected XGBoost: six raw inputs plus calculated torque x tool wear. "
            "Threshold 0.5 retained without tuning. Native model and preprocessor match "
            "the frozen final-test artifacts. Synthetic-data failure-evidence target; "
            "not a verified physical forecast. Registered locally, not deployed."
        ),
    )
    # Step 6: save the registration receipt and example outputs.
    receipt = {
        "registered_model_name": MODEL_NAME, "version": "1",
        "model_uri": model_uri, "alias": MODEL_ALIAS,
        "alias_uri": f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        "package_run_id": package_run_id, "package_fingerprint": fingerprint,
        "tracking_uri": tracking_uri, "target_policy": TARGET_POLICY,
        "threshold": 0.5, "raw_inputs": 6, "model_inputs": 7,
        "source_model_run_id": contract["source_model_run_id"],
        "shap_run_id": contract["shap_run_id"], "final_test_run_id": contract["final_test_run_id"],
        "model_sha256": contract["model_sha256"],
        "preprocessor_sha256": contract["preprocessor_sha256"],
        "code_hashes": code_hashes, "pip_requirements": requirements,
        "python": platform.python_version(),
        "model_refitted": False, "dataset_evaluated": False, "deployed": False,
        "smoke_test": "Three fabricated input rows; packaged scores match native scores.",
    }
    (output / "registration.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8",
    )
    examples.to_json(output / "synthetic_inputs.json", orient="records", indent=2)
    predictions.to_csv(output / "synthetic_predictions.csv", index=False)
    print("Registered:", model_uri, "; alias:", receipt["alias_uri"])
    print("Receipt:", output / "registration.json")
    print(predictions.to_string(index=False))
    return receipt


if __name__ == "__main__":
    main()
