r"""Prepare a local, reviewable Hub folder; never upload, train, or read dataset rows.

    .\.venv\Scripts\python.exe .\scripts\12_prepare_huggingface.py
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def metrics_table(contract):
    lines = [
        "| Partition | AP | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for partition in ["validation", "test"]:
        metrics = contract[f"{partition}_metrics"]
        values = ["N/A" if metrics[key] is None else f"{metrics[key]:.4f}" for key in
                  ["average_precision", "accuracy", "precision", "recall", "f1"]]
        counts = [str(metrics["confusion_matrix"][key]) for key in ["tn", "fp", "fn", "tp"]]
        lines.append("| " + " | ".join([partition.title(), *values, *counts]) + " |")
    return "\n".join(lines)


def main(project_root=PROJECT_ROOT):
    root = Path(project_root).resolve()
    # 1. Verify unchanged artifacts and saved evaluation evidence, without scoring data.
    registry = load_module(PROJECT_ROOT / "scripts/11_register_model.py", "registration_source")
    contract = registry.release_contract(root)
    registration = read_json(root / "artifacts/registry/registration.json")
    saved_contract = read_json(root / "artifacts/registry/model_contract.json")
    if contract != saved_contract or any(
        registration[key] != contract[key]
        for key in ["model_sha256", "preprocessor_sha256", "source_model_run_id", "final_test_run_id"]
    ) or registration["registered_model_name"] != "maintain-predict" or str(registration["version"]) != "1":
        raise ValueError("Current artifacts do not match the registered v1 release.")

    # 2. Copy only the files intended for the model repository. Never copy the project tree.
    output = root / "artifacts/huggingface/model"
    if output.exists():
        raise FileExistsError(f"Export folder already exists; review it before preparing another export: {output}")
    template = PROJECT_ROOT / "hub/model"
    card = (template / "README.md").read_text(encoding="utf-8")
    if card.count("<!-- SAVED_METRICS -->") != 1:
        raise ValueError("Model card must contain one saved-metrics insertion point.")
    runtime_names = {"numpy", "pandas", "scipy", "scikit-learn", "skops", "xgboost"}
    requirements = [item for item in registration["pip_requirements"] if item.split("==")[0] in runtime_names]
    if {item.split("==")[0] for item in requirements} != runtime_names:
        raise ValueError("Registered inference dependency versions are incomplete.")
    output.mkdir(parents=True)
    for name in ["model.ubj", "preprocessor.skops"]:
        shutil.copyfile(root / "artifacts/torque_wear" / name, output / name)
    shutil.copyfile(template / "predict.py", output / "predict.py")
    shutil.copyfile(PROJECT_ROOT / "LICENSE", output / "LICENSE")
    (output / "README.md").write_text(card.replace("<!-- SAVED_METRICS -->", metrics_table(contract)), encoding="utf-8")
    (output / "model_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (output / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")

    # 3. Check only fabricated rows against the existing registered-release smoke receipt.
    predictor = load_module(output / "predict.py", "prepared_hub_predict").FailurePredictor(output)
    examples = registry.synthetic_inputs()
    reference = read_json(root / "artifacts/registry/synthetic_inputs.json")
    if json.loads(examples.to_json(orient="records")) != reference:
        raise ValueError("Registered smoke examples differ; inspect the release before export.")
    # Reference CSV contains only the three fabricated rows, never held-out dataset rows.
    import pandas as pd
    reference_predictions = pd.read_csv(root / "artifacts/registry/synthetic_predictions.csv")
    predictions = predictor.predict(examples)
    registry.check_predictions(predictions, reference_predictions["failure_score"].to_numpy())
    if not np.array_equal(predictions["predicted_label"], reference_predictions["predicted_label"]):
        raise ValueError("Hub predictions differ from the registered release.")

    # 4. Record portable provenance. No credentials or local tracking URI are exported.
    manifest = {
        "destination_repo_id": "hadissuryaalamin/maintain-predict",
        "registered_model_name": registration["registered_model_name"], "version": "1",
        "source_package_fingerprint": registration["package_fingerprint"],
        "source_model_run_id": contract["source_model_run_id"],
        "final_test_run_id": contract["final_test_run_id"],
        "python": registration["python"], "threshold": 0.5,
        "model_refitted": False, "dataset_evaluated": False, "uploaded": False,
        "smoke_check": "Three fabricated rows match the registered-release reference scores.",
        "license": "mit", "dataset_license": "cc-by-4.0",
        "publication_pending": ["Approve Hub upload"],
        "sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Prepared locally:", output)
    print("Native artifacts unchanged; fabricated example predictions match. No upload performed.")
    print(predictions.to_string(index=False))
    return manifest


if __name__ == "__main__":
    main()
