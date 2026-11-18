import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pytest

from test_explain import saved_selected
from test_final_test import final_case


def exercise_module():
    path = Path(__file__).resolve().parents[1] / "scripts/11_register_model.py"
    spec = importlib.util.spec_from_file_location("registry_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registry_case(final_case):
    root, _, _ = final_case
    module = exercise_module()
    selection = json.loads((root / "artifacts/torque_wear/results.json").read_text())
    shap = json.loads((root / "artifacts/shap/summary.json").read_text())
    final = {
        **selection, "test_evaluated": True, "model_refitted": False,
        "selected_before_test": True, "evaluation_partition": "test",
        "selected_model_run_id": selection["mlflow_run_id"],
        "mlflow_run_id": "synthetic-final-run",
        "model_sha256": shap["model_sha256"],
        "preprocessor_sha256": shap["preprocessor_sha256"],
        # Saved synthetic metrics stand in for a completed evaluation receipt.
        "test_metrics": selection["validation_metrics"],
    }
    folder = root / "artifacts/final_test"
    folder.mkdir()
    (folder / "results.json").write_text(json.dumps(final))
    return root, module


def test_registration_is_portable_and_idempotent_without_fit_or_dataset_reads(registry_case, monkeypatch):
    root, module = registry_case
    from sklearn.compose import ColumnTransformer
    from xgboost import XGBClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError("Registration must not train or read dataset CSVs.")

    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.chdir(root)
    old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    original = {p.name: p.read_bytes() for p in (root / "artifacts/torque_wear").iterdir()}
    try:
        receipt = module.main(root)
        client = MlflowClient()
        assert receipt["version"] == "1"
        assert receipt["model_refitted"] is receipt["dataset_evaluated"] is False
        assert str(client.get_model_version_by_alias("maintain-predict", "selected").version) == "1"
        assert client.get_run(receipt["package_run_id"]).info.status == "FINISHED"
        assert module.main(root) == receipt
        assert len(client.search_model_versions("name='maintain-predict'")) == 1
        model = mlflow.pyfunc.load_model(receipt["alias_uri"])
        assert len(model.metadata.signature.inputs.inputs) == 6
        inputs = module.synthetic_inputs()
        assert len(model.predict(inputs)) == 3
        for bad in [inputs.drop(columns="Type"), inputs.assign(Type="invalid"),
                    inputs.assign(**{"Torque [Nm]": float("nan")})]:
            with pytest.raises(Exception):
                model.predict(bad)
        # Fresh isolated interpreter must import the bundled package, not this repo's source.
        code = """
import json, pathlib, sys
import mlflow, pandas as pd
mlflow.set_tracking_uri(sys.argv[1])
mlflow.set_registry_uri(sys.argv[1])
model = mlflow.pyfunc.load_model('models:/maintain-predict/1')
rows = pd.DataFrame(json.loads(pathlib.Path(sys.argv[2]).read_text()))
assert len(model.predict(rows)) == 3
import maintain_predict.registry_model as implementation
assert 'code' in pathlib.Path(implementation.__file__).parts, implementation.__file__
"""
        result = subprocess.run([
            sys.executable, "-I", "-c", code, receipt["tracking_uri"],
            str(root / "artifacts/registry/synthetic_inputs.json"),
        ], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert original == {p.name: p.read_bytes() for p in (root / "artifacts/torque_wear").iterdir()}
    finally:
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)


def test_release_contract_rejects_changed_model_before_registration(registry_case):
    root, module = registry_case
    path = root / "artifacts/torque_wear/model.ubj"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="frozen configuration"):
        module.release_contract(root)
    assert not (root / "mlflow.db").exists()
