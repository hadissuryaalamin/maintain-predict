from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import skops.io
from xgboost import XGBClassifier

from test_explain import saved_selected
from test_final_test import final_case
from test_registry import registry_case
from maintain_predict.features import add_torque_wear


def export_module():
    path = Path(__file__).resolve().parents[1] / "scripts/12_prepare_huggingface.py"
    spec = importlib.util.spec_from_file_location("hub_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export_case(registry_case):
    root, registration = registry_case
    contract = registration.release_contract(root)
    folder = root / "artifacts/registry"
    folder.mkdir()
    (folder / "model_contract.json").write_text(json.dumps(contract))
    receipt = {
        **{key: contract[key] for key in [
            "model_sha256", "preprocessor_sha256", "source_model_run_id", "final_test_run_id",
        ]},
        "registered_model_name": "maintain-predict", "version": "1",
        "package_fingerprint": "synthetic-package", "python": "synthetic-environment",
        "pip_requirements": [f"{name}=={version(name)}" for name in [
            "mlflow", "numpy", "pandas", "scipy", "scikit-learn", "skops", "xgboost",
        ]],
    }
    (folder / "registration.json").write_text(json.dumps(receipt))
    examples = registration.synthetic_inputs()
    examples.to_json(folder / "synthetic_inputs.json", orient="records")
    model_dir = root / "artifacts/torque_wear"
    model = XGBClassifier()
    model.load_model(model_dir / "model.ubj")
    preprocess = skops.io.load(model_dir / "preprocessor.skops", trusted=[])
    scores = model.predict_proba(preprocess.transform(add_torque_wear(examples)))[:, 1]
    pd.DataFrame({
        "failure_score": scores, "predicted_label": (scores >= 0.5).astype(int),
        "threshold": 0.5,
    }).to_csv(folder / "synthetic_predictions.csv", index=False)
    return root, export_module(), examples


def test_export_contains_only_portable_release_files_without_training_or_data_evaluation(export_case, monkeypatch):
    root, module, examples = export_case
    from sklearn.compose import ColumnTransformer

    def forbidden(*args, **kwargs):
        raise AssertionError("Export must not train anything.")

    monkeypatch.setattr(XGBClassifier, "fit", forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit", forbidden)
    read_csv = pd.read_csv

    def only_smoke_receipt(path, *args, **kwargs):
        assert Path(path) == root / "artifacts/registry/synthetic_predictions.csv"
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", only_smoke_receipt)
    original = {p.name: p.read_bytes() for p in (root / "artifacts/torque_wear").iterdir()}
    manifest = module.main(root)
    output = root / "artifacts/huggingface/model"
    assert {p.name for p in output.iterdir()} == {
        "README.md", "model.ubj", "preprocessor.skops", "predict.py",
        "model_contract.json", "requirements.txt", "export_manifest.json",
        "LICENSE",
    }
    assert manifest["uploaded"] is manifest["dataset_evaluated"] is manifest["model_refitted"] is False
    for name, digest in manifest["sha256"].items():
        assert module.sha256(output / name) == digest
    assert "mlflow" not in (output / "requirements.txt").read_text()
    assert (output / "LICENSE").read_bytes() == (module.PROJECT_ROOT / "LICENSE").read_bytes()
    assert manifest["license"] == "mit" and manifest["dataset_license"] == "cc-by-4.0"
    assert manifest["publication_pending"] == ["Approve Hub upload"]
    card = (output / "README.md").read_text(encoding="utf-8")
    assert "<!-- SAVED_METRICS -->" not in card and "| Validation |" in card and "| Test |" in card
    assert "license: mit" in card and "[MIT License](LICENSE)" in card
    assert original == {p.name: p.read_bytes() for p in (root / "artifacts/torque_wear").iterdir()}
    with pytest.raises(FileExistsError):
        module.main(root)
    code = """
import importlib.util, pathlib, sys
import pandas as pd
folder = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('standalone_predict', folder / 'predict.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
predictor = module.FailurePredictor(folder)
rows = pd.DataFrame({'Type':['L'], 'Air temperature [K]':[300.0],
 'Process temperature [K]':[310.0], 'Rotational speed [rpm]':[1500.0],
 'Torque [Nm]':[40.0], 'Tool wear [min]':[100.0]})
assert len(predictor.predict(rows)) == 1
assert 'maintain_predict' not in sys.modules
assert 'mlflow' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(output)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_standalone_loader_rejects_invalid_inputs_and_tampered_artifacts(export_case):
    root, module, examples = export_case
    module.main(root)
    folder = root / "artifacts/huggingface/model"
    helper = module.load_module(folder / "predict.py", "test_hub_predict")
    predictor = helper.FailurePredictor(folder)
    baseline = predictor.predict(examples)
    extra = examples.assign(**{"Machine failure": 1, "RNF": 1, "Torque wear [Nm min]": -999})
    pd.testing.assert_frame_equal(baseline, predictor.predict(extra))
    for bad in [examples.drop(columns="Type"), examples.assign(Type="bad"),
                examples.assign(**{"Torque [Nm]": float("nan")}),
                examples.assign(**{"Torque [Nm]": 1e308, "Tool wear [min]": 1e308})]:
        with pytest.raises(ValueError):
            predictor.predict(bad)
    path = folder / "model.ubj"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        helper.FailurePredictor(folder)
