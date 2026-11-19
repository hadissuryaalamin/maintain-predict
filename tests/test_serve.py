import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from maintain_predict import serve
from maintain_predict.features import FEATURE_COLUMNS, TARGET_POLICY, TORQUE_WEAR_COLUMN
from maintain_predict.inference import FailurePredictor
from test_explain import saved_selected

EXAMPLE = {
    "type": "L", "air_temp_k": 300, "process_temp_k": 310,
    "rpm": 1500, "torque_nm": 40, "tool_wear_min": 100,
}


@pytest.fixture
def client(monkeypatch):
    class NoInference:
        def predict(self, *args):
            raise AssertionError("Invalid requests must not reach the model.")

    monkeypatch.setattr(serve, "load_predictor", NoInference)
    with TestClient(serve.app) as active:
        yield active


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("changes", [
    {"type": "unknown"}, {"rpm": 0}, {"torque_nm": -1},
    {"tool_wear_min": -1}, {"air_temp_k": 0}, {"rpm": "1500"},
    {"rpm": True}, {"rpm": "NaN"}, {"Machine failure": 1}, {"threshold": 0.1},
    {"torque_nm": 1e308, "tool_wear_min": 1e308},
])
def test_invalid_inputs_are_rejected_before_inference(client, changes):
    response = client.post("/predict", json={**EXAMPLE, **changes})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_missing_and_nonfinite_inputs_return_valid_json_errors(client):
    missing = {key: value for key, value in EXAMPLE.items() if key != "type"}
    assert client.post("/predict", json=missing).status_code == 422
    raw = json.dumps(EXAMPLE).replace('"rpm": 1500', '"rpm": 1e999')
    response = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["detail"]


def test_prediction_matches_native_model_and_loads_once_without_fitting(saved_selected, monkeypatch):
    root, _, _ = saved_selected
    folder = root / "artifacts/torque_wear"
    contract = {
        "raw_feature_columns": list(FEATURE_COLUMNS),
        "model_feature_columns": [*FEATURE_COLUMNS, TORQUE_WEAR_COLUMN],
        "target_policy": TARGET_POLICY, "threshold": 0.5,
        "model_sha256": hashlib.sha256((folder / "model.ubj").read_bytes()).hexdigest(),
        "preprocessor_sha256": hashlib.sha256((folder / "preprocessor.skops").read_bytes()).hexdigest(),
    }
    (folder / "model_contract.json").write_text(json.dumps(contract))
    monkeypatch.setenv("MAINTAIN_PREDICT_MODEL_DIR", str(folder))
    reference = FailurePredictor(folder).predict(serve.PredictionInput(**EXAMPLE).as_frame()).iloc[0]
    from xgboost import XGBClassifier
    from sklearn.compose import ColumnTransformer

    def no_fit(*args, **kwargs):
        raise AssertionError("API startup and prediction must not fit a model.")

    monkeypatch.setattr(XGBClassifier, "fit", no_fit)
    monkeypatch.setattr(ColumnTransformer, "fit", no_fit)
    original_load = serve.load_predictor
    loaded = []

    def tracked_load():
        model = original_load()
        loaded.append(model)
        return model

    monkeypatch.setattr(serve, "load_predictor", tracked_load)
    with TestClient(serve.app) as active:
        for _ in range(2):
            response = active.post("/predict", json=EXAMPLE)
            assert response.status_code == 200
            result = response.json()
            assert result["failure_score"] == pytest.approx(float(reference["failure_score"]), abs=1e-8)
            assert result["predicted_label"] == int(reference["predicted_label"])
            assert result["threshold"] == 0.5 and result["target_policy"] == TARGET_POLICY
        assert len(loaded) == 1
    assert not hasattr(serve.app.state, "predictor")


def test_default_download_pins_revision_and_excludes_remote_code(monkeypatch, tmp_path):
    monkeypatch.delenv("MAINTAIN_PREDICT_MODEL_DIR", raising=False)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(serve, "snapshot_download", download)
    monkeypatch.setattr(serve, "FailurePredictor", lambda folder: folder)
    assert serve.load_predictor() == tmp_path
    assert calls == [{
        "repo_id": serve.MODEL_REPO_ID, "revision": serve.MODEL_REVISION,
        "allow_patterns": ["model.ubj", "preprocessor.skops", "model_contract.json"], "token": False,
    }]


def test_requests_without_loaded_model_are_not_healthy():
    # No lifespan context: no downloads or startup loading should occur.
    active = TestClient(serve.app)
    assert active.get("/health").status_code == 503
    assert active.post("/predict", json=EXAMPLE).status_code == 503
