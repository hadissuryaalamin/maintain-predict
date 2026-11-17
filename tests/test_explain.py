import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from maintain_predict.explain import (
    SELECTED_FEATURE_COLUMNS, group_encoded_contributions,
    reconstruction_errors, representative_positions,
)
from maintain_predict.features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY, add_torque_wear


def test_type_grouping_sums_signed_contributions_and_preserves_additivity():
    names = ["type__Type_L", "type__Type_M"] + [
        "sensors__" + name for name in SELECTED_FEATURE_COLUMNS if name != "Type"
    ]
    values = np.array([[2, -1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    grouped = group_encoded_contributions(values, names)
    np.testing.assert_allclose(grouped, [[1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    np.testing.assert_allclose(grouped.sum(axis=1), values.sum(axis=1))


def test_type_grouping_rejects_features_outside_the_selected_schema():
    names = ["type__Type_L"] + [
        "sensors__" + name for name in SELECTED_FEATURE_COLUMNS if name != "Type"
    ]
    names[-1] = "sensors__RNF"
    with pytest.raises(ValueError, match="outside"):
        group_encoded_contributions(np.zeros((2, 7)), names)


def test_reconstruction_checks_margin_and_probability_spaces():
    values = np.array([[0.3, -0.1], [1.0, -0.2]])
    margins = np.array([0.4, 1.0])
    result = reconstruction_errors(0.2, values, margins, expit(margins))
    assert result["max_abs_margin_error"] < 1e-12
    assert result["max_abs_probability_error"] < 1e-12
    with pytest.raises(ValueError, match="margins"):
        reconstruction_errors(0.2, values, margins + 0.3, expit(margins))
    with pytest.raises(ValueError, match="prediction scores"):
        reconstruction_errors(0.2, values, margins, expit(margins) / 2)


def test_examples_use_smallest_source_index_per_outcome_and_allow_missing_categories():
    assert representative_positions(
        [1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0], [5, 3, 4, 9, 1, 2],
    ) == {"TP": 4, "FN": 1, "FP": 2, "TN": 5}
    assert representative_positions([0, 0], [0, 0], [3, 2]) == {"TN": 1}


def exercise_module():
    path = Path(__file__).resolve().parents[1] / "scripts/09_shap.py"
    spec = importlib.util.spec_from_file_location("shap_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def readings(n, offset=0):
    values = np.arange(n, dtype=float) + offset
    return pd.DataFrame({
        "UDI": np.arange(n) + offset + 1,
        "Type": ["L" if i % 2 else "M" for i in range(n)],
        "Air temperature [K]": 300 + values / 10,
        "Process temperature [K]": 310 + values / 5,
        "Rotational speed [rpm]": 1200 + values,
        "Torque [Nm]": 20 + values,
        "Tool wear [min]": 10 + values,
        "Machine failure": [int(i % 3 == 0) for i in range(n)],
        **{c: np.zeros(n, dtype=int) for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
    }, index=np.arange(n) + offset)


@pytest.fixture
def saved_selected(tmp_path):
    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    import skops.io
    from maintain_predict.train import make_xgboost_pipeline
    from maintain_predict.evaluate import classification_metrics

    train, validation = readings(48), readings(12, 1000)
    model = make_xgboost_pipeline(n_jobs=1, include_torque_wear=True)
    model.set_params(model__n_estimators=6, model__max_depth=2)
    model.fit(add_torque_wear(train), train["Machine failure"])
    features = add_torque_wear(validation)
    scores = model.predict_proba(features)[:, 1]
    folder = tmp_path / "artifacts/torque_wear"
    folder.mkdir(parents=True)
    model.named_steps["model"].save_model(folder / "model.ubj")
    skops.io.dump(model.named_steps["preprocess"], folder / "preprocessor.skops")
    manifest = {"source_sha256": "synthetic-only", "random_state": 42}
    path = tmp_path / "data/splits/type_target_seed42.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    summary = {
        **manifest, "target_policy": TARGET_POLICY, "threshold": 0.5,
        "evaluation_partition": "validation", "test_evaluated": False,
        "feature_columns": list(SELECTED_FEATURE_COLUMNS),
        "validation_metrics": classification_metrics(validation["Machine failure"], scores),
        "mlflow_run_id": "synthetic-selected-model",
    }
    (folder / "results.json").write_text(json.dumps(summary))
    pd.DataFrame({
        "source_row_index": validation.index, "UDI": validation["UDI"],
        TARGET_COLUMN: validation["Machine failure"],
        "failure_score": scores, "predicted_label": (scores >= 0.5).astype(int),
    }).to_csv(folder / "validation_predictions.csv", index=False)
    return tmp_path, validation, manifest


def test_shap_rejects_mismatched_saved_membership(saved_selected):
    root, validation, manifest = saved_selected
    module = exercise_module()
    with pytest.raises(ValueError, match="membership"):
        module.load_selected_model(root, manifest, validation.iloc[::-1])


def test_explanation_runs_without_refitting_or_accessing_train_or_test(saved_selected, monkeypatch):
    root, validation, _ = saved_selected
    import mlflow
    from mlflow.tracking import MlflowClient
    from xgboost import XGBClassifier
    from sklearn.compose import ColumnTransformer

    module = exercise_module()

    class OnlyValidation:
        @property
        def train(self):
            raise AssertionError("Explanations must not use raw training rows.")

        @property
        def test(self):
            raise AssertionError("Test rows must remain reserved.")

    fake = OnlyValidation()
    fake.validation = validation
    monkeypatch.setattr(module, "load_saved_splits", lambda *_: fake)

    def no_refit(*args, **kwargs):
        raise AssertionError("The saved model and preprocessor must never be refitted.")

    monkeypatch.setattr(XGBClassifier, "fit", no_refit)
    monkeypatch.setattr(ColumnTransformer, "fit", no_refit)
    before = {
        p.name: p.read_bytes() for p in (root / "artifacts/torque_wear").iterdir()
    }
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(root)
    try:
        module.main(root)
        pointer = json.loads((root / "artifacts/shap/latest.json").read_text())
        output = Path(pointer["output_dir"])
        summary = json.loads((output / "summary.json").read_text())
        assert summary["observations"] == 12
        assert summary["test_evaluated"] is False and summary["model_refitted"] is False
        assert summary["selected_model_run_id"] == "synthetic-selected-model"
        assert summary["units"] == "log-odds" and summary["threshold"] == 0.5
        assert summary["feature_columns"] == list(SELECTED_FEATURE_COLUMNS)
        grouped = pd.read_csv(output / "shap_values_grouped.csv")
        encoded = pd.read_csv(output / "shap_values_encoded.csv")
        rows = pd.read_csv(output / "row_explanations.csv")
        assert rows["source_row_index"].tolist() == validation.index.tolist()
        assert grouped["source_row_index"].tolist() == validation.index.tolist()
        np.testing.assert_allclose(
            grouped[list(SELECTED_FEATURE_COLUMNS)].sum(axis=1) + rows["base_log_odds"],
            rows["model_log_odds"], atol=1e-4,
        )
        np.testing.assert_allclose(expit(rows["model_log_odds"]), rows["score"], atol=1e-6)
        np.testing.assert_allclose(
            encoded[summary["encoded_feature_names"]].sum(axis=1),
            grouped[list(SELECTED_FEATURE_COLUMNS)].sum(axis=1), atol=1e-6,
        )
        importance = pd.read_csv(output / "global_importance.csv").set_index("feature")
        for name in SELECTED_FEATURE_COLUMNS:
            assert importance.loc[name, "mean_abs_shap_log_odds"] == pytest.approx(
                np.abs(grouped[name]).mean(), abs=1e-8,
            )
        for filename in ["global_importance.png", "beeswarm_encoded.png"]:
            assert (output / filename).stat().st_size > 0
        assert summary["examples"]
        for example in summary["examples"]:
            assert (output / example["file"]).stat().st_size > 0
        assert before == {
            p.name: p.read_bytes() for p in (root / "artifacts/torque_wear").iterdir()
        }
        client = MlflowClient()
        assert client.get_run(pointer["mlflow_run_id"]).info.status == "FINISHED"
        logged = {a.path for a in client.list_artifacts(pointer["mlflow_run_id"])}
        assert {"summary.json", "global_importance.png", "shap_values_grouped.csv"} <= logged
    finally:
        mlflow.set_tracking_uri(previous_uri)
