import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maintain_predict.features import (
    FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY, TEMPERATURE_DIFFERENCE_COLUMN,
    add_temperature_difference,
)


def readings(n, offset=0):
    values = np.arange(n, dtype=float) + offset
    return pd.DataFrame({
        "UDI": np.arange(n) + offset + 1,
        "Type": ["L" if i % 2 else "M" for i in range(n)],
        **{c: values for c in FEATURE_COLUMNS if c != "Type"},
        "Process temperature [K]": values + 10 + np.arange(n) / 10,
        "Machine failure": [int(i % 4 == 0) for i in range(n)],
        **{c: np.zeros(n, dtype=int) for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
    }, index=np.arange(n) + offset)


def test_difference_keeps_original_inputs_and_excludes_outcomes_without_mutation():
    raw = readings(4)
    raw["Air temperature [K]"] = [300, 301, 302, 303]
    raw["Process temperature [K]"] = [310, 309, 312, 308]
    before = raw.copy(deep=True)
    result = add_temperature_difference(raw)
    assert result.columns.tolist() == [*FEATURE_COLUMNS, TEMPERATURE_DIFFERENCE_COLUMN]
    assert result[TEMPERATURE_DIFFERENCE_COLUMN].tolist() == [10, 8, 10, 5]
    pd.testing.assert_frame_equal(result[list(FEATURE_COLUMNS)], raw[list(FEATURE_COLUMNS)])
    pd.testing.assert_frame_equal(raw, before)
    assert result.index.equals(raw.index)
    # Calling again replaces the derived column from raw temperatures, not duplicates.
    pd.testing.assert_frame_equal(add_temperature_difference(result), result)


def exercise_module():
    path = Path(__file__).resolve().parents[1] / "scripts/05_temperature_difference.py"
    spec = importlib.util.spec_from_file_location("temperature_difference_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_reference(root, validation):
    from maintain_predict.evaluate import classification_metrics

    folder = root / "artifacts/xgboost"
    folder.mkdir(parents=True)
    scores = np.zeros(len(validation))
    pd.DataFrame({
        "source_row_index": validation.index, "UDI": validation["UDI"],
        TARGET_COLUMN: validation["Machine failure"], "failure_score": scores,
        "predicted_label": np.zeros(len(validation), dtype=int),
    }).to_csv(folder / "validation_predictions.csv", index=False)
    summary = {
        "source_sha256": "synthetic-only", "target_policy": TARGET_POLICY,
        "random_state": 42, "threshold": 0.5, "test_evaluated": False,
        "evaluation_partition": "validation", "feature_columns": list(FEATURE_COLUMNS),
        "best_params": {
            "model__n_estimators": 2, "model__max_depth": 2, "model__learning_rate": 0.1,
        },
        "mlflow_run_id": "synthetic-reference",
        "validation_metrics": classification_metrics(validation["Machine failure"], scores),
    }
    (folder / "results.json").write_text(json.dumps(summary))
    return folder


def test_reference_rejects_a_different_input_feature_policy(tmp_path):
    pytest.importorskip("xgboost")
    module = exercise_module()
    validation = readings(12, 1000)
    folder = save_reference(tmp_path, validation)
    path = folder / "results.json"
    summary = json.loads(path.read_text())
    summary["feature_columns"].append(TEMPERATURE_DIFFERENCE_COLUMN)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="does not match"):
        module.load_reference(
            tmp_path, {"source_sha256": "synthetic-only", "random_state": 42}, validation,
        )


def test_feature_exercise_uses_paired_training_folds_and_roundtrips_model(tmp_path, monkeypatch):
    xgb = pytest.importorskip("xgboost")
    import mlflow
    from mlflow.tracking import MlflowClient
    import skops.io
    from maintain_predict.evaluate import classification_metrics

    module = exercise_module()
    train, validation = readings(60), readings(12, 1000)

    class FakeSplits:
        @property
        def test(self):
            raise AssertionError("Test rows must remain reserved.")

    fake = FakeSplits()
    fake.train, fake.validation = train, validation
    manifest_path = tmp_path / "data/splits/type_target_seed42.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "source_sha256": "synthetic-only", "random_state": 42,
    }))
    folder = save_reference(tmp_path, validation)
    previous = {p.name: p.read_bytes() for p in folder.iterdir()}
    monkeypatch.setattr(module, "load_saved_splits", lambda *_: fake)
    factory = module.make_xgboost_pipeline
    monkeypatch.setattr(module, "make_xgboost_pipeline", lambda **kwargs: factory(n_jobs=1, **kwargs))
    fitting_inputs = []
    fit = xgb.XGBClassifier.fit

    def fitting_training_only(self, X, y, **kwargs):
        values = np.asarray(X)
        assert values.max() < 100  # Validation sensor values start at 1000.
        assert self.get_params()["n_estimators"] == 2
        assert self.get_params()["max_depth"] == 2
        assert self.get_params()["scale_pos_weight"] == 3
        assert not kwargs.get("eval_set")
        fitting_inputs.append(values.copy())
        return fit(self, X, y, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", fitting_training_only)
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    try:
        module.main(tmp_path)
        assert [len(x) for x in fitting_inputs] == [48] * 10 + [60]
        # Same rows, original sensors and encoder columns in corresponding folds.
        for original, augmented in zip(fitting_inputs[:5], fitting_inputs[5:10]):
            np.testing.assert_array_equal(original, augmented[:, :-1])
            np.testing.assert_allclose(augmented[:, -1], original[:, 3] - original[:, 2])
        output = tmp_path / "artifacts/temperature_difference"
        summary = json.loads((output / "results.json").read_text())
        assert summary["test_evaluated"] is False
        assert summary["new_hyperparameter_search"] is False
        assert summary["feature_columns"] == [*FEATURE_COLUMNS, TEMPERATURE_DIFFERENCE_COLUMN]
        cv = pd.read_csv(output / "cv_results.csv")
        assert len(cv) == 5
        np.testing.assert_allclose(
            cv["ap_difference"], cv["temperature_difference_ap"] - cv["original_ap"], atol=1e-12,
        )
        assert summary["cv"]["paired_mean_ap_difference"] == pytest.approx(cv["ap_difference"].mean())
        p = pd.read_csv(output / "validation_predictions.csv")
        assert p["source_row_index"].tolist() == validation.index.tolist()
        assert summary["validation_metrics"] == classification_metrics(p[TARGET_COLUMN], p["failure_score"])
        restored = xgb.XGBClassifier()
        restored.load_model(output / "model.ubj")
        preprocess = skops.io.load(output / "preprocessor.skops", trusted=[])
        scores = restored.predict_proba(preprocess.transform(add_temperature_difference(validation)))[:, 1]
        np.testing.assert_allclose(scores, p["failure_score"], atol=1e-7)
        assert set(pd.read_csv(output / "comparison.csv")["model"]) == {
            "xgboost_original", "xgboost_temperature_difference",
        }
        for name in ["validation_confusion_matrix.png", "validation_precision_recall.png"]:
            assert (output / name).stat().st_size > 0
        client = MlflowClient()
        assert client.get_run(summary["mlflow_run_id"]).info.status == "FINISHED"
        logged = {a.path for a in client.list_artifacts(summary["mlflow_run_id"])}
        assert {"model.ubj", "preprocessor.skops", "cv_results.csv", "comparison.csv"} <= logged
        assert previous == {p.name: p.read_bytes() for p in folder.iterdir()}
    finally:
        mlflow.set_tracking_uri(previous_uri)
