import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maintain_predict.features import (
    FEATURE_COLUMNS, POWER_COLUMN, TARGET_COLUMN, TARGET_POLICY,
    TEMPERATURE_DIFFERENCE_COLUMN, add_power,
)


def readings(n, offset=0):
    values = np.arange(n, dtype=float) + offset
    return pd.DataFrame({
        "UDI": np.arange(n) + offset + 1,
        "Type": ["L" if i % 2 else "M" for i in range(n)],
        "Air temperature [K]": 300 + values,
        "Process temperature [K]": 310 + values,
        "Rotational speed [rpm]": 60 + values,
        "Torque [Nm]": 2 + values,
        "Tool wear [min]": values,
        "Machine failure": [int(i % 4 == 0) for i in range(n)],
        **{c: np.zeros(n, dtype=int) for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
        TEMPERATURE_DIFFERENCE_COLUMN: np.full(n, 999.0),
    }, index=np.arange(n) + offset)


def test_power_converts_rpm_to_radians_per_second_and_retains_raw_inputs_only():
    raw = readings(3)
    raw["Rotational speed [rpm]"] = [60, 120, 0]
    raw["Torque [Nm]"] = [10, 5, 20]
    before = raw.copy(deep=True)
    result = add_power(raw)
    assert result.columns.tolist() == [*FEATURE_COLUMNS, POWER_COLUMN]
    np.testing.assert_allclose(result[POWER_COLUMN], [20 * np.pi, 20 * np.pi, 0])
    pd.testing.assert_frame_equal(result[list(FEATURE_COLUMNS)], raw[list(FEATURE_COLUMNS)])
    pd.testing.assert_frame_equal(raw, before)
    assert result.index.equals(raw.index)
    pd.testing.assert_frame_equal(add_power(result), result)


def exercise_module():
    path = Path(__file__).resolve().parents[1] / "scripts/06_power.py"
    spec = importlib.util.spec_from_file_location("power_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_reference(root, validation):
    from maintain_predict.evaluate import classification_metrics

    folder = root / "artifacts/xgboost"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "source_row_index": validation.index, "UDI": validation["UDI"],
        TARGET_COLUMN: validation["Machine failure"],
        "failure_score": np.zeros(len(validation)), "predicted_label": 0,
    }).to_csv(folder / "validation_predictions.csv", index=False)
    summary = {
        "source_sha256": "synthetic-only", "target_policy": TARGET_POLICY,
        "random_state": 42, "threshold": 0.5, "test_evaluated": False,
        "evaluation_partition": "validation", "feature_columns": list(FEATURE_COLUMNS),
        "best_params": {
            "model__n_estimators": 2, "model__max_depth": 2, "model__learning_rate": 0.1,
        },
        "mlflow_run_id": "synthetic-reference",
        "validation_metrics": classification_metrics(
            validation["Machine failure"], np.zeros(len(validation)),
        ),
    }
    (folder / "results.json").write_text(json.dumps(summary))


def test_power_reference_rejects_an_engineered_feature_reference(tmp_path):
    pytest.importorskip("xgboost")
    module = exercise_module()
    validation = readings(12, 1000)
    save_reference(tmp_path, validation)
    path = tmp_path / "artifacts/xgboost/results.json"
    summary = json.loads(path.read_text())
    summary["feature_columns"].append(TEMPERATURE_DIFFERENCE_COLUMN)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="does not match"):
        module.load_reference(
            tmp_path, {"source_sha256": "synthetic-only", "random_state": 42}, validation,
        )


def test_power_exercise_pairs_training_folds_excludes_temperature_and_roundtrips_model(
    tmp_path, monkeypatch,
):
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
            raise AssertionError("Test data must remain reserved.")

    fake = FakeSplits()
    fake.train, fake.validation = train, validation
    manifest_path = tmp_path / "data/splits/type_target_seed42.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "source_sha256": "synthetic-only", "random_state": 42,
    }))
    save_reference(tmp_path, validation)
    temperature_folder = tmp_path / "artifacts/temperature_difference"
    temperature_folder.mkdir()
    (temperature_folder / "prior_result.txt").write_text("Preserve the earlier experiment.")
    previous = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in (tmp_path / "artifacts").rglob("*") if p.is_file()
    }
    monkeypatch.setattr(module, "load_saved_splits", lambda *_: fake)
    factory = module.make_xgboost_pipeline
    monkeypatch.setattr(module, "make_xgboost_pipeline", lambda **kwargs: factory(n_jobs=1, **kwargs))
    fitted_inputs = []
    fit = xgb.XGBClassifier.fit

    def training_only_fit(self, X, y, **kwargs):
        values = np.asarray(X)
        assert values[:, 2].max() < 1000  # Validation air temperature starts at 1300.
        assert self.get_params()["n_estimators"] == 2
        assert self.get_params()["max_depth"] == 2
        assert self.get_params()["scale_pos_weight"] == 3
        assert not kwargs.get("eval_set")
        fitted_inputs.append(values.copy())
        return fit(self, X, y, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", training_only_fit)
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    try:
        module.main(tmp_path)
        assert [len(x) for x in fitted_inputs] == [48] * 10 + [60]
        for original, augmented in zip(fitted_inputs[:5], fitted_inputs[5:10]):
            assert original.shape[1] == 7 and augmented.shape[1] == 8
            np.testing.assert_array_equal(original, augmented[:, :-1])
            np.testing.assert_allclose(
                augmented[:, -1], original[:, 5] * original[:, 4] * (2 * np.pi / 60),
            )
        output = tmp_path / "artifacts/power"
        summary = json.loads((output / "results.json").read_text())
        assert summary["feature_columns"] == [*FEATURE_COLUMNS, POWER_COLUMN]
        assert TEMPERATURE_DIFFERENCE_COLUMN not in summary["feature_columns"]
        assert summary["test_evaluated"] is False and summary["threshold"] == 0.5
        assert summary["new_hyperparameter_search"] is False
        cv = pd.read_csv(output / "cv_results.csv")
        assert len(cv) == 5
        np.testing.assert_allclose(cv["ap_difference"], cv["power_ap"] - cv["original_ap"], atol=1e-12)
        assert summary["cv"]["paired_mean_ap_difference"] == pytest.approx(cv["ap_difference"].mean())
        p = pd.read_csv(output / "validation_predictions.csv")
        assert p["source_row_index"].tolist() == validation.index.tolist()
        assert summary["validation_metrics"] == classification_metrics(p[TARGET_COLUMN], p["failure_score"])
        restored = xgb.XGBClassifier()
        restored.load_model(output / "model.ubj")
        preprocess = skops.io.load(output / "preprocessor.skops", trusted=[])
        scores = restored.predict_proba(preprocess.transform(add_power(validation)))[:, 1]
        np.testing.assert_allclose(scores, p["failure_score"], atol=1e-7)
        assert set(pd.read_csv(output / "comparison.csv")["model"]) == {
            "xgboost_original", "xgboost_power",
        }
        assert (output / "validation_confusion_matrix.png").stat().st_size > 0
        assert (output / "validation_precision_recall.png").stat().st_size > 0
        client = MlflowClient()
        assert client.get_run(summary["mlflow_run_id"]).info.status == "FINISHED"
        logged = {a.path for a in client.list_artifacts(summary["mlflow_run_id"])}
        assert {"model.ubj", "preprocessor.skops", "cv_results.csv", "comparison.csv"} <= logged
        assert all((tmp_path / p).read_bytes() == data for p, data in previous.items())
    finally:
        mlflow.set_tracking_uri(previous_uri)
