import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

xgb = pytest.importorskip("xgboost")

from maintain_predict.evaluate import classification_metrics
from maintain_predict.features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY
from maintain_predict.train import make_xgboost_pipeline, make_xgboost_search
from maintain_predict.xgboost_model import FoldWeightedXGBClassifier


def fake_readings(n, offset=0):
    values = np.arange(n, dtype=float) + offset
    return pd.DataFrame({
        "UDI": np.arange(n) + offset + 1,
        "Type": ["L" if i % 2 else "M" for i in range(n)],
        **{c: values for c in FEATURE_COLUMNS if c != "Type"},
        "Machine failure": [int(i % 4 == 0) for i in range(n)],
        **{c: np.zeros(n, dtype=int) for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
    }, index=np.arange(n) + offset)


def exercise_module():
    path = Path(__file__).resolve().parents[1] / "scripts/04_xgboost.py"
    spec = importlib.util.spec_from_file_location("xgboost_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_clones_and_handles_unseen_type_without_learning_validation():
    train = fake_readings(20)
    before = train.copy(deep=True)
    model = clone(make_xgboost_pipeline(n_jobs=1))
    with pytest.raises(Exception, match="need to call fit|not fitted"):
        model.named_steps["model"].get_booster()
    model.set_params(model__n_estimators=3)
    model.fit(train[list(FEATURE_COLUMNS)], train["Machine failure"])
    assert model.named_steps["model"].get_params()["scale_pos_weight"] == 3.0
    validation = fake_readings(4, 1000)
    validation["Type"] = "H"
    preprocessing = model.named_steps["preprocess"]
    transformed = preprocessing.transform(validation[list(FEATURE_COLUMNS)])
    numeric = [c for c in FEATURE_COLUMNS if c != "Type"]
    np.testing.assert_array_equal(transformed[:, -5:], validation[numeric].to_numpy())
    scores = model.predict_proba(validation[list(FEATURE_COLUMNS)])
    np.testing.assert_allclose(scores.sum(axis=1), 1)
    assert preprocessing.named_transformers_["type"].categories_[0].tolist() == ["L", "M"]
    pd.testing.assert_frame_equal(train, before)


def test_weight_is_recomputed_from_each_fit_instead_of_reusing_a_global_ratio():
    model = FoldWeightedXGBClassifier(
        n_estimators=2, max_depth=2, n_jobs=1, scale_pos_weight=999,
    )
    inputs = np.arange(12).reshape(-1, 1)
    model.fit(inputs, [0] * 9 + [1] * 3)
    assert model.get_params()["scale_pos_weight"] == 3
    model.fit(inputs, [0] * 6 + [1] * 6)
    assert model.get_params()["scale_pos_weight"] == 1


@pytest.mark.parametrize("target", [[0, 0], [1, 1], [0, 2], [0, np.nan]])
def test_invalid_fitting_labels_are_rejected(target):
    with pytest.raises(ValueError, match="both binary target classes"):
        FoldWeightedXGBClassifier(n_estimators=2).fit([[0], [1]], target)


def test_search_configuration_has_twelve_candidates_and_five_training_folds():
    search = make_xgboost_search()
    assert len(list(ParameterGrid(search.param_grid))) == 12
    assert search.cv.n_splits == 5
    assert search.cv.shuffle and search.cv.random_state == 42
    assert search.scoring == "average_precision" and search.refit
    assert search.estimator.named_steps["model"].get_params().get("early_stopping_rounds") is None


def save_fake_previous_results(root, validation):
    target = validation["Machine failure"]
    metrics = classification_metrics(target, np.zeros(len(validation)))
    common = {
        "source_sha256": "synthetic-only", "target_policy": TARGET_POLICY,
        "random_state": 42, "threshold": 0.5, "test_evaluated": False,
        "evaluation_partition": "validation",
    }
    baseline = {**common, "models": {
        name: {"validation_metrics": metrics}
        for name in ["always_no_failure", "logistic_regression"]
    }}
    for name in ["always_no_failure", "logistic_regression", "random_forest"]:
        directory = (
            root / "artifacts/random_forest" if name == "random_forest"
            else root / "artifacts/baselines" / name
        )
        directory.mkdir(parents=True)
        pd.DataFrame({
            "source_row_index": validation.index, "UDI": validation["UDI"],
            TARGET_COLUMN: target, "failure_score": np.zeros(len(validation)),
            "predicted_label": np.zeros(len(validation), dtype=int),
        }).to_csv(directory / "validation_predictions.csv", index=False)
    (root / "artifacts/baselines/results.json").write_text(json.dumps(baseline))
    (root / "artifacts/random_forest/results.json").write_text(json.dumps({
        **common, "validation_metrics": metrics,
    }))


def test_stale_comparison_labels_are_rejected(tmp_path):
    module = exercise_module()
    validation = fake_readings(12, 1000)
    save_fake_previous_results(tmp_path, validation)
    changed = validation.copy()
    changed["Machine failure"] = 1 - changed["Machine failure"]
    with pytest.raises(ValueError, match="rows or labels"):
        module.load_previous_results(
            tmp_path, {"source_sha256": "synthetic-only", "random_state": 42}, changed,
        )


def test_full_exercise_keeps_test_reserved_and_roundtrips_native_model(tmp_path, monkeypatch):
    import mlflow
    from mlflow.tracking import MlflowClient
    import skops.io

    module = exercise_module()
    train, validation = fake_readings(60), fake_readings(12, 1000)

    class FakeSplits:
        @property
        def test(self):
            raise AssertionError("Test data must not be accessed.")

    fake = FakeSplits()
    fake.train, fake.validation = train, validation
    manifest_path = tmp_path / "data/splits/type_target_seed42.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "source_sha256": "synthetic-only", "random_state": 42,
    }))
    save_fake_previous_results(tmp_path, validation)
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in (tmp_path / "artifacts").rglob("*") if p.is_file()
    }
    monkeypatch.setattr(module, "load_saved_splits", lambda *_: fake)
    search = make_xgboost_search()
    search.param_grid = {"model__n_estimators": [2], "model__max_depth": [1, 2]}
    search.estimator.set_params(model__n_jobs=1)
    monkeypatch.setattr(module, "make_xgboost_search", lambda **_: search)
    original_fit = xgb.XGBClassifier.fit
    fit_sizes = []

    def fitting_partition_only(self, X, y, **kwargs):
        assert np.asarray(X)[:, -5:].max() < 60
        assert self.get_params()["scale_pos_weight"] == pytest.approx(
            np.count_nonzero(np.asarray(y) == 0) / np.count_nonzero(np.asarray(y) == 1)
        )
        assert not kwargs.get("eval_set")  # No validation-based early stopping.
        fit_sizes.append(len(X))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", fitting_partition_only)
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    try:
        module.main(tmp_path)
        output = tmp_path / "artifacts/xgboost"
        summary = json.loads((output / "results.json").read_text())
        assert summary["test_evaluated"] is False
        assert summary["threshold"] == 0.5 and summary["evaluation_partition"] == "validation"
        assert fit_sizes == [48] * 10 + [60]
        assert summary["scale_pos_weight"] == 3.0
        predictions = pd.read_csv(output / "validation_predictions.csv")
        assert predictions["source_row_index"].tolist() == validation.index.tolist()
        restored = xgb.XGBClassifier()
        restored.load_model(output / "model.ubj")
        preprocessing = skops.io.load(output / "preprocessor.skops", trusted=[])
        scores = restored.predict_proba(
            preprocessing.transform(validation[list(FEATURE_COLUMNS)]),
        )[:, 1]
        np.testing.assert_allclose(scores, predictions["failure_score"], atol=1e-7)
        assert summary["validation_metrics"] == classification_metrics(
            predictions[TARGET_COLUMN], predictions["failure_score"],
        )
        assert set(pd.read_csv(output / "comparison.csv")["model"]) == {
            "always_no_failure", "logistic_regression", "random_forest", "xgboost",
        }
        for filename in ["validation_confusion_matrix.png", "validation_precision_recall.png"]:
            assert (output / filename).stat().st_size > 0
        client = MlflowClient()
        run = client.get_run(summary["mlflow_run_id"])
        assert run.info.status == "FINISHED"
        logged = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert {"model.ubj", "preprocessor.skops", "cv_results.csv", "comparison.csv"} <= logged
        assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    finally:
        mlflow.set_tracking_uri(previous_uri)
