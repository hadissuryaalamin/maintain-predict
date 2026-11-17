import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from maintain_predict.evaluate import classification_metrics
from maintain_predict.features import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_POLICY
from maintain_predict.train import make_random_forest_pipeline, make_random_forest_search


def fake_readings(n, offset=0):
    values = np.arange(n, dtype=float) + offset
    return pd.DataFrame({
        "UDI": np.arange(n) + offset + 1,
        "Type": ["L" if i % 2 else "M" for i in range(n)],
        **{c: values for c in FEATURE_COLUMNS if c != "Type"},
        "Machine failure": [i % 2 for i in range(n)],
        **{c: np.zeros(n, dtype=int) for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
    }, index=np.arange(n) + offset)


def exercise_module():
    path = Path(__file__).resolve().parents[1] / "scripts/03_random_forest.py"
    spec = importlib.util.spec_from_file_location("random_forest_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Import must not train or write outputs.
    return module


def test_forest_handles_unseen_type_without_refitting_or_scaling_sensors():
    train = fake_readings(20)
    before = train.copy(deep=True)
    model = make_random_forest_pipeline(n_jobs=1)
    assert not hasattr(model.named_steps["model"], "estimators_")
    model.set_params(model__n_estimators=4)
    model.fit(train[list(FEATURE_COLUMNS)], train["Machine failure"])
    preprocessing = model.named_steps["preprocess"]
    validation = fake_readings(2, offset=1000)
    validation["Type"] = "H"
    transformed = preprocessing.transform(validation[list(FEATURE_COLUMNS)])
    numeric = [c for c in FEATURE_COLUMNS if c != "Type"]
    np.testing.assert_array_equal(transformed[:, -5:], validation[numeric].to_numpy())
    probabilities = model.predict_proba(validation[list(FEATURE_COLUMNS)])
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert preprocessing.named_transformers_["type"].categories_[0].tolist() == ["L", "M"]
    pd.testing.assert_frame_equal(train, before)


def test_search_folds_are_disjoint_and_cover_training_only():
    search = make_random_forest_search()
    assert len(list(ParameterGrid(search.param_grid))) == 12
    assert search.scoring == "average_precision"
    assert search.refit is True
    data = fake_readings(60)
    held_out = []
    for train_indices, held_indices in search.cv.split(data, data["Machine failure"]):
        assert not set(train_indices).intersection(held_indices)
        assert data.iloc[held_indices]["Machine failure"].value_counts().to_dict() == {0: 6, 1: 6}
        held_out.extend(held_indices)
    assert sorted(held_out) == list(range(60))


def save_fake_baseline(root, validation):
    folder = root / "artifacts/baselines/always_no_failure"
    folder.mkdir(parents=True)
    pd.DataFrame({"source_row_index": validation.index}).to_csv(
        folder / "validation_predictions.csv", index=False,
    )
    summary = {
        "source_sha256": "synthetic-only", "target_policy": TARGET_POLICY,
        "random_state": 42, "threshold": 0.5, "test_evaluated": False,
        "evaluation_partition": "validation",
        "models": {"always_no_failure": {"validation_metrics": classification_metrics(
            validation["Machine failure"], np.zeros(len(validation)),
        )}},
    }
    (folder.parent / "results.json").write_text(json.dumps(summary), encoding="utf-8")
    return folder.parent


def test_stale_baseline_membership_is_rejected_before_training(tmp_path):
    module = exercise_module()
    validation = fake_readings(12, 1000)
    save_fake_baseline(tmp_path, validation)
    with pytest.raises(ValueError, match="membership"):
        module.load_baseline_comparison(
            tmp_path, {"source_sha256": "synthetic-only", "random_state": 42},
            list(reversed(validation.index)),
        )


def test_exercise_saves_artifacts_without_training_on_validation_or_accessing_test(
    tmp_path, monkeypatch,
):
    import mlflow
    from mlflow.tracking import MlflowClient

    module = exercise_module()
    train = fake_readings(60)
    validation = fake_readings(12, 1000)

    class FakeSplits:
        @property
        def test(self):
            raise AssertionError("The exercise must never access test rows.")

    fake = FakeSplits()
    fake.train, fake.validation = train, validation
    manifest_path = tmp_path / "data/splits/type_target_seed42.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "source_sha256": "synthetic-only", "random_state": 42,
    }), encoding="utf-8")
    baseline_dir = save_fake_baseline(tmp_path, validation)
    original_baseline_bytes = {
        p.relative_to(baseline_dir): p.read_bytes()
        for p in baseline_dir.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(module, "load_saved_splits", lambda *_: fake)
    search = make_random_forest_search()
    search.param_grid = {"model__n_estimators": [3], "model__max_depth": [2, None]}
    search.estimator.set_params(model__n_jobs=1)
    monkeypatch.setattr(module, "make_random_forest_search", lambda **_: search)
    original_fit = RandomForestClassifier.fit
    fit_sizes = []

    def training_only_fit(self, X, y, **kwargs):
        # Numeric validation inputs start at 1000; no fitted forest may see them.
        assert np.asarray(X)[:, -5:].max() < 60
        fit_sizes.append(len(X))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(RandomForestClassifier, "fit", training_only_fit)
    old_tracking_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)  # Contain any MLflow defaults in the synthetic case.
    try:
        module.main(tmp_path)
        output = tmp_path / "artifacts/random_forest"
        summary = json.loads((output / "results.json").read_text())
        assert summary["test_evaluated"] is False
        assert summary["evaluation_partition"] == "validation"
        assert summary["threshold"] == 0.5
        assert fit_sizes == [48] * 10 + [60]
        predictions = pd.read_csv(output / "validation_predictions.csv")
        assert predictions["source_row_index"].tolist() == validation.index.tolist()
        assert predictions[TARGET_COLUMN].tolist() == validation["Machine failure"].tolist()
        expected = classification_metrics(
            predictions[TARGET_COLUMN], predictions["failure_score"],
        )
        assert summary["validation_metrics"] == expected
        for filename in [
            "model.skops", "validation_confusion_matrix.png",
            "validation_precision_recall.png", "cv_results.csv", "comparison.csv",
        ]:
            assert (output / filename).stat().st_size > 0
        run = MlflowClient().get_run(summary["mlflow_run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.metrics["validation_tp"] == expected["confusion_matrix"]["tp"]
        logged = {a.path for a in MlflowClient().list_artifacts(run.info.run_id)}
        assert {"model.skops", "comparison.csv", "cv_results.csv"} <= logged
        assert original_baseline_bytes == {
            p.relative_to(baseline_dir): p.read_bytes()
            for p in baseline_dir.rglob("*") if p.is_file()
        }
    finally:
        mlflow.set_tracking_uri(old_tracking_uri)
