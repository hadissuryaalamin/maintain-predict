import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from test_explain import saved_selected
from maintain_predict.explain import SELECTED_FEATURE_COLUMNS
from maintain_predict.evaluate import classification_metrics, save_evaluation_plots
from maintain_predict.features import TARGET_POLICY


def exercise_module():
    path = Path(__file__).resolve().parents[1] / "scripts/10_final_test.py"
    spec = importlib.util.spec_from_file_location("final_test_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def final_case(saved_selected):
    root, test_rows, manifest = saved_selected
    module = exercise_module()
    folder = root / "artifacts/torque_wear"
    selection_path = folder / "results.json"
    selection = json.loads(selection_path.read_text())
    selection["fixed_params"] = {
        "model__n_estimators": 6, "model__max_depth": 2, "model__learning_rate": 0.1,
    }
    selection_path.write_text(json.dumps(selection))
    shap_folder = root / "artifacts/shap"
    shap_folder.mkdir()
    shap_summary = {
        "selected_model_run_id": selection["mlflow_run_id"],
        "source_sha256": manifest["source_sha256"], "target_policy": TARGET_POLICY,
        "feature_columns": list(SELECTED_FEATURE_COLUMNS), "threshold": 0.5,
        "model_sha256": module.sha256(folder / "model.ubj"),
        "preprocessor_sha256": module.sha256(folder / "preprocessor.skops"),
        "mlflow_run_id": "synthetic-shap-run",
    }
    (shap_folder / "summary.json").write_text(json.dumps(shap_summary))
    (shap_folder / "latest.json").write_text(json.dumps({
        "summary_file": str(shap_folder / "summary.json"),
    }))
    return root, test_rows, module


def test_final_evaluation_predicts_once_without_fitting_and_reuses_completed_results(final_case, monkeypatch):
    root, test_rows, module = final_case
    import mlflow
    from mlflow.tracking import MlflowClient
    from matplotlib.figure import Figure
    from sklearn.compose import ColumnTransformer
    from xgboost import XGBClassifier

    class OnlyTest:
        @property
        def train(self):
            raise AssertionError("No training rows should be used for final evaluation.")

        @property
        def validation(self):
            raise AssertionError("Do not predict on validation again.")

    split = OnlyTest()
    split.test = test_rows
    load_calls = []

    def load_split(*args):
        load_calls.append(1)
        return split

    monkeypatch.setattr(module, "load_saved_splits", load_split)
    predict = XGBClassifier.predict_proba
    prediction_calls = []

    def predict_once(self, X, **kwargs):
        prediction_calls.append(1)
        assert len(prediction_calls) == 1
        assert X.shape == (12, 8)  # Two Type indicators in the synthetic fixture.
        return predict(self, X, **kwargs)

    def no_fit(*args, **kwargs):
        raise AssertionError("Final evaluation must not refit anything.")

    monkeypatch.setattr(XGBClassifier, "predict_proba", predict_once)
    monkeypatch.setattr(XGBClassifier, "fit", no_fit)
    monkeypatch.setattr(ColumnTransformer, "fit", no_fit)
    savefig = Figure.savefig
    titles = []

    def save_labeled_figure(self, *args, **kwargs):
        titles.append(self.axes[0].get_title())
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save_labeled_figure)
    before = {
        p.relative_to(root): p.read_bytes()
        for p in (root / "artifacts").rglob("*") if p.is_file()
    }
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(root)
    try:
        result = module.main(root)
        output = root / "artifacts/final_test"
        assert result["test_evaluated"] is True and result["model_refitted"] is False
        assert result["threshold"] == 0.5 and result["evaluation_partition"] == "test"
        assert result["selected_before_test"] is True
        predictions = pd.read_csv(output / "test_predictions.csv")
        assert predictions["source_row_index"].tolist() == test_rows.index.tolist()
        assert result["test_metrics"] == classification_metrics(
            predictions["Failure evidence"], predictions["failure_score"],
        )
        configuration = json.loads((output / "evaluation_config.json").read_text())
        assert configuration["feature_columns"] == list(SELECTED_FEATURE_COLUMNS)
        assert configuration["test_source_row_indices"] == test_rows.index.tolist()
        assert len(titles) == 2 and all("Test" in title and "Validation" not in title for title in titles)
        for filename in ["test_confusion_matrix.png", "test_precision_recall.png"]:
            assert (output / filename).stat().st_size > 0
        client = MlflowClient()
        assert client.get_run(result["mlflow_run_id"]).info.status == "FINISHED"
        logged = {a.path for a in client.list_artifacts(result["mlflow_run_id"])}
        assert {"results.json", "test_predictions.csv", "evaluation_config.json"} <= logged
        saved = {p.name: p.read_bytes() for p in output.iterdir()}
        assert module.main(root) == result
        assert len(prediction_calls) == len(load_calls) == 1
        assert saved == {p.name: p.read_bytes() for p in output.iterdir()}
        assert all((root / p).read_bytes() == content for p, content in before.items())
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_changed_model_is_rejected_before_test_data_access(final_case, monkeypatch):
    root, _, module = final_case
    path = root / "artifacts/torque_wear/model.ubj"
    path.write_bytes(path.read_bytes() + b"changed")

    def no_test_access(*args):
        raise AssertionError("Changed selection must be rejected before loading test rows.")

    monkeypatch.setattr(module, "load_saved_splits", no_test_access)
    with pytest.raises(ValueError, match="frozen selection"):
        module.main(root)
    assert not (root / "artifacts/final_test").exists()


def test_incomplete_evaluation_blocks_new_predictions(tmp_path):
    pytest.importorskip("xgboost")
    module = exercise_module()
    (tmp_path / "artifacts/final_test").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Do not rerun inference"):
        module.main(tmp_path)


def test_plot_partition_rejects_an_invalid_label(tmp_path):
    with pytest.raises(ValueError, match="partition"):
        save_evaluation_plots([0, 1], [0.1, 0.9], {}, tmp_path, title="Example", partition="train")
