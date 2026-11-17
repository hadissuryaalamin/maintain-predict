import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from maintain_predict.evaluate import classification_metrics
from maintain_predict.features import FEATURE_COLUMNS
from maintain_predict.train import make_logistic_pipeline


def test_metrics_match_a_known_confusion_matrix_and_average_precision():
    metrics = classification_metrics([1, 0, 1, 0], [0.8, 0.6, 0.2, 0.1])
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert metrics[key] == pytest.approx(0.5)
    assert metrics["average_precision"] == pytest.approx(5 / 6)


def test_always_negative_baseline_has_zero_recall_and_undefined_precision():
    model = DummyClassifier(strategy="constant", constant=0)
    model.fit(np.zeros((4, 1)), [0, 1, 0, 1])
    scores = model.predict_proba(np.zeros((4, 1)))[:, 1]
    metrics = classification_metrics([1, 0, 1, 0], scores)
    assert metrics["precision"] is None
    assert metrics["recall"] == 0
    assert metrics["f1"] == 0
    assert metrics["average_precision"] == pytest.approx(0.5)
    assert metrics["predicted_positives"] == 0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 2, "tp": 0}


@pytest.mark.parametrize("scores", [[0.1], [0.1, np.nan], [0.1, 1.1]])
def test_metrics_reject_invalid_or_misaligned_scores(scores):
    with pytest.raises(ValueError):
        classification_metrics([0, 1], scores)


def test_metrics_reject_an_evaluation_set_without_positive_cases():
    with pytest.raises(ValueError, match="both binary target classes"):
        classification_metrics([0, 0], [0.1, 0.2])


def test_threshold_includes_the_boundary():
    metrics = classification_metrics([0, 1], [0.499, 0.5], threshold=0.5)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}


def test_pipeline_learns_categories_and_scaling_only_from_training_rows():
    numeric = [column for column in FEATURE_COLUMNS if column != "Type"]
    train = pd.DataFrame({
        "Type": ["L", "M", "L", "M"],
        **{column: [10.0, 20.0, 30.0, 40.0] for column in numeric},
    })
    before = train.copy(deep=True)
    validation = pd.DataFrame({"Type": ["H", "H"], **{column: [1000.0, 2000.0] for column in numeric}})
    pipeline = make_logistic_pipeline()
    assert not hasattr(pipeline.named_steps["model"], "coef_")
    pipeline.fit(train, [0, 1, 0, 1])
    preprocessing = pipeline.named_steps["preprocess"]
    scaler = preprocessing.named_transformers_["sensors"]
    encoder = preprocessing.named_transformers_["type"]
    np.testing.assert_allclose(scaler.mean_, train[numeric].mean().to_numpy())
    assert encoder.categories_[0].tolist() == ["L", "M"]
    mean_before_prediction = scaler.mean_.copy()
    probabilities = pipeline.predict_proba(validation)
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(scaler.mean_, mean_before_prediction)
    assert encoder.categories_[0].tolist() == ["L", "M"]
    pd.testing.assert_frame_equal(train, before)
