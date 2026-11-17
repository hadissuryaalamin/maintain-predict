"""Validation-only explanation helpers; do not fit or change a model."""

import numpy as np
from scipy.special import expit

from maintain_predict.features import FEATURE_COLUMNS, TORQUE_WEAR_COLUMN

SELECTED_FEATURE_COLUMNS = (*FEATURE_COLUMNS, TORQUE_WEAR_COLUMN)


def group_encoded_contributions(values, encoded_names):
    """Sum signed Type indicator contributions into one post-hoc group.

    This preserves additivity. It does not recompute a separate grouped
    Shapley game, and mean(abs(sum(Type))) differs from sum(mean(abs(Type))).
    """
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 2 or values.shape[1] != len(encoded_names)
        or len(set(encoded_names)) != len(encoded_names)
        or not np.isfinite(values).all()
    ):
        raise ValueError("SHAP values must be a finite matrix aligned to unique encoded names.")
    grouped = np.zeros((len(values), len(SELECTED_FEATURE_COLUMNS)))
    seen = set()
    for index, name in enumerate(encoded_names):
        if name.startswith("type__Type_"):
            feature = "Type"
        elif name.startswith("sensors__"):
            feature = name.removeprefix("sensors__")
        else:
            raise ValueError(f"Unexpected encoded feature: {name}")
        if feature not in SELECTED_FEATURE_COLUMNS:
            raise ValueError(f"Feature is outside the selected seven-input schema: {feature}")
        grouped[:, SELECTED_FEATURE_COLUMNS.index(feature)] += values[:, index]
        seen.add(feature)
    if seen != set(SELECTED_FEATURE_COLUMNS):
        raise ValueError("Encoded names do not cover all seven selected inputs.")
    return grouped


def reconstruction_errors(base_values, contributions, margins, probabilities):
    """Require additive margins and their sigmoid to match actual predictions."""
    contributions = np.asarray(contributions, dtype=float)
    margins = np.asarray(margins, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    bases = np.asarray(base_values, dtype=float)
    if contributions.ndim != 2 or len(contributions) == 0:
        raise ValueError("Explanation contributions must be a nonempty matrix.")
    n = len(contributions)
    if bases.ndim == 0:
        bases = np.full(n, bases.item())
    if bases.shape != (n,) or margins.shape != (n,) or probabilities.shape != (n,):
        raise ValueError("Explanation baselines and predictions must align row by row.")
    if not all(np.isfinite(x).all() for x in [bases, contributions, margins, probabilities]):
        raise ValueError("Explanation reconstruction requires finite values.")
    reconstructed = bases + contributions.sum(axis=1)
    if not np.allclose(reconstructed, margins, atol=1e-4, rtol=1e-5):
        raise ValueError("SHAP contributions do not reconstruct model margins.")
    if not np.allclose(expit(reconstructed), probabilities, atol=1e-6, rtol=1e-5):
        raise ValueError("Reconstructed SHAP margins do not match prediction scores.")
    return {
        "max_abs_margin_error": float(np.max(np.abs(reconstructed - margins))),
        "max_abs_probability_error": float(np.max(np.abs(expit(reconstructed) - probabilities))),
    }


def representative_positions(target, predicted, source_indices):
    """Pick the smallest source-row index in each available outcome category."""
    target, predicted, source_indices = (
        np.asarray(target), np.asarray(predicted), np.asarray(source_indices)
    )
    if (
        target.ndim != 1 or predicted.shape != target.shape
        or source_indices.shape != target.shape
        or not np.isin(target, [0, 1]).all() or not np.isin(predicted, [0, 1]).all()
        or len(set(source_indices)) != len(source_indices)
    ):
        raise ValueError("Representative examples require aligned binary labels and unique row IDs.")
    result = {}
    for label, actual, prediction in [("TP", 1, 1), ("FN", 1, 0), ("FP", 0, 1), ("TN", 0, 0)]:
        positions = np.flatnonzero((target == actual) & (predicted == prediction))
        if len(positions):
            result[label] = int(positions[np.argmin(source_indices[positions])])
    return result
