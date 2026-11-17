"""Binary validation metrics and diagnostic plots for the manual exercise."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def classification_metrics(target, probabilities, *, threshold: float = 0.5) -> dict:
    """Summarize binary scores without choosing a threshold from the data.

    Precision is None when no positives are predicted; F1 is zero in that case.
    AP means average precision, not trapezoidal area under a PR curve.
    """
    target = np.asarray(target)
    probabilities = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or probabilities.ndim != 1 or len(target) != len(probabilities):
        raise ValueError("Target and probabilities must be aligned one-dimensional arrays.")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Evaluation requires both binary target classes.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted)) if predicted.any() else None,
        "recall": float(recall_score(target, predicted)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "threshold": float(threshold),
        "observations": int(len(target)),
        "actual_positives": int(np.count_nonzero(target)),
        "predicted_positives": int(np.count_nonzero(predicted)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_validation_plots(target, probabilities, metrics: dict, output_dir: str | Path, *, title: str) -> list[Path]:
    """Save PR and confusion-matrix plots for the supplied validation outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(target, probabilities)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.step(recall, precision, where="post", label=f"AP = {metrics['average_precision']:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05))
    ax.set_title(title + "\nValidation precision–recall curve")
    ax.legend(frameon=False)
    pr_path = output_dir / "validation_precision_recall.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    counts = metrics["confusion_matrix"]
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(5.4, 4), layout="constrained")
    ConfusionMatrixDisplay(matrix, display_labels=["No evidence", "Evidence"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title + f"\nValidation; threshold {metrics['threshold']:g}")
    confusion_path = output_dir / "validation_confusion_matrix.png"
    fig.savefig(confusion_path, dpi=150)
    plt.close(fig)
    return [pr_path, confusion_path]
