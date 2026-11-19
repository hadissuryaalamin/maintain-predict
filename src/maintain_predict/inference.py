"""Load this repository's native XGBoost model and skops preprocessor locally."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io
from xgboost import XGBClassifier

RAW_FEATURES = (
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
)
TORQUE_WEAR = "Torque wear [Nm min]"


class FailurePredictor:
    """Inference only: six raw columns become seven model features internally."""

    def __init__(self, model_dir=None):
        folder = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
        self.contract = json.loads((folder / "model_contract.json").read_text(encoding="utf-8"))
        if (
            self.contract["raw_feature_columns"] != list(RAW_FEATURES)
            or self.contract["model_feature_columns"] != [*RAW_FEATURES, TORQUE_WEAR]
            or self.contract["target_policy"] != "failure_evidence_or_v1"
            or self.contract["threshold"] != 0.5
        ):
            raise ValueError("This loader requires the selected seven-feature model at threshold 0.5.")
        for filename, key in [
            ("model.ubj", "model_sha256"),
            ("preprocessor.skops", "preprocessor_sha256"),
        ]:
            digest = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
            if digest != self.contract[key]:
                raise ValueError(f"Artifact checksum mismatch: {filename}")
        self.preprocessor = skops.io.load(folder / "preprocessor.skops", trusted=[])
        if self.preprocessor.feature_names_in_.tolist() != self.contract["model_feature_columns"]:
            raise ValueError("Preprocessor feature order differs from the contract.")
        self.classifier = XGBClassifier()
        self.classifier.load_model(folder / "model.ubj")
        if self.classifier.classes_.tolist() != [0, 1]:
            raise ValueError("Expected binary classes in order [0, 1].")

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(rows, pd.DataFrame):
            raise ValueError("Provide a pandas DataFrame with the six raw columns.")
        if rows.columns.duplicated().any():
            raise ValueError("Input column names must be unique.")
        missing = set(RAW_FEATURES) - set(rows.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        # Allowlist excludes all labels, identifiers, and supplied derived features.
        features = rows.loc[:, list(RAW_FEATURES)].copy()
        if not features["Type"].isin(["L", "M", "H"]).all():
            raise ValueError("Type must be L, M, or H.")
        numeric = list(RAW_FEATURES[1:])
        for column in numeric:
            features[column] = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(features[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Numeric inputs must be finite and nonmissing.")
        with np.errstate(over="ignore", invalid="ignore"):
            features[TORQUE_WEAR] = features["Torque [Nm]"] * features["Tool wear [min]"]
        if not np.isfinite(features[TORQUE_WEAR].to_numpy(dtype=float)).all():
            raise ValueError("Torque times tool wear must be finite.")
        scores = self.classifier.predict_proba(self.preprocessor.transform(features))[:, 1]
        return pd.DataFrame({
            "failure_score": scores.astype(float),
            "predicted_label": (scores >= self.contract["threshold"]).astype("int64"),
            "threshold": float(self.contract["threshold"]),
        }, index=rows.index)
