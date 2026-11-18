"""MLflow prediction wrapper for the selected native model and preprocessor."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel
import skops.io
from xgboost import XGBClassifier

from maintain_predict.features import (
    FEATURE_COLUMNS, TARGET_POLICY, TORQUE_WEAR_COLUMN, add_torque_wear,
)


class MaintainPredictModel(PythonModel):
    """Accept six raw inputs; derive torque-wear and return score plus label."""

    def load_context(self, context):
        self.contract = json.loads(Path(context.artifacts["contract"]).read_text(encoding="utf-8"))
        if (
            self.contract["target_policy"] != TARGET_POLICY
            or self.contract["raw_feature_columns"] != list(FEATURE_COLUMNS)
            or self.contract["model_feature_columns"] != [*FEATURE_COLUMNS, TORQUE_WEAR_COLUMN]
            or self.contract["threshold"] != 0.5
        ):
            raise ValueError("Packaged contract differs from the selected model policy.")
        for key, artifact in [("model_sha256", "classifier"), ("preprocessor_sha256", "preprocessor")]:
            digest = hashlib.sha256(Path(context.artifacts[artifact]).read_bytes()).hexdigest()
            if digest != self.contract[key]:
                raise ValueError(f"Packaged {artifact} checksum does not match the contract.")
        self.preprocessor = skops.io.load(context.artifacts["preprocessor"], trusted=[])
        if self.preprocessor.feature_names_in_.tolist() != self.contract["model_feature_columns"]:
            raise ValueError("Packaged preprocessor has an unexpected feature schema.")
        self.classifier = XGBClassifier()
        self.classifier.load_model(context.artifacts["classifier"])
        if self.classifier.classes_.tolist() != [0, 1]:
            raise ValueError("Expected binary class order [0, 1].")

    def predict(self, context, model_input, params=None):
        if params:
            raise ValueError("This model uses its fixed packaged threshold and has no prediction parameters.")
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Provide a DataFrame containing the six raw inputs.")
        missing = set(FEATURE_COLUMNS) - set(model_input.columns)
        if missing:
            raise ValueError(f"Missing raw inputs: {sorted(missing)}")
        features = model_input.loc[:, list(FEATURE_COLUMNS)].copy()
        if not features["Type"].isin(["L", "M", "H"]).all():
            raise ValueError("Type must be L, M, or H.")
        numeric = [name for name in FEATURE_COLUMNS if name != "Type"]
        for name in numeric:
            features[name] = pd.to_numeric(features[name], errors="raise")
        if not np.isfinite(features[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Numeric inputs must be finite and nonmissing.")
        transformed = self.preprocessor.transform(add_torque_wear(features))
        scores = self.classifier.predict_proba(transformed)[:, 1]
        return pd.DataFrame({
            "failure_score": scores.astype(float),
            "predicted_label": (scores >= self.contract["threshold"]).astype("int64"),
            "threshold": float(self.contract["threshold"]),
        }, index=model_input.index)
