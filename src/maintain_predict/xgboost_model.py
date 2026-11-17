"""Optional XGBoost estimator with fitting-partition class weights."""

import numpy as np
from xgboost import XGBClassifier


class FoldWeightedXGBClassifier(XGBClassifier):
    """Recompute scale_pos_weight from y on every fit, including CV folds.

    Other constructor parameters and sklearn cloning follow XGBClassifier.
    No held-out labels or externally calculated class weights are needed.
    """

    def fit(self, X, y, **kwargs):
        target = np.asarray(y)
        if target.ndim != 1 or set(np.unique(target)) != {0, 1}:
            raise ValueError("XGBoost fitting requires both binary target classes.")
        ratio = float(np.count_nonzero(target == 0) / np.count_nonzero(target == 1))
        self.set_params(scale_pos_weight=ratio)
        return super().fit(X, y, **kwargs)
