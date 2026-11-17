"""Unfitted pipelines and searches for the manual training exercises."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from maintain_predict.data import DEFAULT_RANDOM_STATE
from maintain_predict.features import FEATURE_COLUMNS


def make_logistic_pipeline(*, random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    """Construct an unfitted pipeline; the caller supplies training data to fit.

    Type is one-hot encoded. Only the five numeric inputs are scaled.
    No preprocessing is fitted during construction or module import.
    """
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "Type"]
    preprocessing = ColumnTransformer(
        [
            ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
            ("sensors", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=2000,
            random_state=random_state,
        )),
    ])


def make_random_forest_pipeline(
    *, random_state: int = DEFAULT_RANDOM_STATE, n_jobs: int = -1
) -> Pipeline:
    """One-hot Type and pass sensors through unchanged; do not fit anything."""
    numeric_columns = [column for column in FEATURE_COLUMNS if column != "Type"]
    preprocessing = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
        ("sensors", "passthrough", numeric_columns),
    ], remainder="drop")
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=random_state, n_jobs=n_jobs,
        )),
    ])


def make_random_forest_search(*, random_state: int = DEFAULT_RANDOM_STATE) -> GridSearchCV:
    """Prepare 12 configurations, selected by training-only five-fold AP."""
    return GridSearchCV(
        estimator=make_random_forest_pipeline(random_state=random_state),
        param_grid={
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 5],
        },
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        refit=True, n_jobs=1, error_score="raise", verbose=1,
    )
