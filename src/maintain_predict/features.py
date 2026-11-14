"""Select allowed inputs and derive the recorded-failure-evidence target."""

import pandas as pd

FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
SOURCE_TARGET_COLUMN = "Machine failure"
FAILURE_FLAG_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")
TARGET_COLUMN = "Failure evidence"
TARGET_POLICY = "failure_evidence_or_v1"


def derive_failure_target(readings: pd.DataFrame) -> pd.Series:
    """Treat any recorded positive outcome as sufficient failure evidence.

    The original overall label and all five failure-mode flags are ORed
    row by row. This is a documented label assumption, not a verified
    correction of the source labels. The input DataFrame is not modified.
    """
    recorded_labels = readings.loc[:, [SOURCE_TARGET_COLUMN, *FAILURE_FLAG_COLUMNS]]
    if not recorded_labels.isin([0, 1]).to_numpy().all():
        raise ValueError("Recorded failure labels must be nonmissing binary values (0 or 1).")

    return recorded_labels.eq(1).any(axis=1).astype("int64").rename(TARGET_COLUMN)


def select_features_and_target(
    readings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select the six allowed inputs and the derived failure-evidence target.

    Outcome flags help define the label; they are never predictor inputs.
    """
    return readings.loc[:, list(FEATURE_COLUMNS)], derive_failure_target(readings)
