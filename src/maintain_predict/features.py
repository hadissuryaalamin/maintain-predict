"""Select the input measurements and failure target."""

import pandas as pd

FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
TARGET_COLUMN = "Machine failure"


def select_features_and_target(
    readings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select the feature allowlist and target from raw readings."""
    return readings.loc[:, list(FEATURE_COLUMNS)], readings[TARGET_COLUMN]
