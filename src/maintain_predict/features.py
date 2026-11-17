"""Select allowed inputs and derive the recorded-failure-evidence target."""

from math import pi

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
TEMPERATURE_DIFFERENCE_COLUMN = "Temperature difference [K]"
POWER_COLUMN = "Power [W]"
TORQUE_WEAR_COLUMN = "Torque wear [Nm min]"


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


def add_temperature_difference(features: pd.DataFrame) -> pd.DataFrame:
    """Keep six allowed inputs and append process minus air temperature.

    This is a deterministic row-wise calculation with no fitted statistics.
    Never mutate the source or carry outcome columns into model inputs.
    """
    result = features.loc[:, list(FEATURE_COLUMNS)].copy()
    result[TEMPERATURE_DIFFERENCE_COLUMN] = (
        result["Process temperature [K]"] - result["Air temperature [K]"]
    )
    return result


def add_power(features: pd.DataFrame) -> pd.DataFrame:
    """Keep six raw inputs and append torque times angular speed in watts.

    rpm * 2*pi/60 converts rotational speed to radians per second. No target,
    fitted statistics, or other engineered feature is used or retained.
    """
    result = features.loc[:, list(FEATURE_COLUMNS)].copy()
    result[POWER_COLUMN] = (
        result["Torque [Nm]"] * result["Rotational speed [rpm]"] * (2 * pi / 60)
    )
    return result


def add_torque_wear(features: pd.DataFrame) -> pd.DataFrame:
    """Keep six raw inputs and add their torque-by-tool-wear interaction.

    This is an interaction feature, not a measurement of physical strain.
    Outcome columns and other engineered features are excluded.
    """
    result = features.loc[:, list(FEATURE_COLUMNS)].copy()
    result[TORQUE_WEAR_COLUMN] = result["Torque [Nm]"] * result["Tool wear [min]"]
    return result


def add_torque_wear_and_temperature_difference(features: pd.DataFrame) -> pd.DataFrame:
    """Retain raw inputs and calculate both selected features from their sensors.

    Do not chain the single-feature helpers: each intentionally keeps only its
    own extra column. Start with torque-wear, then append temperature difference.
    """
    result = add_torque_wear(features)
    result[TEMPERATURE_DIFFERENCE_COLUMN] = (
        result["Process temperature [K]"] - result["Air temperature [K]"]
    )
    return result
