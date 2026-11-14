"""Load readings and make reproducible Type-by-target data partitions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from maintain_predict.features import select_features_and_target

DEFAULT_DATA_PATH = Path("data/ai4i2020.csv")
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class ReadingSplits:
    """Raw rows with their original indices, retained for auditability."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def load_readings(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the locally downloaded AI4I CSV."""
    return pd.read_csv(path)


def split_readings(
    readings: pd.DataFrame, *, random_state: int = DEFAULT_RANDOM_STATE
) -> ReadingSplits:
    """Split 70/15/15, stratified jointly by Type and the derived target.

    Tool wear stays numeric and is not a splitting stratum. Original row
    indices are preserved; preprocessing must be fitted later on train only.
    """
    if not readings.index.is_unique:
        raise ValueError("Source row indices must be unique to audit partition overlap.")

    features, target = select_features_and_target(readings)
    if not features["Type"].isin(["L", "M", "H"]).all():
        raise ValueError("Product Type must contain only L, M, or H.")
    if target.nunique() != 2:
        raise ValueError("Splitting for classification requires both target classes.")

    strata = features["Type"].astype(str) + ":" + target.astype(str)
    positions = np.arange(len(readings))
    try:
        train_positions, holdout_positions = train_test_split(
            positions,
            test_size=0.30,
            random_state=random_state,
            stratify=strata,
        )
        validation_positions, test_positions = train_test_split(
            holdout_positions,
            test_size=0.50,
            random_state=random_state,
            stratify=strata.iloc[holdout_positions],
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot make a 70/15/15 Type-by-target split; each joint stratum "
            "needs enough rows for all three partitions."
        ) from exc

    return ReadingSplits(
        train=readings.iloc[train_positions].copy(),
        validation=readings.iloc[validation_positions].copy(),
        test=readings.iloc[test_positions].copy(),
    )
