"""Load readings and make reproducible Type-by-target data partitions."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from maintain_predict.features import TARGET_COLUMN, TARGET_POLICY, select_features_and_target

DEFAULT_DATA_PATH = Path("data/ai4i2020.csv")
DEFAULT_RANDOM_STATE = 42
DEFAULT_SPLIT_PATH = Path("data/splits/type_target_seed42.json")


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


def load_saved_splits(
    data_path: str | Path = DEFAULT_DATA_PATH,
    manifest_path: str | Path = DEFAULT_SPLIT_PATH,
) -> ReadingSplits:
    """Restore the frozen allocation without generating another random split.

    Validate the source bytes, target policy, row coverage, and UDI alignment
    before returning raw rows. This does not fit or evaluate any model.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    source_bytes = Path(data_path).read_bytes()
    if manifest.get("source_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("The source CSV does not match the saved split checksum.")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("target_policy") != TARGET_POLICY
        or manifest.get("stratify") != ["Type", TARGET_COLUMN]
        or manifest.get("random_state") != DEFAULT_RANDOM_STATE
        or manifest.get("tool_wear_stratum") is not False
        or manifest.get("fractions") != {"train": 0.70, "validation": 0.15, "test": 0.15}
    ):
        raise ValueError("The saved split does not match the agreed split and target policy.")
    if set(manifest.get("partitions", {})) != {"train", "validation", "test"}:
        raise ValueError("The manifest must contain train, validation, and test partitions.")

    readings = pd.read_csv(BytesIO(source_bytes))
    partitions, seen = {}, set()
    for name in ["train", "validation", "test"]:
        saved = manifest["partitions"][name]
        indices = saved["source_row_indices"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(index) is not int or index < 0 or index >= len(readings) for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(f"Invalid or duplicate source-row indices in {name}.")
        if seen.intersection(indices):
            raise ValueError("Saved partitions contain overlapping source rows.")
        seen.update(indices)
        partition = readings.iloc[indices].copy()
        if partition["UDI"].tolist() != saved["udi"]:
            raise ValueError(f"Saved UDI values do not match the source rows in {name}.")
        partitions[name] = partition

    if seen != set(readings.index):
        raise ValueError("Saved partitions do not cover every source row.")
    # train_test_split rounds each requested holdout size upward.
    remaining_count = int(np.ceil(len(readings) * 0.30))
    test_count = int(np.ceil(remaining_count * 0.50))
    expected = {
        "train": len(readings) - remaining_count,
        "validation": remaining_count - test_count,
        "test": test_count,
    }
    if {name: len(partition) for name, partition in partitions.items()} != expected:
        raise ValueError("Saved partition sizes do not match the agreed 70/15/15 split.")
    return ReadingSplits(**partitions)
