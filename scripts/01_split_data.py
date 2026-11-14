# %% Step 1: load the original CSV and the agreed target
r"""Manual exercise: a 70/15/15 split using Type x Failure evidence.

Read each numbered step before running it. The # %% markers also let you run
the steps individually in a Python interactive editor. This exercise does not
train a model or calculate test-set performance.

From the repository root:
    .\.venv\Scripts\python.exe .\scripts\01_split_data.py
"""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# Locate the repository both as a script and in an interactive editor.
candidate_roots = [Path.cwd(), *Path.cwd().parents]
if "__file__" in globals():
    candidate_roots.insert(0, Path(__file__).resolve().parents[1])
PROJECT_ROOT = next(
    (path for path in candidate_roots if (path / "src/maintain_predict").is_dir()),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("Open or run this exercise from the maintain-predict repository.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from maintain_predict.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_POLICY,
    select_features_and_target,
)

RANDOM_STATE = 42
SAVE_SPLIT = True  # Set False to explore without saving a manifest.
DATA_PATH = PROJECT_ROOT / "data/ai4i2020.csv"
source_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
readings = pd.read_csv(DATA_PATH)
features, target = select_features_and_target(readings)

assert readings.index.is_unique
assert readings["UDI"].is_unique
print("Step 1: loaded", len(readings), "rows")
print("Model inputs:", list(FEATURE_COLUMNS))
print("Target:", TARGET_COLUMN, "with policy", TARGET_POLICY)


# %% Step 2: label the six strata, without splitting by tool wear
# Examples: L:0, L:1, M:0, M:1, H:0, H:1.
# These labels help allocate rows. They are not an extra model input.
strata = features["Type"].astype(str) + ":" + target.astype(str)
print("\nStep 2: source counts in each Type-by-target stratum")
print(pd.crosstab(features["Type"], target).to_string())


# %% Step 3: allocate 70% to training and reserve the remaining 30%
# test_size=0.30 means validation PLUS test here, not the final test set alone.
# stratify keeps approximately the same six-group proportions on each side.
train_rows, remaining_rows = train_test_split(
    readings,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=strata,
)
print("\nStep 3: training rows:", len(train_rows))
print("Remaining rows for validation + test:", len(remaining_rows))


# %% Step 4: divide the remaining 30% into two equal parts
# Half of 30% is 15% of the original data.
# .loc keeps the stratum labels aligned with the remaining rows and their order.
remaining_strata = strata.loc[remaining_rows.index]
validation_rows, test_rows = train_test_split(
    remaining_rows,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=remaining_strata,
)
partitions = {"train": train_rows, "validation": validation_rows, "test": test_rows}
print("\nStep 4: final partition sizes")
print(pd.Series({name: len(rows) for name, rows in partitions.items()}).to_string())


# %% Step 5: verify every source row occurs exactly once
# Keep the original indices. Resetting them before this check would hide overlap.
train_indices = set(train_rows.index)
validation_indices = set(validation_rows.index)
test_indices = set(test_rows.index)
assert train_indices.isdisjoint(validation_indices)
assert train_indices.isdisjoint(test_indices)
assert validation_indices.isdisjoint(test_indices)
assert train_indices | validation_indices | test_indices == set(readings.index)
print("\nStep 5: no overlapping rows; all source rows are accounted for")


# %% Step 6: inspect allocation and wear balance, without fitting anything
# Test labels appear only in allocation counts. No test predictions are made.
allocation_rows = []
for name, rows in partitions.items():
    partition_target = target.loc[rows.index]
    wear = rows["Tool wear [min]"]
    allocation_rows.append({
        "partition": name,
        "rows": len(rows),
        "positive evidence": int(partition_target.sum()),
        "positive rate (%)": partition_target.mean() * 100,
        "wear mean (min)": wear.mean(),
        "wear median (min)": wear.median(),
        "wear std (min)": wear.std(),
    })
allocation = pd.DataFrame(allocation_rows).set_index("partition")
print("\nStep 6: allocation and tool-wear balance")
print(allocation.round(3).to_string())

stratum_order = [f"{product_type}:{label}" for product_type in ["L", "M", "H"] for label in [0, 1]]
stratum_counts = pd.DataFrame({
    name: strata.loc[rows.index].value_counts().reindex(stratum_order, fill_value=0)
    for name, rows in partitions.items()
})
print("\nType-by-target counts after splitting")
print(stratum_counts.to_string())

# Prepare training and validation inputs for a LATER baseline exercise.
# Tool wear stays numeric. Encoding and scaling will be fitted on training only.
X_train, y_train = select_features_and_target(train_rows)
X_validation, y_validation = select_features_and_target(validation_rows)
print("\nTraining input shape:", X_train.shape)
print("Validation input shape:", X_validation.shape)
print("The raw test_rows partition is reserved for final evaluation.")


# %% Step 7: optionally save the exact row allocation for future runs
# This writes indices/UDI values, not a modified copy of the raw CSV.
# An existing, different manifest is rejected instead of silently overwritten.
manifest_path = PROJECT_ROOT / f"data/splits/type_target_seed{RANDOM_STATE}.json"
manifest = {
    "schema_version": 1,
    "source_sha256": source_sha256,
    "random_state": RANDOM_STATE,
    "stratify": ["Type", TARGET_COLUMN],
    "target_policy": TARGET_POLICY,
    "tool_wear_stratum": False,
    "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "partitions": {
        name: {
            "source_row_indices": rows.index.tolist(),
            "udi": rows["UDI"].tolist(),
        }
        for name, rows in partitions.items()
    },
}
if SAVE_SPLIT:
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("The source CSV changed. Rerun the exercise from step 1.")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("The existing split manifest differs; review the source, seed, and policy.")
        print("\nStep 7: existing manifest matches this allocation:", manifest_path)
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("\nStep 7: saved split manifest:", manifest_path)
else:
    print("\nStep 7: SAVE_SPLIT is False; no manifest was written")
