import hashlib
import json
from unittest.mock import patch

import pandas as pd
import pytest

from maintain_predict.data import load_saved_splits, split_readings
from maintain_predict.features import FAILURE_FLAG_COLUMNS, TARGET_COLUMN, TARGET_POLICY, select_features_and_target


@pytest.fixture
def synthetic_readings():
    """Small artificial rows, independent of the user's uploaded dataset."""
    rows = []
    for product_type in ["L", "M", "H"]:
        for label in [0, 1]:
            for offset in range(20):
                number = len(rows)
                flags = dict.fromkeys(FAILURE_FLAG_COLUMNS, 0)
                # Half of the positives exist only through RNF, so the test
                # distinguishes derived-target stratification from source labels.
                original = int(label == 1 and offset % 2 == 0)
                flags["RNF"] = int(label == 1 and original == 0)
                rows.append({
                    "UDI": number + 1, "Product ID": f"{product_type}{number:05d}",
                    "Type": product_type, "Air temperature [K]": 300 + offset / 10,
                    "Process temperature [K]": 310 + offset / 10,
                    "Rotational speed [rpm]": 1400 + number,
                    "Torque [Nm]": 30 + number / 10, "Tool wear [min]": number * 2,
                    "Machine failure": original, **flags,
                })
    return pd.DataFrame(rows, index=pd.Index(range(1000, 1840, 7), name="source_row"))


def test_split_has_no_overlap_and_preserves_all_source_rows(synthetic_readings):
    before = synthetic_readings.copy(deep=True)
    splits = split_readings(synthetic_readings)
    groups = splits.as_dict()
    assert {name: len(rows) for name, rows in groups.items()} == {
        "train": 84, "validation": 18, "test": 18,
    }
    index_sets = [set(rows.index) for rows in groups.values()]
    assert index_sets[0].isdisjoint(index_sets[1])
    assert index_sets[0].isdisjoint(index_sets[2])
    assert index_sets[1].isdisjoint(index_sets[2])
    assert set.union(*index_sets) == set(synthetic_readings.index)
    assert sum(len(group) for group in index_sets) == len(synthetic_readings)
    for rows in groups.values():
        pd.testing.assert_frame_equal(rows, before.loc[rows.index])
    pd.testing.assert_frame_equal(synthetic_readings, before)


def test_split_preserves_all_six_type_by_derived_target_strata(synthetic_readings):
    splits = split_readings(synthetic_readings)
    for name, expected_count in [("train", 14), ("validation", 3), ("test", 3)]:
        rows = splits.as_dict()[name]
        features, target = select_features_and_target(rows)
        counts = (features["Type"].astype(str) + ":" + target.astype(str)).value_counts()
        assert len(counts) == 6
        assert counts.eq(expected_count).all()


def test_split_is_reproducible_for_a_fixed_seed(synthetic_readings):
    first = split_readings(synthetic_readings, random_state=42)
    repeated = split_readings(synthetic_readings, random_state=42)
    different = split_readings(synthetic_readings, random_state=43)
    for name, rows in first.as_dict().items():
        pd.testing.assert_frame_equal(rows, repeated.as_dict()[name])
    assert not first.train.index.equals(different.train.index)


def test_split_rejects_ambiguous_source_indices(synthetic_readings):
    duplicate_index = synthetic_readings.copy()
    duplicate_index.index = [0] * len(duplicate_index)
    with pytest.raises(ValueError, match="indices must be unique"):
        split_readings(duplicate_index)


def test_split_explains_when_joint_strata_are_too_small(synthetic_readings):
    tiny = synthetic_readings.groupby(["Type", "Machine failure"]).head(1)
    with pytest.raises(ValueError, match="each joint stratum"):
        split_readings(tiny)


@pytest.fixture
def saved_allocation(tmp_path, synthetic_readings):
    source_path = tmp_path / "readings.csv"
    synthetic_readings.to_csv(source_path, index=False)
    # Match the source CSV's parsed numeric values and zero-based row indices.
    source = pd.read_csv(source_path)
    expected = split_readings(source)
    manifest = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "random_state": 42,
        "target_policy": TARGET_POLICY,
        "stratify": ["Type", TARGET_COLUMN],
        "tool_wear_stratum": False,
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "partitions": {
            name: {"source_row_indices": part.index.tolist(), "udi": part["UDI"].tolist()}
            for name, part in expected.as_dict().items()
        },
    }
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source_path, manifest_path, expected


def test_saved_split_loader_reuses_indices_without_resplitting(saved_allocation):
    source_path, manifest_path, expected = saved_allocation
    before_source, before_manifest = source_path.read_bytes(), manifest_path.read_bytes()
    with patch("maintain_predict.data.train_test_split", side_effect=AssertionError("Must not resplit")):
        actual = load_saved_splits(source_path, manifest_path)
    for name, part in expected.as_dict().items():
        pd.testing.assert_frame_equal(actual.as_dict()[name], part)
    assert source_path.read_bytes() == before_source
    assert manifest_path.read_bytes() == before_manifest


def test_saved_split_loader_rejects_a_changed_csv(saved_allocation):
    source_path, manifest_path, _ = saved_allocation
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        load_saved_splits(source_path, manifest_path)


@pytest.mark.parametrize(
    ("key", "value"), [("target_policy", "different_policy"), ("tool_wear_stratum", True)]
)
def test_saved_split_loader_rejects_another_protocol(saved_allocation, key, value):
    source_path, manifest_path, _ = saved_allocation
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="agreed split and target policy"):
        load_saved_splits(source_path, manifest_path)


def test_saved_split_loader_rejects_cross_partition_overlap(saved_allocation):
    source_path, manifest_path, _ = saved_allocation
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, validation = [manifest["partitions"][name] for name in ["train", "validation"]]
    validation["source_row_indices"][0] = train["source_row_indices"][0]
    validation["udi"][0] = train["udi"][0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        load_saved_splits(source_path, manifest_path)


def test_saved_split_loader_rejects_a_wrong_record_identity(saved_allocation):
    source_path, manifest_path, _ = saved_allocation
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partitions"]["train"]["udi"][0] = -1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="UDI values"):
        load_saved_splits(source_path, manifest_path)


def test_saved_split_loader_rejects_missing_rows(saved_allocation):
    source_path, manifest_path, _ = saved_allocation
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partitions"]["test"]["source_row_indices"].pop()
    manifest["partitions"]["test"]["udi"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="cover every"):
        load_saved_splits(source_path, manifest_path)


def test_saved_split_loader_rejects_wrong_partition_sizes(saved_allocation):
    source_path, manifest_path, _ = saved_allocation
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, validation = [manifest["partitions"][name] for name in ["train", "validation"]]
    validation["source_row_indices"].append(train["source_row_indices"].pop())
    validation["udi"].append(train["udi"].pop())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="partition sizes"):
        load_saved_splits(source_path, manifest_path)
