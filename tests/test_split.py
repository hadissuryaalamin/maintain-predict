import pandas as pd
import pytest

from maintain_predict.data import split_readings
from maintain_predict.features import FAILURE_FLAG_COLUMNS, select_features_and_target


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
