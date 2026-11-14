import pandas as pd
import pytest

from maintain_predict.data import load_readings
from maintain_predict.features import (
    FAILURE_FLAG_COLUMNS,
    FEATURE_COLUMNS,
    SOURCE_TARGET_COLUMN,
    TARGET_COLUMN,
    derive_failure_target,
    select_features_and_target,
)


def test_only_prediction_features_are_selected(tmp_path):
    csv = tmp_path / "ai4i2020.csv"
    csv.write_text(
        "UDI,Product ID,Type,Air temperature [K],Process temperature [K],"
        "Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,"
        "TWF,HDF,PWF,OSF,RNF\n"
        "1,M00001,M,298.1,308.6,1551,42.8,0,1,0,1,0,0,0\n"
        "2,L00002,L,298.2,308.7,1408,46.3,3,0,0,0,0,0,1\n"
        "3,H00003,H,298.3,308.8,1498,49.4,5,1,0,0,0,0,0\n"
        "4,L00004,L,298.1,308.6,1550,40.0,8,0,0,0,0,0,0\n",
        encoding="utf-8",
    )

    features, target = select_features_and_target(load_readings(csv))

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert {
        "UDI", "Product ID", SOURCE_TARGET_COLUMN, TARGET_COLUMN, *FAILURE_FLAG_COLUMNS
    }.isdisjoint(features.columns)
    assert target.tolist() == [1, 1, 1, 0]
    assert target.name == TARGET_COLUMN


@pytest.mark.parametrize(
    ("original", "positive_flags", "expected"),
    [
        (0, (), 0),
        (1, (), 1),
        *[(0, (flag,), 1) for flag in FAILURE_FLAG_COLUMNS],
        (0, ("HDF", "PWF", "OSF"), 1),
        (1, ("TWF", "RNF"), 1),
    ],
)
def test_failure_evidence_policy_preserves_any_recorded_positive(
    original, positive_flags, expected
):
    readings = pd.DataFrame(
        [{
            SOURCE_TARGET_COLUMN: original,
            **{flag: int(flag in positive_flags) for flag in FAILURE_FLAG_COLUMNS},
        }],
        index=pd.Index([42], name="source_row"),
    )
    original_readings = readings.copy(deep=True)

    target = derive_failure_target(readings)

    assert target.tolist() == [expected]
    assert target.name == TARGET_COLUMN
    assert target.dtype == "int64"
    pd.testing.assert_index_equal(target.index, readings.index)
    pd.testing.assert_frame_equal(readings, original_readings)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [(SOURCE_TARGET_COLUMN, None), ("RNF", 2), ("TWF", "unknown")],
)
def test_failure_evidence_policy_rejects_unknown_or_nonbinary_labels(column, invalid_value):
    row = {SOURCE_TARGET_COLUMN: 0, **dict.fromkeys(FAILURE_FLAG_COLUMNS, 0)}
    row[column] = invalid_value

    with pytest.raises(ValueError, match="nonmissing binary"):
        derive_failure_target(pd.DataFrame([row]))


def test_failure_evidence_policy_requires_all_source_label_columns():
    readings = pd.DataFrame([{SOURCE_TARGET_COLUMN: 0}])

    with pytest.raises(KeyError):
        derive_failure_target(readings)
