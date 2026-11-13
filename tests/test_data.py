from maintain_predict.data import load_readings
from maintain_predict.features import FEATURE_COLUMNS, select_features_and_target


def test_only_prediction_features_are_selected(tmp_path):
    csv = tmp_path / "ai4i2020.csv"
    csv.write_text(
        "UDI,Product ID,Type,Air temperature [K],Process temperature [K],"
        "Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,"
        "TWF,HDF,PWF,OSF,RNF\n"
        "1,M00001,M,298.1,308.6,1551,42.8,0,1,0,1,0,0,0\n",
        encoding="utf-8",
    )

    features, target = select_features_and_target(load_readings(csv))

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert {
        "UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"
    }.isdisjoint(features.columns)
    assert target.tolist() == [1]
