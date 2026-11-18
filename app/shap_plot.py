"""Plot local Tree SHAP contributions for one submitted observation."""

import numpy as np
from matplotlib.figure import Figure
from scipy.special import expit
from xgboost import DMatrix


def make_shap_plot(model, inputs):
    """Use the loaded model's native Tree SHAP implementation; never refit."""
    if len(inputs) != 1:
        raise ValueError("This plot explains one submitted row at a time.")
    features = inputs.loc[:, model.contract["raw_feature_columns"]].copy()
    features["Torque wear [Nm min]"] = features["Torque [Nm]"] * features["Tool wear [min]"]
    encoded = model.preprocessor.transform(features)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    booster = model.classifier.get_booster()
    matrix = DMatrix(encoded)
    # The last value is the baseline; all earlier values explain encoded features.
    contributions = booster.predict(matrix, pred_contribs=True, approx_contribs=False)[0]
    score = float(model.predict(inputs)["failure_score"].iloc[0])
    if not np.isclose(expit(contributions.sum()), score, atol=1e-6):
        raise ValueError("SHAP contributions do not reconstruct the prediction score.")

    # Sum signed one-hot contributions to display Type as a single feature.
    grouped = {name: 0.0 for name in model.contract["model_feature_columns"]}
    names = model.preprocessor.get_feature_names_out()
    if len(names) != len(contributions) - 1:
        raise ValueError("Unexpected encoded feature count.")
    for name, value in zip(names, contributions[:-1]):
        key = "Type" if name.startswith("type__Type_") else name.removeprefix("sensors__")
        if key not in grouped:
            raise ValueError(f"Unexpected encoded feature: {name}")
        grouped[key] += float(value)

    ordered = sorted(grouped, key=lambda name: abs(grouped[name]))
    values = [grouped[name] for name in ordered]
    figure = Figure(figsize=(9, 5), layout="constrained")
    axis = figure.subplots()
    axis.barh(ordered, values, color=["#d35454" if value > 0 else "#347bb7" for value in values])
    axis.axvline(0, color="#666666", linewidth=1)
    axis.set_xlabel("SHAP contribution (log-odds): positive raises the class 1 score")
    axis.set_title("Feature contributions for this input")
    return figure
