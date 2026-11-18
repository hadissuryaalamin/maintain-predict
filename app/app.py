import spaces
import gradio as gr
import pandas as pd
from predict import FailurePredictor
from shap_plot import make_shap_plot

@spaces.GPU(duration=10)
def predict(type_, air, process, rpm, torque, wear):
    print("Prediction function started", flush=True)
    model = FailurePredictor()
    inputs = pd.DataFrame({
        "Type": [type_],
        "Air temperature [K]": [air],
        "Process temperature [K]": [process],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [wear],
    })

    # Torque × tool wear is calculated inside the predictor.
    print("Calling model", flush=True)
    result = model.predict(inputs).iloc[0]
    print("Prediction completed", flush=True)

    label = (
        "Failure evidence predicted (class 1)"
        if result["predicted_label"] == 1
        else "No failure evidence predicted (class 0)"
    )
    
    figure = make_shap_plot(model, inputs)
    return float(result["failure_score"]), label, figure


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["L", "M", "H"], value="L", label="Product type"),
        gr.Number(value=300, label="Air temperature [K]"),
        gr.Number(value=310, label="Process temperature [K]"),
        gr.Number(value=1500, label="Rotational speed [rpm]"),
        gr.Number(value=40, label="Torque [Nm]"),
        gr.Number(value=100, label="Tool wear [min]"),
    ],
    outputs=[
        gr.Number(label="Failure score"),
        gr.Textbox(label="Prediction"),
        gr.Plot(label="Why this prediction?"),
    ],
    title="Maintain Predict",
    description=(
        "A demonstration trained on the synthetic AI4I dataset. "
        "Decision threshold: 0.5. "
        "The score is not a calibrated estimate of physical failure risk."
    ),
)

if __name__ == "__main__":
    demo.launch(show_error=True)