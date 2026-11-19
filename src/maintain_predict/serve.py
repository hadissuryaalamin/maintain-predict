"""Serve the selected model: startup loading, validated JSON, and prediction."""

from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from maintain_predict.inference import FailurePredictor

MODEL_REPO_ID = "hadissuryaalamin/maintain-predict"
MODEL_REVISION = "f5cb6c7890cffb30a1ff29644c217d3fcd4ea1a5"
MODEL_FILES = ["model.ubj", "preprocessor.skops", "model_contract.json"]


class PredictionInput(BaseModel):
    """Six raw inputs; targets and engineered inputs are not accepted."""

    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={
        "example": {
            "type": "L", "air_temp_k": 300, "process_temp_k": 310,
            "rpm": 1500, "torque_nm": 40, "tool_wear_min": 100,
        },
    })
    type: Literal["L", "M", "H"]
    air_temp_k: FiniteFloat = Field(gt=0, description="Air temperature in kelvin")
    process_temp_k: FiniteFloat = Field(gt=0, description="Process temperature in kelvin")
    rpm: FiniteFloat = Field(gt=0, description="Rotational speed in rpm")
    torque_nm: FiniteFloat = Field(ge=0, description="Torque in newton-metres")
    tool_wear_min: FiniteFloat = Field(ge=0, description="Accumulated tool wear in minutes")

    @model_validator(mode="after")
    def finite_interaction(self):
        if not math.isfinite(self.torque_nm * self.tool_wear_min):
            raise ValueError("Torque times tool wear must remain finite.")
        return self

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Type": [self.type], "Air temperature [K]": [self.air_temp_k],
            "Process temperature [K]": [self.process_temp_k],
            "Rotational speed [rpm]": [self.rpm], "Torque [Nm]": [self.torque_nm],
            "Tool wear [min]": [self.tool_wear_min],
        })


class PredictionOutput(BaseModel):
    failure_score: float = Field(ge=0, le=1)
    predicted_label: Literal[0, 1]
    threshold: Literal[0.5]
    target_policy: Literal["failure_evidence_or_v1"]


def load_predictor() -> FailurePredictor:
    """Load local artifacts when configured, otherwise fetch the pinned Hub release."""
    local_directory = os.environ.get("MAINTAIN_PREDICT_MODEL_DIR")
    if local_directory:
        directory = Path(local_directory).expanduser().resolve()
    else:
        directory = Path(snapshot_download(
            repo_id=MODEL_REPO_ID, revision=MODEL_REVISION,
            allow_patterns=MODEL_FILES, token=False,
        ))
    # Only our checked-in loader is executed; remote Python files are not downloaded.
    return FailurePredictor(directory)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Standard CPU serving: load once per process, then reuse across requests.
    application.state.predictor = load_predictor()
    try:
        yield
    finally:
        del application.state.predictor


app = FastAPI(
    title="maintain-predict", lifespan=lifespan,
    description=(
        "Predict recorded failure evidence under failure_evidence_or_v1. "
        "Synthetic-data demonstration; scores are not calibrated physical failure risks."
    ),
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Omit raw input/context: nonfinite JSON numbers cannot be echoed in valid JSON.
    details = [{key: error[key] for key in ["type", "loc", "msg"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    if getattr(request.app.state, "predictor", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, request: Request) -> PredictionOutput:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    result = predictor.predict(payload.as_frame()).iloc[0]
    return PredictionOutput(
        failure_score=float(result["failure_score"]),
        predicted_label=int(result["predicted_label"]), threshold=float(result["threshold"]),
        target_policy=predictor.contract["target_policy"],
    )
