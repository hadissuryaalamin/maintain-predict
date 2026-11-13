"""FastAPI entry point for the prediction service."""

from fastapi import FastAPI

app = FastAPI(title="maintain-predict")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
