"""MLflow models-from-code entry point; not a training script."""

import mlflow
from maintain_predict.registry_model import MaintainPredictModel

mlflow.models.set_model(MaintainPredictModel())
