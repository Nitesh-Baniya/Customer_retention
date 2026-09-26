"""REST API service for customer retention predictions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from retention.settings import MODEL_REGISTRY_NAME, TRACKING_SERVER_URI

PRODUCTION_MODEL_URI = f"models:/{MODEL_REGISTRY_NAME}/Production"

_loaded_model = None
_loaded_model_version: str | None = None


class CustomerProfile(BaseModel):
    """Customer feature data matching training schema."""

    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class BatchPredictionRequest(BaseModel):
    """Request containing multiple customer profiles for scoring."""

    customers: list[CustomerProfile]


class RetentionPrediction(BaseModel):
    """Individual prediction result for a customer."""

    attrition_probability: float
    attrition_prediction: int
    attrition_label: str


class BatchPredictionResponse(BaseModel):
    """Response containing predictions for all requested customers."""

    model_name: str
    model_version: str | None
    predictions: list[RetentionPrediction]


def _retrieve_production_model() -> tuple[Any, str | None]:
    """Load the current Production model from MLflow registry."""
    mlflow.set_tracking_uri(TRACKING_SERVER_URI)
    model = mlflow.sklearn.load_model(PRODUCTION_MODEL_URI)
    client = MlflowClient()
    production_versions = client.get_latest_versions(
        MODEL_REGISTRY_NAME, stages=["Production"]
    )
    version = (
        str(production_versions[0].version) if production_versions else None
    )
    return model, version


@asynccontextmanager
async def model_lifespan(app: FastAPI):
    """Manage model loading during API startup/shutdown."""
    global _loaded_model, _loaded_model_version
    try:
        _loaded_model, _loaded_model_version = _retrieve_production_model()
        print(
            f"Loaded model from {PRODUCTION_MODEL_URI} "
            f"(version={_loaded_model_version})"
        )
    except Exception as exc:
        print(f"WARNING: Failed to load Production model: {exc}")
        _loaded_model, _loaded_model_version = None, None
    yield


prediction_api = FastAPI(
    title="Customer Retention Prediction API",
    description="ML-powered API for predicting customer attrition risk",
    version="1.0.0",
    lifespan=model_lifespan,
)


@prediction_api.get("/health")
def health_check() -> dict[str, Any]:
    """API health check endpoint."""
    if _loaded_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and promote a model first.",
        )
    return {
        "status": "healthy",
        "model_name": MODEL_REGISTRY_NAME,
        "model_uri": PRODUCTION_MODEL_URI,
        "model_version": _loaded_model_version,
    }


@prediction_api.post("/predict", response_model=BatchPredictionResponse)
def generate_predictions(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Generate attrition predictions for customer profiles."""
    if _loaded_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and promote a model first.",
        )
    if not request.customers:
        raise HTTPException(
            status_code=400, detail="customers list cannot be empty"
        )

    input_dataframe = pd.DataFrame(
        [profile.model_dump() for profile in request.customers]
    )
    try:
        probabilities = _loaded_model.predict_proba(input_dataframe)[:, 1]
        predictions = _loaded_model.predict(input_dataframe)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Prediction failed: {exc}"
        ) from exc

    prediction_results = [
        RetentionPrediction(
            attrition_probability=float(prob),
            attrition_prediction=int(pred),
            attrition_label="Yes" if int(pred) == 1 else "No",
        )
        for prob, pred in zip(probabilities, predictions)
    ]
    return BatchPredictionResponse(
        model_name=MODEL_REGISTRY_NAME,
        model_version=_loaded_model_version,
        predictions=prediction_results,
    )


app = prediction_api
