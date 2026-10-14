"""FastAPI prediction and monitoring service."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from mlops_app.config import MODEL_PATH
from mlops_app.data import load_dataset
from mlops_app.monitoring import get_monitoring_snapshot, validate_frame

MODEL = None
FEATURE_NAMES: list[str] = []
PREDICTION_COUNT = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, FEATURE_NAMES
    if Path(MODEL_PATH).exists():
        MODEL = joblib.load(MODEL_PATH)
        FEATURE_NAMES = list(load_dataset()[0].columns)
    yield


app = FastAPI(title="MLOps Lifecycle API", version="0.1.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[dict[str, float]]


def _require_model():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run `python -m mlops_app.training` first.")


def _predict(rows: list[dict[str, float]]):
    global PREDICTION_COUNT
    _require_model()
    try:
        frame = pd.DataFrame(rows)
        validate_frame(frame, FEATURE_NAMES)
        frame = frame[FEATURE_NAMES]
        predictions = MODEL.predict(frame).astype(int).tolist()
        probabilities = MODEL.predict_proba(frame)[:, 1].astype(float).tolist()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    PREDICTION_COUNT += len(rows)
    return [{"prediction": p, "probability_class_1": probability}
            for p, probability in zip(predictions, probabilities)]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "predictions_served": PREDICTION_COUNT}


@app.get("/model")
def model_info():
    _require_model()
    return {"model": "StandardScaler + RandomForestClassifier", "features": FEATURE_NAMES}


@app.post("/predict")
def predict(request: PredictionRequest):
    return _predict([request.features])[0]


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    if not request.instances:
        raise HTTPException(status_code=422, detail="instances must contain at least one row")
    return {"predictions": _predict(request.instances)}


@app.get("/monitoring")
def monitoring():
    _require_model()
    return get_monitoring_snapshot(MODEL, FEATURE_NAMES)
