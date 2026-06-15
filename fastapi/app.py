import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi import HTTPException
from models import (
    CarPrediction,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
import numpy as np

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER


app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
    version="0.0.1",
    debug=True,
)

_model: CarPriceModel = None
_start_time: float = None


def get_model() -> CarPriceModel:
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "sklearn_gbr.pkl")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        _model = CarPriceModel.from_joblib(model_path)
        _model.schema.validate()
    return _model


@app.on_event("startup")
async def startup_event():
    global _start_time
    _start_time = time.time()
    try:
        model = get_model()
        print(f"Model loaded successfully. Mode: {model.mode}")
        print(f"Feature order: {model.feature_order}")
        print(f"Expected features: {model.schema.n_features()}")
        print(f"Model n_features_in_: {model.model.n_features_in_}")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        raise


@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Car Price Prediction API 🙌🏻
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
    return note


favicon_path = "favicon.png"


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


@app.get("/schema")
async def get_schema():
    model = get_model()
    return {
        "feature_order": model.feature_order,
        "numeric_features": model.numeric_features,
        "categorical_features": model.categorical_features,
        "target_column": model.target_column,
        "categorical_options": {
            f: model.categorical_options(f) for f in model.categorical_features
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CarPrediction):
    try:
        model = get_model()
        predictions = model.predict_from_pydantic(data)
        value = float(predictions[0])
        return PredictionResponse(prediction=value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    try:
        model = get_model()
        predictions = []
        for row in request.rows:
            pred = model.predict_from_pydantic(row)
            predictions.append(float(pred[0]))
        return BatchPredictionResponse(predictions=predictions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metadata")
async def get_metadata():
    try:
        model = get_model()
        return {
            "service_name": "Car Price Prediction API",
            "version": "0.0.1",
            "model_name": "sklearn_gbr",
            "model_mode": model.mode,
            "n_features": model.schema.n_features(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@app.get("/status")
async def get_status():
    global _start_time
    model_loaded = _model is not None
    uptime = None
    if _start_time is not None:
        uptime = time.time() - _start_time
    return {
        "status": "running" if model_loaded else "loading",
        "uptime_seconds": uptime,
        "model_loaded": model_loaded,
    }
