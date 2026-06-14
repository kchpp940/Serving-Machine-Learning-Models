import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi import HTTPException
from models import CarPrediction, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse, BatchPredictionItem
import numpy as np

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER


MODEL_NAME = "sklearn_gbr"
MODEL_CURRENCY = "USD"

app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
    version="0.0.1",
    debug=True,
)

_model: CarPriceModel = None


def get_model() -> CarPriceModel:
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "sklearn_gbr.pkl")
        if not os.path.exists(model_path):
            raise RuntimeError(f"模型文件不存在: {model_path}")
        _model = CarPriceModel.from_joblib(model_path)
        _model.schema.validate()
    return _model


@app.on_event("startup")
async def startup_event():
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
    from fastapi.responses import FileResponse
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
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    try:
        model = get_model()
        batch_result = model.predict_records(request.records)

        results = []
        for item in batch_result.results:
            results.append(BatchPredictionItem(
                row_index=item.row_index,
                prediction=item.prediction,
                currency=MODEL_CURRENCY,
                model_name=MODEL_NAME,
                error=item.error,
            ))

        return BatchPredictionResponse(
            total_records=batch_result.total_records,
            valid_count=batch_result.valid_count,
            invalid_count=batch_result.invalid_count,
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")
