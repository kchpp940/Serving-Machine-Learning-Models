import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
import numpy as np

from car_pricing.model_runtime import CarPriceModel
from models import build_car_prediction_model, PredictionResponse


app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
    version="0.0.1",
    debug=True,
)

_model: CarPriceModel = None
_CarPrediction: type = None


def get_model() -> CarPriceModel:
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "sklearn_gbr.pkl")
        if not os.path.exists(model_path):
            raise RuntimeError(f"模型文件不存在: {model_path}")
        _model = CarPriceModel.from_joblib(model_path)
        _model.schema.validate()
    return _model


def get_prediction_model() -> type:
    global _CarPrediction
    if _CarPrediction is None:
        model = get_model()
        schema = model.to_schema_dict(include_encoders=False)
        _CarPrediction = build_car_prediction_model(schema)
    return _CarPrediction


@app.on_event("startup")
async def startup_event():
    try:
        model = get_model()
        schema = model.to_schema_dict(include_encoders=False)
        CarPred = get_prediction_model()
        print(f"Model loaded successfully. Mode: {model.mode}")
        print(f"Schema version: {schema['schema_version']}")
        print(f"Feature order: {schema['feature_order']}")
        print(f"Expected features: {len(schema['feature_order'])}")
        print(f"Model n_features_in_: {model.model.n_features_in_}")
        print(f"Pydantic fields: {list(CarPred.__fields__.keys())}")

        @app.post("/predict", response_model=PredictionResponse)
        async def predict(data: CarPred):
            try:
                m = get_model()
                predictions = m.predict_from_pydantic(data)
                value = float(predictions[0])
                return PredictionResponse(prediction=value)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

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
    return model.to_schema_dict(include_encoders=True)
