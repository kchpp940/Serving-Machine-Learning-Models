import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi import HTTPException
from models import CarPrediction, PredictionResponse
import numpy as np

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER
from car_pricing.model_lineage import (
    ModelLineage,
    build_fastapi_status,
    compute_file_hash,
    verify_artifact_hash,
)


app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
    version="0.0.1",
    debug=True,
)

_model: CarPriceModel = None
_lineage: ModelLineage = None


def get_model() -> CarPriceModel:
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "sklearn_gbr.pkl")
        if not os.path.exists(model_path):
            raise RuntimeError(f"模型文件不存在: {model_path}")
        _model = CarPriceModel.from_joblib(model_path)
        _model.schema.validate()
    return _model


def get_lineage() -> ModelLineage:
    global _lineage
    if _lineage is None:
        model = get_model()
        model_path = os.path.join(os.path.dirname(__file__), "models", "sklearn_gbr.pkl")
        metadata_path = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            _lineage = ModelLineage(
                run_id=metadata.get("run_id", ""),
                experiment_id=metadata.get("experiment_id", ""),
                model_name=metadata.get("model_name", "sklearn_gbr"),
                model_type=metadata.get("model_type", ""),
                schema_version=metadata.get("schema_version", model.schema.schema_version()),
                data_version=metadata.get("data_version", ""),
                model_artifact_hash=metadata.get("model_artifact_hash", compute_file_hash(model_path)),
                metrics=dict(metadata.get("metrics", {})),
                params=dict(metadata.get("params", {})),
                parent_run_id=metadata.get("parent_run_id"),
                schema=model.schema,
            )
        else:
            model_artifact_hash = compute_file_hash(model_path)
            _lineage = ModelLineage(
                run_id="",
                experiment_id="",
                model_name="sklearn_gbr",
                model_type="GradientBoostingRegressor",
                schema_version=model.schema.schema_version(),
                data_version="",
                model_artifact_hash=model_artifact_hash,
                metrics={},
                params={"n_features": model.schema.n_features()},
                schema=model.schema,
            )
    return _lineage


@app.on_event("startup")
async def startup_event():
    try:
        model = get_model()
        lineage = get_lineage()
        print(f"Model loaded successfully. Mode: {model.mode}")
        print(f"Feature order: {model.feature_order}")
        print(f"Expected features: {model.schema.n_features()}")
        print(f"Model n_features_in_: {model.model.n_features_in_}")
        print(f"Model name: {lineage.model_name}")
        print(f"Model type: {lineage.model_type}")
        print(f"Schema version: {lineage.schema_version}")
        print(f"Data version: {lineage.data_version}")
        print(f"Model artifact hash: {lineage.model_artifact_hash}")
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


@app.get("/status")
async def get_status():
    lineage = get_lineage()
    return build_fastapi_status(lineage)


@app.get("/metadata")
async def get_metadata():
    lineage = get_lineage()
    return lineage.to_runtime_metadata()


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
