import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi import HTTPException
from models import (
    CarPrediction,
    PredictionResponse,
    ExplainResponse,
)
import numpy as np
from typing import Union

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER


app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.

## Features

- **Predict**: Get car price predictions based on vehicle features
- **Explain**: Understand which features most influence the prediction
- **Schema**: Query available features and their options
""",
    version="0.1.0",
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
        print(f"Supports feature importance: {model.supports_feature_importance}")
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


@app.get("/schema", summary="Get feature schema and options")
async def get_schema():
    model = get_model()
    return {
        "feature_order": model.feature_order,
        "numeric_features": model.numeric_features,
        "categorical_features": model.categorical_features,
        "target_column": model.target_column,
        "feature_labels": {f: model.schema.label(f) for f in model.feature_order},
        "categorical_options": {
            f: model.categorical_options(f) for f in model.categorical_features
        },
        "model_name": model.model_name,
        "supports_feature_importance": model.supports_feature_importance,
    }


@app.get("/feature_importance", summary="Get global feature importance")
async def get_feature_importance():
    try:
        model = get_model()
        if not model.supports_feature_importance:
            raise HTTPException(
                status_code=501,
                detail=f"模型 {model.model_name} 不支持特征重要性",
            )
        importances = model.feature_importances()
        sorted_features = sorted(
            [
                {
                    "feature": f,
                    "label": model.schema.label(f),
                    "global_importance": score,
                    "global_importance_percent": round(score * 100, 2),
                }
                for f, score in importances.items()
            ],
            key=lambda x: x["global_importance"],
            reverse=True,
        )
        return {
            "model_name": model.model_name,
            "features": sorted_features,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征重要性失败: {str(e)}")


@app.post(
    "/predict",
    response_model=Union[PredictionResponse, ExplainResponse],
    summary="Predict car price",
    response_description="Predicted car price, optionally with feature explanation",
)
def predict(
    data: CarPrediction,
    explain: bool = Query(
        default=False,
        description="Whether to include feature importance explanation with the prediction",
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of top features to return when explain is enabled",
    ),
):
    try:
        model = get_model()
        predictions = model.predict_from_pydantic(data)
        value = float(predictions[0])

        if not explain:
            return PredictionResponse(prediction=value)

        if not model.supports_feature_importance:
            raise HTTPException(
                status_code=501,
                detail=f"模型 {model.model_name} 不支持特征重要性解释",
            )

        explanation = model.explain_from_pydantic(data, top_k=top_k)
        return ExplainResponse(
            prediction=value,
            model_name=explanation["model_name"],
            top_features=explanation["top_features"],
            feature_values=explanation["feature_values"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain a car price prediction",
    response_description="Prediction with global feature importance and input feature values",
)
def explain_prediction(
    data: CarPrediction,
    top_k: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of top features to highlight",
    ),
):
    try:
        model = get_model()
        if not model.supports_feature_importance:
            raise HTTPException(
                status_code=501,
                detail=f"模型 {model.model_name} 不支持特征重要性解释",
            )

        predictions = model.predict_from_pydantic(data)
        value = float(predictions[0])
        explanation = model.explain_from_pydantic(data, top_k=top_k)

        return ExplainResponse(
            prediction=value,
            model_name=explanation["model_name"],
            top_features=explanation["top_features"],
            feature_values=explanation["feature_values"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解释失败: {str(e)}")
