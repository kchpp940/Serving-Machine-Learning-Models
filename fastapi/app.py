import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi import HTTPException
from models import (
    CarPrediction,
    PredictionResponsePydantic,
    BatchPredictionRequest,
    BatchPredictionResponsePydantic,
    ExplainResponsePydantic,
)
import numpy as np
from typing import Dict, Any, List

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER
from car_pricing.prediction_protocol import (
    PredictionResult,
    ExplainResult,
    InputFeatureValueItem,
    GlobalFeatureImportanceItem,
    BatchRowResult,
    BatchPredictionResponse,
)


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
    return model.schema.to_api_dict()


def _build_prediction_result(model: CarPriceModel, values: Dict[str, Any]) -> PredictionResult:
    try:
        prediction = model.predict_raw(values)
        return PredictionResult(
            prediction=float(prediction[0]),
            error=None,
            status="ok",
        )
    except ValueError as e:
        return PredictionResult(
            prediction=None,
            error=str(e),
            status="error",
        )
    except Exception as e:
        return PredictionResult(
            prediction=None,
            error=f"Prediction failed: {str(e)}",
            status="error",
        )


@app.post("/predict", response_model=PredictionResponsePydantic)
def predict(data: CarPrediction):
    try:
        model = get_model()
        values = data.dict()
        proto = _build_prediction_result(model, values)
        if proto.error:
            raise HTTPException(status_code=400, detail=proto.error)
        return PredictionResponsePydantic.from_protocol(proto)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


def _build_input_features(model: CarPriceModel, values: Dict[str, Any]) -> List[InputFeatureValueItem]:
    schema = model.schema
    items: List[InputFeatureValueItem] = []
    for f in schema.feature_order:
        raw = values.get(f)
        encoded: float | None = None
        try:
            encoded = schema.encode_feature(f, raw)
        except Exception:
            encoded = None
        items.append(InputFeatureValueItem(
            field_name=f,
            display_name=schema.get_display_name(f),
            raw_value=raw,
            encoded_value=encoded,
        ))
    return items


def _build_global_importance(model: CarPriceModel) -> List[GlobalFeatureImportanceItem]:
    schema = model.schema
    items: List[GlobalFeatureImportanceItem] = []
    estimator = getattr(model.model, "feature_importances_", None)
    if estimator is None:
        return items
    importances = list(estimator)
    pairs = [(schema.feature_order[i], float(importances[i])) for i in range(min(len(schema.feature_order), len(importances)))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    for rank, (fname, imp) in enumerate(pairs, start=1):
        items.append(GlobalFeatureImportanceItem(
            field_name=fname,
            display_name=schema.get_display_name(fname),
            importance=imp,
            rank=rank,
        ))
    return items


@app.post("/explain", response_model=ExplainResponsePydantic)
def explain(data: CarPrediction):
    try:
        model = get_model()
        values = data.dict()
        pred_proto = _build_prediction_result(model, values)
        if pred_proto.error:
            raise HTTPException(status_code=400, detail=pred_proto.error)

        input_features = _build_input_features(model, values)
        global_importance = _build_global_importance(model)

        proto = ExplainResult(
            prediction=pred_proto.prediction,
            input_features=input_features,
            global_importance=global_importance,
            error=None,
        )
        return ExplainResponsePydantic.from_protocol(proto)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解释失败: {str(e)}")


def _validate_and_predict_row(values: Dict[str, Any], model: CarPriceModel) -> BatchRowResult:
    schema = model.schema
    field_errors: Dict[str, str] = {}
    numeric_set = set(schema.numeric_features)
    categorical_set = set(schema.categorical_features)
    categorical_options = {f: model.categorical_options(f) for f in schema.categorical_features}

    for f in schema.feature_order:
        if f not in values:
            field_errors[f] = "Missing value"
            continue
        val = values[f]
        if f in numeric_set:
            try:
                float(val)
            except (ValueError, TypeError):
                field_errors[f] = "Must be a number"
        elif f in categorical_set:
            opts = categorical_options.get(f, [])
            if opts:
                valid_values = {opt["form_value"] for opt in opts}
                if val not in valid_values:
                    field_errors[f] = f"Invalid option: {val}"

    if field_errors:
        return BatchRowResult(
            prediction=None,
            error="Invalid field values",
            field_errors=field_errors,
        )

    proto = _build_prediction_result(model, values)
    return BatchRowResult(
        prediction=proto.prediction,
        error=proto.error,
        field_errors=None,
    )


@app.post("/predict_batch", response_model=BatchPredictionResponsePydantic)
def predict_batch(request: BatchPredictionRequest):
    model = get_model()
    results: List[BatchRowResult] = []

    for i, row in enumerate(request.rows):
        row_id = request.row_ids[i] if request.row_ids and i < len(request.row_ids) else str(i)
        result = _validate_and_predict_row(row, model)
        result.row_id = row_id
        results.append(result)

    success_count = sum(1 for r in results if r.prediction is not None)
    error_count = len(results) - success_count

    proto = BatchPredictionResponse(
        results=results,
        success_count=success_count,
        error_count=error_count,
        total_count=len(results),
    )
    return BatchPredictionResponsePydantic.from_protocol(proto)
