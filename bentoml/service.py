import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON, Text

import numpy as np
import pandas as pd
import json

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER


predictor = bentoml.sklearn.load_runner("gbr:latest")

service = bentoml.Service("gbr", runners=[predictor])

_start_time = time.time()


def _get_schema():
    raw_bundle = bentoml.sklearn.load_model("gbr:latest")
    model = CarPriceModel.from_sklearn_object(raw_bundle)
    return model


_model = None


def get_model():
    global _model
    if _model is None:
        _model = _get_schema()
        _model.schema.validate()
    return _model


@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:
    model = get_model()

    missing_cols = set(FEATURE_ORDER) - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入数据缺少列: {missing_cols}")

    extra_cols = set(df.columns) - set(FEATURE_ORDER)
    if extra_cols:
        df = df[FEATURE_ORDER]

    result = model.predict_dataframe(df)
    return np.array(result)


@service.api(input=JSON(), output=JSON())
def predict_json(input_data: dict) -> dict:
    model = get_model()
    try:
        values = {f: input_data[f] for f in FEATURE_ORDER}
        pred = model.predict_raw(values)
        return {
            "prediction": float(pred[0]),
            "status": "ok",
        }
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"缺少必填字段: {missing}")
    except ValueError as e:
        raise


@service.api(input=JSON(), output=JSON())
def predict_batch(input_data: dict) -> dict:
    model = get_model()
    rows = input_data.get("rows", [])
    predictions = []
    for row in rows:
        values = {f: row[f] for f in FEATURE_ORDER}
        pred = model.predict_raw(values)
        predictions.append(float(pred[0]))
    return {
        "predictions": predictions,
        "status": "ok",
        "count": len(predictions),
    }


@service.api(input=Text(), output=JSON())
def schema(_: str = "") -> dict:
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


@service.api(input=Text(), output=JSON())
def metadata(_: str = "") -> dict:
    model = get_model()
    return {
        "service_name": "gbr",
        "version": "1.0.0",
        "model_name": "gbr",
        "model_mode": model.mode,
        "n_features": model.schema.n_features(),
    }


@service.api(input=Text(), output=JSON())
def status(_: str = "") -> dict:
    global _start_time
    model_loaded = _model is not None
    uptime = time.time() - _start_time if _start_time else None
    return {
        "status": "running" if model_loaded else "loading",
        "uptime_seconds": uptime,
        "model_loaded": model_loaded,
    }
