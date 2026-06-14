import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON

import numpy as np
import pandas as pd
import json

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER


MODEL_NAME = "sklearn_gbr"
MODEL_CURRENCY = "USD"

predictor = bentoml.sklearn.load_runner("gbr:latest")

service = bentoml.Service("gbr", runners=[predictor])


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
def predict_batch(input_json) -> dict:
    model = get_model()
    records = input_json.get("records", [])

    batch_result = model.predict_records(records)

    results = []
    for item in batch_result.results:
        results.append({
            "row_index": item.row_index,
            "prediction": item.prediction,
            "currency": MODEL_CURRENCY,
            "model_name": MODEL_NAME,
            "error": item.error,
        })

    return {
        "status": "ok",
        "total_records": batch_result.total_records,
        "valid_count": batch_result.valid_count,
        "invalid_count": batch_result.invalid_count,
        "results": results,
    }


@service.api(input=JSON(), output=JSON())
def schema(_) -> dict:
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
