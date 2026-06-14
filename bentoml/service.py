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
    feature_order = model.feature_order
    required_fields = set(feature_order)

    records = input_json.get("records", [])
    results = []
    valid_rows = []
    valid_indices = []

    for idx, record in enumerate(records):
        missing = required_fields - set(record.keys())
        if missing:
            results.append({
                "row_index": idx,
                "prediction": None,
                "currency": MODEL_CURRENCY,
                "model_name": MODEL_NAME,
                "error": f"缺少字段: {', '.join(sorted(missing))}",
            })
            continue

        row_errors = []
        for field in feature_order:
            try:
                model.encode_feature(field, record[field])
            except (ValueError, TypeError) as e:
                row_errors.append(f"{field}: {str(e)}")

        if row_errors:
            results.append({
                "row_index": idx,
                "prediction": None,
                "currency": MODEL_CURRENCY,
                "model_name": MODEL_NAME,
                "error": "; ".join(row_errors),
            })
            continue

        valid_rows.append({f: record[f] for f in feature_order})
        valid_indices.append(idx)

    if valid_rows:
        df = pd.DataFrame(valid_rows)
        predictions = model.predict_dataframe(df)
        for pos, (idx, pred) in enumerate(zip(valid_indices, predictions)):
            results.append({
                "row_index": idx,
                "prediction": float(pred),
                "currency": MODEL_CURRENCY,
                "model_name": MODEL_NAME,
                "error": None,
            })

    results.sort(key=lambda x: x["row_index"])
    valid_count = sum(1 for r in results if r["error"] is None)
    invalid_count = len(results) - valid_count

    return {
        "status": "ok",
        "total_records": len(records),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
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
