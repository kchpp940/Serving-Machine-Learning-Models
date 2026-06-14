import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel


_model_tag = "gbr:latest"
predictor = bentoml.sklearn.load_runner(_model_tag)

service = bentoml.Service("gbr", runners=[predictor])


def _load_model_and_metadata():
    bento_model = bentoml.sklearn.get(_model_tag)
    raw_bundle = bento_model.to_sklearn()
    model = CarPriceModel.from_sklearn_object(raw_bundle)
    metadata = bento_model.info.metadata
    labels = bento_model.info.labels
    return model, metadata, labels


_model = None
_model_metadata = None
_model_labels = None


def get_model():
    global _model, _model_metadata, _model_labels
    if _model is None:
        _model, _model_metadata, _model_labels = _load_model_and_metadata()
        _model.schema.validate()
    return _model


def get_model_metadata():
    if _model is None:
        get_model()
    return _model_metadata, _model_labels


@service.api(input=JSON(), output=JSON())
def metadata(_) -> dict:
    model = get_model()
    metadata, labels = get_model_metadata()

    result = {
        "service": "Car Price Prediction API",
        "model_mode": model.mode,
        "feature_schema": model.schema.to_dict(),
        "labels": labels or {},
        "training_metadata": metadata.get("training_metadata", {}) if metadata else {},
        "mlflow_run_id": metadata.get("mlflow_run_id") if metadata else None,
        "mlflow_params": metadata.get("mlflow_params", {}) if metadata else {},
        "mlflow_metrics": metadata.get("mlflow_metrics", {}) if metadata else {},
    }

    if metadata and "training_metadata" in metadata:
        tm = metadata["training_metadata"]
        result["data_version"] = tm.get("data_version")
        result["data_path"] = tm.get("data_path")
        result["metrics"] = tm.get("metrics", {})

    return result


@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:
    model = get_model()
    feature_order = model.feature_order

    missing_cols = set(feature_order) - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入数据缺少列: {missing_cols}")

    extra_cols = set(df.columns) - set(feature_order)
    if extra_cols:
        df = df[feature_order]

    result = model.predict_dataframe(df)
    return np.array(result)
