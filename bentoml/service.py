import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER
from car_pricing.model_lineage import (
    ModelLineage,
    build_bentoml_metadata,
    verify_artifact_hash,
)


MODEL_TAG = "gbr:latest"

predictor = bentoml.sklearn.load_runner(MODEL_TAG)

service = bentoml.Service("gbr", runners=[predictor])


def _load_model_and_metadata():
    bento_model = bentoml.sklearn.get(MODEL_TAG)
    raw_bundle = bentoml.sklearn.load_model(MODEL_TAG)
    model = CarPriceModel.from_sklearn_object(raw_bundle)

    metadata = dict(bento_model.info.metadata) if bento_model.info.metadata else {}
    labels = dict(bento_model.info.labels) if bento_model.info.labels else {}

    return model, metadata, labels


_model = None
_metadata = None
_labels = None
_lineage = None


def get_model():
    global _model, _metadata, _labels
    if _model is None:
        _model, _metadata, _labels = _load_model_and_metadata()
        _model.schema.validate()
    return _model


def get_metadata():
    global _metadata
    if _metadata is None:
        get_model()
    return _metadata


def get_labels():
    global _labels
    if _labels is None:
        get_model()
    return _labels


def get_lineage():
    global _lineage
    if _lineage is None:
        metadata = get_metadata()
        model = get_model()
        _lineage = ModelLineage(
            run_id=metadata.get("run_id", ""),
            experiment_id=metadata.get("experiment_id", ""),
            model_name=metadata.get("model_name", ""),
            model_type=metadata.get("model_type", ""),
            schema_version=metadata.get("schema_version", ""),
            data_version=metadata.get("data_version", ""),
            model_artifact_hash=metadata.get("model_artifact_hash", ""),
            metrics=dict(metadata.get("metrics", {})),
            params=dict(metadata.get("params", {})),
            parent_run_id=metadata.get("parent_run_id"),
            schema=model.schema,
        )
    return _lineage


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
def metadata(input_data: dict) -> dict:
    lineage = get_lineage()
    return build_bentoml_metadata(lineage)


@service.api(input=JSON(), output=JSON())
def schema_info(input_data: dict) -> dict:
    model = get_model()
    return model.schema.to_dict(include_encoders=True)


@service.api(input=JSON(), output=JSON())
def health(input_data: dict) -> dict:
    try:
        model = get_model()
        lineage = get_lineage()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_name": lineage.model_name,
            "model_type": lineage.model_type,
            "schema_version": lineage.schema_version,
            "data_version": lineage.data_version,
            "n_features": model.schema.n_features(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e),
        }
