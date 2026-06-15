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
from car_pricing.service_status import build_service_status


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


@service.api(input=JSON(), output=JSON())
def health(_) -> dict:
    try:
        model = get_model()
        model.schema.validate()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "model_loaded": False, "error": str(e)}


@service.api(input=JSON(), output=JSON())
def metadata(_) -> dict:
    model = get_model()
    return {
        "feature_order": model.feature_order,
        "numeric_features": model.numeric_features,
        "categorical_features": model.categorical_features,
        "target_column": model.target_column,
        "categorical_options": {
            f: model.categorical_options(f) for f in model.categorical_features
        },
        "model_mode": model.mode,
        "n_features": model.schema.n_features(),
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


@service.api(input=JSON(), output=JSON())
def status(_) -> dict:
    try:
        model = get_model()
    except Exception:
        model = None
    data_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "Data", "cars.csv")
    )
    svc_status = build_service_status(
        service_type="bentoml",
        model=model,
        data_csv_path=data_csv,
    )
    return svc_status.to_dict()


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
