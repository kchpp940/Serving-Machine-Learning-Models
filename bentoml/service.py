import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel


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
    schema = model.to_schema_dict(include_encoders=False)

    feature_order = schema["feature_order"]
    missing_cols = set(feature_order) - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入数据缺少列: {missing_cols}")

    extra_cols = set(df.columns) - set(feature_order)
    if extra_cols:
        df = df[feature_order]

    result = model.predict_dataframe(df)
    return np.array(result)
