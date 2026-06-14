import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER, SchemaMismatchError


predictor = bentoml.sklearn.load_runner("gbr:latest")

service = bentoml.Service("gbr", runners=[predictor])

_INTERFACE_FIELDS = list(FEATURE_ORDER)

_model: CarPriceModel = None


def get_model() -> CarPriceModel:
    global _model
    if _model is None:
        raw_bundle = bentoml.sklearn.load_model("gbr:latest")
        _model = CarPriceModel.from_sklearn_object(raw_bundle)
        _model.validate_service(_INTERFACE_FIELDS)
    return _model


@service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:
    model = get_model()

    missing_cols = set(model.feature_order) - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入数据缺少列: {sorted(missing_cols)}")

    extra_cols = set(df.columns) - set(model.feature_order)
    if extra_cols:
        df = df[model.feature_order]

    result = model.predict_dataframe(df)
    return np.array(result)
