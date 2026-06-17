import os

import bentoml
import numpy as np
import pandas as pd
from bentoml.io import NumpyNdarray, PandasDataFrame
from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER

predictor = bentoml.sklearn.get("gbr:latest").to_runner()
svc = bentoml.Service("gbr", runners=[predictor])


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


@svc.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(df: pd.DataFrame) -> np.ndarray:
    model = get_model()

    missing_cols = set(FEATURE_ORDER) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

    extra_cols = set(df.columns) - set(FEATURE_ORDER)
    if extra_cols:
        df = df[FEATURE_ORDER]

    result = model.predict_dataframe(df)
    return np.array(result)
