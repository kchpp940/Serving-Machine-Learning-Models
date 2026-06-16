import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bentoml
import bentoml.sklearn
from bentoml.io import NumpyNdarray, PandasDataFrame

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER
from car_pricing.artifact_paths import BENTOML_MODEL_TAG


predictor = bentoml.sklearn.load_runner(f"{BENTOML_MODEL_TAG}:latest")

service = bentoml.Service(BENTOML_MODEL_TAG, runners=[predictor])


def _get_schema():
    raw_bundle = bentoml.sklearn.load_model(f"{BENTOML_MODEL_TAG}:latest")
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
