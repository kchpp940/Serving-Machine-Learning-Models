import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER
from car_pricing.config import get_config

CONFIG = get_config()

try:
    import bentoml
    import bentoml.sklearn
    from bentoml.io import NumpyNdarray, PandasDataFrame

    _BENTOML_AVAILABLE = True
except ImportError:
    _BENTOML_AVAILABLE = False
    bentoml = None
    bentoml_sklearn = None
    NumpyNdarray = None
    PandasDataFrame = None


if _BENTOML_AVAILABLE:
    predictor = bentoml.sklearn.load_runner(CONFIG.bentoml_model_tag)
    service = bentoml.Service("gbr", runners=[predictor])
else:
    predictor = None
    service = None


def _get_schema():
    if not _BENTOML_AVAILABLE:
        raise RuntimeError("BentoML is not installed")
    raw_bundle = bentoml.sklearn.load_model(CONFIG.bentoml_model_tag)
    model = CarPriceModel.from_sklearn_object(raw_bundle)
    return model


_model = None


def get_model():
    global _model
    if _model is None:
        _model = _get_schema()
        _model.schema.validate()
    return _model


if _BENTOML_AVAILABLE:
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
else:
    def predict(df: pd.DataFrame) -> np.ndarray:
        raise RuntimeError("BentoML is not installed")
