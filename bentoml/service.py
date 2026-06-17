import os

import numpy as np
import pandas as pd

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER

try:
    import bentoml
    from bentoml.io import NumpyNdarray, PandasDataFrame

    try:
        predictor = bentoml.models.get("gbr:latest").to_runner()
        svc = bentoml.Service("gbr", runners=[predictor])
    except Exception:
        svc = bentoml.Service("gbr", runners=[])
except ImportError:
    bentoml = None
    svc = None


def _get_schema():
    if bentoml is None:
        raise RuntimeError("BentoML is not installed")
    raw_bundle = bentoml.picklable_model.load_model("gbr:latest")
    model = CarPriceModel.from_sklearn_object(raw_bundle)
    return model


_model = None


def get_model():
    global _model
    if _model is None:
        _model = _get_schema()
        _model.schema.validate()
    return _model


if svc is not None:
    @svc.api(input=PandasDataFrame(), output=NumpyNdarray())
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
