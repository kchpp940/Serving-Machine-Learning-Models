import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import (
    FEATURE_ORDER,
    find_model_path,
)


_INTERFACE_FIELDS = list(FEATURE_ORDER)

_model: CarPriceModel = None


def _get_model() -> CarPriceModel:
    global _model
    if _model is None:
        local_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = find_model_path(local_dir=local_dir)
        _model = CarPriceModel.from_joblib(model_path)
        _model.validate_service(_INTERFACE_FIELDS)
    return _model


def predict_price(**kwargs):
    model = _get_model()
    return model.predict_raw(kwargs)
