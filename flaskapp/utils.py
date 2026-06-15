import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np

from car_pricing.model_runtime import CarPriceModel


_model = None


def _get_model():
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), "models", "sklearn_gbr.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "shared_models", "sklearn_gbr.pkl"
            )
        _model = CarPriceModel.from_joblib(model_path)
        _model.schema.validate()
    return _model


def predict_price(values: dict):
    model = _get_model()
    return model.predict_raw(values)
