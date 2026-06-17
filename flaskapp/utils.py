import os

import joblib
import numpy as np

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER


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


def predict_price(
    enginesize,
    curbweight,
    horsepower,
    highwaympg,
    carwidth,
    wheelbase,
    drivewheel,
    citympg,
    boreratio,
    cylindernumber,
):
    model = _get_model()
    values = {
        "enginesize": enginesize,
        "curbweight": curbweight,
        "horsepower": horsepower,
        "highwaympg": highwaympg,
        "carwidth": carwidth,
        "wheelbase": wheelbase,
        "drivewheel": drivewheel,
        "citympg": citympg,
        "boreratio": boreratio,
        "cylindernumber": cylindernumber,
    }
    return model.predict_raw(values)
