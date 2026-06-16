import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER
from car_pricing.artifact_paths import ArtifactPaths
from car_pricing.config import RuntimeConfig


_model = None
_model_path: str = None


def _get_model():
    global _model, _model_path
    if _model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config = RuntimeConfig.from_env(base_dir=script_dir)
        paths = ArtifactPaths.from_config(config=config, base_dir=script_dir)
        paths.assert_model_file_exists()
        _model_path = paths.model_file
        _model = CarPriceModel.from_joblib(paths.model_file)
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
