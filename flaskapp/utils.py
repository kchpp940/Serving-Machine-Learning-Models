import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import FEATURE_ORDER
from car_pricing.artifact_paths import ArtifactPaths


_model = None
_model_path: str = None


def _get_model():
    global _model, _model_path
    if _model is None:
        paths = ArtifactPaths.for_flaskapp(__file__)
        shared_models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "shared_models"
        )
        resolved_path = paths.find_existing_model_file(fallback_dirs=[shared_models_dir])
        if resolved_path is None:
            raise FileNotFoundError(
                f"模型文件不存在，已查找: {paths.model_file}, {os.path.join(shared_models_dir, paths.model_filename)}"
            )
        _model_path = resolved_path
        _model = CarPriceModel.from_joblib(resolved_path)
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
