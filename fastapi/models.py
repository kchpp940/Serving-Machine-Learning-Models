import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from car_pricing.feature_schema import (
    build_car_prediction_model,
    interface_fields_from_model,
)

CarPrediction = build_car_prediction_model()

INTERFACE_FIELDS = interface_fields_from_model(CarPrediction)
