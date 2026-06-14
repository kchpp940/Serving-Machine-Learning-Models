import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic import BaseModel

from car_pricing.feature_schema import FEATURE_ORDER, CATEGORICAL_FEATURES


INTERFACE_FIELDS = list(FEATURE_ORDER)

_CATEGORICAL_SET = set(CATEGORICAL_FEATURES)


class CarPrediction(BaseModel):
    enginesize: float
    curbweight: float
    horsepower: float
    highwaympg: float
    carwidth: float
    wheelbase: float
    drivewheel: str
    citympg: float
    boreratio: float
    cylindernumber: str

    class Config:
        json_schema_extra = {
            "example": {
                "enginesize": 130,
                "curbweight": 2548,
                "horsepower": 111,
                "highwaympg": 27,
                "carwidth": 64.1,
                "wheelbase": 88.6,
                "drivewheel": "rwd",
                "citympg": 21,
                "boreratio": 3.47,
                "cylindernumber": "four",
            }
        }


assert list(CarPrediction.model_fields.keys()) == INTERFACE_FIELDS, (
    f"CarPrediction 字段 {list(CarPrediction.model_fields.keys())} "
    f"与 INTERFACE_FIELDS {INTERFACE_FIELDS} 不一致"
)
