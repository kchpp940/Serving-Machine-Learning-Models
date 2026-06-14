import enum
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import joblib

from models import CarPrediction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sklearn_gbr.pkl")
MODEL_NAME = "sklearn_gbr"
CURRENCY = "USD"
FAVICON_PATH = os.path.join(BASE_DIR, "favicon.png")


class ErrorKind(enum.Enum):
    MODEL_UNAVAILABLE = "model_unavailable"
    BAD_REQUEST = "bad_request"
    PREDICTION_FAILED = "prediction_failed"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    INTERNAL_ERROR = "internal_error"


@dataclass
class ServiceError:
    kind: ErrorKind
    detail: str


@dataclass
class PredictionResult:
    prediction: float
    currency: str
    model_name: str


@dataclass
class HealthResult:
    status: str
    model_available: bool
    model_error: Optional[str]


@dataclass
class ModelState:
    model: Optional[object] = None
    available: bool = False
    error_message: Optional[str] = None

    @classmethod
    def load(cls) -> "ModelState":
        try:
            if not os.path.exists(MODEL_PATH):
                return cls(
                    model=None,
                    available=False,
                    error_message=f"Model file not found at {MODEL_PATH}",
                )
            loaded = joblib.load(MODEL_PATH)
            return cls(model=loaded, available=True, error_message=None)
        except Exception as e:
            return cls(
                model=None, available=False, error_message=str(e)
            )


def build_features(data: CarPrediction) -> Union[np.ndarray, ServiceError]:
    try:
        return np.array(
            [
                [
                    data.enginesize,
                    data.curbweight,
                    data.horsepower,
                    data.highwaympg,
                    data.carwidth,
                    data.wheelbase,
                    data.drivewheel,
                    data.citympg,
                    data.boreratio,
                    data.cylindernumber,
                ]
            ],
            dtype=np.float64,
        )
    except (ValueError, TypeError) as e:
        return ServiceError(
            kind=ErrorKind.BAD_REQUEST,
            detail=f"Failed to construct feature array from input data: {str(e)}",
        )


def run_prediction(
    features: np.ndarray, model: object
) -> Union[float, ServiceError]:
    try:
        predictions = model.predict(features)
        return float(predictions[0])
    except (IndexError, ValueError, TypeError) as e:
        return ServiceError(
            kind=ErrorKind.PREDICTION_FAILED,
            detail=f"Failed to parse prediction result: {str(e)}",
        )
    except Exception as e:
        return ServiceError(
            kind=ErrorKind.PREDICTION_FAILED,
            detail=f"Model inference failed: {str(e)}",
        )


def predict(data: CarPrediction, state: ModelState) -> Union[PredictionResult, ServiceError]:
    if not state.available or state.model is None:
        return ServiceError(
            kind=ErrorKind.MODEL_UNAVAILABLE,
            detail=(
                "Prediction service is unavailable. "
                f"Model not loaded: {state.error_message or 'Unknown reason'}"
            ),
        )

    features_result = build_features(data)
    if isinstance(features_result, ServiceError):
        return features_result

    prediction_result = run_prediction(features_result, state.model)
    if isinstance(prediction_result, ServiceError):
        return prediction_result

    return PredictionResult(
        prediction=float(prediction_result),
        currency=CURRENCY,
        model_name=MODEL_NAME,
    )


def check_health(state: ModelState) -> HealthResult:
    if state.available:
        return HealthResult(
            status="ok",
            model_available=True,
            model_error=None,
        )
    return HealthResult(
        status="degraded",
        model_available=False,
        model_error=state.error_message,
    )


def format_validation_errors(errors: list) -> ServiceError:
    formatted = []
    for err in errors:
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "Unknown validation error")
        formatted.append(f"{loc}: {msg}" if loc else msg)
    detail = "; ".join(formatted) if formatted else "Invalid request input"
    return ServiceError(
        kind=ErrorKind.VALIDATION_ERROR,
        detail=f"Input validation failed: {detail}",
    )
