import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import joblib
from fastapi.responses import JSONResponse

from models import CarPrediction, PredictionResponse, ErrorResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sklearn_gbr.pkl")
MODEL_NAME = "sklearn_gbr"
CURRENCY = "USD"
FAVICON_PATH = os.path.join(BASE_DIR, "favicon.png")

HTTP_400_BAD_REQUEST = 400
HTTP_404_NOT_FOUND = 404
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_503_SERVICE_UNAVAILABLE = 503


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


def build_features(data: CarPrediction) -> np.ndarray:
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


def run_prediction(
    features: np.ndarray, model: object
) -> Tuple[Optional[float], Optional[str]]:
    try:
        predictions = model.predict(features)
        predicted_value = float(predictions[0])
        return predicted_value, None
    except (IndexError, ValueError, TypeError) as e:
        return None, f"Failed to parse prediction result: {str(e)}"
    except Exception as e:
        return None, f"Model inference failed: {str(e)}"


def build_success_response(predicted_value: float) -> PredictionResponse:
    return PredictionResponse(
        prediction=predicted_value,
        currency=CURRENCY,
        model_name=MODEL_NAME,
    )


def build_error_response(detail: str, status_code: int = HTTP_500_INTERNAL_SERVER_ERROR) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
    )


def build_validation_error_response(errors: list) -> JSONResponse:
    formatted = []
    for err in errors:
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "Unknown validation error")
        formatted.append(f"{loc}: {msg}" if loc else msg)
    detail = "; ".join(formatted) if formatted else "Invalid request input"
    return build_error_response(
        f"Input validation failed: {detail}",
        HTTP_422_UNPROCESSABLE_ENTITY,
    )


def build_health_response(model_state: ModelState) -> Dict[str, Any]:
    if model_state.available:
        return {
            "status": "ok",
            "model_available": True,
            "model_error": None,
        }
    return {
        "status": "degraded",
        "model_available": False,
        "model_error": model_state.error_message,
    }
