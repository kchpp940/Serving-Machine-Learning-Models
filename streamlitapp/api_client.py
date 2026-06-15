from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests as re

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))

DEFAULT_SCHEMA = {
    "feature_order": [
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "drivewheel", "citympg",
        "boreratio", "cylindernumber",
    ],
    "numeric_features": [
        "enginesize", "curbweight", "horsepower", "highwaympg",
        "carwidth", "wheelbase", "citympg", "boreratio",
    ],
    "categorical_features": ["drivewheel", "cylindernumber"],
    "target_column": "price",
    "categorical_options": {
        "drivewheel": [
            {"display": "Four Wheel Drive (4WD)", "form_value": "4wd", "model_code": 0},
            {"display": "Front Wheel Drive (FWD)", "form_value": "fwd", "model_code": 1},
            {"display": "Rear Wheel Drive (RWD)", "form_value": "rwd", "model_code": 2},
        ],
        "cylindernumber": [
            {"display": "2 cylinders", "form_value": "two", "model_code": 6},
            {"display": "3 cylinders", "form_value": "three", "model_code": 4},
            {"display": "4 cylinders", "form_value": "four", "model_code": 2},
            {"display": "5 cylinders", "form_value": "five", "model_code": 1},
            {"display": "6 cylinders", "form_value": "six", "model_code": 3},
            {"display": "8 cylinders", "form_value": "eight", "model_code": 0},
            {"display": "12 cylinders", "form_value": "twelve", "model_code": 5},
        ],
    },
}

FIELD_DISPLAY_NAMES = {
    "enginesize": "Engine Size",
    "curbweight": "Curb Weight",
    "horsepower": "Horsepower",
    "highwaympg": "Highway MPG",
    "carwidth": "Car Width",
    "wheelbase": "Wheel Base",
    "drivewheel": "Drive Wheel",
    "citympg": "City MPG",
    "boreratio": "Bore Ratio",
    "cylindernumber": "Number of Cylinders",
}


@dataclass
class PredictionResult:
    scenario_id: str
    name: str
    prediction: Optional[float] = None
    error: Optional[str] = None
    values: Dict[str, Any] = field(default_factory=dict)


def get_display_name(field: str) -> str:
    return FIELD_DISPLAY_NAMES.get(field, field.replace("_", " ").title())


def fetch_schema() -> Tuple[Optional[dict], Optional[str]]:
    url = f"{API_BASE_URL.rstrip('/')}/schema"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
        missing = [k for k in required if k not in data]
        if missing:
            return None, f"Schema missing fields: {', '.join(missing)}"
        return data, None
    except re.exceptions.ConnectionError:
        return None, "Unable to connect to the prediction service to fetch schema."
    except re.exceptions.Timeout:
        return None, "Schema request timed out."
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        return None, f"Server returned error when fetching schema: {detail or str(e)}"
    except ValueError:
        return None, "Schema response was not valid JSON."
    except Exception as e:
        return None, f"Unexpected error fetching schema: {str(e)}"


def predict_single(values: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    url = f"{API_BASE_URL.rstrip('/')}/predict"
    try:
        res = re.post(url, json=values, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        body = res.json()
        prediction = body.get("prediction")
        if prediction is None:
            return None, f"Unexpected response format from server: {body}"
        return float(prediction), None
    except re.exceptions.ConnectionError:
        return None, "Unable to connect to the prediction service. Please check that the API server is running."
    except re.exceptions.Timeout:
        return None, "The request to the prediction service timed out. Please try again later."
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = res.json().get("detail", "")
        except Exception:
            pass
        return None, f"Server returned an error ({res.status_code}): {detail or str(e)}"
    except ValueError:
        return None, "The server returned an invalid response. Please try again later."
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"


def predict_batch(scenarios: List[PredictionResult]) -> List[PredictionResult]:
    results: List[PredictionResult] = []
    for sc in scenarios:
        pred, err = predict_single(sc.values)
        results.append(PredictionResult(
            scenario_id=sc.scenario_id,
            name=sc.name,
            prediction=pred,
            error=err,
            values=sc.values,
        ))
    return results


def get_default_values(schema: dict) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    numeric_set = set(schema["numeric_features"])
    categorical_set = set(schema["categorical_features"])
    categorical_options = schema.get("categorical_options", {})

    for field in schema["feature_order"]:
        if field in numeric_set:
            defaults[field] = 0.0
        elif field in categorical_set:
            opts = categorical_options.get(field, [])
            if opts:
                defaults[field] = opts[0]["form_value"]
            else:
                defaults[field] = ""
        else:
            defaults[field] = ""
    return defaults


def validate_values(values: Dict[str, Any], schema: dict) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    numeric_set = set(schema["numeric_features"])
    categorical_set = set(schema["categorical_features"])
    categorical_options = schema.get("categorical_options", {})

    for field in schema["feature_order"]:
        if field not in values:
            errors[field] = "Missing value"
            continue
        val = values[field]
        if field in numeric_set:
            try:
                float(val)
            except (ValueError, TypeError):
                errors[field] = "Must be a number"
        elif field in categorical_set:
            opts = categorical_options.get(field, [])
            if opts:
                valid_values = {opt["form_value"] for opt in opts}
                if val not in valid_values:
                    errors[field] = f"Invalid option: {val}"
    return errors
