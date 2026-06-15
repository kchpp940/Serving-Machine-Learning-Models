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
class BatchRowResult:
    row_id: Optional[str] = None
    prediction: Optional[float] = None
    error: Optional[str] = None
    field_errors: Optional[Dict[str, str]] = None


@dataclass
class BatchResponse:
    results: List[BatchRowResult] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    total_count: int = 0
    transport_error: Optional[str] = None


@dataclass
class SchemaResponse:
    schema: Optional[dict] = None
    error: Optional[str] = None
    using_fallback: bool = False


def get_display_name(field: str) -> str:
    return FIELD_DISPLAY_NAMES.get(field, field.replace("_", " ").title())


def get_api_base_url() -> str:
    return API_BASE_URL


def get_default_values(schema: dict) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    numeric_set = set(schema.get("numeric_features", []))
    categorical_set = set(schema.get("categorical_features", []))
    categorical_options = schema.get("categorical_options", {})

    for field in schema.get("feature_order", []):
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
    numeric_set = set(schema.get("numeric_features", []))
    categorical_set = set(schema.get("categorical_features", []))
    categorical_options = schema.get("categorical_options", {})

    for field in schema.get("feature_order", []):
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


def format_value_for_display(field: str, value, schema: dict) -> str:
    categorical_options = schema.get("categorical_options", {})
    opts = categorical_options.get(field, [])
    if opts:
        for opt in opts:
            if opt["form_value"] == value:
                return opt["display"]
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value)}"
        return f"{value:.2f}"
    return str(value)


def fetch_schema() -> SchemaResponse:
    url = f"{API_BASE_URL.rstrip('/')}/schema"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
        missing = [k for k in required if k not in data]
        if missing:
            return SchemaResponse(
                schema=DEFAULT_SCHEMA,
                error=f"Schema missing fields: {', '.join(missing)}",
                using_fallback=True,
            )
        return SchemaResponse(schema=data, error=None, using_fallback=False)
    except re.exceptions.ConnectionError:
        return SchemaResponse(
            schema=DEFAULT_SCHEMA,
            error="Unable to connect to the prediction service to fetch schema.",
            using_fallback=True,
        )
    except re.exceptions.Timeout:
        return SchemaResponse(
            schema=DEFAULT_SCHEMA,
            error="Schema request timed out.",
            using_fallback=True,
        )
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        return SchemaResponse(
            schema=DEFAULT_SCHEMA,
            error=f"Server returned error when fetching schema: {detail or str(e)}",
            using_fallback=True,
        )
    except ValueError:
        return SchemaResponse(
            schema=DEFAULT_SCHEMA,
            error="Schema response was not valid JSON.",
            using_fallback=True,
        )
    except Exception as e:
        return SchemaResponse(
            schema=DEFAULT_SCHEMA,
            error=f"Unexpected error fetching schema: {str(e)}",
            using_fallback=True,
        )


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


def predict_batch(rows: List[Dict[str, Any]], row_ids: Optional[List[str]] = None) -> BatchResponse:
    url = f"{API_BASE_URL.rstrip('/')}/predict_batch"
    try:
        payload = {"rows": rows}
        if row_ids:
            payload["row_ids"] = row_ids
        res = re.post(url, json=payload, timeout=REQUEST_TIMEOUT * max(1, len(rows)))
        res.raise_for_status()
        body = res.json()
        results_data = body.get("results", [])
        results: List[BatchRowResult] = []
        for r in results_data:
            results.append(BatchRowResult(
                row_id=r.get("row_id"),
                prediction=r.get("prediction"),
                error=r.get("error"),
                field_errors=r.get("field_errors"),
            ))
        return BatchResponse(
            results=results,
            success_count=body.get("success_count", 0),
            error_count=body.get("error_count", 0),
            total_count=body.get("total_count", 0),
            transport_error=None,
        )
    except re.exceptions.ConnectionError:
        return BatchResponse(
            results=[],
            transport_error="Unable to connect to the prediction service. Please check that the API server is running.",
        )
    except re.exceptions.Timeout:
        return BatchResponse(
            results=[],
            transport_error="The batch prediction request timed out. Please try again later.",
        )
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = res.json().get("detail", "")
        except Exception:
            pass
        return BatchResponse(
            results=[],
            transport_error=f"Server returned an error ({res.status_code}): {detail or str(e)}",
        )
    except ValueError:
        return BatchResponse(
            results=[],
            transport_error="The server returned an invalid response. Please try again later.",
        )
    except Exception as e:
        return BatchResponse(
            results=[],
            transport_error=f"An unexpected error occurred: {str(e)}",
        )
