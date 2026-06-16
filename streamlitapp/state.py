from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests as re
import streamlit as st


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))

APP_TITLE = "Car Price Prediction Web App"

CURRENT_PAGE_KEY = "current_page"
SCHEMA_CACHE_KEY = "schema_cache"
SCHEMA_FETCHED_AT_KEY = "schema_fetched_at"
SCHEMA_ERROR_KEY = "schema_error"
PREDICTION_RESULTS_KEY = "prediction_results"
SCENARIO_LIST_KEY = "scenario_list"
SERVICE_STATUS_KEY = "service_status"
SERVICE_STATUS_REFRESHED_AT_KEY = "service_status_refreshed_at"
SERVICE_HEALTH_KEY = "service_health"
SERVICE_METADATA_KEY = "service_metadata"
ERROR_MESSAGE_KEY = "error_message"
SUCCESS_MESSAGE_KEY = "success_message"

PAGE_PREDICTION = "Prediction"
PAGE_SCENARIO_COMPARE = "Scenario Compare"
PAGE_SYSTEM_STATUS = "System Status"

ALL_PAGES = [PAGE_PREDICTION, PAGE_SCENARIO_COMPARE, PAGE_SYSTEM_STATUS]


@dataclass
class PredictionResult:
    scenario_name: str
    features: Dict[str, Any]
    prediction: float
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "features": self.features,
            "prediction": self.prediction,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionResult":
        return cls(
            scenario_name=data["scenario_name"],
            features=data["features"],
            prediction=data["prediction"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


def _base_url() -> str:
    return API_BASE_URL.rstrip("/")


def _ensure_key(key: str, default: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = default


def init_state() -> None:
    _ensure_key(CURRENT_PAGE_KEY, PAGE_PREDICTION)
    _ensure_key(SCHEMA_CACHE_KEY, None)
    _ensure_key(SCHEMA_FETCHED_AT_KEY, None)
    _ensure_key(SCHEMA_ERROR_KEY, None)
    _ensure_key(PREDICTION_RESULTS_KEY, [])
    _ensure_key(SCENARIO_LIST_KEY, [])
    _ensure_key(SERVICE_STATUS_KEY, None)
    _ensure_key(SERVICE_STATUS_REFRESHED_AT_KEY, None)
    _ensure_key(SERVICE_HEALTH_KEY, None)
    _ensure_key(SERVICE_METADATA_KEY, None)
    _ensure_key(ERROR_MESSAGE_KEY, None)
    _ensure_key(SUCCESS_MESSAGE_KEY, None)


def reset_all_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def get_current_page() -> str:
    return st.session_state[CURRENT_PAGE_KEY]


def set_current_page(page: str) -> None:
    if page not in ALL_PAGES:
        raise ValueError(f"Invalid page: {page}. Must be one of {ALL_PAGES}")
    st.session_state[CURRENT_PAGE_KEY] = page


def get_schema_cache() -> Optional[Dict[str, Any]]:
    return st.session_state[SCHEMA_CACHE_KEY]


def get_schema_fetched_at() -> Optional[datetime]:
    return st.session_state[SCHEMA_FETCHED_AT_KEY]


def get_schema_error() -> Optional[str]:
    return st.session_state[SCHEMA_ERROR_KEY]


def has_schema() -> bool:
    return st.session_state[SCHEMA_CACHE_KEY] is not None


def ensure_schema(force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    if not force_refresh and has_schema():
        return st.session_state[SCHEMA_CACHE_KEY]
    if not force_refresh and st.session_state[SCHEMA_ERROR_KEY] is not None:
        return None

    url = f"{_base_url()}/schema"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
        missing = [k for k in required if k not in data]
        if missing:
            st.session_state[SCHEMA_ERROR_KEY] = f"Schema missing fields: {', '.join(missing)}"
            return None
        st.session_state[SCHEMA_CACHE_KEY] = data
        st.session_state[SCHEMA_FETCHED_AT_KEY] = datetime.now()
        st.session_state[SCHEMA_ERROR_KEY] = None
        return data
    except re.exceptions.ConnectionError:
        st.session_state[SCHEMA_ERROR_KEY] = "Unable to connect to the prediction service to fetch schema."
    except re.exceptions.Timeout:
        st.session_state[SCHEMA_ERROR_KEY] = "Schema request timed out."
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        st.session_state[SCHEMA_ERROR_KEY] = f"Server returned error when fetching schema: {detail or str(e)}"
    except ValueError:
        st.session_state[SCHEMA_ERROR_KEY] = "Schema response was not valid JSON."
    except Exception as e:
        st.session_state[SCHEMA_ERROR_KEY] = f"Unexpected error fetching schema: {str(e)}"
    return None


def predict(values: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    url = f"{_base_url()}/predict"
    try:
        resp = re.post(url, json=values, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
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
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        return None, f"Server returned an error ({resp.status_code}): {detail or str(e)}"
    except ValueError:
        return None, "The server returned an invalid response. Please try again later."
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"


def get_prediction_results() -> List[PredictionResult]:
    raw = st.session_state[PREDICTION_RESULTS_KEY]
    return [PredictionResult.from_dict(r) if isinstance(r, dict) else r for r in raw]


def add_prediction_result(result: PredictionResult) -> None:
    st.session_state[PREDICTION_RESULTS_KEY].append(result.to_dict())


def clear_prediction_results() -> None:
    st.session_state[PREDICTION_RESULTS_KEY] = []


def remove_prediction_result(index: int) -> None:
    results = st.session_state[PREDICTION_RESULTS_KEY]
    if 0 <= index < len(results):
        results.pop(index)


def get_scenario_list() -> List[str]:
    return list(st.session_state[SCENARIO_LIST_KEY])


def add_scenario(name: str) -> None:
    if name and name not in st.session_state[SCENARIO_LIST_KEY]:
        st.session_state[SCENARIO_LIST_KEY].append(name)


def remove_scenario(name: str) -> None:
    if name in st.session_state[SCENARIO_LIST_KEY]:
        st.session_state[SCENARIO_LIST_KEY].remove(name)


def clear_scenarios() -> None:
    st.session_state[SCENARIO_LIST_KEY] = []


def get_service_status() -> Optional[Dict[str, Any]]:
    return st.session_state[SERVICE_STATUS_KEY]


def get_service_status_refreshed_at() -> Optional[datetime]:
    return st.session_state[SERVICE_STATUS_REFRESHED_AT_KEY]


def get_service_health() -> Optional[Dict[str, Any]]:
    return st.session_state[SERVICE_HEALTH_KEY]


def get_service_metadata() -> Optional[Dict[str, Any]]:
    return st.session_state[SERVICE_METADATA_KEY]


def refresh_service_status() -> Tuple[bool, Optional[str]]:
    base = _base_url()
    last_error: Optional[str] = None

    try:
        resp = re.get(f"{base}/status", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        st.session_state[SERVICE_STATUS_KEY] = resp.json()
        st.session_state[SERVICE_STATUS_REFRESHED_AT_KEY] = datetime.now()
    except Exception as e:
        last_error = str(e)

    try:
        resp = re.get(f"{base}/health", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        st.session_state[SERVICE_HEALTH_KEY] = resp.json()
    except Exception as e:
        if last_error is None:
            last_error = str(e)

    try:
        resp = re.get(f"{base}/metadata", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        st.session_state[SERVICE_METADATA_KEY] = resp.json()
    except Exception as e:
        if last_error is None:
            last_error = str(e)

    success = st.session_state[SERVICE_STATUS_KEY] is not None
    return success, last_error


def get_error_message() -> Optional[str]:
    return st.session_state[ERROR_MESSAGE_KEY]


def set_error_message(message: Optional[str]) -> None:
    st.session_state[ERROR_MESSAGE_KEY] = message


def clear_error_message() -> None:
    st.session_state[ERROR_MESSAGE_KEY] = None


def get_success_message() -> Optional[str]:
    return st.session_state[SUCCESS_MESSAGE_KEY]


def set_success_message(message: Optional[str]) -> None:
    st.session_state[SUCCESS_MESSAGE_KEY] = message


def clear_success_message() -> None:
    st.session_state[SUCCESS_MESSAGE_KEY] = None


def consume_messages() -> Tuple[Optional[str], Optional[str]]:
    error = get_error_message()
    success = get_success_message()
    clear_error_message()
    clear_success_message()
    return error, success
