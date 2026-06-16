from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from constants import DEFAULT_SCHEMA


CURRENT_PAGE_KEY = "current_page"
SCHEMA_CACHE_KEY = "schema_cache"
SCHEMA_FETCHED_AT_KEY = "schema_fetched_at"
SCHEMA_ERROR_KEY = "schema_error"
SCHEMA_USING_FALLBACK_KEY = "schema_using_fallback"
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


def _ensure_key(key: str, default: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = default


def init_state() -> None:
    _ensure_key(CURRENT_PAGE_KEY, PAGE_PREDICTION)
    _ensure_key(SCHEMA_CACHE_KEY, None)
    _ensure_key(SCHEMA_FETCHED_AT_KEY, None)
    _ensure_key(SCHEMA_ERROR_KEY, None)
    _ensure_key(SCHEMA_USING_FALLBACK_KEY, False)
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


def set_schema_cache(schema: Dict[str, Any]) -> None:
    st.session_state[SCHEMA_CACHE_KEY] = schema
    st.session_state[SCHEMA_FETCHED_AT_KEY] = datetime.now()
    st.session_state[SCHEMA_ERROR_KEY] = None
    st.session_state[SCHEMA_USING_FALLBACK_KEY] = False


def get_schema_fetched_at() -> Optional[datetime]:
    return st.session_state[SCHEMA_FETCHED_AT_KEY]


def get_schema_error() -> Optional[str]:
    return st.session_state[SCHEMA_ERROR_KEY]


def set_schema_error(error: str) -> None:
    st.session_state[SCHEMA_CACHE_KEY] = DEFAULT_SCHEMA
    st.session_state[SCHEMA_ERROR_KEY] = error
    st.session_state[SCHEMA_USING_FALLBACK_KEY] = True


def is_schema_using_fallback() -> bool:
    return st.session_state[SCHEMA_USING_FALLBACK_KEY]


def get_effective_schema() -> Dict[str, Any]:
    schema = st.session_state[SCHEMA_CACHE_KEY]
    return schema if schema is not None else DEFAULT_SCHEMA


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


def set_service_status(status: Dict[str, Any]) -> None:
    st.session_state[SERVICE_STATUS_KEY] = status
    st.session_state[SERVICE_STATUS_REFRESHED_AT_KEY] = datetime.now()


def get_service_status_refreshed_at() -> Optional[datetime]:
    return st.session_state[SERVICE_STATUS_REFRESHED_AT_KEY]


def get_service_health() -> Optional[Dict[str, Any]]:
    return st.session_state[SERVICE_HEALTH_KEY]


def set_service_health(health: Dict[str, Any]) -> None:
    st.session_state[SERVICE_HEALTH_KEY] = health


def get_service_metadata() -> Optional[Dict[str, Any]]:
    return st.session_state[SERVICE_METADATA_KEY]


def set_service_metadata(metadata: Dict[str, Any]) -> None:
    st.session_state[SERVICE_METADATA_KEY] = metadata


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


def consume_messages() -> tuple[Optional[str], Optional[str]]:
    error = get_error_message()
    success = get_success_message()
    clear_error_message()
    clear_success_message()
    return error, success
