from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests as re

from car_pricing.prediction_protocol import (
    PredictionResult,
    InputFeatureValueItem,
    GlobalFeatureImportanceItem,
    ExplainResult,
    BatchRowResult,
    BatchPredictionResponse,
    build_fallback_schema,
    get_display_name_from_schema,
    get_default_values_from_schema,
    validate_values_from_schema,
    format_value_for_display,
)


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.environ.get("API_REQUEST_TIMEOUT", "10"))


@dataclass
class SchemaResponse:
    schema: Optional[dict] = None
    error: Optional[str] = None
    using_fallback: bool = False


@dataclass
class BatchTransportResponse:
    results: List[BatchRowResult] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    total_count: int = 0
    transport_error: Optional[str] = None


@dataclass
class SingleTransportResponse:
    result: Optional[PredictionResult] = None
    transport_error: Optional[str] = None


@dataclass
class ExplainTransportResponse:
    result: Optional[ExplainResult] = None
    transport_error: Optional[str] = None


def get_api_base_url() -> str:
    return API_BASE_URL


def get_display_name(field_name: str, schema: dict) -> str:
    return get_display_name_from_schema(field_name, schema)


def get_default_values(schema: dict) -> Dict[str, Any]:
    return get_default_values_from_schema(schema)


def validate_values(values: Dict[str, Any], schema: dict) -> Dict[str, str]:
    return validate_values_from_schema(values, schema)


def format_value(field_name: str, value: Any, schema: dict) -> str:
    return format_value_for_display(field_name, value, schema)


def fetch_schema() -> SchemaResponse:
    url = f"{API_BASE_URL.rstrip('/')}/schema"
    fallback = build_fallback_schema()
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
        missing = [k for k in required if k not in data]
        if missing:
            return SchemaResponse(
                schema=fallback,
                error=f"Schema missing fields: {', '.join(missing)}",
                using_fallback=True,
            )
        return SchemaResponse(schema=data, error=None, using_fallback=False)
    except re.exceptions.ConnectionError:
        return SchemaResponse(
            schema=fallback,
            error="Unable to connect to the prediction service to fetch schema.",
            using_fallback=True,
        )
    except re.exceptions.Timeout:
        return SchemaResponse(
            schema=fallback,
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
            schema=fallback,
            error=f"Server returned error when fetching schema: {detail or str(e)}",
            using_fallback=True,
        )
    except ValueError:
        return SchemaResponse(
            schema=fallback,
            error="Schema response was not valid JSON.",
            using_fallback=True,
        )
    except Exception as e:
        return SchemaResponse(
            schema=fallback,
            error=f"Unexpected error fetching schema: {str(e)}",
            using_fallback=True,
        )


def predict_single(values: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    resp = predict(values)
    if resp.transport_error:
        return None, resp.transport_error
    if resp.result and resp.result.error:
        return None, resp.result.error
    if resp.result and resp.result.prediction is not None:
        return resp.result.prediction, None
    return None, "Unexpected response format from server"


def predict(values: Dict[str, Any]) -> SingleTransportResponse:
    url = f"{API_BASE_URL.rstrip('/')}/predict"
    try:
        res = re.post(url, json=values, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        body = res.json()
        if "prediction" not in body:
            return SingleTransportResponse(
                transport_error=f"Unexpected response format from server: {body}"
            )
        result = PredictionResult(
            prediction=body.get("prediction"),
            error=body.get("error"),
            status=body.get("status", "ok"),
        )
        return SingleTransportResponse(result=result, transport_error=None)
    except re.exceptions.ConnectionError:
        return SingleTransportResponse(
            transport_error="Unable to connect to the prediction service. Please check that the API server is running.",
        )
    except re.exceptions.Timeout:
        return SingleTransportResponse(
            transport_error="The request to the prediction service timed out. Please try again later.",
        )
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = res.json().get("detail", "")
        except Exception:
            pass
        return SingleTransportResponse(
            transport_error=f"Server returned an error ({res.status_code}): {detail or str(e)}",
        )
    except ValueError:
        return SingleTransportResponse(
            transport_error="The server returned an invalid response. Please try again later.",
        )
    except Exception as e:
        return SingleTransportResponse(
            transport_error=f"An unexpected error occurred: {str(e)}",
        )


def explain(values: Dict[str, Any]) -> ExplainTransportResponse:
    url = f"{API_BASE_URL.rstrip('/')}/explain"
    try:
        res = re.post(url, json=values, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        body = res.json()
        if "prediction" not in body and "error" not in body:
            return ExplainTransportResponse(
                transport_error=f"Unexpected response format from server: {body}"
            )
        input_features = [
            InputFeatureValueItem(
                field_name=x.get("field_name", ""),
                display_name=x.get("display_name", ""),
                raw_value=x.get("raw_value"),
                encoded_value=x.get("encoded_value"),
            )
            for x in body.get("input_features", [])
        ]
        global_importance = [
            GlobalFeatureImportanceItem(
                field_name=x.get("field_name", ""),
                display_name=x.get("display_name", ""),
                importance=float(x.get("importance", 0.0)),
                rank=int(x.get("rank", 0)),
            )
            for x in body.get("global_importance", [])
        ]
        result = ExplainResult(
            prediction=body.get("prediction"),
            input_features=input_features,
            global_importance=global_importance,
            error=body.get("error"),
        )
        return ExplainTransportResponse(result=result, transport_error=None)
    except re.exceptions.ConnectionError:
        return ExplainTransportResponse(
            transport_error="Unable to connect to the prediction service. Please check that the API server is running.",
        )
    except re.exceptions.Timeout:
        return ExplainTransportResponse(
            transport_error="The explain request timed out. Please try again later.",
        )
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = res.json().get("detail", "")
        except Exception:
            pass
        return ExplainTransportResponse(
            transport_error=f"Server returned an error ({res.status_code}): {detail or str(e)}",
        )
    except ValueError:
        return ExplainTransportResponse(
            transport_error="The server returned an invalid response. Please try again later.",
        )
    except Exception as e:
        return ExplainTransportResponse(
            transport_error=f"An unexpected error occurred: {str(e)}",
        )


def predict_batch(rows: List[Dict[str, Any]], row_ids: Optional[List[str]] = None) -> BatchTransportResponse:
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
        return BatchTransportResponse(
            results=results,
            success_count=body.get("success_count", 0),
            error_count=body.get("error_count", 0),
            total_count=body.get("total_count", 0),
            transport_error=None,
        )
    except re.exceptions.ConnectionError:
        return BatchTransportResponse(
            results=[],
            transport_error="Unable to connect to the prediction service. Please check that the API server is running.",
        )
    except re.exceptions.Timeout:
        return BatchTransportResponse(
            results=[],
            transport_error="The batch prediction request timed out. Please try again later.",
        )
    except re.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = res.json().get("detail", "")
        except Exception:
            pass
        return BatchTransportResponse(
            results=[],
            transport_error=f"Server returned an error ({res.status_code}): {detail or str(e)}",
        )
    except ValueError:
        return BatchTransportResponse(
            results=[],
            transport_error="The server returned an invalid response. Please try again later.",
        )
    except Exception as e:
        return BatchTransportResponse(
            results=[],
            transport_error=f"An unexpected error occurred: {str(e)}",
        )


__all__ = [
    "API_BASE_URL",
    "SchemaResponse",
    "SingleTransportResponse",
    "ExplainTransportResponse",
    "BatchTransportResponse",
    "PredictionResult",
    "InputFeatureValueItem",
    "GlobalFeatureImportanceItem",
    "ExplainResult",
    "BatchRowResult",
    "BatchPredictionResponse",
    "get_api_base_url",
    "get_display_name",
    "get_default_values",
    "validate_values",
    "format_value",
    "fetch_schema",
    "predict",
    "predict_single",
    "explain",
    "predict_batch",
    "build_fallback_schema",
]
