from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

from car_pricing.feature_schema import (
    FeatureSchema,
    FIELD_DISPLAY_NAMES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    WORD_TO_NUM_CYLINDERS,
    DRIVEWheel_DISPLAY,
)


def _display_name(field_name: str, raw_class: str) -> str:
    if field_name == "drivewheel":
        return DRIVEWheel_DISPLAY.get(raw_class, raw_class.upper())
    if field_name == "cylindernumber":
        num = WORD_TO_NUM_CYLINDERS.get(raw_class.lower())
        if num is not None:
            return f"{num} cylinders"
        return raw_class
    return raw_class


class CarPricingClient:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        self._base_url = (base_url or os.environ.get("API_BASE_URL", "http://localhost:8000")).rstrip("/")
        self._timeout = int(timeout or os.environ.get("API_REQUEST_TIMEOUT", "10"))
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout(self) -> int:
        return self._timeout

    def _url(self, endpoint: str) -> str:
        return f"{self._base_url}/{endpoint.lstrip('/')}"

    def fetch_schema(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            resp = self._session.get(self._url("/schema"), timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
            missing = [k for k in required if k not in data]
            if missing:
                return None, f"Schema missing fields: {', '.join(missing)}"
            return data, None
        except requests.exceptions.ConnectionError:
            return None, "Unable to connect to the prediction service to fetch schema."
        except requests.exceptions.Timeout:
            return None, "Schema request timed out."
        except requests.exceptions.HTTPError as e:
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

    def predict(self, values: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
        try:
            resp = self._session.post(self._url("/predict"), json=values, timeout=self._timeout)
            resp.raise_for_status()
            body = resp.json()
            prediction = body.get("prediction")
            if prediction is None:
                return None, f"Unexpected response format from server: {body}"
            return float(prediction), None
        except requests.exceptions.ConnectionError:
            return None, "Unable to connect to the prediction service. Please check that the API server is running."
        except requests.exceptions.Timeout:
            return None, "The request to the prediction service timed out. Please try again later."
        except requests.exceptions.HTTPError as e:
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

    def health(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            resp = self._session.get(self._url("/health"), timeout=self._timeout)
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            return None, str(e)

    def metadata(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            resp = self._session.get(self._url("/metadata"), timeout=self._timeout)
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            return None, str(e)

    def status(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            resp = self._session.get(self._url("/status"), timeout=self._timeout)
            resp.raise_for_status()
            return resp.json(), None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def build_fallback_schema() -> Dict[str, Any]:
        schema = FeatureSchema.default()
        categorical_options: Dict[str, list] = {}
        for f in schema.categorical_features:
            if f == "drivewheel":
                categorical_options[f] = [
                    {"display": _display_name(f, cls), "form_value": cls}
                    for cls in ("fwd", "rwd", "4wd")
                ]
            elif f == "cylindernumber":
                categorical_options[f] = [
                    {"display": _display_name(f, cls), "form_value": cls}
                    for cls in sorted(WORD_TO_NUM_CYLINDERS.keys())
                ]
            else:
                categorical_options[f] = []
        return {
            "feature_order": list(schema.feature_order),
            "numeric_features": list(schema.numeric_features),
            "categorical_features": list(schema.categorical_features),
            "target_column": schema.target_column,
            "categorical_options": categorical_options,
        }
