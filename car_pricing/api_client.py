from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests

from car_pricing.feature_schema import (
    FeatureSchema,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    FIELD_DISPLAY_NAMES,
)
from car_pricing.runtime_config import RuntimeConfig


class CarPricingClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        config: Optional[RuntimeConfig] = None,
    ):
        if config is not None:
            self._base_url = base_url or config.api_base_url
            self._timeout = timeout or config.request_timeout
        else:
            self._base_url = base_url or "http://localhost:8000"
            self._timeout = timeout or 10
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def timeout(self) -> int:
        return self._timeout

    def _url(self, endpoint: str) -> str:
        return f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"

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

    @staticmethod
    def build_fallback_schema() -> Dict[str, Any]:
        schema = FeatureSchema.default()
        return {
            "feature_order": list(schema.feature_order),
            "numeric_features": list(schema.numeric_features),
            "categorical_features": list(schema.categorical_features),
            "target_column": schema.target_column,
            "categorical_options": {
                f: schema.categorical_options(f) for f in schema.categorical_features
            } if schema.categorical_encoders else {},
        }
