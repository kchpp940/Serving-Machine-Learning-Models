from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 10


class ApiClientError(Exception):
    pass


class CarPricingApiClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)).rstrip("/")
        self.timeout = timeout or int(os.environ.get("API_REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT)))

    def _get(self, path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.ConnectionError:
            return None, "Unable to connect to the prediction service."
        except requests.exceptions.Timeout:
            return None, "Request timed out."
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass
            return None, f"Server returned error: {detail or str(e)}"
        except ValueError:
            return None, "Response was not valid JSON."
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"

    def _post(self, path: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.ConnectionError:
            return None, "Unable to connect to the prediction service."
        except requests.exceptions.Timeout:
            return None, "Request timed out."
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass
            return None, f"Server returned error ({resp.status_code}): {detail or str(e)}"
        except ValueError:
            return None, "Response was not valid JSON."
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"

    def fetch_schema(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        data, err = self._get("/schema")
        if err:
            return None, err
        required = ["feature_order", "numeric_features", "categorical_features", "categorical_options"]
        missing = [k for k in required if k not in data]
        if missing:
            return None, f"Schema missing fields: {', '.join(missing)}"
        return data, None

    def predict(self, values: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
        data, err = self._post("/predict", values)
        if err:
            return None, err
        prediction = data.get("prediction")
        if prediction is None:
            return None, f"Unexpected response format from server: {data}"
        return float(prediction), None

    def get_status(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return self._get("/status")

    def get_health(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return self._get("/health")

    def get_metadata(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return self._get("/metadata")
