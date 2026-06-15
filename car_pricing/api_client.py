from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

try:
    import requests as re
except ImportError:  # pragma: no cover
    re = None


API_VERSION = "1.0.0"

API_BASE_SUGGESTIONS = {
    "development": "http://localhost:8000",
    "bentoml_local": "http://localhost:3000",
    "docker": "http://localhost:8080",
    "production": "https://your-api-domain.com",
}


def detect_api_base() -> str:
    env_base = os.environ.get("API_BASE_URL")
    if env_base:
        return env_base
    if os.environ.get("DOCKER_RUNTIME") or os.path.exists("/.dockerenv"):
        return API_BASE_SUGGESTIONS["docker"]
    return API_BASE_SUGGESTIONS["development"]


def _ensure_requests():
    if re is None:
        raise RuntimeError("需要先安装 requests 库")


def _get_json(url: str, label: str, timeout: int = 10) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    _ensure_requests()
    try:
        resp = re.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), None
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        return None, f"{label} returned HTTP {resp.status_code}: {detail or resp.text}"
    except re.exceptions.ConnectionError:
        return None, f"Cannot connect to {label} at {url}"
    except re.exceptions.Timeout:
        return None, f"{label} request timed out"
    except ValueError:
        return None, f"{label} returned invalid JSON"
    except Exception as e:
        return None, f"{label} error: {str(e)}"


def _post_json(url: str, label: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    _ensure_requests()
    try:
        resp = re.post(url, json=payload or {}, timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), None
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            pass
        return None, f"{label} returned HTTP {resp.status_code}: {detail or resp.text}"
    except re.exceptions.ConnectionError:
        return None, f"Cannot connect to {label} at {url}"
    except re.exceptions.Timeout:
        return None, f"{label} request timed out"
    except ValueError:
        return None, f"{label} returned invalid JSON"
    except Exception as e:
        return None, f"{label} error: {str(e)}"


class PredictionAPIClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 10):
        self.base_url = (base_url or detect_api_base()).rstrip("/")
        self.timeout = timeout

    def health(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return _get_json(f"{self.base_url}/health", "FastAPI /health", self.timeout)

    def metadata(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return _get_json(f"{self.base_url}/metadata", "FastAPI /metadata", self.timeout)

    def schema(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return _get_json(f"{self.base_url}/schema", "FastAPI /schema", self.timeout)

    def status(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return _get_json(f"{self.base_url}/status", "FastAPI /status", self.timeout)

    def predict(self, values: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        url = f"{self.base_url}/predict"
        _ensure_requests()
        try:
            resp = re.post(url, json=values, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json(), None
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                pass
            return None, f"Server returned an error ({resp.status_code}): {detail or str(resp)}"
        except re.exceptions.ConnectionError:
            return None, "Unable to connect to the prediction service. Please check that the API server is running."
        except re.exceptions.Timeout:
            return None, "The request to the prediction service timed out. Please try again later."
        except ValueError:
            return None, "The server returned an invalid response. Please try again later."
        except Exception as e:
            return None, f"An unexpected error occurred: {str(e)}"


class BentoMLAPIClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 10):
        default_base = os.environ.get("BENTOML_BASE_URL", "http://localhost:3000")
        self.base_url = (base_url or default_base).rstrip("/")
        self.timeout = timeout

    def health(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return _post_json(f"{self.base_url}/health", "BentoML /health", {}, self.timeout)

    def metadata(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return _post_json(f"{self.base_url}/metadata", "BentoML /metadata", {}, self.timeout)

    def schema(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return _post_json(f"{self.base_url}/schema", "BentoML /schema", {}, self.timeout)

    def status(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return _post_json(f"{self.base_url}/status", "BentoML /status", {}, self.timeout)
