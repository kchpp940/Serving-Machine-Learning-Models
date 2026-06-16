from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests as re

from constants import API_BASE_URL, REQUEST_TIMEOUT


def _base_url() -> str:
    return API_BASE_URL.rstrip("/")


def fetch_schema() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = f"{_base_url()}/schema"
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


def fetch_health() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = f"{_base_url()}/health"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except re.exceptions.ConnectionError:
        return None, "Unable to connect to the prediction service."
    except re.exceptions.Timeout:
        return None, "Health request timed out."
    except re.exceptions.HTTPError as e:
        return None, f"Server returned error when fetching health: {str(e)}"
    except ValueError:
        return None, "Health response was not valid JSON."
    except Exception as e:
        return None, f"Unexpected error fetching health: {str(e)}"


def fetch_status() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = f"{_base_url()}/status"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except re.exceptions.ConnectionError:
        return None, "Unable to connect to the prediction service."
    except re.exceptions.Timeout:
        return None, "Status request timed out."
    except re.exceptions.HTTPError as e:
        return None, f"Server returned error when fetching status: {str(e)}"
    except ValueError:
        return None, "Status response was not valid JSON."
    except Exception as e:
        return None, f"Unexpected error fetching status: {str(e)}"


def fetch_metadata() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = f"{_base_url()}/metadata"
    try:
        resp = re.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except re.exceptions.ConnectionError:
        return None, "Unable to connect to the prediction service."
    except re.exceptions.Timeout:
        return None, "Metadata request timed out."
    except re.exceptions.HTTPError as e:
        return None, f"Server returned error when fetching metadata: {str(e)}"
    except ValueError:
        return None, "Metadata response was not valid JSON."
    except Exception as e:
        return None, f"Unexpected error fetching metadata: {str(e)}"
