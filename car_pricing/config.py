from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _project_root() -> str:
    return _PROJECT_ROOT


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw


@dataclass
class RuntimeConfig:
    api_host: str
    api_port: int
    model_dir: str
    model_filename: str
    model_metadata_filename: str
    model_status_filename: str
    data_dir: str
    data_csv_filename: str
    api_base_url: str
    request_timeout: int
    bentoml_model_tag: str

    @property
    def project_root(self) -> str:
        return _project_root()

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_filename)

    @property
    def model_metadata_path(self) -> str:
        return os.path.join(self.model_dir, self.model_metadata_filename)

    @property
    def model_status_path(self) -> str:
        return os.path.join(self.model_dir, self.model_status_filename)

    @property
    def data_csv_path(self) -> str:
        return os.path.join(self.data_dir, self.data_csv_filename)

    @property
    def bind_address(self) -> str:
        return f"{self.api_host}:{self.api_port}"

    def as_dict(self) -> dict:
        return {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "model_dir": self.model_dir,
            "model_path": self.model_path,
            "model_metadata_path": self.model_metadata_path,
            "model_status_path": self.model_status_path,
            "data_dir": self.data_dir,
            "data_csv_path": self.data_csv_path,
            "api_base_url": self.api_base_url,
            "request_timeout": self.request_timeout,
            "bentoml_model_tag": self.bentoml_model_tag,
            "project_root": self.project_root,
            "bind_address": self.bind_address,
        }


_DEFAULT_API_HOST = "0.0.0.0"
_DEFAULT_API_PORT = 8000
_DEFAULT_MODEL_FILENAME = "sklearn_gbr.pkl"
_DEFAULT_MODEL_METADATA_FILENAME = "model_metadata.json"
_DEFAULT_MODEL_STATUS_FILENAME = "model_status.json"
_DEFAULT_DATA_CSV_FILENAME = "cars.csv"
_DEFAULT_REQUEST_TIMEOUT = 10
_DEFAULT_BENTOML_MODEL_TAG = "gbr:latest"


def _resolve_model_dir(explicit: Optional[str] = None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    env_val = os.environ.get("MODEL_DIR")
    if env_val:
        return os.path.abspath(env_val)
    cwd_models = os.path.abspath(os.path.join(os.getcwd(), "models"))
    if os.path.isdir(cwd_models) and os.path.exists(
        os.path.join(cwd_models, _DEFAULT_MODEL_FILENAME)
    ):
        return cwd_models
    cwd_shared = os.path.abspath(os.path.join(os.getcwd(), "shared_models"))
    if os.path.isdir(cwd_shared) and os.path.exists(
        os.path.join(cwd_shared, _DEFAULT_MODEL_FILENAME)
    ):
        return cwd_shared
    root_shared = os.path.abspath(os.path.join(_project_root(), "shared_models"))
    if os.path.isdir(root_shared) and os.path.exists(
        os.path.join(root_shared, _DEFAULT_MODEL_FILENAME)
    ):
        return root_shared
    root_models = os.path.abspath(os.path.join(_project_root(), "models"))
    return root_models


def _resolve_data_dir(explicit: Optional[str] = None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    env_val = os.environ.get("DATA_DIR")
    if env_val:
        return os.path.abspath(env_val)
    cwd_data = os.path.abspath(os.path.join(os.getcwd(), "Data"))
    if os.path.isdir(cwd_data) and os.path.exists(
        os.path.join(cwd_data, _DEFAULT_DATA_CSV_FILENAME)
    ):
        return cwd_data
    root_data = os.path.abspath(os.path.join(_project_root(), "Data"))
    return root_data


def load_config(
    model_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> RuntimeConfig:
    api_host = _env_str("API_HOST", _DEFAULT_API_HOST)
    api_port = _env_int("API_PORT", _DEFAULT_API_PORT)
    resolved_model_dir = _resolve_model_dir(model_dir)
    resolved_data_dir = _resolve_data_dir(data_dir)
    model_filename = _env_str("MODEL_FILENAME", _DEFAULT_MODEL_FILENAME)
    model_metadata_filename = _env_str(
        "MODEL_METADATA_FILENAME", _DEFAULT_MODEL_METADATA_FILENAME
    )
    model_status_filename = _env_str(
        "MODEL_STATUS_FILENAME", _DEFAULT_MODEL_STATUS_FILENAME
    )
    data_csv_filename = _env_str("DATA_CSV_FILENAME", _DEFAULT_DATA_CSV_FILENAME)
    default_api_base = f"http://localhost:{api_port}"
    api_base_url = _env_str("API_BASE_URL", default_api_base)
    request_timeout = _env_int("REQUEST_TIMEOUT", _DEFAULT_REQUEST_TIMEOUT)
    bentoml_model_tag = _env_str("BENTOML_MODEL_TAG", _DEFAULT_BENTOML_MODEL_TAG)

    os.makedirs(resolved_model_dir, exist_ok=True)

    return RuntimeConfig(
        api_host=api_host,
        api_port=api_port,
        model_dir=resolved_model_dir,
        model_filename=model_filename,
        model_metadata_filename=model_metadata_filename,
        model_status_filename=model_status_filename,
        data_dir=resolved_data_dir,
        data_csv_filename=data_csv_filename,
        api_base_url=api_base_url,
        request_timeout=request_timeout,
        bentoml_model_tag=bentoml_model_tag,
    )


_CONFIG: Optional[RuntimeConfig] = None


def get_config() -> RuntimeConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def reload_config() -> RuntimeConfig:
    global _CONFIG
    _CONFIG = load_config()
    return _CONFIG


__all__ = [
    "RuntimeConfig",
    "load_config",
    "get_config",
    "reload_config",
]
