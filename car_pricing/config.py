from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


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


@dataclass
class EnvVarMeta:
    """环境变量元数据（单一真相来源，包括所有部署目标的默认值）。"""
    name: str
    default: Any
    type: str
    description: str
    sensitive: bool = False
    expose_in_docs: bool = True
    required_in_deployment: bool = True
    category: str = "general"
    deployment_defaults: Optional[Dict[str, Any]] = None

    def value_for(self, target: str) -> Any:
        """获取指定部署目标的默认值。

        Args:
            target: 部署目标名 (docker/heroku/vercel/shell/procfile)
        Returns:
            该目标的默认值；未覆盖时返回全局 default
        """
        if self.deployment_defaults and target in self.deployment_defaults:
            return self.deployment_defaults[target]
        return self.default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "default": self.default,
            "type": self.type,
            "description": self.description,
            "sensitive": self.sensitive,
            "expose_in_docs": self.expose_in_docs,
            "required_in_deployment": self.required_in_deployment,
            "category": self.category,
            "deployment_defaults": self.deployment_defaults,
        }


def export_env_schema() -> Dict[str, EnvVarMeta]:
    """导出完整的环境变量 schema（单一真相来源）。

    所有配置文件、.env.example、测试脚本、部署文档均应从此 schema 生成，
    避免在多处维护重复的变量清单和默认值。
    """
    schema: Dict[str, EnvVarMeta] = {}

    _add = lambda meta: schema.update({meta.name: meta})

    _add(EnvVarMeta(
        name="API_HOST",
        default=_DEFAULT_API_HOST,
        type="str",
        description="API 监听地址",
        category="api",
    ))

    _add(EnvVarMeta(
        name="API_PORT",
        default=_DEFAULT_API_PORT,
        type="int",
        description="API 监听端口（Heroku 会自动注入 PORT 环境变量，启动时会映射到此变量）",
        category="api",
    ))

    _add(EnvVarMeta(
        name="MODEL_DIR",
        default="<auto-resolved>",
        type="str",
        description="模型文件目录。未设置时按优先级解析：./models → ./shared_models → <project_root>/shared_models → <project_root>/models",
        category="model",
        deployment_defaults={
            "docker": "/app/shared_models",
            "heroku": "/app/shared_models",
            "procfile": "/app/shared_models",
            "shell": "$PROJECT_ROOT/shared_models",
            "vercel": "./shared_models",
        },
    ))

    _add(EnvVarMeta(
        name="MODEL_FILENAME",
        default=_DEFAULT_MODEL_FILENAME,
        type="str",
        description="模型文件名",
        category="model",
    ))

    _add(EnvVarMeta(
        name="MODEL_METADATA_FILENAME",
        default=_DEFAULT_MODEL_METADATA_FILENAME,
        type="str",
        description="模型元数据文件名",
        category="model",
    ))

    _add(EnvVarMeta(
        name="MODEL_STATUS_FILENAME",
        default=_DEFAULT_MODEL_STATUS_FILENAME,
        type="str",
        description="模型状态文件名",
        category="model",
    ))

    _add(EnvVarMeta(
        name="DATA_DIR",
        default="<auto-resolved>",
        type="str",
        description="训练数据目录。未设置时按优先级解析：./Data → <project_root>/Data",
        category="data",
        deployment_defaults={
            "docker": "/app/Data",
            "heroku": "/app/Data",
            "procfile": "/app/Data",
            "shell": "$PROJECT_ROOT/Data",
            "vercel": "./Data",
        },
    ))

    _add(EnvVarMeta(
        name="DATA_CSV_FILENAME",
        default=_DEFAULT_DATA_CSV_FILENAME,
        type="str",
        description="训练数据 CSV 文件名",
        category="data",
    ))

    _add(EnvVarMeta(
        name="API_BASE_URL",
        default=f"http://localhost:{_DEFAULT_API_PORT}",
        type="str",
        description="API 基础 URL（供 Streamlit 等客户端调用）",
        category="client",
        deployment_defaults={
            "docker": "http://localhost:${API_PORT}",
            "procfile": "http://localhost:${API_PORT}",
            "shell": "http://localhost:$API_PORT",
        },
    ))

    _add(EnvVarMeta(
        name="REQUEST_TIMEOUT",
        default=_DEFAULT_REQUEST_TIMEOUT,
        type="int",
        description="请求超时时间（秒）",
        category="client",
    ))

    _add(EnvVarMeta(
        name="BENTOML_MODEL_TAG",
        default=_DEFAULT_BENTOML_MODEL_TAG,
        type="str",
        description="BentoML 模型标签",
        category="bentoml",
    ))

    return schema


def env_schema_to_list() -> List[EnvVarMeta]:
    """将 schema 转换为按 category 排序的列表。"""
    schema = export_env_schema()
    categories = ["api", "model", "data", "client", "bentoml", "general"]
    return sorted(
        schema.values(),
        key=lambda m: (categories.index(m.category) if m.category in categories else 99, m.name)
    )


def validate_env_coverage(env_vars: List[str]) -> List[str]:
    """校验给定的环境变量列表是否覆盖了所有 required_in_deployment 的变量。

    返回缺失的变量名列表。
    """
    schema = export_env_schema()
    required = {name for name, meta in schema.items() if meta.required_in_deployment}
    provided = set(env_vars)
    return sorted(required - provided)


__all__ = [
    "RuntimeConfig",
    "load_config",
    "get_config",
    "reload_config",
    "EnvVarMeta",
    "export_env_schema",
    "env_schema_to_list",
    "validate_env_coverage",
]
