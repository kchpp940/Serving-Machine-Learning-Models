from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


API_HOST_DEFAULT = "0.0.0.0"
API_PORT_DEFAULT = 8000
MODEL_FILENAME_DEFAULT = "sklearn_gbr.pkl"
MODEL_METADATA_FILENAME_DEFAULT = "model_metadata.json"
MODEL_STATUS_FILENAME_DEFAULT = "model_status.json"
SCHEMA_SNAPSHOT_FILENAME_DEFAULT = "schema_snapshot.json"
LINEAGE_FILENAME_DEFAULT = "model_lineage.json"
DATA_CSV_FILENAME_DEFAULT = "cars.csv"
API_BASE_URL_DEFAULT = "http://localhost:8000"
REQUEST_TIMEOUT_DEFAULT = 10
BENTOML_MODEL_TAG_DEFAULT = "gbr:latest"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

AUTO_RESOLVED_MARKER = "<auto-resolved>"

_DEPLOYMENT_COLUMNS: List[str] = [
    "Schema本地默认",
    "Docker/Heroku",
    "Shell 脚本",
    "Vercel",
]


@dataclass
class EnvVarMeta:
    name: str
    default: Any
    type: str
    sensitive: bool
    required: bool
    description: str
    group: str
    deployment_defaults: Dict[str, Any] = field(default_factory=dict)
    py_type: Any = str

    def value_for(self, deployment: str) -> Any:
        if deployment in self.deployment_defaults:
            return self.deployment_defaults[deployment]
        return self.default

    def to_schema_dict(self) -> Dict[str, Any]:
        schema_default = self.value_for("Schema本地默认")
        return {
            "name": self.name,
            "default": schema_default,
            "type": self.type,
            "sensitive": self.sensitive,
            "required": self.required,
            "description": self.description,
            "group": self.group,
        }


_ENV_VAR_META_LIST: List[EnvVarMeta] = []


def _register_env_var(meta: EnvVarMeta) -> EnvVarMeta:
    if "Schema本地默认" not in meta.deployment_defaults:
        if meta.default == "" and meta.name in ("MODEL_DIR", "DATA_DIR"):
            meta.deployment_defaults["Schema本地默认"] = AUTO_RESOLVED_MARKER
        else:
            meta.deployment_defaults["Schema本地默认"] = meta.default
    _ENV_VAR_META_LIST.append(meta)
    return meta


def _build_env_var_registry() -> List[EnvVarMeta]:
    if _ENV_VAR_META_LIST:
        return _ENV_VAR_META_LIST

    _register_env_var(EnvVarMeta(
        name="API_HOST",
        default=API_HOST_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="API 监听地址",
        group="API server configuration",
        py_type=str,
    ))
    _register_env_var(EnvVarMeta(
        name="API_PORT",
        default=API_PORT_DEFAULT,
        type="int",
        sensitive=False,
        required=True,
        description="API 监听端口（Heroku 会自动注入 PORT 环境变量，启动时会映射到此变量）",
        group="API server configuration",
        py_type=int,
    ))
    _register_env_var(EnvVarMeta(
        name="MODEL_DIR",
        default="",
        type="str",
        sensitive=False,
        required=True,
        description="模型文件目录。未设置时按优先级解析：./models → ./shared_models → <project_root>/shared_models → <project_root>/models",
        group="Model file locations",
        py_type=str,
        deployment_defaults={
            "Docker/Heroku": "/app/shared_models",
            "Shell 脚本": "$PROJECT_ROOT/shared_models",
            "Vercel": AUTO_RESOLVED_MARKER,
        },
    ))
    _register_env_var(EnvVarMeta(
        name="MODEL_FILENAME",
        default=MODEL_FILENAME_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="模型文件名",
        group="Model file locations",
        py_type=str,
    ))
    _register_env_var(EnvVarMeta(
        name="MODEL_METADATA_FILENAME",
        default=MODEL_METADATA_FILENAME_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="模型元数据文件名",
        group="Model file locations",
        py_type=str,
    ))
    _register_env_var(EnvVarMeta(
        name="MODEL_STATUS_FILENAME",
        default=MODEL_STATUS_FILENAME_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="模型状态文件名",
        group="Model file locations",
        py_type=str,
    ))
    _register_env_var(EnvVarMeta(
        name="SCHEMA_SNAPSHOT_FILENAME",
        default=SCHEMA_SNAPSHOT_FILENAME_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="Schema snapshot 文件名（训练时由 lineage.persist() 写入完整特征/编码器快照）",
        group="Model file locations",
        py_type=str,
    ))
    _register_env_var(EnvVarMeta(
        name="LINEAGE_FILENAME",
        default=LINEAGE_FILENAME_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="Model lineage 文件名（训练时由 lineage.persist() 写入完整血缘记录）",
        group="Model file locations",
        py_type=str,
    ))
    _register_env_var(EnvVarMeta(
        name="DATA_CSV_FILENAME",
        default=DATA_CSV_FILENAME_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="训练数据 CSV 文件名",
        group="Training data locations",
        py_type=str,
    ))
    _register_env_var(EnvVarMeta(
        name="DATA_DIR",
        default="",
        type="str",
        sensitive=False,
        required=True,
        description="训练数据目录。未设置时按优先级解析：./Data → <project_root>/Data",
        group="Training data locations",
        py_type=str,
        deployment_defaults={
            "Docker/Heroku": "/app/Data",
            "Shell 脚本": "$PROJECT_ROOT/Data",
            "Vercel": AUTO_RESOLVED_MARKER,
        },
    ))
    _register_env_var(EnvVarMeta(
        name="API_BASE_URL",
        default=API_BASE_URL_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="API 基础 URL（供 Streamlit 等客户端调用）",
        group="Client configuration",
        py_type=str,
        deployment_defaults={
            "Docker/Heroku": "http://localhost:${API_PORT}",
            "Shell 脚本": "http://localhost:$API_PORT",
        },
    ))
    _register_env_var(EnvVarMeta(
        name="REQUEST_TIMEOUT",
        default=REQUEST_TIMEOUT_DEFAULT,
        type="int",
        sensitive=False,
        required=True,
        description="请求超时时间（秒）",
        group="Client configuration",
        py_type=int,
    ))
    _register_env_var(EnvVarMeta(
        name="BENTOML_MODEL_TAG",
        default=BENTOML_MODEL_TAG_DEFAULT,
        type="str",
        sensitive=False,
        required=True,
        description="BentoML 模型标签",
        group="BentoML configuration",
        py_type=str,
    ))

    return _ENV_VAR_META_LIST


def get_env_var_registry() -> List[EnvVarMeta]:
    return _build_env_var_registry()


def find_env_var_meta(name: str) -> Optional[EnvVarMeta]:
    for meta in _build_env_var_registry():
        if meta.name == name:
            return meta
    return None


@dataclass
class RuntimeConfig:
    api_host: str = API_HOST_DEFAULT
    api_port: int = API_PORT_DEFAULT
    model_dir: str = ""
    model_filename: str = MODEL_FILENAME_DEFAULT
    model_metadata_filename: str = MODEL_METADATA_FILENAME_DEFAULT
    model_status_filename: str = MODEL_STATUS_FILENAME_DEFAULT
    schema_snapshot_filename: str = SCHEMA_SNAPSHOT_FILENAME_DEFAULT
    lineage_filename: str = LINEAGE_FILENAME_DEFAULT
    data_dir: str = ""
    data_csv_filename: str = DATA_CSV_FILENAME_DEFAULT
    api_base_url: str = API_BASE_URL_DEFAULT
    request_timeout: int = REQUEST_TIMEOUT_DEFAULT
    bentoml_model_tag: str = BENTOML_MODEL_TAG_DEFAULT

    def resolve_model_dir(self, base_dir: Optional[str] = None) -> str:
        if self.model_dir:
            return os.path.abspath(self.model_dir)

        search_base = base_dir or os.getcwd()
        candidates = [
            os.path.join(search_base, "models"),
            os.path.join(search_base, "shared_models"),
            os.path.join(PROJECT_ROOT, "shared_models"),
            os.path.join(PROJECT_ROOT, "models"),
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.isfile(
                os.path.join(c, self.model_filename)
            ):
                return os.path.abspath(c)

        return os.path.abspath(candidates[0])

    def resolve_data_dir(self, base_dir: Optional[str] = None) -> str:
        if self.data_dir:
            return os.path.abspath(self.data_dir)

        search_base = base_dir or os.getcwd()
        candidates = [
            os.path.join(search_base, "Data"),
            os.path.join(PROJECT_ROOT, "Data"),
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.isfile(
                os.path.join(c, self.data_csv_filename)
            ):
                return os.path.abspath(c)

        return os.path.abspath(candidates[0])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls, base_dir: Optional[str] = None) -> "RuntimeConfig":
        registry = _build_env_var_registry()
        meta_by_name = {m.name: m for m in registry}

        def _env(name: str, default: Any) -> Any:
            val = os.environ.get(name)
            if val is None or val == "":
                return default
            meta = meta_by_name.get(name)
            py_type = meta.py_type if meta else str
            if py_type is int:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default
            if py_type is float:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default
            if py_type is bool:
                s = val.strip().lower()
                if s in ("1", "true", "yes", "on"):
                    return True
                if s in ("0", "false", "no", "off"):
                    return False
                return default
            return val

        config = cls(
            api_host=_env("API_HOST", API_HOST_DEFAULT),
            api_port=_env("API_PORT", API_PORT_DEFAULT),
            model_dir=_env("MODEL_DIR", ""),
            model_filename=_env("MODEL_FILENAME", MODEL_FILENAME_DEFAULT),
            model_metadata_filename=_env("MODEL_METADATA_FILENAME", MODEL_METADATA_FILENAME_DEFAULT),
            model_status_filename=_env("MODEL_STATUS_FILENAME", MODEL_STATUS_FILENAME_DEFAULT),
            schema_snapshot_filename=_env("SCHEMA_SNAPSHOT_FILENAME", SCHEMA_SNAPSHOT_FILENAME_DEFAULT),
            lineage_filename=_env("LINEAGE_FILENAME", LINEAGE_FILENAME_DEFAULT),
            data_dir=_env("DATA_DIR", ""),
            data_csv_filename=_env("DATA_CSV_FILENAME", DATA_CSV_FILENAME_DEFAULT),
            api_base_url=_env("API_BASE_URL", API_BASE_URL_DEFAULT),
            request_timeout=_env("REQUEST_TIMEOUT", REQUEST_TIMEOUT_DEFAULT),
            bentoml_model_tag=_env("BENTOML_MODEL_TAG", BENTOML_MODEL_TAG_DEFAULT),
        )

        if "PORT" in os.environ and not os.environ.get("API_PORT"):
            try:
                config.api_port = int(os.environ["PORT"])
            except (ValueError, TypeError):
                pass

        if base_dir is not None:
            config.model_dir = config.resolve_model_dir(base_dir)
            config.data_dir = config.resolve_data_dir(base_dir)

        return config

    @classmethod
    def default(cls, base_dir: Optional[str] = None) -> "RuntimeConfig":
        config = cls()
        if base_dir is not None:
            config.model_dir = config.resolve_model_dir(base_dir)
            config.data_dir = config.resolve_data_dir(base_dir)
        return config


def export_env_schema() -> Dict[str, Any]:
    registry = _build_env_var_registry()
    groups: Dict[str, int] = {}
    for meta in registry:
        groups[meta.group] = groups.get(meta.group, 0) + 1

    return {
        "version": "1.0",
        "variables": [meta.to_schema_dict() for meta in registry],
        "variable_count": len(registry),
        "groups": [
            {"name": group, "count": count}
            for group, count in groups.items()
        ],
        "deployment_columns": list(_DEPLOYMENT_COLUMNS),
        "deployment_defaults": {
            meta.name: {
                col: meta.value_for(col)
                for col in _DEPLOYMENT_COLUMNS
            }
            for meta in registry
        },
    }
