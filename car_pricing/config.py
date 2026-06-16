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
        def _env(name: str, default: Any, cast=None) -> Any:
            val = os.environ.get(name)
            if val is None or val == "":
                return default
            if cast is not None:
                try:
                    return cast(val)
                except (ValueError, TypeError):
                    return default
            return val

        config = cls(
            api_host=_env("API_HOST", API_HOST_DEFAULT),
            api_port=_env("API_PORT", API_PORT_DEFAULT, int),
            model_dir=_env("MODEL_DIR", ""),
            model_filename=_env("MODEL_FILENAME", MODEL_FILENAME_DEFAULT),
            model_metadata_filename=_env(
                "MODEL_METADATA_FILENAME", MODEL_METADATA_FILENAME_DEFAULT
            ),
            model_status_filename=_env(
                "MODEL_STATUS_FILENAME", MODEL_STATUS_FILENAME_DEFAULT
            ),
            schema_snapshot_filename=_env(
                "SCHEMA_SNAPSHOT_FILENAME", SCHEMA_SNAPSHOT_FILENAME_DEFAULT
            ),
            lineage_filename=_env("LINEAGE_FILENAME", LINEAGE_FILENAME_DEFAULT),
            data_dir=_env("DATA_DIR", ""),
            data_csv_filename=_env("DATA_CSV_FILENAME", DATA_CSV_FILENAME_DEFAULT),
            api_base_url=_env("API_BASE_URL", API_BASE_URL_DEFAULT),
            request_timeout=_env("REQUEST_TIMEOUT", REQUEST_TIMEOUT_DEFAULT, int),
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
    return {
        "version": "1.0",
        "variables": [
            {
                "name": "API_HOST",
                "default": API_HOST_DEFAULT,
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "API 监听地址",
                "group": "API server configuration",
            },
            {
                "name": "API_PORT",
                "default": API_PORT_DEFAULT,
                "type": "int",
                "sensitive": False,
                "required": True,
                "description": "API 监听端口（Heroku 会自动注入 PORT 环境变量，启动时会映射到此变量）",
                "group": "API server configuration",
            },
            {
                "name": "MODEL_DIR",
                "default": "<auto-resolved>",
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "模型文件目录。未设置时按优先级解析：./models → ./shared_models → <project_root>/shared_models → <project_root>/models",
                "group": "Model file locations",
            },
            {
                "name": "MODEL_FILENAME",
                "default": MODEL_FILENAME_DEFAULT,
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "模型文件名",
                "group": "Model file locations",
            },
            {
                "name": "MODEL_METADATA_FILENAME",
                "default": MODEL_METADATA_FILENAME_DEFAULT,
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "模型元数据文件名",
                "group": "Model file locations",
            },
            {
                "name": "MODEL_STATUS_FILENAME",
                "default": MODEL_STATUS_FILENAME_DEFAULT,
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "模型状态文件名",
                "group": "Model file locations",
            },
            {
                "name": "DATA_CSV_FILENAME",
                "default": DATA_CSV_FILENAME_DEFAULT,
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "训练数据 CSV 文件名",
                "group": "Training data locations",
            },
            {
                "name": "DATA_DIR",
                "default": "<auto-resolved>",
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "训练数据目录。未设置时按优先级解析：./Data → <project_root>/Data",
                "group": "Training data locations",
            },
            {
                "name": "API_BASE_URL",
                "default": API_BASE_URL_DEFAULT,
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "API 基础 URL（供 Streamlit 等客户端调用）",
                "group": "Client configuration",
            },
            {
                "name": "REQUEST_TIMEOUT",
                "default": REQUEST_TIMEOUT_DEFAULT,
                "type": "int",
                "sensitive": False,
                "required": True,
                "description": "请求超时时间（秒）",
                "group": "Client configuration",
            },
            {
                "name": "BENTOML_MODEL_TAG",
                "default": BENTOML_MODEL_TAG_DEFAULT,
                "type": "str",
                "sensitive": False,
                "required": True,
                "description": "BentoML 模型标签",
                "group": "BentoML configuration",
            },
        ],
    }
