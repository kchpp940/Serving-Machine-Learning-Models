"""统一的路径解析。

三端的模型 pkl 优先使用项目级 `shared_models/` 目录作为唯一真源，
回退到各自子目录下的 `models/`（保留私有部署灵活性）。

项目根注入在 `car_pricing/__init__.py` 最开头完成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from car_pricing import PROJECT_ROOT

Deployment = Literal["flaskapp", "fastapi", "bentoml"]

_MODEL_FILE_NAME = "sklearn_gbr.pkl"
_SHARED_MODELS_DIR = PROJECT_ROOT / "shared_models"


def get_model_path(deployment: Deployment = "fastapi") -> Path:
    """返回模型 pkl 的绝对路径。

    查找优先级：
    1. 项目级 `shared_models/sklearn_gbr.pkl`（唯一真源，三端共享）
    2. 子目录级 `<deployment>/models/sklearn_gbr.pkl`（私有部署回退）

    Args:
        deployment: "flaskapp" / "fastapi" / "bentoml"

    Returns:
        找到的 pkl 绝对路径（不保证存在，由调用方按需校验）

    Raises:
        ValueError: deployment 不是合法取值
    """
    if deployment not in ("flaskapp", "fastapi", "bentoml"):
        raise ValueError(
            f"deployment 必须是 'flaskapp' / 'fastapi' / 'bentoml'，收到 {deployment!r}"
        )

    shared = _SHARED_MODELS_DIR / _MODEL_FILE_NAME
    if shared.exists():
        return shared

    fallback = PROJECT_ROOT / deployment / "models" / _MODEL_FILE_NAME
    return fallback


def resolve_relative_path(relative: str) -> Path:
    """相对于项目根的路径解析，方便入口找 favicon、静态文件等。"""
    return (PROJECT_ROOT / relative).resolve()
