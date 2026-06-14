"""Car Price Prediction 共享运行时包。

三端（Flask / FastAPI / BentoML）统一从本包导入推理适配器与常量，
避免裸文件导入和 sys.path hack。

本包被导入时会自动确保项目根在 sys.path 中，因此即使从子目录
直接启动也能正常 import。

示例::

    from car_pricing import CarPriceModel, load_model, get_model_path
    from car_pricing import build_model_info, ErrorMessages

    model = load_model("fastapi")
    info = build_model_info(model, get_model_path("fastapi"))
"""

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_ROOT.parent

# 确保从任何子目录启动都能 import 项目根下的模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .paths import get_model_path, resolve_relative_path
from .model_runtime import (
    CarPriceModel,
    _is_bundle,
    _DEFAULT_FEATURE_ORDER_LEGACY,
    _NUMERIC_FEATURE_SET,
    _CATEGORICAL_FEATURE_SET,
)
from .model_info import (
    ModelInfo,
    CategoricalOption,
    build_model_info,
    ErrorMessages,
    format_invalid_categorical,
)

__all__ = [
    # 核心运行时
    "CarPriceModel",
    "load_model",
    # 路径
    "get_model_path",
    "PROJECT_ROOT",
    "resolve_relative_path",
    # 模型信息
    "ModelInfo",
    "CategoricalOption",
    "build_model_info",
    "ErrorMessages",
    "format_invalid_categorical",
    # 内部常量（向后兼容）
    "_is_bundle",
    "_DEFAULT_FEATURE_ORDER_LEGACY",
    "_NUMERIC_FEATURE_SET",
    "_CATEGORICAL_FEATURE_SET",
]


def load_model(deployment: str = "fastapi") -> CarPriceModel:
    """统一加载入口：按部署环境找到对应的 pkl，返回 CarPriceModel。

    Args:
        deployment: 部署环境名，"flaskapp" / "fastapi" / "bentoml"。
                    决定从哪个子目录的 models/ 下查找 pkl（如无 shared_models/）。

    Returns:
        CarPriceModel 适配器实例（bundle / legacy 自动识别）。

    Raises:
        FileNotFoundError: 找不到对应的 pkl 文件
        RuntimeError: 模型加载或校验失败
    """
    path = get_model_path(deployment)
    return CarPriceModel.from_joblib(path)
