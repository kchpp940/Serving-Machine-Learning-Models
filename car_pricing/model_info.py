"""统一的模型信息结构和错误提示。

三端入口都可以复用这里的 build_model_info() / ErrorMessages，
保证暴露给用户/调用方的信息格式一致。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

from .model_runtime import CarPriceModel


@dataclass
class CategoricalOption:
    display: str
    form_value: str
    model_code: int


@dataclass
class ModelInfo:
    """统一的模型元数据结构，三端 `/model_info` 接口均返回此格式。"""
    model_path: str
    mode: str
    feature_order: list[str]
    categorical_features: list[str]
    categorical_options: dict[str, list[CategoricalOption]]
    numeric_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_info(model: CarPriceModel, model_path: str) -> ModelInfo:
    """从 CarPriceModel 实例构造标准 ModelInfo。"""
    options = {
        col: [
            CategoricalOption(**opt)
            for opt in model.categorical_options(col)
        ]
        for col in model.categorical_features
    }
    return ModelInfo(
        model_path=str(model_path),
        mode=model.mode,
        feature_order=list(model.feature_order),
        categorical_features=list(model.categorical_features),
        categorical_options=options,
        numeric_features=list(model.numeric_features),
    )


class ErrorMessages:
    """统一的错误提示文案，三端共用。"""
    MODEL_MISSING = "Model file not found at {path}"
    MODEL_LOAD_FAILED = "Failed to load model: {error}"
    MODEL_DIMENSION_MISMATCH = (
        "Model n_features_in_={actual} does not match feature_order "
        "length {expected}"
    )
    PREDICTION_FAILED = "Prediction failed: {error}"
    INVALID_CATEGORICAL_VALUE = (
        "Invalid value {value!r} for field {field!r}. "
        "Allowed values: {allowed}"
    )
    INVALID_NUMERIC_VALUE = (
        "Invalid numeric value {value!r} for field {field!r}"
    )
    FIELD_EMPTY = "Field {field!r} cannot be empty"


def format_invalid_categorical(field: str, value, allowed: list | set) -> str:
    allowed_str = ", ".join(sorted(str(x) for x in allowed))
    return ErrorMessages.INVALID_CATEGORICAL_VALUE.format(
        field=field, value=value, allowed=allowed_str
    )
