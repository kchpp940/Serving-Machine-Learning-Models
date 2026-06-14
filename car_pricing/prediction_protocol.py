"""单条和批量预测的共享协议定义。

所有端（FastAPI Pydantic、BentoML dict、api_client dataclass、model_runtime）
都必须从本模块取字段名、默认值和转换函数，避免三处同步。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Dict, Any, Optional, TypeVar, Union


# ---------------------------------------------------------------------------
# 常量协议：字段名、默认值
# ---------------------------------------------------------------------------

# 单条预测 canonical 字段
FIELD_PREDICTION = "prediction"
FIELD_CURRENCY = "currency"
FIELD_MODEL_NAME = "model_name"
# 旧响应字段（仅做兼容，不进入新协议类型）
FIELD_STATUS = "status"

DEFAULT_CURRENCY = "USD"
DEFAULT_MODEL_NAME = ""
DEFAULT_STATUS = "ok"

SINGLE_PREDICTION_FIELDS: tuple = (FIELD_PREDICTION, FIELD_CURRENCY, FIELD_MODEL_NAME)

# 批量单条结果字段
FIELD_ROW_INDEX = "row_index"
FIELD_ERROR = "error"

BATCH_ITEM_FIELDS: tuple = (
    FIELD_ROW_INDEX,
    FIELD_PREDICTION,
    FIELD_CURRENCY,
    FIELD_MODEL_NAME,
    FIELD_ERROR,
)

# 批量汇总响应字段
FIELD_TOTAL_RECORDS = "total_records"
FIELD_VALID_COUNT = "valid_count"
FIELD_INVALID_COUNT = "invalid_count"
FIELD_RESULTS = "results"

BATCH_RESPONSE_FIELDS: tuple = (
    FIELD_STATUS,
    FIELD_TOTAL_RECORDS,
    FIELD_VALID_COUNT,
    FIELD_INVALID_COUNT,
    FIELD_RESULTS,
)

# 批量请求字段
FIELD_RECORDS = "records"
BATCH_REQUEST_FIELDS: tuple = (FIELD_RECORDS,)


# ---------------------------------------------------------------------------
# 统一的协议数据类（供 model_runtime / api_client 直接使用）
# ---------------------------------------------------------------------------

@dataclass
class PredictionProtocolResult:
    """单条预测的协议结果。"""
    prediction: float
    currency: str = DEFAULT_CURRENCY
    model_name: str = DEFAULT_MODEL_NAME

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionProtocolResult":
        """从响应 dict 解析，旧响应（带 status）自动容错。"""
        if FIELD_PREDICTION not in data:
            raise ValueError(f"响应缺少必填字段 {FIELD_PREDICTION!r}: {data}")
        return cls(
            prediction=float(data[FIELD_PREDICTION]),
            currency=data.get(FIELD_CURRENCY, DEFAULT_CURRENCY),
            model_name=data.get(FIELD_MODEL_NAME, DEFAULT_MODEL_NAME),
            # FIELD_STATUS 存在但被忽略（旧响应容错）
        )


@dataclass
class BatchProtocolItem:
    """批量预测中单条记录的协议结果。"""
    row_index: int
    prediction: Optional[float] = None
    currency: str = DEFAULT_CURRENCY
    model_name: str = DEFAULT_MODEL_NAME
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchProtocolItem":
        if FIELD_ROW_INDEX not in data:
            raise ValueError(f"响应缺少必填字段 {FIELD_ROW_INDEX!r}: {data}")
        return cls(
            row_index=int(data[FIELD_ROW_INDEX]),
            prediction=(float(data[FIELD_PREDICTION]) if data.get(FIELD_PREDICTION) is not None else None),
            currency=data.get(FIELD_CURRENCY, DEFAULT_CURRENCY),
            model_name=data.get(FIELD_MODEL_NAME, DEFAULT_MODEL_NAME),
            error=data.get(FIELD_ERROR),
        )


@dataclass
class BatchProtocolResponse:
    """批量预测的完整协议响应。"""
    total_records: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    results: List[BatchProtocolItem] = field(default_factory=list)
    status: str = DEFAULT_STATUS

    def to_dict(self) -> Dict[str, Any]:
        return {
            FIELD_STATUS: self.status,
            FIELD_TOTAL_RECORDS: self.total_records,
            FIELD_VALID_COUNT: self.valid_count,
            FIELD_INVALID_COUNT: self.invalid_count,
            FIELD_RESULTS: [item.to_dict() for item in self.results],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchProtocolResponse":
        raw_results = data.get(FIELD_RESULTS, []) or []
        results = [BatchProtocolItem.from_dict(item) for item in raw_results]
        return cls(
            status=data.get(FIELD_STATUS, DEFAULT_STATUS),
            total_records=int(data.get(FIELD_TOTAL_RECORDS, len(results))),
            valid_count=int(data.get(FIELD_VALID_COUNT, sum(1 for r in results if r.error is None))),
            invalid_count=int(data.get(FIELD_INVALID_COUNT, sum(1 for r in results if r.error is not None))),
            results=results,
        )

    def to_dataframe_rows(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.results]


# ---------------------------------------------------------------------------
# 便捷构造函数（各端调用，避免直接手写默认值）
# ---------------------------------------------------------------------------

def build_prediction_result(
    prediction: float,
    *,
    currency: str = DEFAULT_CURRENCY,
    model_name: str = DEFAULT_MODEL_NAME,
) -> PredictionProtocolResult:
    """构造单条预测结果，使用协议默认值。"""
    return PredictionProtocolResult(
        prediction=float(prediction),
        currency=currency,
        model_name=model_name,
    )


def build_batch_success_item(
    row_index: int,
    prediction: float,
    *,
    currency: str = DEFAULT_CURRENCY,
    model_name: str = DEFAULT_MODEL_NAME,
) -> BatchProtocolItem:
    """构造批量预测中成功的单条结果。"""
    return BatchProtocolItem(
        row_index=int(row_index),
        prediction=float(prediction),
        currency=currency,
        model_name=model_name,
        error=None,
    )


def build_batch_error_item(
    row_index: int,
    error: str,
    *,
    currency: str = DEFAULT_CURRENCY,
    model_name: str = DEFAULT_MODEL_NAME,
) -> BatchProtocolItem:
    """构造批量预测中失败的单条结果。"""
    return BatchProtocolItem(
        row_index=int(row_index),
        prediction=None,
        currency=currency,
        model_name=model_name,
        error=error,
    )


def build_batch_response(
    items: List[BatchProtocolItem],
    *,
    status: str = DEFAULT_STATUS,
) -> BatchProtocolResponse:
    """从批量条目列表构造汇总响应，自动统计数量。"""
    total = len(items)
    valid = sum(1 for it in items if it.error is None)
    invalid = total - valid
    return BatchProtocolResponse(
        status=status,
        total_records=total,
        valid_count=valid,
        invalid_count=invalid,
        results=items,
    )


# ---------------------------------------------------------------------------
# 协议校验：任何实现（dict / dataclass / pydantic）都可以用这些函数验证
# ---------------------------------------------------------------------------

def validate_single_result_fields(d: Dict[str, Any]) -> None:
    """校验单条结果 dict 是否包含 canonical 字段。不检查 status。"""
    if FIELD_PREDICTION not in d:
        raise ValueError(f"缺少必填字段 {FIELD_PREDICTION!r}")
    for k in SINGLE_PREDICTION_FIELDS:
        if k in d and d[k] is None and k == FIELD_PREDICTION:
            raise ValueError(f"字段 {FIELD_PREDICTION!r} 不能为空")


def validate_batch_item_fields(d: Dict[str, Any]) -> None:
    """校验批量条目 dict 是否包含所有必填字段。"""
    if FIELD_ROW_INDEX not in d:
        raise ValueError(f"缺少必填字段 {FIELD_ROW_INDEX!r}")


__all__ = [
    # 常量
    "FIELD_PREDICTION",
    "FIELD_CURRENCY",
    "FIELD_MODEL_NAME",
    "FIELD_STATUS",
    "FIELD_ROW_INDEX",
    "FIELD_ERROR",
    "FIELD_TOTAL_RECORDS",
    "FIELD_VALID_COUNT",
    "FIELD_INVALID_COUNT",
    "FIELD_RESULTS",
    "FIELD_RECORDS",
    "DEFAULT_CURRENCY",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_STATUS",
    "SINGLE_PREDICTION_FIELDS",
    "BATCH_ITEM_FIELDS",
    "BATCH_RESPONSE_FIELDS",
    "BATCH_REQUEST_FIELDS",
    # 协议数据类
    "PredictionProtocolResult",
    "BatchProtocolItem",
    "BatchProtocolResponse",
    # 构造函数
    "build_prediction_result",
    "build_batch_success_item",
    "build_batch_error_item",
    "build_batch_response",
    # 校验函数
    "validate_single_result_fields",
    "validate_batch_item_fields",
]
