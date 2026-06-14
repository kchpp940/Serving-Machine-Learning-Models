from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 10


class APIClientError(Exception):
    """API 客户端基础异常。"""
    pass


class APIConnectionError(APIClientError):
    """连接失败。"""
    pass


class APITimeoutError(APIClientError):
    """请求超时。"""
    pass


class APIHTTPError(APIClientError):
    """HTTP 错误。"""

    def __init__(self, status_code: int, detail: str = "", message: str = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(message or f"HTTP {status_code}: {detail}")


@dataclass
class SchemaInfo:
    feature_order: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    target_column: str
    categorical_options: Dict[str, List[Dict[str, Any]]]

    @classmethod
    def from_dict(cls, data: dict) -> "SchemaInfo":
        return cls(
            feature_order=list(data["feature_order"]),
            numeric_features=list(data["numeric_features"]),
            categorical_features=list(data["categorical_features"]),
            target_column=data.get("target_column", "price"),
            categorical_options=data.get("categorical_options", {}),
        )

    def to_dict(self) -> dict:
        return {
            "feature_order": list(self.feature_order),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "target_column": self.target_column,
            "categorical_options": {
                k: list(v) for k, v in self.categorical_options.items()
            },
        }


@dataclass
class BatchPredictionResultItem:
    row_index: int
    prediction: Optional[float] = None
    currency: str = "USD"
    model_name: str = ""
    error: Optional[str] = None


@dataclass
class BatchPredictionResponse:
    status: str = "ok"
    total_records: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    results: List[BatchPredictionResultItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "BatchPredictionResponse":
        results = []
        for item in data.get("results", []):
            results.append(BatchPredictionResultItem(
                row_index=item["row_index"],
                prediction=item.get("prediction"),
                currency=item.get("currency", "USD"),
                model_name=item.get("model_name", ""),
                error=item.get("error"),
            ))
        return cls(
            status=data.get("status", "ok"),
            total_records=data.get("total_records", 0),
            valid_count=data.get("valid_count", 0),
            invalid_count=data.get("invalid_count", 0),
            results=results,
        )

    def to_dataframe_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "row_index": item.row_index,
                "prediction": item.prediction,
                "currency": item.currency,
                "model_name": item.model_name,
                "error": item.error,
            }
            for item in self.results
        ]


class CarPriceAPIClient:
    """车价预测 API 的共享客户端。

    封装了 schema 获取、单条预测、批量预测等操作，
    统一处理 URL 拼接、错误解析和数据序列化。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        if requests is None:
            raise RuntimeError("需要先安装 requests 库")

        self.base_url = (base_url or os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)).rstrip("/")
        self.timeout = timeout or int(os.environ.get("API_REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT)))

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = self._url(path)
        kwargs.setdefault("timeout", self.timeout)
        try:
            resp = requests.request(method, url, **kwargs)
        except requests.exceptions.ConnectionError as e:
            raise APIConnectionError(f"无法连接到 {self.base_url}: {e}")
        except requests.exceptions.Timeout as e:
            raise APITimeoutError(f"请求超时 ({self.timeout}s): {path}")

        if resp.status_code >= 400:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text
            raise APIHTTPError(resp.status_code, detail=detail)

        try:
            return resp.json()
        except ValueError as e:
            raise APIClientError(f"响应不是有效的 JSON: {e}")

    # ---------- Schema ----------

    def get_schema(self) -> SchemaInfo:
        """获取服务端 schema。"""
        data = self._request("GET", "/schema")
        return SchemaInfo.from_dict(data)

    # ---------- 单条预测 ----------

    def predict(self, values: dict) -> float:
        """单条预测，返回预测价格。"""
        data = self._request("POST", "/predict", json=values)
        prediction = data.get("prediction")
        if prediction is None:
            raise APIClientError(f"响应中缺少 prediction 字段: {data}")
        return float(prediction)

    # ---------- 批量预测 ----------

    def predict_batch(self, records: list) -> BatchPredictionResponse:
        """批量预测，返回结构化结果。

        Args:
            records: 记录列表，每条为字段名到值的字典

        Returns:
            BatchPredictionResponse: 包含每条的预测或错误信息
        """
        data = self._request("POST", "/predict_batch", json={"records": records})
        return BatchPredictionResponse.from_dict(data)

    # ---------- 工具方法 ----------

    def validate_csv_columns(self, columns: List[str], schema: SchemaInfo) -> Tuple[List[str], List[str]]:
        """校验 CSV 列是否符合 schema。

        返回 (缺失列列表, 多余列列表)。
        """
        required = set(schema.feature_order)
        actual = set(columns)
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        return missing, extra

    def validate_csv_rows(
        self,
        rows: List[Dict[str, Any]],
        schema: SchemaInfo,
    ) -> List[Dict[str, Any]]:
        """本地预校验 CSV 行，返回错误列表。

        注意：这是客户端轻量校验，最终以服务端校验为准。
        错误项格式: {"row_index": int, "errors": List[str]}
        """
        numeric_features = set(schema.numeric_features)
        categorical_features = set(schema.categorical_features)
        categorical_options = schema.categorical_options

        errors = []
        for idx, row in enumerate(rows):
            row_errors = []
            for field in schema.feature_order:
                val = row.get(field)
                if field in numeric_features:
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        row_errors.append(f"{field}: 无法解析为数值 (值={val!r})")
                elif field in categorical_features:
                    opts = categorical_options.get(field, [])
                    valid_values = {opt["form_value"] for opt in opts}
                    s_val = str(val).strip() if val is not None else ""
                    if s_val not in valid_values:
                        valid_str = ", ".join(sorted(valid_values))
                        row_errors.append(f"{field}: 非法值 {val!r}, 合法值为 [{valid_str}]")
            if row_errors:
                errors.append({
                    "row_index": idx,
                    "errors": row_errors,
                })
        return errors
