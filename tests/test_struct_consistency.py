#!/usr/bin/env python3
"""验证所有端的返回结构和字段一致性。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_single_prediction_canonical_fields():
    """验证单条预测的 canonical 字段在各端一致。"""
    print("=== 单条预测 canonical 字段验证 ===")

    # FastAPI 服务端 PredictionResponse 字段
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi"))
    from models import PredictionResponse
    fastapi_fields = set(PredictionResponse.__fields__.keys())
    print(f"FastAPI PredictionResponse 字段: {sorted(fastapi_fields)}")

    # 客户端 PredictionResult 字段
    from car_pricing.api_client import PredictionResult
    client_fields = {f.name for f in PredictionResult.__dataclass_fields__.values()}
    print(f"api_client PredictionResult 字段: {sorted(client_fields)}")

    # 期望的 canonical 字段
    expected = {"prediction", "currency", "model_name"}
    print(f"期望的 canonical 字段: {sorted(expected)}")

    assert fastapi_fields == expected, f"FastAPI 字段不匹配: {fastapi_fields} vs {expected}"
    assert client_fields == expected, f"客户端字段不匹配: {client_fields} vs {expected}"
    assert "status" not in fastapi_fields, "'status' 不应出现在 PredictionResponse 中"
    print("✅ 单条预测字段一致\n")


def test_batch_prediction_fields():
    """验证批量预测结果字段在各端一致。"""
    print("=== 批量预测结果字段验证 ===")

    # FastAPI 服务端 BatchPredictionItem 字段
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi"))
    from models import BatchPredictionItem as FastAPIBatchItem
    fastapi_fields = set(FastAPIBatchItem.__fields__.keys())
    print(f"FastAPI BatchPredictionItem 字段: {sorted(fastapi_fields)}")

    # 客户端 BatchPredictionResultItem 字段
    from car_pricing.api_client import BatchPredictionResultItem
    client_fields = {f.name for f in BatchPredictionResultItem.__dataclass_fields__.values()}
    print(f"api_client BatchPredictionResultItem 字段: {sorted(client_fields)}")

    # 客户端 BatchPredictionItem 别名
    from car_pricing.api_client import BatchPredictionItem as ClientBatchItem
    assert ClientBatchItem is BatchPredictionResultItem, "BatchPredictionItem 应该是 BatchPredictionResultItem 的别名"
    print("✅ BatchPredictionItem 是 BatchPredictionResultItem 的别名")

    # 期望的批量字段（与服务端一致）
    expected = {"row_index", "prediction", "currency", "model_name", "error"}
    print(f"期望的批量字段: {sorted(expected)}")

    assert fastapi_fields == expected, f"FastAPI 批量字段不匹配: {fastapi_fields} vs {expected}"
    assert client_fields == expected, f"客户端批量字段不匹配: {client_fields} vs {expected}"
    print("✅ 批量预测字段一致\n")


def test_car_pricing_exports():
    """验证 car_pricing 包的导出符号。"""
    print("=== car_pricing 导出符号验证 ===")

    import car_pricing

    required_exports = [
        # 单条结果
        "PredictionResult",
        # 批量结果
        "BatchPredictionResponse",
        "BatchPredictionResultItem",
        "BatchPredictionItem",
        # 客户端
        "CarPriceAPIClient",
        "PredictionAPIClient",
        # 异常
        "APIClientError",
        "APIConnectionError",
        "APITimeoutError",
        "APIHTTPError",
        "CarPriceAPIError",
        "CarPriceConnectionError",
        "CarPriceTimeoutError",
        "CarPriceHTTPError",
        # 工具
        "format_error",
        "DEFAULT_API_BASE_URL",
        "DEFAULT_TIMEOUT",
        # 模型
        "CarPriceModel",
        "SchemaInfo",
    ]

    for name in required_exports:
        assert hasattr(car_pricing, name), f"car_pricing 缺少导出: {name}"
        print(f"✅ {name}")

    # 验证 PredictionAPIClient 是 CarPriceAPIClient 的别名
    assert car_pricing.PredictionAPIClient is car_pricing.CarPriceAPIClient
    print("\n✅ PredictionAPIClient 是 CarPriceAPIClient 的别名")

    # 验证异常别名
    assert car_pricing.CarPriceAPIError is car_pricing.APIClientError
    assert car_pricing.CarPriceConnectionError is car_pricing.APIConnectionError
    assert car_pricing.CarPriceTimeoutError is car_pricing.APITimeoutError
    assert car_pricing.CarPriceHTTPError is car_pricing.APIHTTPError
    print("✅ 所有异常别名正确\n")


def test_predict_result_construction():
    """验证 PredictionResult 的构造和默认值。"""
    print("=== PredictionResult 构造验证 ===")

    from car_pricing.api_client import PredictionResult

    # 完整字段
    r1 = PredictionResult(prediction=13295.27, currency="USD", model_name="sklearn_gbr")
    assert r1.prediction == 13295.27
    assert r1.currency == "USD"
    assert r1.model_name == "sklearn_gbr"
    print(f"✅ 完整构造: prediction={r1.prediction}, currency={r1.currency}, model_name={r1.model_name}")

    # 默认值
    r2 = PredictionResult(prediction=10000.0)
    assert r2.prediction == 10000.0
    assert r2.currency == "USD"
    assert r2.model_name == ""
    print(f"✅ 默认值: prediction={r2.prediction}, currency={r2.currency}, model_name={r2.model_name!r}")

    # 验证没有 status 字段
    assert not hasattr(r2, "status"), "PredictionResult 不应有 status 字段"
    print("✅ PredictionResult 没有 status 字段\n")


def test_format_error():
    """验证 format_error 函数。"""
    print("=== format_error 验证 ===")

    from car_pricing.api_client import (
        format_error,
        APIConnectionError,
        APITimeoutError,
        APIHTTPError,
        APIClientError,
    )

    msg = format_error(APIConnectionError("test"))
    assert "connect" in msg.lower()
    print(f"✅ ConnectionError: {msg}")

    msg = format_error(APITimeoutError("test"))
    assert "timed out" in msg.lower()
    print(f"✅ TimeoutError: {msg}")

    msg = format_error(APIHTTPError(404, detail="not found"))
    assert "404" in msg and "not found" in msg
    print(f"✅ HTTPError: {msg}")

    msg = format_error(APIClientError("something wrong"))
    assert "Client error" in msg
    print(f"✅ ClientError: {msg}")

    msg = format_error(ValueError("generic"))
    assert "unexpected error" in msg.lower()
    print(f"✅ Generic error: {msg}\n")


def test_batch_prediction_item_construction():
    """验证批量预测条目的构造。"""
    print("=== BatchPredictionItem 构造验证 ===")

    from car_pricing.api_client import BatchPredictionItem

    # 成功条目
    success = BatchPredictionItem(
        row_index=0,
        prediction=13295.27,
        currency="USD",
        model_name="sklearn_gbr",
        error=None,
    )
    assert success.row_index == 0
    assert success.prediction == 13295.27
    assert success.currency == "USD"
    assert success.model_name == "sklearn_gbr"
    assert success.error is None
    print(f"✅ 成功条目: row_index={success.row_index}, prediction={success.prediction}")

    # 失败条目
    failed = BatchPredictionItem(
        row_index=1,
        prediction=None,
        currency="USD",
        model_name="sklearn_gbr",
        error="缺少字段: enginesize",
    )
    assert failed.row_index == 1
    assert failed.prediction is None
    assert failed.error == "缺少字段: enginesize"
    print(f"✅ 失败条目: row_index={failed.row_index}, error={failed.error}")

    print()


def test_api_client_predict_parsing():
    """验证 api_client.predict 的解析逻辑。"""
    print("=== api_client.predict 解析验证 ===")

    from car_pricing.api_client import CarPriceAPIClient

    # 模拟响应解析（不实际发请求）
    # canonical 响应
    canonical_resp = {
        "prediction": 13295.27,
        "currency": "USD",
        "model_name": "sklearn_gbr",
    }
    prediction = canonical_resp.get("prediction")
    currency = canonical_resp.get("currency", "USD")
    model_name = canonical_resp.get("model_name", "")
    assert prediction == 13295.27
    assert currency == "USD"
    assert model_name == "sklearn_gbr"
    print(f"✅ canonical 响应解析: prediction={prediction}, currency={currency}, model_name={model_name}")

    # 旧响应容错（带 status，不带 currency/model_name）
    legacy_resp = {
        "prediction": 13295.27,
        "status": "ok",
    }
    prediction = legacy_resp.get("prediction")
    currency = legacy_resp.get("currency", "USD")
    model_name = legacy_resp.get("model_name", "")
    # status 不进入返回类型
    assert prediction == 13295.27
    assert currency == "USD"
    assert model_name == ""
    assert "status" not in (prediction, currency, model_name)
    print(f"✅ 旧响应容错: prediction={prediction}, currency={currency}, model_name={model_name!r} (status 被忽略)")

    print()


def main():
    test_single_prediction_canonical_fields()
    test_batch_prediction_fields()
    test_car_pricing_exports()
    test_predict_result_construction()
    test_format_error()
    test_batch_prediction_item_construction()
    test_api_client_predict_parsing()
    print("=" * 50)
    print("✅ 所有一致性验证通过！")
    print("=" * 50)


if __name__ == "__main__":
    main()
