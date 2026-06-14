#!/usr/bin/env python3
"""验证所有端从同一 prediction_protocol 派生，字段和默认值一致。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_protocol_constants_centralized():
    """验证协议常量集中在 prediction_protocol。"""
    print("=== 协议常量集中验证 ===")

    from car_pricing import prediction_protocol as pp

    # 单条预测常量
    assert pp.DEFAULT_CURRENCY == "USD", f"DEFAULT_CURRENCY 错误: {pp.DEFAULT_CURRENCY}"
    assert pp.DEFAULT_MODEL_NAME == "", f"DEFAULT_MODEL_NAME 错误: {pp.DEFAULT_MODEL_NAME!r}"
    assert pp.DEFAULT_STATUS == "ok", f"DEFAULT_STATUS 错误: {pp.DEFAULT_STATUS}"
    assert pp.SINGLE_PREDICTION_FIELDS == ("prediction", "currency", "model_name")
    print(f"✅ DEFAULT_CURRENCY = {pp.DEFAULT_CURRENCY!r}")
    print(f"✅ DEFAULT_MODEL_NAME = {pp.DEFAULT_MODEL_NAME!r}")
    print(f"✅ DEFAULT_STATUS = {pp.DEFAULT_STATUS!r}")
    print(f"✅ SINGLE_PREDICTION_FIELDS = {pp.SINGLE_PREDICTION_FIELDS}")

    # 批量字段常量
    assert pp.BATCH_ITEM_FIELDS == (
        "row_index", "prediction", "currency", "model_name", "error"
    ), f"BATCH_ITEM_FIELDS 错误: {pp.BATCH_ITEM_FIELDS}"
    assert pp.BATCH_RESPONSE_FIELDS == (
        "status", "total_records", "valid_count", "invalid_count", "results"
    ), f"BATCH_RESPONSE_FIELDS 错误: {pp.BATCH_RESPONSE_FIELDS}"
    print(f"✅ BATCH_ITEM_FIELDS = {pp.BATCH_ITEM_FIELDS}")
    print(f"✅ BATCH_RESPONSE_FIELDS = {pp.BATCH_RESPONSE_FIELDS}")
    print()


def test_protocol_dataclasses_exist():
    """验证 prediction_protocol 定义了 canonical 数据类。"""
    print("=== 协议数据类验证 ===")

    from car_pricing import (
        PredictionProtocolResult,
        BatchProtocolItem,
        BatchProtocolResponse,
    )

    # 单条
    single_fields = {f.name for f in PredictionProtocolResult.__dataclass_fields__.values()}
    assert single_fields == {"prediction", "currency", "model_name"}
    print(f"✅ PredictionProtocolResult 字段: {sorted(single_fields)}")

    # 批量条目
    item_fields = {f.name for f in BatchProtocolItem.__dataclass_fields__.values()}
    assert item_fields == {"row_index", "prediction", "currency", "model_name", "error"}
    print(f"✅ BatchProtocolItem 字段: {sorted(item_fields)}")

    # 批量响应
    resp_fields = {f.name for f in BatchProtocolResponse.__dataclass_fields__.values()}
    assert resp_fields == {"status", "total_records", "valid_count", "invalid_count", "results"}
    print(f"✅ BatchProtocolResponse 字段: {sorted(resp_fields)}")
    print()


def test_build_functions_use_protocol_defaults():
    """验证构造函数使用协议默认值。"""
    print("=== 构造函数默认值验证 ===")

    from car_pricing import (
        build_prediction_result,
        build_batch_success_item,
        build_batch_error_item,
        build_batch_response,
        DEFAULT_CURRENCY,
        DEFAULT_MODEL_NAME,
        DEFAULT_STATUS,
    )

    # 单条构造
    single = build_prediction_result(13295.27)
    assert single.prediction == 13295.27
    assert single.currency == DEFAULT_CURRENCY
    assert single.model_name == DEFAULT_MODEL_NAME
    print(f"✅ build_prediction_result: currency={single.currency!r}, model_name={single.model_name!r}")

    # 批量成功条目
    success = build_batch_success_item(0, 13295.27)
    assert success.row_index == 0
    assert success.prediction == 13295.27
    assert success.currency == DEFAULT_CURRENCY
    assert success.model_name == DEFAULT_MODEL_NAME
    assert success.error is None
    print(f"✅ build_batch_success_item: currency={success.currency!r}, model_name={success.model_name!r}")

    # 批量失败条目
    failed = build_batch_error_item(1, "缺少字段")
    assert failed.row_index == 1
    assert failed.prediction is None
    assert failed.currency == DEFAULT_CURRENCY
    assert failed.model_name == DEFAULT_MODEL_NAME
    assert failed.error == "缺少字段"
    print(f"✅ build_batch_error_item: currency={failed.currency!r}, model_name={failed.model_name!r}")

    # 批量响应
    batch = build_batch_response([success, failed])
    assert batch.status == DEFAULT_STATUS
    assert batch.total_records == 2
    assert batch.valid_count == 1
    assert batch.invalid_count == 1
    assert len(batch.results) == 2
    print(f"✅ build_batch_response: status={batch.status!r}, total={batch.total_records}, valid={batch.valid_count}, invalid={batch.invalid_count}")
    print()


def test_api_client_types_alias_protocol():
    """验证 api_client 的结果类型是协议类型的别名。"""
    print("=== api_client 类型别名验证 ===")

    from car_pricing import prediction_protocol as pp
    from car_pricing.api_client import (
        PredictionResult,
        BatchPredictionResultItem,
        BatchPredictionItem,
        BatchPredictionResponse,
    )

    assert PredictionResult is pp.PredictionProtocolResult, (
        "PredictionResult 应是 PredictionProtocolResult 的别名"
    )
    print("✅ PredictionResult is PredictionProtocolResult")

    assert BatchPredictionResultItem is pp.BatchProtocolItem, (
        "BatchPredictionResultItem 应是 BatchProtocolItem 的别名"
    )
    print("✅ BatchPredictionResultItem is BatchProtocolItem")

    assert BatchPredictionItem is pp.BatchProtocolItem, (
        "BatchPredictionItem 应是 BatchProtocolItem 的别名"
    )
    print("✅ BatchPredictionItem is BatchProtocolItem")

    assert BatchPredictionResponse is pp.BatchProtocolResponse, (
        "BatchPredictionResponse 应是 BatchProtocolResponse 的别名"
    )
    print("✅ BatchPredictionResponse is BatchProtocolResponse")
    print()


def test_model_runtime_types_alias_protocol():
    """验证 model_runtime 的结果类型是协议类型的别名。"""
    print("=== model_runtime 类型别名验证 ===")

    from car_pricing import prediction_protocol as pp
    from car_pricing.model_runtime import (
        BatchPredictionItem,
        BatchPredictionResult,
    )

    assert BatchPredictionItem is pp.BatchProtocolItem, (
        "model_runtime.BatchPredictionItem 应是 BatchProtocolItem 的别名"
    )
    print("✅ model_runtime.BatchPredictionItem is BatchProtocolItem")

    assert BatchPredictionResult is pp.BatchProtocolResponse, (
        "model_runtime.BatchPredictionResult 应是 BatchProtocolResponse 的别名"
    )
    print("✅ model_runtime.BatchPredictionResult is BatchProtocolResponse")
    print()


def test_fastapi_pydantic_matches_protocol():
    """验证 FastAPI Pydantic 模型字段与协议一致（使用协议默认值）。"""
    print("=== FastAPI Pydantic 模型与协议一致性验证 ===")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi"))
    from models import PredictionResponse, BatchPredictionItem, BatchPredictionResponse
    from car_pricing import (
        SINGLE_PREDICTION_FIELDS,
        BATCH_ITEM_FIELDS,
        DEFAULT_CURRENCY,
        DEFAULT_MODEL_NAME,
        DEFAULT_STATUS,
    )

    # PredictionResponse
    fastapi_fields = set(PredictionResponse.__fields__.keys())
    expected = set(SINGLE_PREDICTION_FIELDS)
    assert fastapi_fields == expected, f"PredictionResponse 字段不匹配: {fastapi_fields} vs {expected}"
    default_currency = PredictionResponse.__fields__["currency"].default
    default_model = PredictionResponse.__fields__["model_name"].default
    assert default_currency == DEFAULT_CURRENCY, f"PredictionResponse.currency 默认值错误: {default_currency}"
    assert default_model == DEFAULT_MODEL_NAME, f"PredictionResponse.model_name 默认值错误: {default_model!r}"
    print(f"✅ PredictionResponse 字段匹配协议默认值: currency={default_currency!r}, model_name={default_model!r}")

    # BatchPredictionItem
    item_fields = set(BatchPredictionItem.__fields__.keys())
    expected = set(BATCH_ITEM_FIELDS)
    assert item_fields == expected, f"BatchPredictionItem 字段不匹配: {item_fields} vs {expected}"
    item_currency = BatchPredictionItem.__fields__["currency"].default
    item_model = BatchPredictionItem.__fields__["model_name"].default
    assert item_currency == DEFAULT_CURRENCY
    assert item_model == DEFAULT_MODEL_NAME
    print(f"✅ BatchPredictionItem 字段匹配协议默认值")

    # BatchPredictionResponse
    status_default = BatchPredictionResponse.__fields__["status"].default
    assert status_default == DEFAULT_STATUS, f"BatchPredictionResponse.status 默认值错误: {status_default!r}"
    print(f"✅ BatchPredictionResponse.status 默认值 = {status_default!r}")
    print()


def test_protocol_dict_roundtrip():
    """验证协议类的 from_dict / to_dict 往返转换。"""
    print("=== 协议 dict 往返转换验证 ===")

    from car_pricing import (
        PredictionProtocolResult,
        BatchProtocolItem,
        BatchProtocolResponse,
        build_batch_success_item,
        build_batch_error_item,
        build_batch_response,
    )

    # 单条 - canonical
    s1 = PredictionProtocolResult.from_dict({
        "prediction": 13295.27,
        "currency": "USD",
        "model_name": "sklearn_gbr",
    })
    d1 = s1.to_dict()
    assert d1 == {"prediction": 13295.27, "currency": "USD", "model_name": "sklearn_gbr"}
    print(f"✅ 单条 canonical 往返: {d1}")

    # 单条 - 旧响应容错（带 status，status 被忽略）
    s2 = PredictionProtocolResult.from_dict({
        "prediction": 10000.0,
        "status": "ok",
    })
    d2 = s2.to_dict()
    assert "status" not in d2
    assert d2["prediction"] == 10000.0
    assert d2["currency"] == "USD"
    assert d2["model_name"] == ""
    print(f"✅ 单条 legacy 往返: {d2} (status 被剔除)")

    # 批量条目
    success = build_batch_success_item(0, 13295.27, model_name="gbr")
    d_item = success.to_dict()
    assert d_item == {
        "row_index": 0, "prediction": 13295.27, "currency": "USD",
        "model_name": "gbr", "error": None
    }
    item_back = BatchProtocolItem.from_dict(d_item)
    assert item_back.row_index == 0
    assert item_back.prediction == 13295.27
    print(f"✅ 批量条目往返: {d_item}")

    # 批量响应
    failed = build_batch_error_item(1, "bad input")
    resp = build_batch_response([success, failed])
    d_resp = resp.to_dict()
    assert d_resp["status"] == "ok"
    assert d_resp["total_records"] == 2
    assert d_resp["valid_count"] == 1
    assert d_resp["invalid_count"] == 1
    assert len(d_resp["results"]) == 2
    resp_back = BatchProtocolResponse.from_dict(d_resp)
    assert resp_back.total_records == 2
    assert resp_back.valid_count == 1
    print(f"✅ 批量响应往返: total={resp_back.total_records}, valid={resp_back.valid_count}, invalid={resp_back.invalid_count}")
    print()


def test_bentoml_returns_protocol_dict():
    """验证 BentoML service 返回的 dict 符合协议格式。"""
    print("=== BentoML 服务端协议格式验证 ===")

    from car_pricing import (
        BatchProtocolItem,
        BatchProtocolResponse,
        DEFAULT_STATUS,
        BATCH_ITEM_FIELDS,
    )
    from car_pricing.model_runtime import CarPriceModel
    import tempfile

    # 用 model_runtime 模拟 BentoML 行为
    model = _get_test_model()
    records = [
        {"enginesize": 130, "curbweight": 2548, "horsepower": 111, "highwaympg": 27,
         "carwidth": 64.1, "wheelbase": 88.6, "drivewheel": "rwd", "citympg": 21,
         "boreratio": 3.47, "cylindernumber": "four"},
        {"enginesize": None, "curbweight": None},  # 缺少字段
    ]
    batch = model.predict_records(records)
    d = batch.to_dict()  # 这就是 BentoML 返回的内容

    # 验证响应格式
    assert "status" in d and d["status"] == DEFAULT_STATUS
    assert "total_records" in d and d["total_records"] == 2
    assert "valid_count" in d and d["valid_count"] == 1
    assert "invalid_count" in d and d["invalid_count"] == 1
    assert "results" in d and len(d["results"]) == 2
    print(f"✅ 响应包含协议字段: status/total_records/valid_count/invalid_count/results")

    # 验证每条结果字段
    expected_item_fields = set(BATCH_ITEM_FIELDS)
    for r in d["results"]:
        assert set(r.keys()) == expected_item_fields, f"条目字段不对: {set(r.keys())} vs {expected_item_fields}"
        assert "currency" in r and r["currency"] == "USD"
        assert "model_name" in r
    print(f"✅ 每条结果包含协议字段: {sorted(expected_item_fields)}")
    print()


def _get_test_model():
    """加载测试模型。"""
    from car_pricing import CarPriceModel
    model_path = os.path.join(os.path.dirname(__file__), "..", "fastapi", "models", "sklearn_gbr.pkl")
    if not os.path.exists(model_path):
        raise RuntimeError(f"测试模型不存在: {model_path}")
    m = CarPriceModel.from_joblib(model_path)
    return m


def main():
    test_protocol_constants_centralized()
    test_protocol_dataclasses_exist()
    test_build_functions_use_protocol_defaults()
    test_api_client_types_alias_protocol()
    test_model_runtime_types_alias_protocol()
    test_fastapi_pydantic_matches_protocol()
    test_protocol_dict_roundtrip()
    test_bentoml_returns_protocol_dict()
    print("=" * 60)
    print("✅ 所有协议一致性验证通过！所有端都从 prediction_protocol 派生！")
    print("=" * 60)


if __name__ == "__main__":
    main()
