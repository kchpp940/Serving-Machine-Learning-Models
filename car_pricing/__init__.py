from car_pricing.feature_schema import (
    FeatureSchema,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    load_training_data,
    prepare_training_data,
    bundle_model,
    is_model_bundle,
)

from car_pricing.prediction_protocol import (
    # 常量
    DEFAULT_CURRENCY,
    DEFAULT_MODEL_NAME,
    DEFAULT_STATUS,
    FIELD_PREDICTION,
    FIELD_CURRENCY,
    FIELD_MODEL_NAME,
    FIELD_STATUS,
    FIELD_ROW_INDEX,
    FIELD_ERROR,
    FIELD_TOTAL_RECORDS,
    FIELD_VALID_COUNT,
    FIELD_INVALID_COUNT,
    FIELD_RESULTS,
    FIELD_RECORDS,
    SINGLE_PREDICTION_FIELDS,
    BATCH_ITEM_FIELDS,
    BATCH_RESPONSE_FIELDS,
    BATCH_REQUEST_FIELDS,
    # 协议数据类（canonical）
    PredictionProtocolResult,
    BatchProtocolItem,
    BatchProtocolResponse,
    # 构造函数
    build_prediction_result,
    build_batch_success_item,
    build_batch_error_item,
    build_batch_response,
    # 校验
    validate_single_result_fields,
    validate_batch_item_fields,
)

from car_pricing.model_runtime import (
    CarPriceModel,
    # 向后兼容别名（实际是 Protocol 类型）
    BatchPredictionItem,
    BatchPredictionResult,
)

from car_pricing.api_client import (
    CarPriceAPIClient,
    PredictionAPIClient,
    SchemaInfo,
    # 结果类型（Protocol 类型的别名，对外友好名）
    PredictionResult,
    BatchPredictionResponse,
    BatchPredictionResultItem,
    # 异常 - 新名称
    APIClientError,
    APIConnectionError,
    APITimeoutError,
    APIHTTPError,
    # 异常 - 旧名称 / 兼容别名
    CarPriceAPIError,
    CarPriceConnectionError,
    CarPriceTimeoutError,
    CarPriceHTTPError,
    # 工具
    format_error,
    DEFAULT_API_BASE_URL,
    DEFAULT_TIMEOUT,
)

__all__ = [
    # feature_schema
    "FeatureSchema",
    "FEATURE_ORDER",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "TARGET_COLUMN",
    "load_training_data",
    "prepare_training_data",
    "bundle_model",
    "is_model_bundle",
    # prediction_protocol - 常量
    "DEFAULT_CURRENCY",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_STATUS",
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
    "SINGLE_PREDICTION_FIELDS",
    "BATCH_ITEM_FIELDS",
    "BATCH_RESPONSE_FIELDS",
    "BATCH_REQUEST_FIELDS",
    # prediction_protocol - 协议数据类（canonical）
    "PredictionProtocolResult",
    "BatchProtocolItem",
    "BatchProtocolResponse",
    # prediction_protocol - 构造函数
    "build_prediction_result",
    "build_batch_success_item",
    "build_batch_error_item",
    "build_batch_response",
    # prediction_protocol - 校验
    "validate_single_result_fields",
    "validate_batch_item_fields",
    # model_runtime
    "CarPriceModel",
    "BatchPredictionItem",
    "BatchPredictionResult",
    # api_client - 客户端
    "CarPriceAPIClient",
    "PredictionAPIClient",
    "SchemaInfo",
    # api_client - 结果类型（对外别名）
    "PredictionResult",
    "BatchPredictionResponse",
    "BatchPredictionResultItem",
    # api_client - 异常
    "APIClientError",
    "APIConnectionError",
    "APITimeoutError",
    "APIHTTPError",
    "CarPriceAPIError",
    "CarPriceConnectionError",
    "CarPriceTimeoutError",
    "CarPriceHTTPError",
    # api_client - 工具
    "format_error",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_TIMEOUT",
]
