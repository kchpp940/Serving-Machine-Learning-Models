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

from car_pricing.model_runtime import (
    CarPriceModel,
    BatchPredictionItem,
    BatchPredictionResult,
)

from car_pricing.api_client import (
    CarPriceAPIClient,
    PredictionAPIClient,
    SchemaInfo,
    BatchPredictionResponse,
    BatchPredictionResultItem,
    APIClientError,
    APIConnectionError,
    APITimeoutError,
    APIHTTPError,
    CarPriceAPIError,
    CarPriceConnectionError,
    CarPriceTimeoutError,
    CarPriceHTTPError,
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
    # model_runtime
    "CarPriceModel",
    "BatchPredictionItem",
    "BatchPredictionResult",
    # api_client - 新名称
    "CarPriceAPIClient",
    "SchemaInfo",
    "BatchPredictionResponse",
    "BatchPredictionResultItem",
    "APIClientError",
    "APIConnectionError",
    "APITimeoutError",
    "APIHTTPError",
    # api_client - 旧名称 / 兼容别名
    "PredictionAPIClient",
    "CarPriceAPIError",
    "CarPriceConnectionError",
    "CarPriceTimeoutError",
    "CarPriceHTTPError",
    "format_error",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_TIMEOUT",
]
