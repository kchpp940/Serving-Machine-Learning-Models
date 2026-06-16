from typing import Optional, Dict, Any


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.error_code = error_code or f"E{status_code}"
        self.details = details or {}
        super().__init__(message)


class InvalidInputError(ApiError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=400,
            message=message,
            error_code="INVALID_INPUT",
            details=details,
        )


class ModelNotFoundError(ApiError):
    def __init__(self, model_path: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=503,
            message=f"模型文件未找到: {model_path}",
            error_code="MODEL_NOT_FOUND",
            details=details or {"model_path": model_path},
        )


class PredictionError(ApiError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=500,
            message=f"预测失败: {message}",
            error_code="PREDICTION_ERROR",
            details=details,
        )


class ServiceUnavailableError(ApiError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=503,
            message=f"服务不可用: {message}",
            error_code="SERVICE_UNAVAILABLE",
            details=details,
        )
