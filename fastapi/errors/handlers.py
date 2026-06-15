import sys
import os
import logging
from functools import wraps
from typing import Callable, Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from errors.exceptions import ApiError
from schemas.responses import error_response

logger = logging.getLogger(__name__)


def _error_to_response(exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            code=exc.status_code,
            message=exc.message,
            details={"error_code": exc.code, **exc.details},
        ).dict(),
    )


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    logger.warning(f"API Error: {exc.code} - {exc.message}")
    return _error_to_response(exc)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            code=exc.status_code,
            message=str(exc.detail),
            details={"error_code": "HTTP_ERROR"},
        ).dict(),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "loc": " -> ".join(str(x) for x in error["loc"]),
                "msg": error["msg"],
                "type": error["type"],
            }
        )
    logger.warning(f"Validation Error: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response(
            code=422,
            message="请求参数验证失败",
            details={"error_code": "VALIDATION_ERROR", "errors": errors},
        ).dict(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response(
            code=500,
            message="服务器内部错误",
            details={"error_code": "INTERNAL_ERROR"},
        ).dict(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)


def handle_service_call(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            raise ApiError(
                status_code=400,
                message=str(e),
                code="INVALID_INPUT",
                details={"original_error": str(e)},
            ) from e
        except RuntimeError as e:
            raise ApiError(
                status_code=503,
                message=str(e),
                code="SERVICE_ERROR",
                details={"original_error": str(e)},
            ) from e
        except Exception as e:
            raise ApiError(
                status_code=500,
                message=f"服务调用失败: {str(e)}",
                code="INTERNAL_ERROR",
                details={"original_error": str(e)},
            ) from e

    return wrapper
