import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from errors.exceptions import ApiError, InvalidInputError, PredictionError

logger = logging.getLogger(__name__)


def _error_body(code: int, message: str, error_code: str, **extra) -> dict:
    return {
        "detail": message,
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "error_code": error_code,
            **extra,
        },
    }


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning(f"ValueError: {exc}")
    api_err = InvalidInputError(str(exc))
    return JSONResponse(
        status_code=api_err.status_code,
        content=_error_body(
            api_err.status_code, api_err.message, api_err.error_code,
            **api_err.details,
        ),
    )


async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    logger.error(f"RuntimeError: {exc}")
    api_err = PredictionError(str(exc))
    return JSONResponse(
        status_code=api_err.status_code,
        content=_error_body(
            api_err.status_code, api_err.message, api_err.error_code,
            **api_err.details,
        ),
    )


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    logger.warning(f"API Error: {exc.error_code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(
            exc.status_code, exc.message, exc.error_code,
            **exc.details,
        ),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(
            exc.status_code, str(exc.detail), "HTTP_ERROR",
        ),
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
        content=_error_body(
            422, "请求参数验证失败", "VALIDATION_ERROR",
            errors=errors,
        ),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body(
            500, "服务器内部错误", "INTERNAL_ERROR",
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(RuntimeError, runtime_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
