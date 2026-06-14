# 1. Library imports
import os
import logging
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from models import CarPrediction, PredictionResponse, ErrorResponse, HealthResponse
from services import (
    ModelState,
    predict,
    check_health,
    format_validation_errors,
    ServiceError,
    PredictionResult,
    HealthResult,
    ErrorKind,
    FAVICON_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_state = ModelState()


ERROR_KIND_TO_HTTP_STATUS = {
    ErrorKind.MODEL_UNAVAILABLE: 503,
    ErrorKind.BAD_REQUEST: 400,
    ErrorKind.PREDICTION_FAILED: 500,
    ErrorKind.VALIDATION_ERROR: 422,
    ErrorKind.NOT_FOUND: 404,
    ErrorKind.INTERNAL_ERROR: 500,
}


def to_prediction_response(domain: PredictionResult) -> PredictionResponse:
    return PredictionResponse(
        prediction=domain.prediction,
        currency=domain.currency,
        model_name=domain.model_name,
    )


def to_health_response(domain: HealthResult) -> HealthResponse:
    return HealthResponse(
        status=domain.status,
        model_available=domain.model_available,
        model_error=domain.model_error,
    )


def to_json_error(error: ServiceError) -> JSONResponse:
    status_code = ERROR_KIND_TO_HTTP_STATUS.get(error.kind, 500)
    return JSONResponse(
        status_code=status_code,
        content={"detail": error.detail},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_state
    model_state = ModelState.load()
    if model_state.available:
        logger.info("Model loaded successfully.")
    else:
        logger.warning(
            f"Model failed to load: {model_state.error_message}. "
            "Service will continue running but /predict will return errors."
        )
    yield
    model_state = ModelState()


app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
    version="0.0.1",
    debug=False,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error = format_validation_errors(exc.errors())
    return to_json_error(error)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# 6. Routes
@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Car Price Prediction API 🙌🏻
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
    return note


@app.get("/favicon.png", include_in_schema=False, response_model=None)
async def favicon() -> Union[FileResponse, JSONResponse]:
    if not os.path.exists(FAVICON_PATH):
        error = ServiceError(kind=ErrorKind.NOT_FOUND, detail="Favicon not found")
        return to_json_error(error)
    return FileResponse(FAVICON_PATH)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status and model availability.",
)
def health_check() -> HealthResponse:
    result = check_health(model_state)
    return to_health_response(result)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict car price",
    description="Predict the price of a car based on its features using the loaded ML model.",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 13495.0,
                        "currency": "USD",
                        "model_name": "sklearn_gbr",
                    }
                }
            },
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad request — feature array could not be constructed from input data",
        },
        422: {
            "model": ErrorResponse,
            "description": "Validation error — input fields missing or wrong type",
        },
        500: {
            "model": ErrorResponse,
            "description": "Prediction service error — model inference or result parsing failed",
        },
        503: {
            "model": ErrorResponse,
            "description": "Service unavailable — model is not loaded",
        },
    },
)
def predict_route(data: CarPrediction) -> Union[PredictionResponse, JSONResponse]:
    result = predict(data, model_state)
    if isinstance(result, ServiceError):
        return to_json_error(result)
    return to_prediction_response(result)
