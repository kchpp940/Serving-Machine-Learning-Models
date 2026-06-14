# 1. Library imports
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from models import CarPrediction, PredictionResponse, ErrorResponse
from services import (
    ModelState,
    build_features,
    run_prediction,
    build_success_response,
    build_error_response,
    build_validation_error_response,
    build_health_response,
    FAVICON_PATH,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_state = ModelState()


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


# 5. Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return build_validation_error_response(exc.errors())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return build_error_response(str(exc.detail), exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return build_error_response(
        f"Internal server error: {str(exc)}",
        HTTP_500_INTERNAL_SERVER_ERROR,
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
async def favicon():
    if not os.path.exists(FAVICON_PATH):
        return build_error_response("Favicon not found", HTTP_404_NOT_FOUND)
    return FileResponse(FAVICON_PATH)


@app.get("/health")
def health_check():
    return build_health_response(model_state)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request / Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction service error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
def predict(data: CarPrediction):
    if not model_state.available or model_state.model is None:
        return build_error_response(
            (
                "Prediction service is unavailable. "
                f"Model not loaded: {model_state.error_message or 'Unknown reason'}"
            ),
            HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        features = build_features(data)
    except (ValueError, TypeError) as e:
        return build_error_response(
            f"Failed to construct feature array from input data: {str(e)}",
            HTTP_400_BAD_REQUEST,
        )

    predicted_value, error_msg = run_prediction(features, model_state.model)
    if error_msg is not None:
        return build_error_response(error_msg, HTTP_500_INTERNAL_SERVER_ERROR)

    return build_success_response(float(predicted_value))
