# 1. Library imports
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from models import CarPrediction, PredictionResponse, ErrorResponse
from services import (
    ModelState,
    build_features,
    run_prediction,
    build_success_response,
    FAVICON_PATH,
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
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "Unknown validation error")
        errors.append(f"{loc}: {msg}" if loc else msg)
    detail = "; ".join(errors) if errors else "Invalid request input"
    return JSONResponse(
        status_code=422,
        content={"detail": f"Input validation failed: {detail}"},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
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


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    if not os.path.exists(FAVICON_PATH):
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(FAVICON_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_available": model_state.available,
        "model_error": model_state.error_message,
    }


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
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service is unavailable. "
                f"Model not loaded: {model_state.error_message or 'Unknown reason'}"
            ),
        )

    try:
        features = build_features(data)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to construct feature array from input data: {str(e)}",
        ) from e

    predicted_value, error_msg = run_prediction(features, model_state.model)
    if error_msg is not None:
        raise HTTPException(status_code=500, detail=error_msg)

    return build_success_response(float(predicted_value))
