# 1. Library imports
import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from models import CarPrediction, PredictionResponse, ErrorResponse

# 2. Path resolution based on current module location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sklearn_gbr.pkl")
MODEL_NAME = "sklearn_gbr"
FAVICON_PATH = os.path.join(BASE_DIR, "favicon.png")

# 3. App state storage for the loaded model
ml_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please ensure the trained model exists in the models directory."
            )
        ml_model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model during startup: {str(e)}") from e
    yield
    ml_model = None


# 4. Create the app object
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


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request / Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction service error"},
    },
)
def predict(data: CarPrediction):
    if ml_model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please contact the administrator.",
        )

    try:
        features = np.array(
            [
                [
                    data.enginesize,
                    data.curbweight,
                    data.horsepower,
                    data.highwaympg,
                    data.carwidth,
                    data.wheelbase,
                    data.drivewheel,
                    data.citympg,
                    data.boreratio,
                    data.cylindernumber,
                ]
            ],
            dtype=np.float64,
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to construct feature array from input data: {str(e)}",
        ) from e

    try:
        predictions = ml_model.predict(features)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed during model inference: {str(e)}",
        ) from e

    try:
        predicted_value = float(predictions[0])
    except (IndexError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse model prediction result: {str(e)}",
        ) from e

    return PredictionResponse(
        prediction=predicted_value,
        currency="USD",
        model_name=MODEL_NAME,
    )
