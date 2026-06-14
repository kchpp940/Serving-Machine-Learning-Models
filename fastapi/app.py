from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel

from car_pricing import (
    CarPriceModel,
    get_model_path,
    load_model,
    build_model_info,
    ErrorMessages,
    resolve_relative_path,
)
from models import CarPrediction

app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.

Supports two model file formats under `./models/sklearn_gbr.pkl`:
- **Bundle (recommended)**: dict with `model`, `feature_order`, `categorical_encoders`,
  produced by the updated `train.py`.
- **Legacy bare model**: the original GradientBoostingRegressor pickle, falls back to
  the hard-coded 10-feature order and LabelEncoder alphabetical mapping.
""",
    version="1.0.0",
    debug=True,
)

_model: CarPriceModel | None = None


def _get_model() -> CarPriceModel:
    global _model
    if _model is None:
        _model = load_model("fastapi")
    return _model


@app.on_event("startup")
def _startup_load_model():
    global _model
    _model = load_model("fastapi")
    path = get_model_path("fastapi")
    print(
        f"[FastAPI] 模型加载完成: path={path}, mode={_model.mode}, "
        f"n_features={len(_model.feature_order)}, "
        f"categorical_cols={list(_model.categorical_encoders.keys())}"
    )


@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Car Price Prediction API 🙌🏻
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
    return note


favicon_path = resolve_relative_path("fastapi/favicon.png")


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


@app.get("/model_info")
async def model_info():
    m = _get_model()
    info = build_model_info(m, get_model_path("fastapi"))
    return info.to_dict()


@app.post("/predict")
def predict(data: CarPrediction):
    try:
        model = _get_model()
        predictions = model.predict_from_pydantic(data)
        value = str(predictions)[1:-1]
        return {"price": value}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorMessages.MODEL_MISSING.format(path=get_model_path("fastapi")),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorMessages.MODEL_LOAD_FAILED.format(error=str(e)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorMessages.PREDICTION_FAILED.format(error=str(e)),
        )
