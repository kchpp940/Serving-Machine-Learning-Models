import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel

from model_runtime import CarPriceModel
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

_MODEL_PATH = os.path.join(_HERE, "models", "sklearn_gbr.pkl")
_model: CarPriceModel | None = None


def _get_model() -> CarPriceModel:
    global _model
    if _model is None:
        _model = CarPriceModel.from_joblib(_MODEL_PATH)
    return _model


@app.on_event("startup")
def _startup_load_model():
    global _model
    _model = CarPriceModel.from_joblib(_MODEL_PATH)
    print(
        f"[FastAPI] 模型加载完成: mode={_model.mode}, n_features={len(_model.feature_order)}, "
        f"categorical_cols={list(_model.categorical_encoders.keys())}"
    )


@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Car Price Prediction API 🙌🏻
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
    return note


favicon_path = os.path.join(_HERE, "favicon.png")


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


class ModelInfoResponse(BaseModel):
    mode: str
    feature_order: list
    categorical_features: list
    categorical_options: dict


@app.get("/model_info")
async def model_info() -> ModelInfoResponse:
    m = _get_model()
    options = {col: m.categorical_options(col) for col in m.categorical_features}
    return ModelInfoResponse(
        mode=m.mode,
        feature_order=list(m.feature_order),
        categorical_features=list(m.categorical_features),
        categorical_options=options,
    )


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
        raise HTTPException(status_code=500, detail=f"Model file missing: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
