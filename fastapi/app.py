import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from models import CarPrediction, INTERFACE_FIELDS

from car_pricing.model_runtime import CarPriceModel
from car_pricing.feature_schema import (
    find_model_path,
    SchemaMismatchError,
)


app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
    version="0.0.1",
    debug=True,
)

_model: CarPriceModel = None


def get_model() -> CarPriceModel:
    global _model
    if _model is None:
        local_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = find_model_path(local_dir=local_dir)
        _model = CarPriceModel.from_joblib(model_path)
        _model.validate_service(INTERFACE_FIELDS)
    return _model


@app.on_event("startup")
async def startup_event():
    try:
        model = get_model()
        info = model.model_info()
        print("Model loaded successfully.")
        print(f"  mode:             {info['mode']}")
        print(f"  n_features_in_:   {info['n_features_in_']}")
        print(f"  schema_features:  {info['schema_n_features']}")
        print(f"  feature_order:    {info['feature_order']}")
        print(f"  interface_fields: {INTERFACE_FIELDS}")
        print("Schema validation: PASSED")
    except SchemaMismatchError as e:
        print(f"FATAL - Schema mismatch on startup: {e}")
        raise
    except Exception as e:
        print(f"FATAL - Failed to load model on startup: {e}")
        raise


@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Car Price Prediction API 🙌🏻
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
    return note


favicon_path = "favicon.png"


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    from fastapi.responses import FileResponse
    return FileResponse(favicon_path)


@app.get("/schema")
async def get_schema():
    model = get_model()
    return {
        "feature_order": model.feature_order,
        "numeric_features": model.numeric_features,
        "categorical_features": model.categorical_features,
        "target_column": model.target_column,
        "categorical_options": {
            f: model.categorical_options(f) for f in model.categorical_features
        },
    }


@app.get("/model_info")
async def get_model_info():
    model = get_model()
    return model.model_info()


@app.post("/predict")
def predict(data: CarPrediction):
    try:
        model = get_model()
        predictions = model.predict_from_pydantic(data)
        value = float(predictions[0])
        return {"predicted_price": value}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SchemaMismatchError as e:
        raise HTTPException(status_code=500, detail=f"Schema 不一致: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
