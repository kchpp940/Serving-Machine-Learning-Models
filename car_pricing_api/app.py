import os

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, FileResponse

from services.model_service import ModelService
from routers.prediction import router as prediction_router
from routers.schema import router as schema_router
from routers.meta import router as meta_router


app = FastAPI(
    title="Car Price Prediction API",
    description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
    version="0.0.1",
    debug=True,
)

favicon_path = "favicon.png"


@app.on_event("startup")
async def startup_event():
    try:
        service = ModelService.get_instance()
        service.load()
        model = service.model
        lineage = service.get_lineage()
        print(f"Model loaded successfully. Mode: {model.mode}")
        print(f"Feature order: {model.feature_order}")
        print(f"Expected features: {model.schema.n_features()}")
        print(f"Model n_features_in_: {model.model.n_features_in_}")
        print(f"Model name: {lineage.model_name}")
        print(f"Model type: {lineage.model_type}")
        print(f"Schema version: {lineage.schema_version}")
        print(f"Data version: {lineage.data_version}")
        print(f"Model artifact hash: {lineage.model_artifact_hash}")
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        raise


@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
Car Price Prediction API 🙌🏻
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
    return note


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


app.include_router(meta_router)
app.include_router(schema_router)
app.include_router(prediction_router)
