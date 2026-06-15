import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI

from routers import predict_router, system_router
from errors import register_exception_handlers
from services import get_model_service


def create_app() -> FastAPI:
    app = FastAPI(
        title="Car Price Prediction API",
        description="""An API that utilises a Machine Learning model to predict the price of a given car make and model based on various features.""",
        version="0.0.1",
        debug=True,
    )

    register_exception_handlers(app)

    app.include_router(system_router)
    app.include_router(predict_router)

    @app.on_event("startup")
    async def startup_event():
        try:
            service = get_model_service()
            service.load_model()
            print(f"Model loaded successfully. Mode: {service.model.mode}")
            print(f"Feature order: {service.model.feature_order}")
            print(f"Expected features: {service.model.schema.n_features()}")
            print(f"Model n_features_in_: {service.model.model.n_features_in_}")
        except Exception as e:
            print(f"Failed to load model on startup: {e}")
            raise

    return app


app = create_app()
