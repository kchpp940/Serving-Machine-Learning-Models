from car_pricing_api.routers.prediction import router as prediction_router
from car_pricing_api.routers.schema import router as schema_router
from car_pricing_api.routers.meta import router as meta_router

__all__ = [
    "prediction_router",
    "schema_router",
    "meta_router",
]
