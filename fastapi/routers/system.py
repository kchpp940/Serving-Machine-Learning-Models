import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from schemas import StatusResponse
from services import get_schema_info, get_service_status


def build_system_router() -> APIRouter:
    router = APIRouter(prefix="", tags=["system"])

    @router.get("/", response_class=PlainTextResponse)
    def welcome() -> str:
        return "Welcome to the Car Price Prediction API!"

    @router.get("/schema")
    def schema():
        return get_schema_info()

    @router.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        return get_service_status()

    return router
