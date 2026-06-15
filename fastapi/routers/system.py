import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, FileResponse

from schemas.responses import (
    ApiResponse,
    SchemaResponse,
    StatusResponse,
    success_response,
)
from services.model_service import get_schema_info, get_service_status
from errors.handlers import handle_service_call

router = APIRouter(prefix="", tags=["system"])

favicon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "favicon.png")


@router.get("/", response_class=PlainTextResponse)
async def root():
    note = """
Car Price Prediction API 🙌🏻
Note: add "/docs" to the URL to get the Swagger UI Docs or "/redoc"
  """
    return note


@router.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


@router.get("/schema", response_model=ApiResponse[SchemaResponse])
@handle_service_call
def get_schema() -> ApiResponse[SchemaResponse]:
    result = get_schema_info()
    return success_response(result)


@router.get("/status", response_model=ApiResponse[StatusResponse])
@handle_service_call
def get_status() -> ApiResponse[StatusResponse]:
    result = get_service_status()
    return success_response(result)
