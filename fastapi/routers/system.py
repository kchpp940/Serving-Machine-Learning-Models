import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, FileResponse

from schemas.responses import StatusResponse, ErrorResponse
from services.model_service import get_schema_info, get_service_status

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


@router.get("/schema")
def get_schema():
    return get_schema_info()


@router.get("/status", response_model=StatusResponse, responses={503: {"model": ErrorResponse}})
def get_status():
    return get_service_status()
