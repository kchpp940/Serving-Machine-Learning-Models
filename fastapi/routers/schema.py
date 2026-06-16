import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import APIRouter

from services.model_service import ModelService

router = APIRouter(tags=["Schema"])


@router.get("/schema")
def get_schema():
    service = ModelService.get_instance()
    return service.get_schema()
