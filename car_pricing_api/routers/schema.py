from fastapi import APIRouter

from services.model_service import ModelService

router = APIRouter(tags=["Schema"])


@router.get("/schema")
def get_schema():
    service = ModelService.get_instance()
    return service.get_schema()
