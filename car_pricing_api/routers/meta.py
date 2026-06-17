from fastapi import APIRouter

from services.model_service import ModelService

router = APIRouter(tags=["Metadata"])


@router.get("/status")
def get_status():
    service = ModelService.get_instance()
    return service.get_status()


@router.get("/metadata")
def get_metadata():
    service = ModelService.get_instance()
    return service.get_metadata()


@router.get("/health")
def get_health():
    service = ModelService.get_instance()
    return service.get_health()
