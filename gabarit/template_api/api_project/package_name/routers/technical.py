from fastapi import APIRouter
from starlette.requests import Request

from ..core.config import settings
from ..model.model_base import Model
from .schemas.technical import ReponseInformation, ReponseLiveness, ReponseReadiness

# Technical router
router = APIRouter()


@router.get(
    "/liveness",
    response_model=ReponseLiveness,
    name="liveness",
    tags=["technical"],
)
async def get_liveness() -> ReponseLiveness:
    """Liveness probe for k8s"""
    liveness_msg = ReponseLiveness(alive="ok")
    return liveness_msg


@router.get(
    "/readiness",
    response_model=ReponseReadiness,
    name="readiness",
    tags=["technical"],
)
async def get_readiness(request: Request) -> ReponseReadiness:
    """Readiness probe for k8s"""
    model: Model = (
        request.app.state.model if hasattr(request.app.state, "model") else None
    )

    if model and model.is_model_loaded():
        return ReponseReadiness(ready="ok")
    else:
        return ReponseReadiness(ready="ko")


@router.get(
    "/info",
    response_model=ReponseInformation,
    name="information",
    tags=["technical"],
)
async def info(request: Request) -> ReponseInformation:
    """Rest resource for info"""
    model: Model = request.app.state.model

    return ReponseInformation(
        application=settings.app_name,
        version=settings.app_version,
        model_name=model._model_conf.get("model_name", "?"),
        model_version=model._model_conf.get("package_version", "?"),
    )
