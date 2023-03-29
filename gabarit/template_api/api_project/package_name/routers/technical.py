#!/usr/bin/env python3
# Copyright (C) <2018-2022>  <Agence Data Services, DSI Pôle Emploi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from fastapi import APIRouter

from ..core.config import settings
from ..core.resources import RESOURCES, RESOURCE_MODEL
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
async def get_readiness() -> ReponseReadiness:
    """Readiness probe for k8s"""
    model: Model = RESOURCES.get(RESOURCE_MODEL)

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
async def info() -> ReponseInformation:
    """Rest resource for info"""
    model: Model = RESOURCES.get(RESOURCE_MODEL)

    return ReponseInformation(
        application=settings.app_name,
        version=settings.app_version,
        model_name=model._model_conf.get("model_name", "?"),
        model_version=model._model_conf.get("package_version", "?"),
    )
