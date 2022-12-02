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


"""Main router of the REST API"""


from fastapi import APIRouter
from . import functional, technical

# Create the main router
main_routeur = APIRouter()

# Add the technical and functional routers
main_routeur.include_router(functional.router, tags=["functional"])
main_routeur.include_router(technical.router, tags=["technical"])

__all__ = ["main_routeur"]
