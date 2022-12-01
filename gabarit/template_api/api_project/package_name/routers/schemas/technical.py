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


"""Technical schemas"""
from pydantic import BaseModel, Field


class ReponseLiveness(BaseModel):
    """Return object for liveness probe"""

    alive: str = Field(None, title="Message")


class ReponseReadiness(BaseModel):
    """Return object for readiness probe"""

    ready: str = Field(None, title="Message")


class ReponseInformation(BaseModel):
    """Return object for info resource"""

    application: str = Field(None, title="Application name")
    version: str = Field(None, title="Application version")
    model_name: str = Field(None, title="Model name")
    model_version: str = Field(None, title="Model version")
