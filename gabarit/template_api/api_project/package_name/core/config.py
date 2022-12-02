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


"""Config global settings

This module handle global app configuration
"""


import pkg_resources  # type: ignore
from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "{{package_name}}"
    app_version: str = pkg_resources.get_distribution("{{package_name}}").version
    api_entrypoint: str = "/{{package_name}}/rs/v1"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
