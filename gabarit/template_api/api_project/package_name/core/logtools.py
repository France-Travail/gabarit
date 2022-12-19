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


"""Logs utilities

This module is used to define log pattern, add log filters, etc.
"""


from .config import settings


def get_pattern_log():
    return (
        "{"
        '"date": "%(asctime)s", '
        '"level": "%(levelname)s", '
        '"message": "%(message)s", '
        f'"version": "{settings.app_version}", '
        '"function": "File %(pathname)s, line %(lineno)d, in %(funcName)s", '
        '"logger": "%(name)s"'
        "}"
    )
