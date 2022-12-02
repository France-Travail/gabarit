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


"""Startup and Stop handlers for FastAPI application

This module define event handlers and n particular startup and stop handlers that
are used to instantiate your model when the API first start.

To use your own model instead of the base model, create a module in {{package_name}}.model
such as model_awesome.py and import it as Model instead of the one used here.
"""


import logging
from typing import Callable
from fastapi import FastAPI

{%- if gabarit_package_spec %}
from ..model.model_gabarit import ModelGabarit as Model
{%- else %}
from ..model.model_base import Model
{%- endif %}

logger = logging.getLogger(__name__)


def _startup_model(app: FastAPI) -> None:
    """Create and Load model"""
    model = Model()
    model.loading()
    app.state.model = model


def _shutdown_model(app: FastAPI) -> None:
    """Clean the model state"""
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    """Startup handler: invoke init actions"""

    def startup() -> None:
        logger.info("Startup Handler: Load model.")
        _startup_model(app)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    """Stop handler: invoke shutdown actions"""

    def shutdown() -> None:
        logger.info("Shutdown handler : Clean model.")
        _shutdown_model(app)

    return shutdown
