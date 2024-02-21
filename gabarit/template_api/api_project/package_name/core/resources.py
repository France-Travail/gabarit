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


"""Resources for the FastAPI application

This module define resources that need to be instantiated at startup in a global
variable resources that can be used in routes.

This is the way your machine learning models can be loaded in memory at startup
so they can handle requests.

To use your own model instead of the base model, create a module in {{package_name}}.model
such as model_awesome.py and import it as Model instead of the one used here.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

{%- if gabarit_package_spec %}
from ..model.model_gabarit import ModelGabarit as Model
{%- else %}
from ..model.model_base import Model
{%- endif %}

logger = logging.getLogger(__name__)

RESOURCES = {}
RESOURCE_MODEL = "model"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model = Model()
    model.loading()
    logger.info("Model loaded")

    RESOURCES[RESOURCE_MODEL] = model
    yield

    # Clean up the ML models and release the resources
    RESOURCES.clear()

