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
