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


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette_prometheus import metrics, PrometheusMiddleware

from .core.config import settings
from .routers import main_routeur
from .core.event_handlers import start_app_handler, stop_app_handler


def declare_application() -> FastAPI:
    """Create the FastAPI application

    See https://fastapi.tiangolo.com/tutorial/first-steps/ to learn how to
    customize your FastAPI application
    """
    app = FastAPI(
        title=f"REST API form {settings.app_name}",
        description=f"Use {settings.app_name} thanks to FastAPI",
    )

    # Load the model on startup
    app.add_event_handler("startup", start_app_handler(app))
    app.add_event_handler("shutdown", stop_app_handler(app))

    # Add PrometheusMiddleware
    app.add_middleware(PrometheusMiddleware)
    app.add_route("/metrics", metrics)

    # CORS middleware that allows all origins to avoid CORS problems
    # see https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #
    app.include_router(main_routeur, prefix=settings.api_entrypoint)

    return app


app = declare_application()
