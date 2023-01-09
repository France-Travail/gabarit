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


import json
from typing import Union

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from ..model.model_base import Model
from .schemas.functional import NumpyJSONResponse

# Functional router
router = APIRouter()


# This function is async since it uses starlette Request
# There is no return type annotation because starting from FastAPI 0.89, type annotations are
# interpreted as response_model and response_model must be valid pydantic. Since we use here a
# startlette.Response we remove return annotation (cf. https://fastapi.tiangolo.com/release-notes/#0890)
@router.post("/predict")
async def predict(request: Request):
    """Predict route that exposes your model

    This function is using starlette Request object instead of pydantic since we can not
    know what data your model is expecting.
    See https://fastapi.tiangolo.com/advanced/using-request-directly/ for more infos.

    We also use a custom starlette JSONResponse class (NumpyJSONResponse)
    instead of pydantic for the same reasons

    For a cleaner way to handle requests and reponses you should use pydantic as stated in FastAPI
    doc : https://fastapi.tiangolo.com/tutorial/body/#create-your-data-model

    You can use routes from {{package_name}}.routers.technical as examples of how to create requests and
    responses schemas thanks to pydantic or have a look at the FastAPI documentation :
    https://fastapi.tiangolo.com/tutorial/response-model/
    """
    model: Model = request.app.state.model

    body = await request.body()
    body = json.loads(body) if body else {}

    prediction = model.predict(**body)

    return NumpyJSONResponse(prediction)

# This function is async since it uses starlette Request
# There is no return type annotation because starting from FastAPI 0.89, type annotations are
# interpreted as response_model and response_model must be valid pydantic. Since we use here a
# startlette.Response we remove return annotation (cf. https://fastapi.tiangolo.com/release-notes/#0890)
@router.post("/explain")
async def explain(request: Request):
    """Explain route that expose a model explainer in charge of model explicability

    This function is using starlette Request object instead of pydantic since we can not
    know what data your model is expecting.
    See https://fastapi.tiangolo.com/advanced/using-request-directly/ for more infos.

    We also use the custom starlette JSONResponse class (PredictionResponse)
    instead of pydantic for the same reasons

    For a cleaner way to handle requests and reponses you should use pydantic as stated in FastAPI
    doc : https://fastapi.tiangolo.com/tutorial/body/#create-your-data-model

    You can use routes from example_api_num.routers.technical as examples of how to create requests and
    responses schemas thanks to pydantic or have a look at the FastAPI documentation :
    https://fastapi.tiangolo.com/tutorial/response-model/

    If there is not explainer or the explainer does not implement explain_as_json or explain_as_html
    we return a 501 HTTP error : https://developer.mozilla.org/fr/docs/Web/HTTP/Status/501
    """
    model: Model = request.app.state.model

    body = await request.body()
    body = json.loads(body) if body else {}

    # JSON repsonse (when Accept: application/json in the request)
    if request.headers.get("Accept") == "application/json":
        try:
            explanation_json = model.explain_as_json(**body)

        except (AttributeError, NotImplementedError):
            error_msg = {
                "error": {
                    "code": 501,
                    "message": "No explainer capable of handling explicability"
                }
            }
            return Response(
                content=json.dumps(error_msg),
                status_code=501,
                media_type='application/json',
            )
        else:
            return NumpyJSONResponse(explanation_json)

    # HTML repsonse (otherwise)
    else:
        try:
            explanation_html = model.explain_as_html(**body)

        except (AttributeError, NotImplementedError):
            return Response(
                content="No explainer capable of handling explicability",
                status_code=501,
                media_type='text/plain',
            )
        else:
            return HTMLResponse(explanation_html)
