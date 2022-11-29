import json
from typing import Union

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import HTMLResponse

from ..model.model_base import Model
from .schemas.functional import NumpyJSONResponse

# Functional router
router = APIRouter()


# This function is async since it uses starlette Request
@router.post("/predict")
async def predict(request: Request) -> NumpyJSONResponse:
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

@router.post("/explain")
async def explain(request: Request) -> Union[HTMLResponse, NumpyJSONResponse]:
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
    """
    model: Model = request.app.state.model

    body = await request.body()
    body = json.loads(body) if body else {}
    
    if request.headers.get("Accept") == "application/json":
        return NumpyJSONResponse(model.explain_as_json(**body))
    else:
        return HTMLResponse(model.explain_as_html(**body))