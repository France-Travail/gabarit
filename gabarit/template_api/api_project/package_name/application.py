from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.event_handlers import start_app_handler, stop_app_handler
from .routers import main_routeur


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
