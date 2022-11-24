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
