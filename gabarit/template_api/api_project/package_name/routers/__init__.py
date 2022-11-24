"""Main router of the REST API"""
from fastapi import APIRouter

from . import functional, technical

# Create the main router
main_routeur = APIRouter()

# Add the technical and functional routers
main_routeur.include_router(functional.router, tags=["functional"])
main_routeur.include_router(technical.router, tags=["technical"])

__all__ = ["main_routeur"]
