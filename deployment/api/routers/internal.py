from fastapi import APIRouter

from ..utils import logs

log = logs.get_logger()

healthcheck_router = APIRouter(include_in_schema=True, dependencies=[], tags=['internal'])

@healthcheck_router.get("/ping")
def healthcheck() -> str:
    """Expose healthcheck endpoint"""
    return "pong"