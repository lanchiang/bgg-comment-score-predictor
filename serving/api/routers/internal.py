from fastapi import APIRouter
from pydantic import BaseModel

from ..utils import logs

log = logs.get_logger()

healthcheck_router = APIRouter(include_in_schema=True, dependencies=[], tags=['internal'])


class HealthCheckResponse(BaseModel):
    status: str


@healthcheck_router.get("/ping", response_model=HealthCheckResponse)
async def healthcheck():
    """Expose healthcheck endpoint"""
    return {"status": "pong"}
