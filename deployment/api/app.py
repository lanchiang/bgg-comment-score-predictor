import importlib
from fastapi import FastAPI

from .routers import internal, predict

from .utils import logs

PROJECT_NAME = 'voize-demo'
TITLE = f"[{PROJECT_NAME.upper()}] API"
DESCRIPTION = f"Definition of the {PROJECT_NAME} API."

# __version__ = importlib.metadata.version(PROJECT_NAME)
__version__ = "0.1.0"

log = logs.get_logger()

def setup() -> FastAPI:
    """Set up and configure app."""
    log.info("Set up and configure app")
    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=__version__,
    )

    app.include_router(internal.healthcheck_router)
    app.include_router(predict.predict_router)

    return app
