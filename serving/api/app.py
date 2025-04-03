import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routers import internal, predict
from .utils import logs
from .utils.models import model_container

PROJECT_NAME = 'BoardGameGeek Playground'
TITLE = "BGG dataset API"
DESCRIPTION = f"Definition of the {PROJECT_NAME} API."
__version__ = "0.1.0"

log = logs.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = os.getenv(key='MODEL_NAME', default='bert-base-uncased')
    num_labels = os.getenv(key='NUM_LABELS', default=2)

    try:
        log.info("Loading tokenizer and model...")
        model_container.load_model(model_name, num_labels)

        yield
    except Exception as e:
        log.error(f"Model loading error: {str(e)}")
        raise e
    finally:
        model_container.cleanup()


def setup() -> FastAPI:
    """Set up and configure app."""
    log.info("Set up and configure app")
    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan
    )

    app.include_router(internal.healthcheck_router)
    app.include_router(predict.predict_router)

    return app
