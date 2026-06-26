import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .exceptions import ModelNotLoadedError, ModelOutputError
from .model_service import model_service
from .routers import health, model_info, predict

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(model_info.router)


@app.exception_handler(ModelNotLoadedError)
def handle_model_not_loaded(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": "model_not_loaded", "detail": str(exc)},
    )


@app.exception_handler(ModelOutputError)
def handle_model_output_error(request: Request, exc: ModelOutputError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "model_output_mismatch", "detail": str(exc)},
    )


@app.get("/", tags=["root"])
def root() -> dict:
    return {
        "message": "Predictive Maintenance API",
        "docs": "/docs",
        "health": "/health",
    }
