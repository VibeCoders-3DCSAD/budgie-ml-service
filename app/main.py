from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from anyio import CapacityLimiter
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.responses import Response

from app.api import budget, classification, forecast, health
from app.core.body_limit import BodyLimitMiddleware
from app.core.config import SERVICE_VERSION
from app.models.registry import ModelRegistry

app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.compute_limiter = CapacityLimiter(1)
    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry
    yield
    app.state.registry = None


app = FastAPI(
    title="Budgie ML Service",
    version=SERVICE_VERSION,
    description=(
        "Inference service for financial classification, expense "
        "forecasting, and budget optimization."
    ),
    lifespan=lifespan,
)

app.add_middleware(BodyLimitMiddleware)

app.include_router(health.router)
app.include_router(classification.router)
app.include_router(forecast.router)
app.include_router(budget.router)


@app.exception_handler(RequestValidationError)
async def log_request_validation_error(request: Request, exc: RequestValidationError) -> Response:
    if request.url.path in {"/api/v1/budget/recommend", "/api/v1/budget/recommend/batch"}:
        # Pydantic errors also contain raw input and context; log only diagnostics.
        errors = [
            {"loc": error["loc"], "type": error["type"], "msg": error["msg"]}
            for error in exc.errors()
        ]
        app_logger.warning(
            "Budget request validation failed path=%s errors=%s", request.url.path, errors
        )
    return await request_validation_exception_handler(request, exc)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Budgie ML service is running.", "version": SERVICE_VERSION}
