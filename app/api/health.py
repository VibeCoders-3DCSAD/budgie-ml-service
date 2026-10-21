from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import get_registry
from app.core.config import SERVICE_NAME, SERVICE_VERSION, TRUSTED_HISTORY_SHARED_SECRET_ENV
from app.models.registry import ModelRegistry
from app.services.classification_service import RULE_SET_VERSION

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"service": SERVICE_NAME, "status": "ok"}


@router.get("/ready")
async def ready(registry: ModelRegistry = Depends(get_registry)) -> JSONResponse:
    ready = registry.is_ready and bool(os.getenv(TRUSTED_HISTORY_SHARED_SECRET_ENV))
    return JSONResponse(
        {"status": "ready" if ready else "not ready", "models": registry.loaded_modules},
        status_code=200 if ready else 503,
    )


@router.get("/metrics")
async def metrics(registry: ModelRegistry = Depends(get_registry)) -> dict:
    modules = {"classification": {"model_version": RULE_SET_VERSION}}
    for name, module in (("hfce_sarima_v4", registry.hfce_sarima), ("budget", registry.budget)):
        if module is not None:
            modules[name] = {"model_version": module.model_id}
    return {"service": SERVICE_NAME, "version": SERVICE_VERSION, "modules": modules}
