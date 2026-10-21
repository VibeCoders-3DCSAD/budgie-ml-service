from __future__ import annotations

import logging
import time
import uuid

from anyio import CapacityLimiter
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import (
    TrustedHistoryContext,
    get_compute_limiter,
    get_registry,
    get_trusted_history_context,
    require_trusted_user,
)
from app.models.registry import ModelRegistry
from app.schemas.budget import BudgetRequest, BudgetResponse
from app.schemas.common import ApiMetadata, ModuleStatus
from app.services import budget_service

router = APIRouter(prefix="/api/v1/budget", tags=["budget"])
logger = logging.getLogger(__name__)


def _run(request: BudgetRequest, registry: ModelRegistry) -> BudgetResponse:
    start = time.perf_counter()
    try:
        recommendation, explanations = budget_service.optimize(request)
    except ValueError as exc:
        logger.warning(
            "Budget optimization rejected request request_id=%s user_id=%s reason=%s",
            request.request_id,
            request.user_id,
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model_version = registry.budget.model_id if registry.budget is not None else "n/a"
    return BudgetResponse(
        response_id=str(uuid.uuid4()),
        request_id=request.request_id,
        user_id=request.user_id,
        recommendation=recommendation,
        explanations=explanations if request.include_reasoning else None,
        status=ModuleStatus.SUCCESS,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version=model_version,
            strategy_used="hierarchical_lp_v2",
        ),
    )


class BatchRequest(BaseModel):
    requests: list[BudgetRequest] = Field(min_length=1, max_length=10)


class BatchResponse(BaseModel):
    results: list[BudgetResponse]


@router.post("/recommend", response_model=BudgetResponse)
async def recommend(
    request: BudgetRequest,
    registry: ModelRegistry = Depends(get_registry),
    limiter: CapacityLimiter = Depends(get_compute_limiter),
    history: TrustedHistoryContext = Depends(get_trusted_history_context),
) -> BudgetResponse:
    require_trusted_user(history, request.user_id)
    async with limiter:
        return await run_in_threadpool(_run, request, registry)


@router.post("/recommend/batch", response_model=BatchResponse)
async def recommend_batch(
    batch: BatchRequest,
    registry: ModelRegistry = Depends(get_registry),
    limiter: CapacityLimiter = Depends(get_compute_limiter),
    history: TrustedHistoryContext = Depends(get_trusted_history_context),
) -> BatchResponse:
    for request in batch.requests:
        require_trusted_user(history, request.user_id)
    async with limiter:
        results = await run_in_threadpool(lambda: [_run(req, registry) for req in batch.requests])
    return BatchResponse(results=results)
