from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from anyio import CapacityLimiter
from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.deps import (
    TrustedHistoryContext,
    get_compute_limiter,
    get_registry,
    get_trusted_history_context,
    require_trusted_user,
)
from app.models.registry import ModelRegistry
from app.schemas.common import ApiMetadata, ModuleStatus
from app.schemas.forecast import (
    ExpenseForecastSnapshot,
    ForecastedExpense,
    ForecastQuality,
    ForecastRequest,
    ForecastResponse,
)
from app.services import forecast_service

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])


@router.post("/predict", response_model=ForecastResponse)
async def predict(
    request: ForecastRequest,
    registry: ModelRegistry = Depends(get_registry),
    limiter: CapacityLimiter = Depends(get_compute_limiter),
    history: TrustedHistoryContext = Depends(get_trusted_history_context),
) -> ForecastResponse:
    require_trusted_user(history, request.user_id)
    if registry.hfce_sarima is None:
        raise HTTPException(status_code=503, detail="HFCE forecast is unavailable")

    start = time.perf_counter()
    try:
        async with limiter:
            points = await run_in_threadpool(
                forecast_service.forecast,
                registry.hfce_sarima,
                [transaction.model_dump() for transaction in request.historical_transactions],
                request.categories,
                user_id=request.user_id,
                target_month=request.target_month,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model_version = registry.hfce_sarima.model_id
    snapshot = ExpenseForecastSnapshot(
        month=points[0].month,
        total_amount=round(sum(point.amount for point in points), 2),
        category_forecasts=[
            ForecastedExpense(category=point.category, amount=point.amount) for point in points
        ],
        model_version=model_version,
        generated_at=datetime.now(UTC),
        # Under six months of history the amounts are provisional, not a reliable personal forecast.
        quality=(
            ForecastQuality.INSUFFICIENT_DATA
            if any(point.baseline_method == forecast_service.PROVISIONAL for point in points)
            else ForecastQuality.PERSONALIZED
        ),
    )
    return ForecastResponse(
        response_id=str(uuid.uuid4()),
        request_id=request.user_id,
        user_id=request.user_id,
        forecasts=points,
        snapshot=snapshot,
        model_version=model_version,
        status=ModuleStatus.SUCCESS,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version=model_version,
            strategy_used=f"{points[0].baseline_method}_times_hfce_category_multiplier",
        ),
    )
