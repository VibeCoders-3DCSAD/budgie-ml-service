from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends

from app.api.deps import TrustedHistoryContext, get_trusted_history_context, require_trusted_user
from app.schemas.classification import ClassificationV2Request, ClassificationV2Response
from app.schemas.common import ApiMetadata, ModuleStatus
from app.services import classification_service

router = APIRouter(prefix="/api/v1/classification", tags=["classification"])


@router.post("/v2", response_model=ClassificationV2Response)
async def classify_v2(
    request: ClassificationV2Request,
    history: TrustedHistoryContext = Depends(get_trusted_history_context),
) -> ClassificationV2Response:
    require_trusted_user(history, request.user_id)
    start = time.perf_counter()
    return ClassificationV2Response(
        response_id=str(uuid.uuid4()),
        request_id=request.request_id,
        user_id=request.user_id,
        classification=classification_service.classify(request),
        status=ModuleStatus.SUCCESS,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version=classification_service.RULE_SET_VERSION,
            strategy_used="rule_based",
        ),
    )
