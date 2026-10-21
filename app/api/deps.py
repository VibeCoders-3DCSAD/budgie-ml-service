from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

from anyio import CapacityLimiter
from fastapi import Header, HTTPException, Request

from app.core.config import TRUSTED_HISTORY_SHARED_SECRET_ENV
from app.models.registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


@dataclass(frozen=True)
class TrustedHistoryContext:
    user_id: str
    source: str


async def get_trusted_history_context(
    request: Request,
    x_odin_user_id: str | None = Header(default=None),
    x_odin_history_source: str | None = Header(default=None),
    x_odin_history_signature: str | None = Header(default=None),
) -> TrustedHistoryContext:
    """Verify the gateway-signed user-scoped history payload before scoring it."""
    secret = os.getenv(TRUSTED_HISTORY_SHARED_SECRET_ENV)
    if not secret:
        raise HTTPException(status_code=503, detail="trusted history source is not configured")
    if not x_odin_user_id or x_odin_history_source != "odin_api" or not x_odin_history_signature:
        raise HTTPException(status_code=401, detail="missing trusted history provenance")
    payload = b":".join(
        [x_odin_history_source.encode(), x_odin_user_id.encode(), await request.body()]
    )
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(x_odin_history_signature, expected):
        raise HTTPException(status_code=401, detail="invalid trusted history signature")
    return TrustedHistoryContext(user_id=x_odin_user_id, source=x_odin_history_source)


def require_trusted_user(context: TrustedHistoryContext, user_id: str) -> None:
    """Ensure each scored payload belongs to the gateway-authenticated user."""
    if not hmac.compare_digest(context.user_id, user_id):
        raise HTTPException(status_code=401, detail="payload user does not match trusted identity")


def get_compute_limiter(request: Request) -> CapacityLimiter:
    return request.app.state.compute_limiter
