from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.registry import ModelRegistry


@pytest.fixture(autouse=True)
def trusted_history_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODIN_TRUSTED_HISTORY_SHARED_SECRET", "test-history-secret")


@pytest.fixture(scope="session")
def client() -> TestClient:
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ODIN_TRUSTED_HISTORY_SHARED_SECRET", "test-history-secret")
        with TestClient(app) as c:
            assert c.get("/ready").json()["status"] == "ready"
            yield c


@pytest.fixture(scope="session")
def registry() -> ModelRegistry:
    return app.state.registry


def trusted_post(client: TestClient, path: str, payload: dict, user_id: str = "test-user-1"):
    body = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(
        b"test-history-secret", b":".join([b"odin_api", user_id.encode(), body]), hashlib.sha256
    ).hexdigest()
    return client.post(
        path,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Odin-User-Id": user_id,
            "X-Odin-History-Source": "odin_api",
            "X-Odin-History-Signature": signature,
        },
    )
