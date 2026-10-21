from __future__ import annotations

import shutil
from datetime import date

import pytest

from app.core.config import MODELS_DIR
from app.models.loader import ModelLoader
from app.models.registry import ModelRegistry
from tests.conftest import trusted_post
from tests.test_budget import _budget_payload
from tests.test_classification import _payload
from tests.test_forecast import _tenth_of_month_before


def forecast_payload(categories=None):
    categories = categories or ["food"]
    return {
        "user_id": "test-user-1",
        "categories": categories,
        "historical_transactions": [
            {
                "date": _tenth_of_month_before(1),
                "amount": 1200,
                "category": category,
                "transaction_type": "expense",
            }
            for category in categories
        ],
    }


def test_health_and_metrics_report_active_modules(client):
    assert client.get("/health").json() == {"service": "budgie-ml-service", "status": "ok"}
    assert client.get("/ready").status_code == 200
    modules = client.get("/metrics").json()["modules"]
    assert set(modules) == {"classification", "hfce_sarima_v4", "budget"}
    assert modules["classification"]["model_version"] == "financial_rules_v1"


def test_ready_rejects_missing_signing_configuration(client, monkeypatch):
    monkeypatch.delenv("ODIN_TRUSTED_HISTORY_SHARED_SECRET")
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert trusted_post(client, "/api/v1/classification/v2", _payload()).status_code == 503


@pytest.mark.parametrize("module", ["hfce_sarima", "budget"])
def test_ready_requires_each_serving_module(client, monkeypatch, module):
    monkeypatch.setattr(client.app.state.registry, module, None)
    assert client.get("/ready").status_code == 503


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/v1/classification/v2", _payload()),
        ("/api/v1/forecast/predict", forecast_payload()),
        ("/api/v1/budget/recommend", _budget_payload()),
        ("/api/v1/budget/recommend/batch", {"requests": [_budget_payload()]}),
    ],
)
def test_all_inference_routes_require_valid_signature_and_owner(client, path, payload):
    assert client.post(path, json=payload).status_code == 401
    assert trusted_post(client, path, payload, "another-user").status_code == 401
    assert (
        client.post(
            path,
            json=payload,
            headers={
                "X-Odin-User-Id": "test-user-1",
                "X-Odin-History-Source": "odin_api",
                "X-Odin-History-Signature": "invalid",
            },
        ).status_code
        == 401
    )


def test_batch_rejects_mixed_ownership_before_computation(client, monkeypatch):
    from app.services import budget_service

    def fail_if_called(*args):
        pytest.fail("mixed-ownership batch must not execute")

    monkeypatch.setattr(budget_service, "optimize", fail_if_called)
    other = _budget_payload() | {"user_id": "another-user"}
    response = trusted_post(
        client, "/api/v1/budget/recommend/batch", {"requests": [_budget_payload(), other]}
    )
    assert response.status_code == 401


def test_real_artifacts_forecast_all_twelve_categories(client):
    categories = list(client.app.state.registry.hfce_sarima.model)
    response = trusted_post(client, "/api/v1/forecast/predict", forecast_payload(categories))
    assert response.status_code == 200
    points = response.json()["forecasts"]
    assert len(points) == 12
    assert all(point["amount"] > 0 for point in points)
    assert all(point["user_baseline_amount"] == 1200 for point in points)


def test_corrupt_artifact_is_rejected_before_deserialization(tmp_path, monkeypatch):
    root = tmp_path / "models"
    shutil.copytree(MODELS_DIR, root)
    (root / "hfce_sarima_v4/food.joblib").write_bytes(b"corrupt release")
    loader = ModelLoader(root)

    def fail_if_called(*args):
        pytest.fail("checksum verification must precede deserialization")

    monkeypatch.setattr(loader, "load_joblib", fail_if_called)
    with pytest.raises(ValueError, match="checksum mismatch"):
        ModelRegistry(loader).load_all()


def test_body_size_limit(client):
    response = client.post("/api/v1/classification/v2", content=b"x" * (4 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_forecast_rejects_excessive_horizon(client):
    payload = forecast_payload()
    payload["target_month"] = f"{date.today().year + 2}-01"
    assert trusted_post(client, "/api/v1/forecast/predict", payload).status_code == 422


def test_forecast_rejects_invalid_calendar_date(client):
    payload = forecast_payload()
    payload["historical_transactions"][0]["date"] = "2026-02-31"
    assert trusted_post(client, "/api/v1/forecast/predict", payload).status_code == 422


def test_forecast_rejects_oversized_transaction_list(client):
    payload = forecast_payload()
    payload["historical_transactions"] *= 10_001
    assert trusted_post(client, "/api/v1/forecast/predict", payload).status_code == 422


def test_budget_rejects_oversized_batch(client):
    response = trusted_post(
        client, "/api/v1/budget/recommend/batch", {"requests": [_budget_payload()] * 11}
    )
    assert response.status_code == 422


def test_health_stays_responsive_during_forecast(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from app.services import forecast_service

    entered = Event()
    release = Event()
    original = forecast_service.forecast

    def slow_forecast(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test forecast was not released")
        return original(*args, **kwargs)

    monkeypatch.setattr(forecast_service, "forecast", slow_forecast)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            trusted_post, client, "/api/v1/forecast/predict", forecast_payload()
        )
        try:
            assert entered.wait(2)
            assert client.get("/health").status_code == 200
        finally:
            release.set()
        assert future.result(timeout=3).status_code == 200


@pytest.mark.parametrize("field", ["monthly_disposable_income", "liquid_emergency_savings"])
def test_classification_rejects_nonfinite_values(field):
    from pydantic import ValidationError

    from app.schemas.classification import ClassificationV2Request

    with pytest.raises(ValidationError):
        ClassificationV2Request(**(_payload() | {field: float("inf")}))
