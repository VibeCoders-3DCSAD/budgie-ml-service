from __future__ import annotations

import pytest

from app.schemas.classification import ClassificationV2Request
from app.services.classification_service import classify
from tests.conftest import trusted_post


def _payload(**overrides):
    payload = {
        "request_id": "classification-1",
        "user_id": "test-user-1",
        "classification_as_of": "2026-01-01",
        "liquid_emergency_savings": 3_000,
        "average_monthly_recurring_essential_expenses": 1_000,
        "monthly_disposable_income": 10_000,
        "required_monthly_debt_payments": 4_000,
        "monthly_basic_living_costs": 6_000,
        "credit_card_accounts": [],
    }
    payload.update(overrides)
    return payload


def test_classification_applies_documented_boundaries(client):
    response = trusted_post(client, "/api/v1/classification/v2", _payload())
    assert response.status_code == 200
    body = response.json()
    classification = body["classification"]
    assert classification["emergency_savings"]["label"] == "EMERGENCY_FUND_ADEQUATE"
    assert classification["debt_burden"]["label"] == "NOT_FINANCIALLY_OVERBURDENED"
    assert classification["financial_margin"]["label"] == "POSITIVE_FINANCIAL_MARGIN"
    assert body["metadata"]["model_version"] == "financial_rules_v1"
    assert body["metadata"]["strategy_used"] == "rule_based"


def test_classification_handles_thresholds_and_missing_data(client):
    response = trusted_post(
        client,
        "/api/v1/classification/v2",
        _payload(
            liquid_emergency_savings=2_990,
            required_monthly_debt_payments=4_001,
            monthly_basic_living_costs=6_000,
        ),
    )
    assert response.status_code == 200
    classification = response.json()["classification"]
    assert classification["emergency_savings"]["label"] == "EMERGENCY_FUND_INADEQUATE"
    assert classification["debt_burden"]["label"] == "FINANCIALLY_OVERBURDENED"
    assert classification["financial_margin"]["label"] == "NEGATIVE_FINANCIAL_MARGIN"

    response = trusted_post(
        client,
        "/api/v1/classification/v2",
        _payload(monthly_disposable_income=0),
    )
    assert response.status_code == 200
    classification = response.json()["classification"]
    assert classification["debt_burden"]["status"] == "INSUFFICIENT_DATA"
    assert classification["financial_margin"]["status"] == "INSUFFICIENT_DATA"


def test_classification_requires_twelve_consecutive_card_cycles(client):
    cycles = [
        {"completed_on": f"2025-{month:02d}-28", "revolving_balance": 0} for month in range(1, 13)
    ]
    response = trusted_post(
        client,
        "/api/v1/classification/v2",
        _payload(credit_card_accounts=[{"account_id": "card-1", "cycles": cycles}]),
    )
    assert response.status_code == 200
    assessment = response.json()["classification"]["credit_card_behavior"][0]
    assert assessment["label"] == "TRANSACTOR"

    response = trusted_post(
        client,
        "/api/v1/classification/v2",
        _payload(credit_card_accounts=[{"account_id": "card-1", "cycles": cycles[:-1]}]),
    )
    assert response.status_code == 200
    assessment = response.json()["classification"]["credit_card_behavior"][0]
    assert assessment["status"] == "INSUFFICIENT_HISTORY"


def test_classification_rejects_a_stale_card_history_window(client):
    cycles = [
        {"completed_on": f"2024-{month:02d}-28", "revolving_balance": 0} for month in range(1, 13)
    ]
    response = trusted_post(
        client,
        "/api/v1/classification/v2",
        _payload(credit_card_accounts=[{"account_id": "card-1", "cycles": cycles}]),
    )
    assert response.status_code == 200
    assessment = response.json()["classification"]["credit_card_behavior"][0]
    assert assessment["status"] == "INSUFFICIENT_HISTORY"


def test_classification_scores_self_reported_card_behavior(client):
    for months, label in ((0, "TRANSACTOR"), (5, "LIGHT_REVOLVER"), (12, "HEAVY_REVOLVER")):
        response = trusted_post(
            client,
            "/api/v1/classification/v2",
            _payload(self_reported_revolving_months=months),
        )
        assert response.status_code == 200
        [assessment] = response.json()["classification"]["credit_card_behavior"]
        assert assessment["source"] == "self_reported"
        assert assessment["status"] == "SUCCESS"
        assert assessment["label"] == label
        assert assessment["value"] == months


def test_classification_prefers_observed_card_history_over_self_report(client):
    cycles = [
        {"completed_on": f"2025-{month:02d}-28", "revolving_balance": 0} for month in range(1, 13)
    ]
    response = trusted_post(
        client,
        "/api/v1/classification/v2",
        _payload(
            credit_card_accounts=[{"account_id": "card-1", "cycles": cycles}],
            self_reported_revolving_months=12,
        ),
    )
    [assessment] = response.json()["classification"]["credit_card_behavior"]
    assert assessment["source"] == "observed_history"
    assert assessment["label"] == "TRANSACTOR"


def test_classification_rejects_out_of_range_self_report(client):
    response = trusted_post(
        client,
        "/api/v1/classification/v2",
        _payload(self_reported_revolving_months=13),
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("duration", "label"),
    [
        ("no_credit_card", "NO_CREDIT_CARD"),
        ("always_before_due", "TRANSACTOR"),
        ("1_to_11_months", "LIGHT_REVOLVER"),
        ("over_a_year", "HEAVY_REVOLVER"),
    ],
)
def test_classification_preserves_reported_duration_without_exact_months(duration, label):
    request = ClassificationV2Request.model_validate(
        _payload(self_reported_revolving_range=duration)
    )
    [assessment] = classify(request).credit_card_behavior
    assert assessment.label == label
    assert assessment.value is None
    assert assessment.reported_duration == duration
    assert assessment.source == "self_reported"


def test_classification_observed_history_overrides_reported_duration():
    cycles = [
        {"completed_on": f"2025-{month:02d}-28", "revolving_balance": 0} for month in range(1, 13)
    ]
    request = ClassificationV2Request.model_validate(
        _payload(
            credit_card_accounts=[{"account_id": "card-1", "cycles": cycles}],
            self_reported_revolving_range="over_a_year",
        )
    )
    [assessment] = classify(request).credit_card_behavior
    assert assessment.label == "TRANSACTOR"
    assert assessment.source == "observed_history"
    assert assessment.reported_duration is None
