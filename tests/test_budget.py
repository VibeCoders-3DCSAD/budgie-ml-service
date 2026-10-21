from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.conftest import trusted_post


def _budget_payload():
    return {
        "request_id": "req-budget-1",
        "user_id": "test-user-1",
        "available_funds": 25000.0,
        "period": {"start": "2024-01-01", "end": "2024-01-31"},
        "categories": [
            {
                "category_id": "essentials_food",
                "priority_group": "ESSENTIAL",
                "restriction_level": "MINIMUM",
                "floor": 6000.0,
                "ceiling": 9000.0,
                "priority_weight": 0.4,
                "current_spend": 6800.0,
            },
            {
                "category_id": "essentials_rent",
                "priority_group": "ESSENTIAL",
                "restriction_level": "FIXED",
                "floor": 8000.0,
                "ceiling": 8000.0,
                "priority_weight": 0.3,
                "current_spend": 8000.0,
            },
            {
                "category_id": "discretionary_leisure",
                "priority_group": "NON_ESSENTIAL",
                "restriction_level": "FLEXIBLE",
                "floor": 0.0,
                "ceiling": 5000.0,
                "priority_weight": 0.2,
                "current_spend": 2000.0,
            },
            {
                "category_id": "savings",
                "priority_group": "NON_ESSENTIAL",
                "restriction_level": "FLEXIBLE",
                "floor": 0.0,
                "ceiling": 6000.0,
                "priority_weight": 0.1,
                "current_spend": 0.0,
            },
        ],
        "target_ratios": {
            "essentials_food": 0.28,
            "essentials_rent": 0.32,
            "discretionary_leisure": 0.15,
            "savings": 0.25,
        },
    }


def test_budget_recommend(client):
    resp = trusted_post(client, "/api/v1/budget/recommend", _budget_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == "req-budget-1"
    rec = body["recommendation"]
    allocations = {a["category_id"]: a["amount"] for a in rec["allocations"]}
    total = sum(allocations.values())
    assert abs(total - 24750.0) < 1.0
    assert rec["unallocated_surplus"] == 250.0
    assert allocations["essentials_rent"] == 8000.0  # FIXED
    assert rec["utilization_rate"] == 0.99
    assert rec["feasibility"] == "FEASIBLE"
    assert body["metadata"]["model_version"] == "budget-scipy_linprog"
    assert body["metadata"]["strategy_used"] == "hierarchical_lp_v2"


def test_budget_keeps_forecast_surplus_out_of_expense_categories(client):
    payload = _budget_payload()
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 15_000.0,
        "category_forecasts": [
            {"category": "essentials_food", "amount": 7_000.0},
            {"category": "essentials_rent", "amount": 8_000.0},
            {"category": "discretionary_leisure", "amount": 0.0},
            {"category": "savings", "amount": 0.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    allocations = {item["category_id"]: item["amount"] for item in recommendation["allocations"]}
    assert allocations == {
        "essentials_food": 7_000.0,
        "essentials_rent": 8_000.0,
        "discretionary_leisure": 0.0,
        "savings": 0.0,
    }
    assert recommendation["unallocated_surplus"] == 10_000.0


def test_budget_allocates_scarce_funds_across_essential_targets_by_weight(client):
    payload = _budget_payload()
    payload["available_funds"] = 10_000.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 0.8,
        },
        {
            "category_id": "housing",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 0.2,
        },
        {
            "category_id": "recreation",
            "priority_group": "NON_ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 18_000.0,
        "category_forecasts": [
            {"category": "food", "amount": 8_000.0},
            {"category": "housing", "amount": 6_000.0},
            {"category": "recreation", "amount": 4_000.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations == {"food": 8_000.0, "housing": 2_000.0, "recreation": 0.0}


@pytest.mark.parametrize("priority_group", ["ESSENTIAL", "NON_ESSENTIAL"])
def test_budget_does_not_starve_lower_weight_category(client, priority_group):
    payload = _budget_payload()
    payload["available_funds"] = 5_000.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": priority_group,
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 0.8,
        },
        {
            "category_id": "housing",
            "priority_group": priority_group,
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": 0.2,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 14_000.0,
        "category_forecasts": [
            {"category": "food", "amount": 8_000.0},
            {"category": "housing", "amount": 6_000.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations == pytest.approx({"food": 4210.526315789473, "housing": 789.4736842105264})


@pytest.mark.parametrize(
    ("funds", "expected"),
    [
        (1_000.0, {"food": 1_000.0, "leisure": 0.0, "savings": 0.0}),
        (2_000.0, {"food": 2_000.0, "leisure": 0.0, "savings": 0.0}),
        (
            7_000.0,
            {"food": 2_000.0, "leisure": 4210.526315789473, "savings": 789.4736842105264},
        ),
        (12_000.0, {"food": 2_000.0, "leisure": 8_000.0, "savings": 2_000.0}),
        (17_000.0, {"food": 2_000.0, "leisure": 8_000.0, "savings": 6_000.0}),
    ],
)
def test_budget_protects_non_essential_coverage_after_essentials(client, funds, expected):
    payload = _budget_payload()
    payload["available_funds"] = funds
    payload["categories"] = [
        {
            "category_id": category_id,
            "priority_group": group,
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10_000.0,
            "priority_weight": weight,
        }
        for category_id, group, weight in [
            ("food", "ESSENTIAL", 0.1),
            ("leisure", "NON_ESSENTIAL", 0.8),
            ("savings", "NON_ESSENTIAL", 0.2),
        ]
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 16_000.0,
        "category_forecasts": [
            {"category": "food", "amount": 2_000.0},
            {"category": "leisure", "amount": 8_000.0},
            {"category": "savings", "amount": 6_000.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    allocations = {item["category_id"]: item["amount"] for item in recommendation["allocations"]}
    assert recommendation["feasibility"] == "FEASIBLE"
    assert allocations == pytest.approx(expected)
    assert recommendation["unallocated_surplus"] == pytest.approx(max(funds - 16_000.0, 0.0))


@pytest.mark.parametrize(
    ("funds", "expected"),
    [
        (2_500.0, [1_000.0, 1_000.0, 500.0, 0.0, 0.0]),
        (3_500.0, [2_000.0, 1_000.0, 500.0, 0.0, 0.0]),
        (6_000.0, [2_000.0, 1_000.0, 1_750.0, 1_250.0, 0.0]),
        (12_000.0, [2_000.0, 1_000.0, 2_000.0, 6_000.0, 0.0]),
    ],
)
def test_budget_non_essential_coverage_respects_restrictions_and_caps(client, funds, expected):
    payload = _budget_payload()
    payload["available_funds"] = funds
    payload["categories"] = [
        {
            "category_id": category_id,
            "priority_group": group,
            "restriction_level": restriction,
            "floor": floor,
            "ceiling": ceiling,
            "priority_weight": weight,
        }
        for category_id, group, restriction, floor, ceiling, weight in [
            ("food", "ESSENTIAL", "FLEXIBLE", 0.0, 10_000.0, 0.1),
            ("fixed", "NON_ESSENTIAL", "FIXED", 1_000.0, 1_000.0, 1.0),
            ("minimum", "NON_ESSENTIAL", "MINIMUM", 500.0, 2_000.0, 0.8),
            ("flexible", "NON_ESSENTIAL", "FLEXIBLE", 0.0, 10_000.0, 0.2),
            ("zero_target", "NON_ESSENTIAL", "FLEXIBLE", 0.0, 10_000.0, 1.0),
        ]
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 11_000.0,
        "category_forecasts": [
            {"category": "food", "amount": 2_000.0},
            {"category": "fixed", "amount": 0.0},
            {"category": "minimum", "amount": 3_000.0},
            {"category": "flexible", "amount": 6_000.0},
            {"category": "zero_target", "amount": 0.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    assert recommendation["feasibility"] == "FEASIBLE"
    assert [item["amount"] for item in recommendation["allocations"]] == pytest.approx(expected)
    assert recommendation["unallocated_surplus"] == pytest.approx(max(funds - 11_000.0, 0.0))


def test_budget_preserves_forecast_cap_centavos(client):
    payload = _budget_payload()
    payload["available_funds"] = 10_005.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 20_000.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 8_005.99,
        "category_forecasts": [{"category": "food", "amount": 8_005.99}],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    assert recommendation["allocations"] == [{"category_id": "food", "amount": 8_005.99}]
    assert recommendation["unallocated_surplus"] == 10_005.0 - 8_005.99


def test_budget_keeps_whole_peso_forecast_caps(client):
    payload = _budget_payload()
    payload["available_funds"] = 3_000.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 2_000.0,
            "priority_weight": 1.0,
        },
        {
            "category_id": "housing",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 2_000.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 2_170.0,
        "category_forecasts": [
            {"category": "food", "amount": 1_083.0},
            {"category": "housing", "amount": 1_087.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations == {"food": 1_083.0, "housing": 1_087.0}


def test_budget_preserves_forecast_cap_centavos_across_categories(client):
    payload = _budget_payload()
    payload["available_funds"] = 6_000.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 5_000.0,
            "priority_weight": 1.0,
        },
        {
            "category_id": "housing",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 1_000.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": 5_100.67,
        "category_forecasts": [
            {"category": "food", "amount": 4_460.57},
            {"category": "housing", "amount": 640.10},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations == {"food": 4_460.57, "housing": 640.1}


@pytest.mark.parametrize("target", [1.5, 1.23456789])
def test_budget_preserves_fractional_forecast_caps(client, target):
    payload = _budget_payload()
    payload["available_funds"] = 10.0
    payload["categories"] = [
        {
            "category_id": "food",
            "priority_group": "ESSENTIAL",
            "restriction_level": "FLEXIBLE",
            "floor": 0.0,
            "ceiling": 10.0,
            "priority_weight": 1.0,
        },
    ]
    payload["forecast"] = {
        "month": "2024-01",
        "total_amount": target,
        "category_forecasts": [{"category": "food", "amount": target}],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    assert response.json()["recommendation"]["allocations"] == [
        {"category_id": "food", "amount": target}
    ]
    assert response.json()["recommendation"]["unallocated_surplus"] == 10.0 - target


def test_budget_rejects_bad_restriction(client):
    payload = _budget_payload()
    payload["categories"][0]["ceiling"] = 5000.0  # ceiling below the MINIMUM floor
    resp = trusted_post(client, "/api/v1/budget/recommend", payload)
    assert resp.status_code == 422


def test_budget_rejects_fixed_category_with_unequal_floor_and_ceiling(client):
    payload = _budget_payload()
    payload["categories"][1]["ceiling"] = 9_000.0

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 422
    assert "fixed amount requires equal floor and ceiling" in str(response.json()["detail"])


def test_budget_returns_structured_infeasibility(client):
    payload = _budget_payload()
    payload["available_funds"] = 5_000.0
    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    assert recommendation["feasibility"] == "INFEASIBLE"
    assert recommendation["required_funds"] == 14_000.0
    assert recommendation["available_funds"] == 5_000.0
    assert recommendation["shortfall"] == 9_000.0
    assert "essentials_food" in recommendation["conflicting_categories"]
    assert recommendation["utilization_rate"] == 1.0


def test_budget_uses_configured_fixed_amount_not_current_spend(client):
    payload = _budget_payload()
    payload["categories"][1]["current_spend"] = 1_000.0

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations["essentials_rent"] == 8_000.0


def test_budget_keeps_zero_flexible_allocation_rows(client):
    payload = _budget_payload()
    payload["available_funds"] = 14_000.0

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200
    allocations = {
        allocation["category_id"]: allocation["amount"]
        for allocation in response.json()["recommendation"]["allocations"]
    }
    assert allocations["discretionary_leisure"] == 0.0
    assert allocations["savings"] == 0.0


def test_budget_rejects_forecast_for_another_plan_month(client):
    payload = _budget_payload()
    payload["forecast"] = {
        "month": "2024-02",
        "total_amount": 17_000.0,
        "category_forecasts": [
            {"category": "essentials_food", "amount": 7_000.0},
            {"category": "essentials_rent", "amount": 8_000.0},
            {"category": "discretionary_leisure", "amount": 1_000.0},
            {"category": "savings", "amount": 1_000.0},
        ],
        "model_version": "hfce-sarima-v4",
        "generated_at": datetime.now(UTC).isoformat(),
        "quality": "PERSONALIZED",
    }

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 422
    assert "forecast month must match" in str(response.json()["detail"])


@pytest.mark.parametrize("batch", [False, True])
def test_budget_logs_validation_location_without_request_values(client, caplog, batch):
    payload = _budget_payload()
    payload["categories"][0]["priority_weight"] = "private-invalid-value"
    path = "/api/v1/budget/recommend"
    if batch:
        path += "/batch"
        payload = {"requests": [payload]}

    response = trusted_post(client, path, payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "float_parsing"
    assert "Budget request validation failed" in caplog.text
    assert path in caplog.text
    assert "priority_weight" in caplog.text
    assert "float_parsing" in caplog.text
    assert "private-invalid-value" not in caplog.text
    assert "25000" not in caplog.text
    assert "test-history-secret" not in caplog.text


def test_budget_logs_optimizer_rejection_reason(client, caplog):
    payload = _budget_payload()
    payload["categories"][1]["ceiling"] = 9_000.0

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 422
    assert "Budget optimization rejected request" in caplog.text
    assert "request_id=req-budget-1" in caplog.text
    assert "fixed amount requires equal floor and ceiling" in caplog.text


def test_budget_accepts_full_utilization_with_floating_point_roundoff(client):
    payload = _budget_payload()
    payload["available_funds"] = 0.3
    payload["categories"] = [
        {
            "category_id": category_id,
            "priority_group": "ESSENTIAL",
            "restriction_level": "FIXED",
            "floor": amount,
            "ceiling": amount,
            "priority_weight": 0.5,
        }
        for category_id, amount in [("food", 0.1), ("housing", 0.2)]
    ]

    response = trusted_post(client, "/api/v1/budget/recommend", payload)

    assert response.status_code == 200, response.json()
    recommendation = response.json()["recommendation"]
    assert recommendation["utilization_rate"] == 1.0
    assert recommendation["feasibility"] == "FEASIBLE"
    assert recommendation["unallocated_surplus"] == 0.0
    assert recommendation["allocations"] == [
        {"category_id": "food", "amount": 0.1},
        {"category_id": "housing", "amount": 0.2},
    ]
