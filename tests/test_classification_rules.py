"""Boundary-value and equivalence-partition cases for the financial classification rules.

Run with pytest, or directly for the results table and metrics:
    .venv/bin/python -m tests.test_classification_rules
"""

from __future__ import annotations

import pytest

from app.schemas.classification import ClassificationV2Request
from app.services.classification_service import classify

BASE = {
    "request_id": "classification-rules",
    "user_id": "test-user-1",
    "classification_as_of": "2026-01-01",
    "liquid_emergency_savings": 3_000,
    "average_monthly_recurring_essential_expenses": 1_000,
    "monthly_disposable_income": 10_000,
    "required_monthly_debt_payments": 4_000,
    "monthly_basic_living_costs": 6_000,
    "credit_card_accounts": [],
}


def _cards(revolving_months: int, cycles: int = 12, year: int = 2025) -> list[dict]:
    return [
        {
            "account_id": "card-1",
            "cycles": [
                {
                    "completed_on": f"{year}-{month:02d}-28",
                    "revolving_balance": 100 if month <= revolving_months else 0,
                }
                for month in range(1, cycles + 1)
            ],
        }
    ]


# (id, technique, rule, overrides, expected label or status)
CASES = [
    # Emergency savings: coverage = savings / essentials, adequate at >= 3.0 months.
    (
        "ES-01",
        "BVA below",
        "emergency_savings",
        {"liquid_emergency_savings": 2_999},
        "EMERGENCY_FUND_INADEQUATE",
    ),
    (
        "ES-02",
        "BVA on",
        "emergency_savings",
        {"liquid_emergency_savings": 3_000},
        "EMERGENCY_FUND_ADEQUATE",
    ),
    (
        "ES-03",
        "BVA above",
        "emergency_savings",
        {"liquid_emergency_savings": 3_001},
        "EMERGENCY_FUND_ADEQUATE",
    ),
    (
        "ES-04",
        "EP valid low",
        "emergency_savings",
        {"liquid_emergency_savings": 0},
        "EMERGENCY_FUND_INADEQUATE",
    ),
    (
        "ES-05",
        "EP valid high",
        "emergency_savings",
        {"liquid_emergency_savings": 50_000},
        "EMERGENCY_FUND_ADEQUATE",
    ),
    (
        "ES-06",
        "EP invalid",
        "emergency_savings",
        {"average_monthly_recurring_essential_expenses": 0},
        "INSUFFICIENT_DATA",
    ),
    (
        "ES-07",
        "EP missing",
        "emergency_savings",
        {"liquid_emergency_savings": None},
        "INSUFFICIENT_DATA",
    ),
    # Debt burden: DSTI = payments / income, overburdened at > 0.40.
    (
        "DB-01",
        "BVA below",
        "debt_burden",
        {"required_monthly_debt_payments": 3_999},
        "NOT_FINANCIALLY_OVERBURDENED",
    ),
    (
        "DB-02",
        "BVA on",
        "debt_burden",
        {"required_monthly_debt_payments": 4_000},
        "NOT_FINANCIALLY_OVERBURDENED",
    ),
    (
        "DB-03",
        "BVA above",
        "debt_burden",
        {"required_monthly_debt_payments": 4_001},
        "FINANCIALLY_OVERBURDENED",
    ),
    (
        "DB-04",
        "EP no debt",
        "debt_burden",
        {"required_monthly_debt_payments": 0},
        "NOT_FINANCIALLY_OVERBURDENED",
    ),
    ("DB-05", "EP invalid", "debt_burden", {"monthly_disposable_income": 0}, "INSUFFICIENT_DATA"),
    (
        "DB-06",
        "EP missing",
        "debt_burden",
        {"required_monthly_debt_payments": None},
        "INSUFFICIENT_DATA",
    ),
    # Financial margin: income - payments - living costs, positive at >= 0.
    (
        "FM-01",
        "BVA below",
        "financial_margin",
        {"monthly_basic_living_costs": 6_001},
        "NEGATIVE_FINANCIAL_MARGIN",
    ),
    (
        "FM-02",
        "BVA on",
        "financial_margin",
        {"monthly_basic_living_costs": 6_000},
        "POSITIVE_FINANCIAL_MARGIN",
    ),
    (
        "FM-03",
        "BVA above",
        "financial_margin",
        {"monthly_basic_living_costs": 5_999},
        "POSITIVE_FINANCIAL_MARGIN",
    ),
    (
        "FM-04",
        "EP missing",
        "financial_margin",
        {"monthly_basic_living_costs": None},
        "INSUFFICIENT_DATA",
    ),
    # Credit card: revolving months across the 12 completed cycles before the as-of date.
    (
        "CC-01",
        "BVA 0 months",
        "credit_card_behavior",
        {"credit_card_accounts": _cards(0)},
        "TRANSACTOR",
    ),
    (
        "CC-02",
        "BVA 1 month",
        "credit_card_behavior",
        {"credit_card_accounts": _cards(1)},
        "LIGHT_REVOLVER",
    ),
    (
        "CC-03",
        "BVA 11 months",
        "credit_card_behavior",
        {"credit_card_accounts": _cards(11)},
        "LIGHT_REVOLVER",
    ),
    (
        "CC-04",
        "BVA 12 months",
        "credit_card_behavior",
        {"credit_card_accounts": _cards(12)},
        "HEAVY_REVOLVER",
    ),
    (
        "CC-05",
        "BVA 11 cycles",
        "credit_card_behavior",
        {"credit_card_accounts": _cards(0, cycles=11)},
        "INSUFFICIENT_HISTORY",
    ),
    (
        "CC-06",
        "EP stale window",
        "credit_card_behavior",
        {"credit_card_accounts": _cards(0, year=2024)},
        "INSUFFICIENT_HISTORY",
    ),
    (
        "CC-07",
        "EP short history with a carried balance",
        "credit_card_behavior",
        {"credit_card_accounts": _cards(3, cycles=11)},
        "REVOLVER",
    ),
    (
        "CC-08",
        "EP stale carried balance",
        "credit_card_behavior",
        {"credit_card_accounts": _cards(3, year=2024)},
        "INSUFFICIENT_HISTORY",
    ),
]

RULE_OUTCOMES = {
    "emergency_savings": {
        "EMERGENCY_FUND_ADEQUATE",
        "EMERGENCY_FUND_INADEQUATE",
        "INSUFFICIENT_DATA",
    },
    "debt_burden": {
        "FINANCIALLY_OVERBURDENED",
        "NOT_FINANCIALLY_OVERBURDENED",
        "INSUFFICIENT_DATA",
    },
    "financial_margin": {
        "POSITIVE_FINANCIAL_MARGIN",
        "NEGATIVE_FINANCIAL_MARGIN",
        "INSUFFICIENT_DATA",
    },
    "credit_card_behavior": {
        "TRANSACTOR",
        "LIGHT_REVOLVER",
        "HEAVY_REVOLVER",
        "REVOLVER",
        "INSUFFICIENT_HISTORY",
    },
}


def _outcome(rule: str, overrides: dict) -> str:
    result = classify(ClassificationV2Request(**{**BASE, **overrides})).model_dump(mode="json")
    assessment = result[rule]
    if isinstance(assessment, list):
        assessment = assessment[0]
    return assessment.get("label") or assessment["status"]


@pytest.mark.parametrize(
    ("case_id", "rule", "overrides", "expected"),
    [(case_id, rule, overrides, expected) for case_id, _, rule, overrides, expected in CASES],
    ids=[case[0] for case in CASES],
)
def test_rule_outcome(case_id, rule, overrides, expected):
    assert _outcome(rule, overrides) == expected


def test_cases_cover_every_rule_outcome():
    covered = {rule: set() for rule in RULE_OUTCOMES}
    for _, _, rule, _, expected in CASES:
        covered[rule].add(expected)
    assert covered == RULE_OUTCOMES


if __name__ == "__main__":
    covered = {rule: set() for rule in RULE_OUTCOMES}
    passed = 0
    print(f"{'ID':6} {'Technique':16} {'Expected':30} {'Actual':30} Result")
    for case_id, technique, rule, overrides, expected in CASES:
        actual = _outcome(rule, overrides)
        covered[rule].add(actual)
        passed += actual == expected
        result = "PASS" if actual == expected else "FAIL"
        print(f"{case_id:6} {technique:16} {expected:30} {actual:30} {result}")

    total_outcomes = sum(len(outcomes) for outcomes in RULE_OUTCOMES.values())
    hit = sum(len(covered[rule] & outcomes) for rule, outcomes in RULE_OUTCOMES.items())
    print(f"\nRule coverage: {hit}/{total_outcomes} = {hit / total_outcomes:.0%}")
    print(f"Expected-output agreement: {passed}/{len(CASES)} = {passed / len(CASES):.0%}")
    print(f"Test case pass rate: {passed}/{len(CASES)} = {passed / len(CASES):.0%}")
