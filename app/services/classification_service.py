from __future__ import annotations

from datetime import date

from app.schemas.classification import (
    ClassificationDataStatus,
    ClassificationV2Request,
    ClassificationV2Result,
    CreditCardAccount,
    CreditCardAssessment,
    RevolvingRange,
    RuleAssessment,
)

RULE_SET_VERSION = "financial_rules_v1"
EMERGENCY_FUND_MONTHS = 3.0
DSTI_BOUNDARY = 0.40


def _insufficient_data(threshold: float | int) -> RuleAssessment:
    return RuleAssessment(
        status=ClassificationDataStatus.INSUFFICIENT_DATA,
        threshold=threshold,
        inputs={},
    )


def _emergency_savings(request: ClassificationV2Request) -> RuleAssessment:
    savings = request.liquid_emergency_savings
    essentials = request.average_monthly_recurring_essential_expenses
    if savings is None or essentials is None or essentials <= 0:
        return _insufficient_data(EMERGENCY_FUND_MONTHS)
    coverage = savings / essentials
    return RuleAssessment(
        status=ClassificationDataStatus.SUCCESS,
        label="EMERGENCY_FUND_ADEQUATE"
        if coverage >= EMERGENCY_FUND_MONTHS
        else "EMERGENCY_FUND_INADEQUATE",
        value=round(coverage, 4),
        threshold=EMERGENCY_FUND_MONTHS,
        inputs={"liquid_emergency_savings": savings, "monthly_essential_expenses": essentials},
    )


def _debt_burden(request: ClassificationV2Request) -> RuleAssessment:
    income = request.monthly_disposable_income
    payments = request.required_monthly_debt_payments
    if income is None or payments is None or income <= 0:
        return _insufficient_data(DSTI_BOUNDARY)
    dsti = payments / income
    return RuleAssessment(
        status=ClassificationDataStatus.SUCCESS,
        label="FINANCIALLY_OVERBURDENED"
        if dsti > DSTI_BOUNDARY
        else "NOT_FINANCIALLY_OVERBURDENED",
        value=round(dsti, 4),
        threshold=DSTI_BOUNDARY,
        inputs={"monthly_disposable_income": income, "required_monthly_debt_payments": payments},
    )


def _financial_margin(request: ClassificationV2Request) -> RuleAssessment:
    income = request.monthly_disposable_income
    payments = request.required_monthly_debt_payments
    living_costs = request.monthly_basic_living_costs
    if income is None or payments is None or living_costs is None or income <= 0:
        return _insufficient_data(0)
    margin = income - payments - living_costs
    return RuleAssessment(
        status=ClassificationDataStatus.SUCCESS,
        label="POSITIVE_FINANCIAL_MARGIN" if margin >= 0 else "NEGATIVE_FINANCIAL_MARGIN",
        value=round(margin, 2),
        threshold=0,
        inputs={
            "monthly_disposable_income": income,
            "required_monthly_debt_payments": payments,
            "monthly_basic_living_costs": living_costs,
        },
    )


def _credit_card_behavior(
    account: CreditCardAccount, classification_as_of: date
) -> CreditCardAssessment:
    periods = sorted(
        {cycle.completed_on.year * 12 + cycle.completed_on.month for cycle in account.cycles}
    )
    assessment_month = classification_as_of.year * 12 + classification_as_of.month
    required_periods = list(range(assessment_month - 12, assessment_month))
    if len(account.cycles) != 12 or len(periods) != 12 or periods != required_periods:
        # A balance carried within the window already proves revolving, even before
        # the full 12 months exist to tell light from heavy.
        window = set(required_periods)
        observed_revolving = {
            cycle.completed_on.year * 12 + cycle.completed_on.month
            for cycle in account.cycles
            if cycle.revolving_balance > 0
            and cycle.completed_on.year * 12 + cycle.completed_on.month in window
        }
        if observed_revolving:
            return CreditCardAssessment(
                account_id=account.account_id,
                status=ClassificationDataStatus.SUCCESS,
                label="REVOLVER",
                value=len(observed_revolving),
                threshold=12,
                inputs={
                    "revolving_months": len(observed_revolving),
                    "observed_completed_cycles": len(window.intersection(periods)),
                    "required_completed_cycles": 12,
                },
            )
        return CreditCardAssessment(
            account_id=account.account_id,
            status=ClassificationDataStatus.INSUFFICIENT_HISTORY,
            threshold=12,
            inputs={
                "observed_completed_cycles": len(periods),
                "required_completed_cycles": 12,
            },
        )
    revolving_months = sum(cycle.revolving_balance > 0 for cycle in account.cycles)
    return CreditCardAssessment(
        account_id=account.account_id,
        status=ClassificationDataStatus.SUCCESS,
        label=_revolver_label(revolving_months),
        value=revolving_months,
        threshold=12,
        inputs={"revolving_months": revolving_months, "observed_completed_cycles": len(periods)},
    )


def _revolver_label(revolving_months: int) -> str:
    if revolving_months == 0:
        return "TRANSACTOR"
    if revolving_months == 12:
        return "HEAVY_REVOLVER"
    return "LIGHT_REVOLVER"


def _self_reported_credit_card_behavior(revolving_months: int) -> CreditCardAssessment:
    return CreditCardAssessment(
        account_id="self_reported",
        source="self_reported",
        status=ClassificationDataStatus.SUCCESS,
        label=_revolver_label(revolving_months),
        value=revolving_months,
        threshold=12,
        inputs={"revolving_months": revolving_months},
    )


def _credit_card_assessments(request: ClassificationV2Request) -> list[CreditCardAssessment]:
    # Observed statement history takes precedence over the onboarding self-report.
    if request.credit_card_accounts:
        return [
            _credit_card_behavior(account, request.classification_as_of)
            for account in request.credit_card_accounts
        ]
    if request.self_reported_revolving_range is not None:
        return [_self_reported_credit_card_range(request.self_reported_revolving_range)]
    if request.self_reported_revolving_months is not None:
        return [_self_reported_credit_card_behavior(request.self_reported_revolving_months)]
    return []


def _self_reported_credit_card_range(duration: RevolvingRange) -> CreditCardAssessment:
    labels = {
        "no_credit_card": "NO_CREDIT_CARD",
        "always_before_due": "TRANSACTOR",
        "1_to_11_months": "LIGHT_REVOLVER",
        "over_a_year": "HEAVY_REVOLVER",
    }
    # A range establishes the category, but never an exact observed month count.
    return CreditCardAssessment(
        account_id="self_reported",
        source="self_reported",
        reported_duration=duration,
        status=ClassificationDataStatus.SUCCESS,
        label=labels[duration],
        value=None,
        threshold=12,
        inputs={},
    )


def classify(request: ClassificationV2Request) -> ClassificationV2Result:
    """Apply the documented financial-resilience rules without inference or tuning."""
    return ClassificationV2Result(
        emergency_savings=_emergency_savings(request),
        debt_burden=_debt_burden(request),
        financial_margin=_financial_margin(request),
        credit_card_behavior=_credit_card_assessments(request),
    )
