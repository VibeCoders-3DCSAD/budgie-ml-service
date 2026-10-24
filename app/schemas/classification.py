from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ApiMetadata, ModuleStatus

RevolvingRange = Literal["no_credit_card", "always_before_due", "1_to_11_months", "over_a_year"]


class ClassificationDataStatus(StrEnum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


class CreditCardCycle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    completed_on: date
    revolving_balance: float = Field(ge=0)


class CreditCardAccount(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    account_id: str = Field(min_length=1)
    cycles: list[CreditCardCycle] = Field(min_length=1, max_length=120)


class ClassificationV2Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    classification_as_of: date
    liquid_emergency_savings: float | None = Field(default=None, ge=0)
    average_monthly_recurring_essential_expenses: float | None = Field(default=None, ge=0)
    monthly_disposable_income: float | None = Field(default=None, ge=0)
    required_monthly_debt_payments: float | None = Field(default=None, ge=0)
    monthly_basic_living_costs: float | None = Field(default=None, ge=0)
    credit_card_accounts: list[CreditCardAccount] | None = Field(default=None, max_length=50)
    # Used only when no observed card history is supplied.
    self_reported_revolving_months: int | None = Field(default=None, ge=0, le=12)
    self_reported_revolving_range: RevolvingRange | None = None


class RuleAssessment(BaseModel):
    status: ClassificationDataStatus
    label: str | None = None
    value: float | int | None = None
    threshold: float | int
    inputs: dict[str, float | int]


class CreditCardAssessment(RuleAssessment):
    account_id: str
    source: Literal["observed_history", "self_reported"] = "observed_history"
    reported_duration: RevolvingRange | None = None


class ClassificationV2Result(BaseModel):
    emergency_savings: RuleAssessment
    debt_burden: RuleAssessment
    financial_margin: RuleAssessment
    credit_card_behavior: list[CreditCardAssessment]


class ClassificationV2Response(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    classification: ClassificationV2Result
    status: ModuleStatus
    metadata: ApiMetadata
