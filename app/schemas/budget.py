from __future__ import annotations

from calendar import monthrange
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import ApiMetadata, ModuleStatus, Transaction
from app.schemas.forecast import ExpenseForecastSnapshot


class RestrictionLevel(StrEnum):
    FIXED = "FIXED"
    MINIMUM = "MINIMUM"
    FLEXIBLE = "FLEXIBLE"


class BudgetTargetType(StrEnum):
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"


class BudgetPriorityGroup(StrEnum):
    ESSENTIAL = "ESSENTIAL"
    NON_ESSENTIAL = "NON_ESSENTIAL"


class BudgetPeriod(BaseModel):
    start: str
    end: str

    @model_validator(mode="after")
    def validate_calendar_month(self) -> BudgetPeriod:
        try:
            start = date.fromisoformat(self.start)
            end = date.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError("period start and end must be ISO-8601 dates") from exc
        if start.day != 1 or (start.year, start.month) != (end.year, end.month):
            raise ValueError("period must cover one calendar month")
        if end.day != monthrange(end.year, end.month)[1]:
            raise ValueError("period must end on the final day of its calendar month")
        return self


class BudgetCategory(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    category_id: str
    target_type: BudgetTargetType = BudgetTargetType.SUBCATEGORY
    priority_group: BudgetPriorityGroup
    restriction_level: RestrictionLevel
    floor: float = Field(ge=0)
    ceiling: float = Field(ge=0)
    priority_weight: float = Field(ge=0, le=1)
    current_spend: float = Field(ge=0, default=0)


class BudgetRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    request_id: str
    user_id: str
    available_funds: float = Field(gt=0)
    period: BudgetPeriod
    categories: list[BudgetCategory] = Field(min_length=1, max_length=100)
    target_ratios: dict[str, float] | None = None
    transaction_history: list[Transaction] | None = Field(default=None, max_length=10_000)
    forecast: ExpenseForecastSnapshot | None = None
    include_reasoning: bool = True

    @model_validator(mode="after")
    def validate_forecast_targets(self) -> BudgetRequest:
        category_ids = [category.category_id for category in self.categories]
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("categories must not contain duplicate category_id values")
        if self.forecast is None:
            return self
        if self.forecast.month != self.period.start[:7]:
            raise ValueError("forecast month must match the financial plan month")
        forecast_categories = {point.category for point in self.forecast.category_forecasts}
        if forecast_categories != set(category_ids):
            raise ValueError("forecast categories must match financial plan categories")
        return self


class BudgetAllocation(BaseModel):
    category_id: str
    amount: float = Field(ge=0)


class BudgetRecommendation(BaseModel):
    allocations: list[BudgetAllocation]
    utilization_rate: float = Field(ge=0, le=1)
    constraint_satisfaction: float = Field(ge=0, le=1)
    feasibility: str
    required_funds: float | None = Field(default=None, ge=0)
    available_funds: float | None = Field(default=None, gt=0)
    shortfall: float | None = Field(default=None, ge=0)
    unallocated_surplus: float = Field(default=0, ge=0)
    conflicting_categories: list[str] = Field(default_factory=list)


class BudgetExplanation(BaseModel):
    category_id: str
    reason: str


class BudgetResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    recommendation: BudgetRecommendation
    explanations: list[BudgetExplanation] | None = None
    status: ModuleStatus
    metadata: ApiMetadata
