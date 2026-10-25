from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import ApiMetadata, ModuleStatus, Transaction

MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"


class ForecastRequest(BaseModel):
    user_id: str
    historical_transactions: list[Transaction] = Field(min_length=1, max_length=10_000)
    categories: list[str] = Field(min_length=1, max_length=12)
    # Month to forecast (YYYY-MM); defaults to the next calendar month.
    target_month: str | None = Field(default=None, pattern=MONTH_PATTERN)
    # Several months in one call, e.g. a year of monthly forecasts; the snapshot covers the first.
    target_months: list[Annotated[str, Field(pattern=MONTH_PATTERN)]] | None = Field(
        default=None, min_length=1, max_length=12
    )

    @model_validator(mode="after")
    def allow_one_target_kind(self) -> ForecastRequest:
        if self.target_month is not None and self.target_months is not None:
            raise ValueError("send target_month or target_months, not both")
        months = self.target_months
        if months is not None and len(set(months)) != len(months):
            raise ValueError("target_months must not repeat a month")
        return self


class ForecastPoint(BaseModel):
    category: str
    month: str
    quarter: str
    amount: float
    user_baseline_amount: float
    hfce_multiplier: float
    # Last year's spending above the typical month, grown by HFCE; amount = baseline * multiplier + this.
    seasonal_amount: float = 0.0
    hfce_forecast_amount_million_php: float
    explanation: str
    # same_period_last_year (12+ months), median_monthly (6-11 months), or recency_weighted_provisional (<6 months).
    baseline_method: str
    # True when the history is too short to show this user's seasonal pattern.
    seasonal_uncertainty: bool


class ForecastQuality(StrEnum):
    PERSONALIZED = "PERSONALIZED"
    FALLBACK = "FALLBACK"
    COLD_START = "COLD_START"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ForecastedExpense(BaseModel):
    category: str = Field(min_length=1)
    amount: float = Field(ge=0)


class ExpenseForecastSnapshot(BaseModel):
    """Immutable expense input used when recommending one calendar-month plan."""

    month: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    total_amount: float = Field(ge=0)
    category_forecasts: list[ForecastedExpense] = Field(min_length=1)
    model_version: str = Field(min_length=1)
    generated_at: datetime
    quality: ForecastQuality

    @field_validator("generated_at")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone offset")
        return value

    @model_validator(mode="after")
    def validate_category_forecasts(self) -> ExpenseForecastSnapshot:
        categories = [point.category for point in self.category_forecasts]
        if len(categories) != len(set(categories)):
            raise ValueError("category_forecasts must not contain duplicate categories")
        if abs(sum(point.amount for point in self.category_forecasts) - self.total_amount) > 0.01:
            raise ValueError("total_amount must match category_forecasts")
        return self


class ForecastResponse(BaseModel):
    response_id: str
    request_id: str
    user_id: str
    forecasts: list[ForecastPoint]
    snapshot: ExpenseForecastSnapshot
    model_version: str
    status: ModuleStatus
    metadata: ApiMetadata
