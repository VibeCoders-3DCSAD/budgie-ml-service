from __future__ import annotations

from datetime import date as calendar_date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModuleStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    FALLBACK = "FALLBACK"


class ApiMetadata(BaseModel):
    processing_time_ms: float
    model_version: str
    strategy_used: str | None = None


class Transaction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    transaction_id: str | None = None
    date: str
    amount: float = Field(gt=0)
    category: str
    transaction_type: str = Field(pattern="^(income|expense)$")
    description: str | None = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        parsed = calendar_date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("date must use YYYY-MM-DD")
        return value
