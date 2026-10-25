from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from app.core.config import MAX_FORECAST_MONTHS
from app.models.registry import ModuleModel
from app.schemas.forecast import ForecastPoint
from app.services.features import transactions_to_frame

logger = logging.getLogger(__name__)

# Completed months that feed the baseline, ending with last month.
BASELINE_LOOKBACK_MONTHS = 12
# A category whose latest completed-month spend is this many months old is stale.
STALE_CATEGORY_MONTHS = 6
MONTHS_PER_YEAR = 12
# How much completed history decides the baseline: a full year splits a typical month from the
# seasonal spending above it, six to eleven months use the median month, and less than six months is provisional.
FULL_YEAR_MONTHS = 12
# A month above Q3 + 1.5 * IQR of the year's category totals holds seasonal spending, such as tuition.
SPIKE_IQR_FACTOR = 1.5
ROBUST_BASELINE_MONTHS = 6
SAME_PERIOD_LAST_YEAR = "same_period_last_year"
MEDIAN_MONTHLY = "median_monthly"
PROVISIONAL = "recency_weighted_provisional"


def _round_to_whole_peso(amount: float) -> float:
    return float(Decimal(str(amount)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _inverse_hfce_values(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.asarray(values, dtype="float64")
    if transform == "log1p":
        return np.expm1(values)
    if transform == "log":
        return np.exp(values)
    raise ValueError(f"unknown HFCE target transform: {transform}")


def _hfce_quarterly_series(artifact: dict, forecast_steps: int) -> pd.Series:
    """Observed plus forecast HFCE values, indexed by quarter, for either artifact family."""
    if artifact.get("family", "sarima") != "seasonal_naive":
        transform = artifact["transform"]
        history = _inverse_hfce_values(
            np.asarray(artifact["model"].model.endog, dtype="float64").reshape(-1), transform
        )
        forecast = _inverse_hfce_values(
            np.asarray(artifact["model"].forecast(forecast_steps)), transform
        )
        values = [*history.tolist(), *np.asarray(forecast, dtype="float64").reshape(-1).tolist()]
    else:
        values = np.asarray(artifact["model"], dtype="float64").tolist()
        for _ in range(forecast_steps):
            values.append(values[-4])
    last_quarter = pd.Period(artifact["period_end"], freq="Q") + forecast_steps
    quarters = pd.period_range(end=last_quarter, periods=len(values), freq="Q")
    return pd.Series(values, index=quarters, dtype="float64")


def _monthly_baseline(
    expenses: pd.DataFrame, category: str, as_of: date, stale_months: int = STALE_CATEGORY_MONTHS
) -> tuple[float, np.ndarray, list[str]]:
    current_month = pd.Period(as_of, freq="M")
    expense_months = expenses["date"].dt.to_period("M")
    completed = expenses.loc[
        expense_months.lt(current_month)
        & expense_months.ge(current_month - BASELINE_LOOKBACK_MONTHS)
        & expenses["category"].eq(category)
    ]
    monthly = completed.groupby(completed["date"].dt.to_period("M"))["amount"].sum()
    if monthly.empty:
        raise ValueError(f"forecast requires a completed {category} month")
    if monthly.index.max() <= current_month - stale_months:
        raise ValueError(f"forecast has no recent {category} spending")
    values = monthly.to_numpy(dtype="float64")
    weights = np.arange(1, len(values) + 1, dtype="float64")
    return (
        float(np.average(values, weights=weights)),
        values,
        [str(month) for month in monthly.index],
    )


def _history_window(expenses: pd.DataFrame, as_of: date) -> list[pd.Period]:
    """Completed months from the user's first recorded expense in the lookback window through last month."""
    current_month = pd.Period(as_of, freq="M")
    months = expenses["date"].dt.to_period("M")
    recorded = months.loc[
        months.ge(current_month - BASELINE_LOOKBACK_MONTHS) & months.lt(current_month)
    ]
    if recorded.empty:
        return []
    return list(pd.period_range(recorded.min(), current_month - 1, freq="M"))


def _baseline_method(history_months: int) -> str:
    if history_months >= FULL_YEAR_MONTHS:
        return SAME_PERIOD_LAST_YEAR
    if history_months >= ROBUST_BASELINE_MONTHS:
        return MEDIAN_MONTHLY
    return PROVISIONAL


def _median_monthly_baseline(
    expenses: pd.DataFrame, category: str, window: list[pd.Period]
) -> tuple[float, np.ndarray, list[str]]:
    """Median monthly spend across the recorded window, counting months without spend as zero."""
    in_category = expenses.loc[expenses["category"].eq(category)]
    monthly = in_category.groupby(in_category["date"].dt.to_period("M"))["amount"].sum()
    values = np.array([float(monthly.get(month, 0.0)) for month in window], dtype="float64")
    return float(np.median(values)), values, [str(month) for month in window]


def _spike_months(values: np.ndarray) -> np.ndarray:
    """Months whose total is an IQR outlier among the year's category totals."""
    q1, q3 = np.percentile(values, [25, 75])
    return values > q3 + SPIKE_IQR_FACTOR * (q3 - q1)


def _seasonal_forecast(
    expenses: pd.DataFrame,
    category: str,
    window: list[pd.Period],
    target_period: pd.Period,
    national: pd.Series,
) -> tuple[float, float, float, pd.Period | None]:
    """A full year's forecast: a typical month scaled by HFCE, plus last year's seasonal spending.

    HFCE drives the typical month, so a flat spender still sees the national quarterly shape. Spike
    months and payments the user marked yearly stay out of that typical month and come back only in
    the month they recur, grown by HFCE.
    Returns the typical month, its HFCE multiplier, the seasonal amount, and the month it came from.
    """
    in_category = expenses.loc[expenses["category"].eq(category)]
    months = in_category["date"].dt.to_period("M")
    is_yearly = (
        in_category["recurrence"].eq("yearly")
        if "recurrence" in in_category
        else pd.Series(False, index=in_category.index)
    )
    monthly = in_category.loc[~is_yearly].groupby(months[~is_yearly])["amount"].sum()
    yearly = in_category.loc[is_yearly].groupby(months[is_yearly])["amount"].sum()
    values = np.array([float(monthly.get(month, 0.0)) for month in window], dtype="float64")
    spikes = _spike_months(values)
    typical_quarters = [
        month.asfreq("Q") for month, spike in zip(window, spikes, strict=True) if not spike
    ]
    if any(quarter not in national.index for quarter in typical_quarters):
        raise ValueError("forecast data is unavailable for the baseline months")
    base = float(values[~spikes].mean())
    target_hfce = float(national[target_period.asfreq("Q")])
    multiplier = target_hfce / float(national[typical_quarters].mean())
    prior_month = target_period - MONTHS_PER_YEAR
    if prior_month not in window:
        return base, multiplier, 0.0, None
    prior_index = window.index(prior_month)
    excess = values[prior_index] - base if spikes[prior_index] else 0.0
    last_year = excess + float(yearly.get(prior_month, 0.0))
    if last_year == 0.0:
        return base, multiplier, 0.0, None
    seasonal = last_year * target_hfce / float(national[prior_month.asfreq("Q")])
    return base, multiplier, seasonal, prior_month


def _explanation(
    category: str,
    method: str,
    baseline_months: list[str],
    target_quarter: pd.Period | None = None,
    spike_month: pd.Period | None = None,
) -> str:
    if method == SAME_PERIOD_LAST_YEAR:
        plan = f"Monthly {category} plan based on your typical month and the HFCE {target_quarter} trend"
        return f"{plan}, plus your {spike_month} seasonal spending." if spike_month else f"{plan}."
    if method == MEDIAN_MONTHLY:
        return (
            f"Monthly {category} plan based on your typical (median) month and HFCE growth; "
            "seasonal changes appear once you have 12 months of history."
        )
    return (
        f"Provisional monthly {category} estimate from fewer than {ROBUST_BASELINE_MONTHS} months of spending; "
        "it is not yet a reliable personal forecast."
    )


def forecast(
    model: ModuleModel,
    transactions: list[dict],
    categories: list[str],
    as_of: date | None = None,
    user_id: str | None = None,
    target_month: str | None = None,
) -> list[ForecastPoint]:
    """Project each HFCE category for one month: the next calendar month unless one is given."""
    as_of = as_of or date.today()
    current_month = pd.Period(as_of, freq="M")
    target_period = pd.Period(target_month, freq="M") if target_month else current_month + 1
    if target_period < current_month:
        raise ValueError("forecast target month cannot be in the past")
    if target_period > current_month + MAX_FORECAST_MONTHS:
        raise ValueError("forecast target month exceeds the 12-month serving horizon")
    frame = transactions_to_frame(transactions)
    expenses = frame.loc[
        frame["transaction_type"].eq("expense") & frame["date"].dt.date.le(as_of)
    ].copy()
    if expenses.empty:
        raise ValueError("forecast requires at least one expense transaction")

    target_quarter = target_period.asfreq("Q")
    window = _history_window(expenses, as_of)
    method = _baseline_method(len(window))
    seasonal = method == SAME_PERIOD_LAST_YEAR
    points = []
    for category in dict.fromkeys(categories):
        artifact = model.model.get(category)
        if artifact is None:
            raise ValueError(f"forecast does not support category: {category}")
        # A yearly category such as Christmas gifts stays current through the full year of history.
        user_baseline, monthly_history, baseline_months = _monthly_baseline(
            expenses,
            category,
            as_of,
            BASELINE_LOOKBACK_MONTHS + 1 if seasonal else STALE_CATEGORY_MONTHS,
        )
        baseline_weights = np.arange(1, len(monthly_history) + 1, dtype="float64")
        if method == MEDIAN_MONTHLY:
            user_baseline, monthly_history, baseline_months = _median_monthly_baseline(
                expenses, category, window
            )
            baseline_weights = np.ones(len(monthly_history), dtype="float64")

        artifact_quarter = pd.Period(artifact["period_end"], freq="Q")
        forecast_steps = target_quarter.ordinal - artifact_quarter.ordinal
        if forecast_steps < 1:
            raise ValueError("forecast data is unavailable for the target month")
        national = _hfce_quarterly_series(artifact, forecast_steps)
        national_forecast = float(national[target_quarter])
        seasonal_amount = 0.0
        spike_month = None
        if seasonal:
            user_baseline, multiplier, seasonal_amount, spike_month = _seasonal_forecast(
                expenses, category, window, target_period, national
            )
            baseline_months = [str(month) for month in window]
        else:
            # Compare against national HFCE for the same quarters the user's baseline came from.
            baseline_quarters = [
                pd.Period(month, freq="M").asfreq("Q") for month in baseline_months
            ]
            if any(quarter not in national.index for quarter in baseline_quarters):
                raise ValueError("forecast data is unavailable for the baseline months")
            national_baseline = float(
                np.average(national[baseline_quarters].to_numpy(), weights=baseline_weights)
            )
            multiplier = national_forecast / national_baseline
        amount = user_baseline * multiplier + seasonal_amount
        logger.info(
            "forecast completed category=%s method=%s baseline_month_count=%s target_month=%s",
            category,
            method,
            len(baseline_months),
            str(target_period),
        )
        points.append(
            ForecastPoint(
                category=category,
                month=str(target_period),
                quarter=str(target_quarter),
                amount=_round_to_whole_peso(amount),
                user_baseline_amount=round(user_baseline, 2),
                hfce_multiplier=round(multiplier, 6),
                seasonal_amount=round(seasonal_amount, 2),
                hfce_forecast_amount_million_php=round(national_forecast, 2),
                explanation=_explanation(
                    category, method, baseline_months, target_quarter, spike_month
                ),
                baseline_method=method,
                # Only a full year of history shows when in the year this user's spending rises and falls.
                seasonal_uncertainty=method != SAME_PERIOD_LAST_YEAR,
            )
        )
    return points


def forecast_months(
    model: ModuleModel,
    transactions: list[dict],
    categories: list[str],
    target_months: Sequence[str | None],
    as_of: date | None = None,
    user_id: str | None = None,
) -> list[ForecastPoint]:
    """Forecast each target month in order; None stands for the next calendar month."""
    as_of = as_of or date.today()
    return [
        point
        for target_month in target_months
        for point in forecast(
            model, transactions, categories, as_of=as_of, user_id=user_id, target_month=target_month
        )
    ]
