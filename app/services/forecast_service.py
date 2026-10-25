from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from itertools import groupby

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
# How much completed history decides the baseline: a full year walks last month forward by HFCE and adds
# the seasonal spending above it, six to eleven months use the median month, and less than six months is provisional.
FULL_YEAR_MONTHS = 12
# A month above Q3 + 1.5 * IQR of the year's category totals holds seasonal spending, such as tuition.
SPIKE_IQR_FACTOR = 1.5
# A month counts as spending on a category above this share of its median spending month.
ACTIVE_MONTH_SHARE = 0.1
# A seasonal category, such as school terms, turns on and off in at most this many stretches a year,
# each on and off stretch lasting at least the minimum, so a single skipped month stays noise.
MAX_SEASONS = 2
MIN_SEASON_MONTHS = 2
ROBUST_BASELINE_MONTHS = 6
SAME_PERIOD_LAST_YEAR = "same_period_last_year"
MEDIAN_MONTHLY = "median_monthly"
PROVISIONAL = "recency_weighted_provisional"
# Months follow the Philippine calendar; the Philippines keeps GMT+8 all year, with no daylight saving.
MANILA = timezone(timedelta(hours=8))


def manila_today() -> date:
    """Today in the Philippines, whatever time zone the server runs in."""
    return datetime.now(MANILA).date()


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


SEASONAL = "seasonal"
NEW_EXPENSE = "new_expense"


def _on_months(values: np.ndarray) -> tuple[str | None, list[list[int]]]:
    """How a category's spending turns on and off over the year, with its on stretches as window indices.

    SEASONAL switches on and off with the calendar, such as school terms. NEW_EXPENSE is a single stretch
    still running last month that began within the year: it has not been seen to stop. Spending in every
    month, or in scattered months, returns None.
    """
    positive = values[values > 0]
    if positive.size == 0:
        return None, []
    active = values > ACTIVE_MONTH_SHARE * float(np.median(positive))
    if active.all():
        return None, []
    months = len(active)
    # Walk the year from the first off month after an on month, so no stretch splits at the wrap.
    start = next(i for i in range(months) if not active[i] and active[i - 1])
    order = [(start + offset) % months for offset in range(months)]
    stretches = [
        (on, list(indices)) for on, indices in groupby(order, key=lambda i: bool(active[i]))
    ]
    seasons = [indices for on, indices in stretches if on]
    if len(seasons) > MAX_SEASONS or any(
        len(indices) < MIN_SEASON_MONTHS for _, indices in stretches
    ):
        return None, []
    if len(seasons) == 1 and months - 1 in seasons[0] and 0 not in seasons[0]:
        return NEW_EXPENSE, seasons
    return SEASONAL, seasons


def _season_names(window: list[pd.Period], seasons: list[list[int]]) -> str:
    """The on stretches as calendar months, such as "January to June"."""
    return " and ".join(
        f"{window[indices[0]].strftime('%B')} to {window[indices[-1]].strftime('%B')}"
        for indices in sorted(seasons, key=lambda indices: window[indices[0]].month)
    )


def _walk_forward_multiplier(
    national: pd.Series, last_month: pd.Period, target_period: pd.Period
) -> float:
    """HFCE growth from last month to the target, one month at a time.

    HFCE is quarterly, so each quarter's growth over the one before is split evenly across its three
    months: every month steps by the cube root, and a full quarter compounds to the quarterly change.
    """
    multiplier = 1.0
    for month in pd.period_range(last_month + 1, target_period, freq="M"):
        quarter = month.asfreq("Q")
        if quarter not in national.index or quarter - 1 not in national.index:
            raise ValueError("forecast data is unavailable for the target month")
        multiplier *= (float(national[quarter]) / float(national[quarter - 1])) ** (1 / 3)
    return multiplier


def _seasonal_forecast(
    expenses: pd.DataFrame,
    category: str,
    window: list[pd.Period],
    target_period: pd.Period,
    national: pd.Series,
) -> tuple[float, float, float, pd.Period | None, pd.Period | None, str | None]:
    """A full year's forecast: last month walked forward by HFCE, plus last year's seasonal spending.

    The walk starts from last month unless it was a spike month or had no spending, in which case the
    typical month stands in. Spike months and payments the user marked yearly never start the walk;
    they come back only in the month they recur, grown by HFCE over the year.
    A category that switches on and off with the calendar follows last year's months instead: its
    typical month from the latest on stretch, walked forward from that stretch's end, in the months it
    was on, and nothing in the months it was off. Yearly payments still come back in their month.
    Returns the starting amount, the walk-forward multiplier, the seasonal amount, the month the
    seasonal amount came from, the month the walk started from (None for the typical month), and the
    calendar months the category is on (None unless it is seasonal).
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
    pattern, seasons = _on_months(values)
    prior_month = target_period - MONTHS_PER_YEAR
    in_window = prior_month in window
    if pattern == NEW_EXPENSE:
        # Its months would read as spikes against the empty months before it began.
        start, start_month = float(values[-1]), window[-1]
        multiplier = _walk_forward_multiplier(national, window[-1], target_period)
        last_year = float(yearly.get(prior_month, 0.0)) if in_window else 0.0
    elif pattern == SEASONAL:
        latest = max(seasons, key=max)
        on_months = {window[index].month for indices in seasons for index in indices}
        start = float(np.median(values[latest])) if target_period.month in on_months else 0.0
        multiplier = _walk_forward_multiplier(national, window[max(latest)], target_period)
        start_month = None
        last_year = float(yearly.get(prior_month, 0.0)) if in_window else 0.0
    else:
        spikes = _spike_months(values)
        base = float(values[~spikes].mean())
        last_is_typical = not spikes[-1] and values[-1] > 0
        start = float(values[-1]) if last_is_typical else base
        start_month = window[-1] if last_is_typical else None
        multiplier = _walk_forward_multiplier(national, window[-1], target_period)
        last_year = 0.0
        if in_window:
            prior_index = window.index(prior_month)
            excess = values[prior_index] - base if spikes[prior_index] else 0.0
            last_year = excess + float(yearly.get(prior_month, 0.0))
    season_names = _season_names(window, seasons) if pattern == SEASONAL else None
    if last_year == 0.0:
        return start, multiplier, 0.0, None, start_month, season_names
    prior_quarter = prior_month.asfreq("Q")
    if prior_quarter not in national.index:
        raise ValueError("forecast data is unavailable for the baseline months")
    growth = float(national[target_period.asfreq("Q")]) / float(national[prior_quarter])
    return start, multiplier, last_year * growth, prior_month, start_month, season_names


def _explanation(
    category: str,
    method: str,
    baseline_months: list[str],
    spike_month: pd.Period | None = None,
    start_month: pd.Period | None = None,
    seasons: str | None = None,
    in_season: bool = False,
) -> str:
    if method == SAME_PERIOD_LAST_YEAR:
        start = f"your {start_month} spending" if start_month else "your typical month"
        plan = f"Monthly {category} plan from {start}, grown month by month with HFCE"
        if seasons and in_season:
            plan = f"Monthly {category} plan from your typical month of {seasons}, grown month by month with HFCE"
        elif seasons:
            plan = (
                f"No {category} spending planned outside {seasons}, when you spent on it last year"
            )
        return f"{plan}, plus your {spike_month} seasonal spending." if spike_month else f"{plan}."
    if method == MEDIAN_MONTHLY:
        return (
            f"Monthly {category} plan from your typical (median) month, grown month by month with HFCE; "
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
    as_of = as_of or manila_today()
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
        start_month = None
        seasons = None
        if seasonal:
            user_baseline, multiplier, seasonal_amount, spike_month, start_month, seasons = (
                _seasonal_forecast(expenses, category, window, target_period, national)
            )
            baseline_months = [str(month) for month in window]
        elif method == MEDIAN_MONTHLY:
            # The median month stands for spending as of last month, walked forward like a full year.
            multiplier = _walk_forward_multiplier(national, current_month - 1, target_period)
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
                    category,
                    method,
                    baseline_months,
                    spike_month,
                    start_month,
                    seasons,
                    user_baseline > 0,
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
    as_of = as_of or manila_today()
    return [
        point
        for target_month in target_months
        for point in forecast(
            model, transactions, categories, as_of=as_of, user_id=user_id, target_month=target_month
        )
    ]
