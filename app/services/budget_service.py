from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from app.schemas.budget import (
    BudgetAllocation,
    BudgetCategory,
    BudgetExplanation,
    BudgetPriorityGroup,
    BudgetRecommendation,
    BudgetRequest,
    RestrictionLevel,
)

EPS = 1e-9


def optimize(request: BudgetRequest) -> tuple[BudgetRecommendation, list[BudgetExplanation]]:
    """Solve one monthly Financial Plan allocation without relaxing user rules.

    Forecast targets cap allocations. A first linear program maximizes
    weighted Essential coverage so a lower-weight Essential is not starved.
    The same coverage program protects Non-essentials using funds remaining
    after Essential targets. A final program preserves both coverage floors,
    prioritizes Essentials, and uses category weights to approach forecast caps.

    Constraints:
        - sum of allocations <= available funds,
      - FIXED categories held at their configured amount,
      - MINIMUM categories keep their configured floor,
      - FLEXIBLE categories respect [floor, ceiling].

    Feasibility:
      - FEASIBLE: all constraints met, allocations sum to available funds.
      - INFEASIBLE: no allocation satisfies all hard constraints.
    """
    categories = request.categories
    n = len(categories)
    available_funds = request.available_funds
    funds = available_funds

    forecast_targets = _forecast_targets(request)
    target_ratios = request.target_ratios or {}
    targets = np.array(
        [
            forecast_targets.get(
                category.category_id, funds * target_ratios.get(category.category_id, 0.0)
            )
            for category in categories
        ]
    )
    weights = np.array([max(cat.priority_weight, EPS) for cat in categories])

    for cat in categories:
        if cat.ceiling < cat.floor:
            raise ValueError(f"category {cat.category_id}: ceiling below floor")
        if cat.restriction_level == RestrictionLevel.FIXED and abs(cat.floor - cat.ceiling) > EPS:
            raise ValueError(
                f"category {cat.category_id}: fixed amount requires equal floor and ceiling"
            )
    # Forecast targets cap discretionary allocation. A configured hard floor
    # remains authoritative when it is higher than the forecast.
    allocation_bounds = [
        _allocation_bounds(category, targets[index]) for index, category in enumerate(categories)
    ]
    minimum_required = sum(bound[0] for bound in allocation_bounds)
    if minimum_required > funds + EPS:
        return _infeasible_recommendation(
            categories, allocation_bounds, funds, minimum_required, "minimum_required"
        ), []

    allocation_bounds = _priority_coverage_bounds(categories, allocation_bounds, funds)
    maximum_weighted_allocation = sum(
        weight * ceiling
        for weight, (_floor_amount, ceiling) in zip(weights, allocation_bounds, strict=True)
    )
    essential_priority = maximum_weighted_allocation + 1.0
    objective = np.array(
        [
            -weight
            - (
                essential_priority
                if category.priority_group == BudgetPriorityGroup.ESSENTIAL
                else 0.0
            )
            for category, weight in zip(categories, weights, strict=True)
        ]
    )
    result = linprog(
        c=objective,
        A_ub=[[1.0] * n],
        b_ub=[funds],
        bounds=allocation_bounds,
        method="highs",
    )

    if not result.success:
        return _infeasible_recommendation(
            categories, allocation_bounds, funds, minimum_required, "solver"
        ), []

    allocations = np.maximum(result.x[:n], 0.0)
    # Solver allocations can sum a fraction above funds due to floating-point roundoff.
    utilization = min(float(allocations.sum()) / available_funds, 1.0)
    unallocated_surplus = max(available_funds - float(allocations.sum()), 0.0)

    constraint_satisfaction = 1.0
    for i, cat in enumerate(categories):
        if (
            cat.restriction_level == RestrictionLevel.FIXED
            and abs(allocations[i] - cat.floor) > EPS
        ):
            constraint_satisfaction = 0.0
        if allocations[i] < cat.floor - EPS or allocations[i] > cat.ceiling + EPS:
            constraint_satisfaction = 0.0

    feasibility = "INFEASIBLE" if constraint_satisfaction < 0.99 else "FEASIBLE"
    recommendation = BudgetRecommendation(
        allocations=[
            BudgetAllocation(category_id=cat.category_id, amount=float(a))
            for cat, a in zip(categories, allocations, strict=True)
        ],
        utilization_rate=utilization,
        constraint_satisfaction=constraint_satisfaction,
        feasibility=feasibility,
        unallocated_surplus=unallocated_surplus,
    )

    explanations = []
    for cat, a in zip(categories, allocations, strict=True):
        if cat.restriction_level == RestrictionLevel.FIXED:
            explanations.append(
                BudgetExplanation(
                    category_id=cat.category_id,
                    reason=f"FIXED category held at {a:.2f}",
                )
            )
        elif cat.restriction_level == RestrictionLevel.MINIMUM:
            explanations.append(
                BudgetExplanation(
                    category_id=cat.category_id,
                    reason=f"MINIMUM category held at {a:.2f} or above",
                )
            )
        elif a > 0:
            explanations.append(
                BudgetExplanation(
                    category_id=cat.category_id,
                    reason="Allocation set within floor/ceiling bounds",
                )
            )

    return recommendation, explanations


def _allocation_bounds(category: BudgetCategory, target: float) -> tuple[float, float]:
    if category.restriction_level == RestrictionLevel.FIXED:
        return (category.floor, category.floor)
    minimum = category.floor
    configured_cap = category.ceiling
    if configured_cap < minimum:
        raise ValueError(f"category {category.category_id}: ceiling cannot satisfy the minimum")
    return (minimum, max(minimum, min(configured_cap, target)))


def _forecast_targets(request: BudgetRequest) -> dict[str, float]:
    if request.forecast is None:
        return {}
    return {point.category: point.amount for point in request.forecast.category_forecasts}


def _priority_coverage_bounds(
    categories: list[BudgetCategory],
    allocation_bounds: list[tuple[float, float]],
    funds: float,
) -> list[tuple[float, float]]:
    """Protect each group's weighted coverage, reserving Essential targets first."""
    for group in (BudgetPriorityGroup.ESSENTIAL, BudgetPriorityGroup.NON_ESSENTIAL):
        coverage_bounds = allocation_bounds
        if group == BudgetPriorityGroup.NON_ESSENTIAL:
            coverage_bounds = [
                (ceiling, ceiling)
                if category.priority_group == BudgetPriorityGroup.ESSENTIAL
                else (floor, ceiling)
                for category, (floor, ceiling) in zip(categories, allocation_bounds, strict=True)
            ]
            if sum(floor for floor, _ceiling in coverage_bounds) >= funds:
                break

        coverage_floors = _weighted_coverage_floors(categories, coverage_bounds, funds, group)
        allocation_bounds = [
            (max(floor, coverage_floors[index]), ceiling)
            if category.priority_group == group
            else (floor, ceiling)
            for index, (category, (floor, ceiling)) in enumerate(
                zip(categories, allocation_bounds, strict=True)
            )
        ]
    return allocation_bounds


def _weighted_coverage_floors(
    categories: list[BudgetCategory],
    allocation_bounds: list[tuple[float, float]],
    funds: float,
    priority_group: BudgetPriorityGroup,
) -> list[float]:
    """Maximize a group's common weighted coverage level with an LP."""
    floors = [floor for floor, _ceiling in allocation_bounds]
    ceilings = [ceiling for _floor, ceiling in allocation_bounds]
    weighted_gaps = {
        index: (ceilings[index] - floors[index]) * max(categories[index].priority_weight, EPS)
        for index, category in enumerate(categories)
        if category.priority_group == priority_group and ceilings[index] > floors[index] + EPS
    }
    if not weighted_gaps:
        return floors

    # Variables are allocations x_i and coverage level z. Each group member must
    # receive floor_i + z * weighted_gap_i before the final target optimization.
    n = len(categories)
    objective = [0.0] * n + [-1.0]
    budget_constraint = [1.0] * n + [0.0]
    coverage_constraints = []
    coverage_limits = []
    for index, weighted_gap in weighted_gaps.items():
        constraint = [0.0] * (n + 1)
        constraint[index] = -1.0
        constraint[-1] = weighted_gap
        coverage_constraints.append(constraint)
        coverage_limits.append(-floors[index])

    result = linprog(
        c=objective,
        A_ub=[budget_constraint, *coverage_constraints],
        b_ub=[funds, *coverage_limits],
        bounds=[*allocation_bounds, (0.0, None)],
        method="highs",
    )
    if not result.success:
        raise ValueError(f"{priority_group.value} coverage could not be optimized")

    return list(result.x[:-1])


def _infeasible_recommendation(
    categories: list[BudgetCategory],
    bounds: list[tuple[float, float]],
    funds: float,
    constrained_total: float,
    reason: str,
) -> BudgetRecommendation:
    shortfall = abs(constrained_total - funds)
    allocated_amount = sum(lower for lower, _upper in bounds)
    return BudgetRecommendation(
        allocations=[
            BudgetAllocation(category_id=category.category_id, amount=lower)
            for category, (lower, _upper) in zip(categories, bounds, strict=True)
        ],
        utilization_rate=min(allocated_amount / funds, 1.0),
        constraint_satisfaction=0.0,
        feasibility="INFEASIBLE",
        required_funds=constrained_total,
        available_funds=funds,
        shortfall=shortfall,
        unallocated_surplus=0.0,
        conflicting_categories=[
            category.category_id
            for category, (lower, upper) in zip(categories, bounds, strict=True)
            if lower > 0 or (reason == "maximum_capacity" and upper > 0)
        ],
    )
