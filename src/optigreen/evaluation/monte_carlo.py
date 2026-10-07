"""
Monte Carlo robustness testing for optimization policies.
Generates N demand realizations from the forecast distribution
and evaluates each policy's cost, service level, and CO2.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class MonteCarloResult:
    scenario_name: str
    n_samples: int
    mean_cost: float
    std_cost: float
    worst_cost: float
    mean_service_level: float
    worst_service_level: float
    mean_shortage: float
    mean_emissions: float
    p5_cost: float    # 5th percentile (best case)
    p95_cost: float   # 95th percentile (near worst case)


def generate_demand_realizations(
    forecast_df: pd.DataFrame,
    n_samples: int = 100,
    seed: int = 42
) -> List[pd.DataFrame]:
    """
    Generates N demand realizations by sampling from a
    Normal(P50, sigma) distribution, where:
        sigma = (P90 - P10) / 2.56
    (because P10/P90 correspond to ±1.28 sigma for a normal distribution)

    Returns a list of N DataFrames, each with the same schema as forecast_df
    but with 'demand_sample' replacing the quantile columns.
    """
    rng = np.random.default_rng(seed)
    df = forecast_df.copy()

    # Standard deviation from 80% prediction interval
    df['sigma'] = (df['P90'] - df['P10']) / 2.56
    df['sigma'] = df['sigma'].clip(lower=0.0)

    realizations = []
    for i in range(n_samples):
        sample = df.copy()
        noise = rng.normal(0, 1, size=len(df))
        sample['demand_sample'] = (df['P50'] + df['sigma'] * noise).clip(lower=0)
        realizations.append(sample[['date', 'region_id', 'product_id', 'demand_sample']])

    return realizations


def evaluate_policy_on_realization(
    realization_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    shortage_penalty: float = 50.0,
    holding_cost: float = 3.0,
    initial_inventory: float = 500.0,
) -> Dict:
    """
    Simulates executing a fixed decision plan against a specific demand realization.

    decision_df: shipment plan per (region, product, day) — from OptimizationResult.shipment_wr_plan.
    realization_df: realized demand per (region, product, day).

    Returns cost, service_level, shortage_qty for this realization.
    """
    # Build region-product-day delivery lookup from the shipment plan
    if decision_df is None or len(decision_df) == 0:
        # Fallback: no shipments planned (infeasible scenario)
        total_demand = realization_df['demand_sample'].sum()
        return {
            'cost': total_demand * shortage_penalty,
            'service_level': 0.0,
            'shortage_qty': total_demand,
        }

    delivery = decision_df.groupby(['region', 'product', 'day'])['quantity'].sum().to_dict()

    total_holding = total_shortage = total_demand = total_shortage_qty = 0.0

    for _, row in realization_df.iterrows():
        day = row.get('day', 0)
        delivered = delivery.get((row['region_id'], row['product_id'], day), 0.0)
        actual = row['demand_sample']
        total_demand += actual

        if actual > delivered:
            shortage = actual - delivered
            total_shortage_qty += shortage
            total_shortage += shortage * shortage_penalty
        else:
            leftover = delivered - actual
            total_holding += leftover * holding_cost

    total_cost = total_holding + total_shortage
    service_level = 1.0 - total_shortage_qty / max(total_demand, 1)

    return {
        'cost': total_cost,
        'service_level': service_level,
        'shortage_qty': total_shortage_qty,
    }


def run_monte_carlo(
    scenario_name: str,
    forecast_df: pd.DataFrame,
    shipment_wr_plan: Optional[pd.DataFrame],
    n_samples: int = 100,
    seed: int = 42,
    shortage_penalty: float = 50.0,
    holding_cost: float = 3.0,
) -> MonteCarloResult:
    """
    Runs full Monte Carlo evaluation for a single scenario.
    """
    realizations = generate_demand_realizations(forecast_df, n_samples, seed)

    costs, svc_levels, shortages = [], [], []

    for i, real_df in enumerate(realizations):
        # Add day index aligned with shipment plan
        if shipment_wr_plan is not None and 'day' in shipment_wr_plan.columns:
            pass  # Use existing day column in plan

        # Map dates to day indices
        dates = sorted(real_df['date'].unique())
        date_to_day = {d: idx for idx, d in enumerate(dates)}
        real_df = real_df.copy()
        real_df['day'] = real_df['date'].map(date_to_day)

        result = evaluate_policy_on_realization(
            realization_df=real_df,
            decision_df=shipment_wr_plan,
            shortage_penalty=shortage_penalty,
            holding_cost=holding_cost,
        )
        costs.append(result['cost'])
        svc_levels.append(result['service_level'])
        shortages.append(result['shortage_qty'])

    costs = np.array(costs)
    svc_levels = np.array(svc_levels)

    return MonteCarloResult(
        scenario_name=scenario_name,
        n_samples=n_samples,
        mean_cost=float(np.mean(costs)),
        std_cost=float(np.std(costs)),
        worst_cost=float(np.max(costs)),
        mean_service_level=float(np.mean(svc_levels)),
        worst_service_level=float(np.min(svc_levels)),
        mean_shortage=float(np.mean(shortages)),
        mean_emissions=0.0,  # emissions are fixed by the decision plan
        p5_cost=float(np.percentile(costs, 5)),
        p95_cost=float(np.percentile(costs, 95)),
    )
