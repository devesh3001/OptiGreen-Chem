"""
High-level optimizer orchestrator.
Connects ForecastProvider → Risk Scores → MILP → OptimizationResult.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

from optigreen.forecasting.forecast_provider import ForecastProvider
from optigreen.optimization.milp_model import MILPModel
from optigreen.optimization.result_schema import OptimizationResult


def build_demand_matrix(provider: ForecastProvider,
                        dates: List,
                        demand_mode: str = 'p50') -> Dict:
    """
    Converts ForecastProvider output into a {(region, product, day): qty} dict.
    demand_mode: 'p10', 'p50', 'p90'
    """
    demand_matrix = {}
    for day_idx, date in enumerate(dates):
        fc_df = provider.get_all_for_dates([date])
        for _, row in fc_df.iterrows():
            qty = row[demand_mode.upper()]
            demand_matrix[(row['region_id'], row['product_id'], day_idx)] = max(0.0, qty)
    return demand_matrix



def run_scenario(
    scenario_name: str,
    demand_mode: str,
    weights: Dict,
    provider: ForecastProvider,
    plants_df: pd.DataFrame,
    warehouses_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    routes_pw_df: pd.DataFrame,
    routes_wr_df: pd.DataFrame,
    dates: List,
    risk_scores: Optional[Dict] = None,
    initial_inventory: Optional[Dict] = None,
    shortage_penalty: float = 50.0,
    solver: str = 'highs',
    time_limit: int = 120,
) -> OptimizationResult:
    """
    Runs a single MILP scenario end-to-end.
    """
    horizon = len(dates)

    demand_matrix = build_demand_matrix(provider, dates, demand_mode)

    milp = MILPModel(
        plants_df=plants_df,
        warehouses_df=warehouses_df,
        regions_df=regions_df,
        products_df=products_df,
        routes_pw_df=routes_pw_df,
        routes_wr_df=routes_wr_df,
        demand_matrix=demand_matrix,
        risk_scores=risk_scores or {},
        initial_inventory=initial_inventory or {},
        horizon_days=horizon,
        weights=weights,
        shortage_penalty=shortage_penalty,
    )
    print(f"DEBUG: len(demand_matrix) = {len(demand_matrix)}")
    if len(demand_matrix) > 0:
        print(f"DEBUG: sample demand = {list(demand_matrix.values())[0]}")
        print(f"DEBUG: len(milp.pw_routes) = {len(milp.pw_routes)}, len(milp.wr_routes) = {len(milp.wr_routes)}")
    milp.build()
    return milp.solve(solver=solver, time_limit=time_limit,
                      scenario_name=scenario_name, demand_mode=demand_mode)


def run_baseline_policy(
    provider: ForecastProvider,
    dates: List,
    actual_demand_df: pd.DataFrame,
    initial_inventory: float = 1000.0,
    shortage_penalty: float = 50.0,
    holding_cost: float = 3.0,
) -> OptimizationResult:
    """
    Baseline rule: order exactly P50 forecast each day per (region, product).
    No optimization — pure heuristic reorder policy.
    Evaluated against actual demand.
    """
    total_holding = total_shortage = total_demand_qty = total_shortage_qty = 0.0

    for region in provider.regions:
        for product in provider.products:
            inventory = initial_inventory
            for day_idx, date in enumerate(dates):
                fc = provider.get(date, region, product)
                order_qty = fc.p50 if fc else 0.0
                actual = actual_demand_df[
                    (actual_demand_df['date'] == pd.Timestamp(date)) &
                    (actual_demand_df['region_id'] == region) &
                    (actual_demand_df['product_id'] == product)
                ]['demand'].values
                actual_qty = float(actual[0]) if len(actual) > 0 else 0.0
                total_demand_qty += actual_qty

                inventory += order_qty
                if actual_qty > inventory:
                    shortage = actual_qty - inventory
                    total_shortage_qty += shortage
                    total_shortage += shortage * shortage_penalty
                    inventory = 0.0
                else:
                    inventory -= actual_qty

                total_holding += inventory * holding_cost

    service_level = 1.0 - total_shortage_qty / max(total_demand_qty, 1)
    total_cost = total_holding + total_shortage

    return OptimizationResult(
        status='baseline',
        scenario_name='Baseline (P50 Reorder Policy)',
        demand_mode='p50',
        total_cost=total_cost,
        production_cost=0.0,
        transport_cost=0.0,
        holding_cost=total_holding,
        shortage_cost=total_shortage,
        total_emissions=0.0,
        service_level=service_level,
        total_shortage=total_shortage_qty,
        total_demand=total_demand_qty,
        horizon_days=len(dates),
    )
