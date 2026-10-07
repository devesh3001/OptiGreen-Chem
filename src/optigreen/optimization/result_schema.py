"""
OptimizationResult: Structured output from the MILP optimizer.
Consumed by the API, Streamlit dashboard, RAG, and LLM explanation layers.
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class OptimizationResult:
    # Solver status
    status: str                      # 'optimal', 'feasible', 'infeasible', 'error'
    scenario_name: str
    demand_mode: str                 # 'p50', 'p90', 'risk_aware'

    # Objective and costs
    objective_value: Optional[float] = None
    total_cost: Optional[float] = None
    production_cost: Optional[float] = None
    transport_cost: Optional[float] = None
    holding_cost: Optional[float] = None
    shortage_cost: Optional[float] = None
    carbon_cost: Optional[float] = None    # λ_carbon × total_emissions

    # Sustainability
    total_emissions: Optional[float] = None
    emissions_per_unit: Optional[float] = None

    # Service
    service_level: Optional[float] = None
    total_shortage: Optional[float] = None
    total_demand: Optional[float] = None

    # Inventory
    average_inventory: Optional[float] = None
    total_production: Optional[float] = None

    # Decision plans (DataFrames)
    production_plan: Optional[pd.DataFrame] = None   # [plant, product, day, quantity]
    shipment_pw_plan: Optional[pd.DataFrame] = None  # [plant, warehouse, product, day, quantity]
    shipment_wr_plan: Optional[pd.DataFrame] = None  # [warehouse, region, product, day, quantity]
    inventory_plan: Optional[pd.DataFrame] = None    # [warehouse, product, day, inventory]
    shortage_plan: Optional[pd.DataFrame] = None     # [region, product, day, shortage]

    # Performance
    solver_runtime_s: Optional[float] = None
    horizon_days: Optional[int] = None

    def to_summary_dict(self) -> dict:
        """Returns a flat dict for tabular comparison across scenarios."""
        return {
            'Scenario': self.scenario_name,
            'Status': self.status,
            'Demand Mode': self.demand_mode,
            'Total Cost': round(self.total_cost or 0, 2),
            'Production Cost': round(self.production_cost or 0, 2),
            'Transport Cost': round(self.transport_cost or 0, 2),
            'Holding Cost': round(self.holding_cost or 0, 2),
            'Shortage Cost': round(self.shortage_cost or 0, 2),
            'Total CO2': round(self.total_emissions or 0, 2),
            'CO2/Unit': round(self.emissions_per_unit or 0, 4),
            'Service Level': round(self.service_level or 0, 4),
            'Total Shortage': round(self.total_shortage or 0, 2),
            'Avg Inventory': round(self.average_inventory or 0, 2),
            'Total Production': round(self.total_production or 0, 2),
            'Solver Time (s)': round(self.solver_runtime_s or 0, 2),
        }
