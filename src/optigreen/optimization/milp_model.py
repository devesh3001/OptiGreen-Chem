"""
Core MILP model using Pyomo + HiGHS solver.
All-continuous LP (no binary variables) for tractability across many scenarios.

Model: Multi-period supply chain LP
  Sets: Plants (P), Warehouses (W), Regions (R), Products (PR), Days (T)
  Decisions: Production, Shipment_PW, Shipment_WR, Inventory, Shortage
  Objective: Weighted sum of cost, carbon, risk penalty
"""
import time
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from typing import Dict, Any, Optional

from optigreen.optimization.result_schema import OptimizationResult


class MILPModel:
    """
    Builds and solves the supply chain LP.

    Parameters
    ----------
    plants_df : DataFrame with columns [plant_id, production_capacity, variable_production_cost, production_emission_factor]
    warehouses_df : DataFrame with columns [warehouse_id, capacity, holding_cost]
    regions_df : DataFrame with columns [region_id]
    products_df : DataFrame with columns [product_id]
    routes_pw_df : DataFrame with columns [source (plant), destination (warehouse), transport_cost, capacity, carbon_emission_factor]
    routes_wr_df : DataFrame with columns [source (warehouse), destination (region), transport_cost, carbon_emission_factor]
    demand_matrix : Dict[(region, product, day)] -> demand quantity
    risk_scores : Dict[(region, product, day)] -> risk score in [0,1]  (optional)
    initial_inventory : Dict[(warehouse, product)] -> starting inventory level
    horizon_days : int
    weights : Dict with keys 'cost', 'carbon', 'risk'
    shortage_penalty : float
    """

    def __init__(self,
                 plants_df: pd.DataFrame,
                 warehouses_df: pd.DataFrame,
                 regions_df: pd.DataFrame,
                 products_df: pd.DataFrame,
                 routes_pw_df: pd.DataFrame,
                 routes_wr_df: pd.DataFrame,
                 demand_matrix: Dict,
                 risk_scores: Optional[Dict] = None,
                 initial_inventory: Optional[Dict] = None,
                 horizon_days: int = 7,
                 weights: Optional[Dict] = None,
                 shortage_penalty: float = 50.0):

        self.plants = list(plants_df['plant_id'])
        self.warehouses = list(warehouses_df['warehouse_id'])
        self.regions = list(regions_df['region_id'])
        self.products = list(products_df['product_id'])
        self.days = list(range(horizon_days))
        self.horizon = horizon_days

        # Parameter lookups
        self.plant_capacity = dict(zip(plants_df['plant_id'], plants_df['production_capacity']))
        self.prod_cost = dict(zip(plants_df['plant_id'], plants_df['variable_production_cost']))
        self.prod_emission = dict(zip(plants_df['plant_id'], plants_df['production_emission_factor']))

        self.wh_capacity = dict(zip(warehouses_df['warehouse_id'], warehouses_df['capacity']))
        self.holding_cost = dict(zip(warehouses_df['warehouse_id'], warehouses_df['holding_cost']))

        # Route lookups (plant→warehouse)
        self.pw_routes = {}  # (plant, warehouse) -> {cost, capacity, carbon}
        for _, r in routes_pw_df.iterrows():
            self.pw_routes[(r['source'], r['destination'])] = {
                'cost': r['transport_cost'],
                'capacity': r['capacity'],
                'carbon': r['carbon_emission_factor'],
            }

        # Route lookups (warehouse→region)
        self.wr_routes = {}  # (warehouse, region) -> {cost, carbon}
        for _, r in routes_wr_df.iterrows():
            self.wr_routes[(r['source'], r['destination'])] = {
                'cost': r['transport_cost'],
                'carbon': r['carbon_emission_factor'],
            }

        self.demand = demand_matrix  # (region, product, day) -> quantity
        self.risk_scores = risk_scores or {}  # (region, product, day) -> [0,1]
        self.initial_inventory = initial_inventory or {}  # (warehouse, product) -> quantity
        self.weights = weights or {'cost': 1.0, 'carbon': 0.0, 'risk': 0.0}
        self.shortage_penalty = shortage_penalty

        self.model = None
        self.result = None

    def _get_initial_inv(self, w, pr):
        return self.initial_inventory.get((w, pr), 0.0)

    def build(self) -> pyo.ConcreteModel:
        """Construct the Pyomo LP model."""
        m = pyo.ConcreteModel(name="SupplyChainLP")

        # ------------------------------------------------------------------ #
        # Sets
        # ------------------------------------------------------------------ #
        m.P = pyo.Set(initialize=self.plants)
        m.W = pyo.Set(initialize=self.warehouses)
        m.R = pyo.Set(initialize=self.regions)
        m.PR = pyo.Set(initialize=self.products)
        m.T = pyo.Set(initialize=self.days)
        m.PW = pyo.Set(initialize=list(self.pw_routes.keys()))
        m.WR = pyo.Set(initialize=list(self.wr_routes.keys()))

        # ------------------------------------------------------------------ #
        # Decision Variables
        # ------------------------------------------------------------------ #
        m.Production = pyo.Var(m.P, m.PR, m.T, domain=pyo.NonNegativeReals)
        m.Shipment_PW = pyo.Var(m.PW, m.PR, m.T, domain=pyo.NonNegativeReals)
        m.Shipment_WR = pyo.Var(m.WR, m.PR, m.T, domain=pyo.NonNegativeReals)
        m.Inventory = pyo.Var(m.W, m.PR, m.T, domain=pyo.NonNegativeReals)
        m.Shortage = pyo.Var(m.R, m.PR, m.T, domain=pyo.NonNegativeReals)

        # ------------------------------------------------------------------ #
        # Constraints
        # ------------------------------------------------------------------ #

        # (1) Production capacity: total daily production at plant ≤ capacity
        def prod_cap_rule(m, p, t):
            return sum(m.Production[p, pr, t] for pr in m.PR) <= self.plant_capacity[p]
        m.ProdCapacity = pyo.Constraint(m.P, m.T, rule=prod_cap_rule)

        # (2) Plant→Warehouse route capacity
        def pw_cap_rule(m, p, w, pr, t):
            cap = self.pw_routes.get((p, w), {}).get('capacity', np.inf)
            if cap == np.inf:
                return pyo.Constraint.Skip
            return m.Shipment_PW[p, w, pr, t] <= cap
        m.PW_Capacity = pyo.Constraint(m.PW, m.PR, m.T, rule=pw_cap_rule)

        # (3) Production conservation: total shipped from plant ≤ total produced
        def prod_ship_balance_rule(m, p, pr, t):
            shipped = sum(m.Shipment_PW[p, w, pr, t] for (pp, w) in m.PW if pp == p)
            return shipped <= m.Production[p, pr, t]
        m.ProdShipBalance = pyo.Constraint(m.P, m.PR, m.T, rule=prod_ship_balance_rule)

        # (4) Inventory balance at each warehouse
        def inv_balance_rule(m, w, pr, t):
            inbound = sum(m.Shipment_PW[p, w, pr, t] for (p, ww) in m.PW if ww == w)
            outbound = sum(m.Shipment_WR[w, r, pr, t] for (ww, r) in m.WR if ww == w)
            if t == 0:
                prev_inv = self._get_initial_inv(w, pr)
            else:
                prev_inv = m.Inventory[w, pr, t - 1]
            return m.Inventory[w, pr, t] == prev_inv + inbound - outbound
        m.InvBalance = pyo.Constraint(m.W, m.PR, m.T, rule=inv_balance_rule)

        # (5) Warehouse capacity
        def wh_cap_rule(m, w, t):
            return sum(m.Inventory[w, pr, t] for pr in m.PR) <= self.wh_capacity[w]
        m.WH_Capacity = pyo.Constraint(m.W, m.T, rule=wh_cap_rule)

        # (6) Demand satisfaction: shipments + shortage >= demand
        def demand_rule(m, r, pr, t):
            delivered = sum(m.Shipment_WR[w, r, pr, t] for (w, rr) in m.WR if rr == r)
            demand_qty = self.demand.get((r, pr, t), 0.0)
            return delivered + m.Shortage[r, pr, t] >= demand_qty
        m.DemandSatisfaction = pyo.Constraint(m.R, m.PR, m.T, rule=demand_rule)

        # ------------------------------------------------------------------ #
        # Objective
        # ------------------------------------------------------------------ #
        def objective_rule(m):
            # --- Cost component ---
            prod_cost = sum(
                self.prod_cost[p] * m.Production[p, pr, t]
                for p in m.P for pr in m.PR for t in m.T
            )
            transport_pw_cost = sum(
                self.pw_routes[p, w]['cost'] * m.Shipment_PW[p, w, pr, t]
                for (p, w) in m.PW for pr in m.PR for t in m.T
            )
            transport_wr_cost = sum(
                self.wr_routes[w, r]['cost'] * m.Shipment_WR[w, r, pr, t]
                for (w, r) in m.WR for pr in m.PR for t in m.T
            )
            holding = sum(
                self.holding_cost[w] * m.Inventory[w, pr, t]
                for w in m.W for pr in m.PR for t in m.T
            )
            shortage = sum(
                self.shortage_penalty * m.Shortage[r, pr, t]
                for r in m.R for pr in m.PR for t in m.T
            )
            total_cost = prod_cost + transport_pw_cost + transport_wr_cost + holding + shortage

            # --- Carbon component ---
            carbon_prod = sum(
                self.prod_emission[p] * m.Production[p, pr, t]
                for p in m.P for pr in m.PR for t in m.T
            )
            carbon_pw = sum(
                self.pw_routes[p, w]['carbon'] * m.Shipment_PW[p, w, pr, t]
                for (p, w) in m.PW for pr in m.PR for t in m.T
            )
            carbon_wr = sum(
                self.wr_routes[w, r]['carbon'] * m.Shipment_WR[w, r, pr, t]
                for (w, r) in m.WR for pr in m.PR for t in m.T
            )
            total_carbon = carbon_prod + carbon_pw + carbon_wr

            # --- Risk component ---
            risk_penalty = sum(
                self.risk_scores.get((r, pr, t), 0.0) * m.Shortage[r, pr, t]
                for r in m.R for pr in m.PR for t in m.T
            )

            return (self.weights['cost'] * total_cost
                    + self.weights.get('carbon', 0.0) * total_carbon
                    + self.weights.get('risk', 0.0) * risk_penalty)

        m.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        self.model = m
        return m

    def solve(self, solver: str = 'highs', time_limit: int = 120,
              scenario_name: str = 'unnamed', demand_mode: str = 'p50') -> OptimizationResult:
        """Solve the model and return an OptimizationResult."""
        if self.model is None:
            self.build()

        start = time.time()
        opt = pyo.SolverFactory(solver)
        if solver == 'highs':
            opt.options['time_limit'] = time_limit
            opt.options['mip_rel_gap'] = 0.01

        result = opt.solve(self.model, tee=False)
        elapsed = time.time() - start

        # Check solver status
        tc = result.solver.termination_condition
        is_optimal = tc in (TerminationCondition.optimal, TerminationCondition.feasible)

        if not is_optimal:
            return OptimizationResult(
                status=str(tc),
                scenario_name=scenario_name,
                demand_mode=demand_mode,
                solver_runtime_s=elapsed,
                horizon_days=self.horizon,
            )

        status_str = 'optimal' if tc == TerminationCondition.optimal else 'feasible'
        m = self.model

        # ------------------------------------------------------------------ #
        # Extract results
        # ------------------------------------------------------------------ #
        prod_records, ship_pw_records, ship_wr_records, inv_records, short_records = [], [], [], [], []

        prod_cost_total = transport_pw_total = transport_wr_total = 0.0
        holding_total = shortage_total = carbon_total = 0.0
        total_demand = total_shortage_qty = total_production = 0.0
        inv_sum = inv_count = 0.0

        for p in self.plants:
            for pr in self.products:
                for t in self.days:
                    qty = pyo.value(m.Production[p, pr, t])
                    if qty and qty > 1e-6:
                        prod_records.append({'plant': p, 'product': pr, 'day': t, 'quantity': qty})
                        prod_cost_total += self.prod_cost[p] * qty
                        carbon_total += self.prod_emission[p] * qty
                        total_production += qty

        for (p, w) in self.pw_routes:
            for pr in self.products:
                for t in self.days:
                    qty = pyo.value(m.Shipment_PW[p, w, pr, t])
                    if qty and qty > 1e-6:
                        ship_pw_records.append({'plant': p, 'warehouse': w, 'product': pr, 'day': t, 'quantity': qty})
                        transport_pw_total += self.pw_routes[p, w]['cost'] * qty
                        carbon_total += self.pw_routes[p, w]['carbon'] * qty

        for (w, r) in self.wr_routes:
            for pr in self.products:
                for t in self.days:
                    qty = pyo.value(m.Shipment_WR[w, r, pr, t])
                    if qty and qty > 1e-6:
                        ship_wr_records.append({'warehouse': w, 'region': r, 'product': pr, 'day': t, 'quantity': qty})
                        transport_wr_total += self.wr_routes[w, r]['cost'] * qty
                        carbon_total += self.wr_routes[w, r]['carbon'] * qty

        for w in self.warehouses:
            for pr in self.products:
                for t in self.days:
                    inv = pyo.value(m.Inventory[w, pr, t])
                    if inv is not None:
                        inv_records.append({'warehouse': w, 'product': pr, 'day': t, 'inventory': inv})
                        holding_total += self.holding_cost[w] * inv
                        inv_sum += inv
                        inv_count += 1

        for r in self.regions:
            for pr in self.products:
                for t in self.days:
                    s = pyo.value(m.Shortage[r, pr, t])
                    d = self.demand.get((r, pr, t), 0.0)
                    total_demand += d
                    if s and s > 1e-6:
                        short_records.append({'region': r, 'product': pr, 'day': t, 'shortage': s})
                        shortage_total += self.shortage_penalty * s
                        total_shortage_qty += s

        total_cost = prod_cost_total + transport_pw_total + transport_wr_total + holding_total + shortage_total
        service_level = 1.0 - (total_shortage_qty / total_demand) if total_demand > 0 else 1.0
        avg_inventory = inv_sum / max(inv_count, 1)
        emissions_per_unit = carbon_total / max(total_production, 1)

        return OptimizationResult(
            status=status_str,
            scenario_name=scenario_name,
            demand_mode=demand_mode,
            objective_value=pyo.value(m.Objective),
            total_cost=total_cost,
            production_cost=prod_cost_total,
            transport_cost=transport_pw_total + transport_wr_total,
            holding_cost=holding_total,
            shortage_cost=shortage_total,
            carbon_cost=self.weights.get('carbon', 0.0) * carbon_total,
            total_emissions=carbon_total,
            emissions_per_unit=emissions_per_unit,
            service_level=service_level,
            total_shortage=total_shortage_qty,
            total_demand=total_demand,
            average_inventory=avg_inventory,
            total_production=total_production,
            production_plan=pd.DataFrame(prod_records),
            shipment_pw_plan=pd.DataFrame(ship_pw_records),
            shipment_wr_plan=pd.DataFrame(ship_wr_records),
            inventory_plan=pd.DataFrame(inv_records),
            shortage_plan=pd.DataFrame(short_records),
            solver_runtime_s=elapsed,
            horizon_days=self.horizon,
        )
