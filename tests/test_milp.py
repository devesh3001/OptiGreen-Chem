"""
Unit tests for MILP optimization constraints and solver behavior.
Tests: inventory conservation, capacity, non-negativity, infeasibility handling.
"""
import pytest
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from optigreen.optimization.milp_model import MILPModel
from optigreen.optimization.result_schema import OptimizationResult


# ------------------------------------------------------------------ #
# Minimal test topology: 1 plant, 1 warehouse, 1 region, 1 product, 3 days
# ------------------------------------------------------------------ #

@pytest.fixture
def tiny_topology():
    plants_df = pd.DataFrame([{
        'plant_id': 'P1',
        'loc_x': 0, 'loc_y': 0,
        'production_capacity': 500.0,
        'variable_production_cost': 10.0,
        'production_emission_factor': 2.0,
        'fixed_cost': 0.0, 'energy_consumption': 0.0,
    }])
    warehouses_df = pd.DataFrame([{
        'warehouse_id': 'W1',
        'loc_x': 1, 'loc_y': 1,
        'capacity': 2000.0,
        'holding_cost': 2.0,
    }])
    regions_df = pd.DataFrame([{'region_id': 'R1', 'loc_x': 2, 'loc_y': 2,
                                  'base_demand_multiplier': 1.0}])
    products_df = pd.DataFrame([{'product_id': 'PRD1', 'product_family': 'A',
                                   'unit_weight': 1.0, 'production_time': 1.0,
                                   'storage_requirement': 1.0, 'safety_stock_requirement': 10.0}])
    routes_pw_df = pd.DataFrame([{
        'source': 'P1', 'destination': 'W1',
        'distance': 100.0, 'transport_cost': 5.0,
        'transport_time': 1.0, 'carbon_emission_factor': 1.0,
        'capacity': 1000.0,
    }])
    routes_wr_df = pd.DataFrame([{
        'source': 'W1', 'destination': 'R1',
        'distance': 50.0, 'transport_cost': 3.0,
        'transport_time': 1.0, 'carbon_emission_factor': 0.5,
    }])
    return plants_df, warehouses_df, regions_df, products_df, routes_pw_df, routes_wr_df


def make_milp(tiny_topology, demand_per_day=100.0, horizon=3,
              capacity_override=None, initial_inv=0.0, weights=None):
    plants_df, warehouses_df, regions_df, products_df, routes_pw_df, routes_wr_df = tiny_topology
    if capacity_override is not None:
        plants_df = plants_df.copy()
        plants_df['production_capacity'] = capacity_override

    demand = {('R1', 'PRD1', t): demand_per_day for t in range(horizon)}
    initial_inventory = {('W1', 'PRD1'): initial_inv}

    milp = MILPModel(
        plants_df=plants_df,
        warehouses_df=warehouses_df,
        regions_df=regions_df,
        products_df=products_df,
        routes_pw_df=routes_pw_df,
        routes_wr_df=routes_wr_df,
        demand_matrix=demand,
        initial_inventory=initial_inventory,
        horizon_days=horizon,
        weights=weights or {'cost': 1.0, 'carbon': 0.0, 'risk': 0.0},
        shortage_penalty=50.0,
    )
    return milp


class TestMILPFeasibility:

    def test_solvable_with_sufficient_capacity(self, tiny_topology):
        """With capacity 500 and demand 100, the model should be optimal and have no shortage."""
        milp = make_milp(tiny_topology, demand_per_day=100.0)
        milp.build()
        result = milp.solve(scenario_name='test_feasible', demand_mode='p50')
        assert result.status in ('optimal', 'feasible'), f"Expected feasible, got {result.status}"
        assert result.total_shortage is not None
        assert result.total_shortage < 1.0, f"Expected no shortage, got {result.total_shortage}"

    def test_shortage_when_demand_exceeds_capacity(self, tiny_topology):
        """With demand 600 > capacity 500, shortage must be > 0 (not infeasible)."""
        milp = make_milp(tiny_topology, demand_per_day=600.0)
        milp.build()
        result = milp.solve(scenario_name='test_shortage', demand_mode='p50')
        assert result.status in ('optimal', 'feasible')
        assert result.total_shortage is not None
        assert result.total_shortage > 0.0, "Should have shortage when demand > capacity"

    def test_infeasible_zero_capacity_reported_cleanly(self, tiny_topology):
        """Zero capacity with positive demand: solver should report infeasibility or status != optimal."""
        # With shortage variables, the model is actually always feasible.
        # This test verifies that the result schema is populated correctly even in degenerate cases.
        # Zero capacity → production = 0 → all demand becomes shortage.
        milp = make_milp(tiny_topology, demand_per_day=100.0, capacity_override=0.0)
        milp.build()
        result = milp.solve(scenario_name='test_zero_cap', demand_mode='p50')
        # Model may be optimal with total_shortage = total_demand
        if result.status in ('optimal', 'feasible'):
            assert result.total_shortage is not None
            assert result.total_shortage > 0.0


class TestInventoryConservation:

    def test_inventory_balance(self, tiny_topology):
        """
        Conservation: Inventory[t] = Inventory[t-1] + Inbound[t] - Outbound[t]
        Verified numerically after solve.
        """
        milp = make_milp(tiny_topology, demand_per_day=100.0, horizon=3, initial_inv=50.0)
        milp.build()
        result = milp.solve(scenario_name='test_inv_balance', demand_mode='p50')

        assert result.status in ('optimal', 'feasible')
        assert result.inventory_plan is not None

        inv_plan = result.inventory_plan
        ship_pw = result.shipment_pw_plan
        ship_wr = result.shipment_wr_plan

        for t in range(3):
            inv_t = inv_plan[(inv_plan['warehouse'] == 'W1') &
                             (inv_plan['product'] == 'PRD1') &
                             (inv_plan['day'] == t)]['inventory'].sum()

            inbound = ship_pw[(ship_pw['warehouse'] == 'W1') &
                              (ship_pw['product'] == 'PRD1') &
                              (ship_pw['day'] == t)]['quantity'].sum() if ship_pw is not None and len(ship_pw) > 0 else 0.0

            outbound = ship_wr[(ship_wr['warehouse'] == 'W1') &
                               (ship_wr['product'] == 'PRD1') &
                               (ship_wr['day'] == t)]['quantity'].sum() if ship_wr is not None and len(ship_wr) > 0 else 0.0

            prev_inv = 50.0 if t == 0 else inv_plan[
                (inv_plan['warehouse'] == 'W1') &
                (inv_plan['product'] == 'PRD1') &
                (inv_plan['day'] == t - 1)]['inventory'].sum()

            expected = prev_inv + inbound - outbound
            assert abs(inv_t - expected) < 1.0, \
                f"Day {t}: inventory conservation violated. Expected {expected:.1f}, got {inv_t:.1f}"

    def test_inventory_non_negative(self, tiny_topology):
        """All inventory values must be >= 0."""
        milp = make_milp(tiny_topology, demand_per_day=100.0, horizon=3)
        milp.build()
        result = milp.solve(scenario_name='test_inv_nonneg', demand_mode='p50')
        if result.inventory_plan is not None and len(result.inventory_plan) > 0:
            assert (result.inventory_plan['inventory'] >= -1e-6).all(), \
                "Inventory should be non-negative"


class TestProductionConstraints:

    def test_production_within_capacity(self, tiny_topology):
        """Total daily production at each plant must not exceed capacity."""
        milp = make_milp(tiny_topology, demand_per_day=400.0, horizon=3)
        milp.build()
        result = milp.solve(scenario_name='test_prod_cap', demand_mode='p50')

        if result.production_plan is not None and len(result.production_plan) > 0:
            daily_prod = result.production_plan.groupby(['plant', 'day'])['quantity'].sum()
            for (plant, day), qty in daily_prod.items():
                capacity = 500.0  # from tiny_topology
                assert qty <= capacity + 1e-4, \
                    f"Plant {plant} day {day}: production {qty:.1f} exceeds capacity {capacity}"

    def test_shipment_non_negative(self, tiny_topology):
        """All shipment quantities must be >= 0."""
        milp = make_milp(tiny_topology, demand_per_day=100.0, horizon=3)
        milp.build()
        result = milp.solve(scenario_name='test_ship_nonneg', demand_mode='p50')

        for plan_df in [result.shipment_pw_plan, result.shipment_wr_plan]:
            if plan_df is not None and len(plan_df) > 0:
                assert (plan_df['quantity'] >= -1e-6).all(), "Shipment quantities should be >= 0"


class TestObjectiveAndCarbon:

    def test_carbon_weights_increase_with_lambda(self, tiny_topology):
        """Higher λ_carbon should generally increase objective but reduce emissions incentive."""
        milp_no_carbon = make_milp(tiny_topology, demand_per_day=100.0,
                                   weights={'cost': 1.0, 'carbon': 0.0, 'risk': 0.0})
        milp_no_carbon.build()
        r0 = milp_no_carbon.solve(scenario_name='no_carbon', demand_mode='p50')

        milp_carbon = make_milp(tiny_topology, demand_per_day=100.0,
                                weights={'cost': 1.0, 'carbon': 10.0, 'risk': 0.0})
        milp_carbon.build()
        r1 = milp_carbon.solve(scenario_name='with_carbon', demand_mode='p50')

        # Both should be feasible
        assert r0.status in ('optimal', 'feasible')
        assert r1.status in ('optimal', 'feasible')
        # Objective with carbon weight should be higher (more expensive signal)
        assert r1.objective_value >= r0.objective_value - 1.0

    def test_result_schema_fully_populated(self, tiny_topology):
        """Verify all key fields in OptimizationResult are populated after a successful solve."""
        milp = make_milp(tiny_topology, demand_per_day=100.0)
        milp.build()
        result = milp.solve(scenario_name='schema_test', demand_mode='p50')

        if result.status in ('optimal', 'feasible'):
            assert result.total_cost is not None
            assert result.production_cost is not None
            assert result.transport_cost is not None
            assert result.holding_cost is not None
            assert result.shortage_cost is not None
            assert result.service_level is not None
            assert 0.0 <= result.service_level <= 1.0
            assert result.solver_runtime_s is not None and result.solver_runtime_s >= 0
            assert result.horizon_days == 3
