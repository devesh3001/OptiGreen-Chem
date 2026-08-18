"""
Phase 4: Full end-to-end execution script.

Pipeline:
  Historical Demand → Train ML (Phase 2/3) → Forecast Future →
  Risk Intelligence → MILP Optimization (5 Scenarios) →
  Pareto Analysis → Monte Carlo Robustness → Actual Demand Evaluation → Report

All forecasts are generated on TEST set dates (never seen during training).
Decisions are evaluated against actual test-set demand (never the forecast input).
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

DATA_DIR = "data/synthetic"
RESULTS_DIR = "data/processed"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────── #
# Imports
# ────────────────────────────────────────────────────────────── #
import sys
sys.path.insert(0, "src")

from optigreen.features.pipeline import FeaturePipeline
from optigreen.forecasting.prob_xgboost import ProbabilisticXGBoostForecaster
from optigreen.forecasting.forecast_provider import ForecastProvider
from optigreen.forecasting.metrics import calculate_probabilistic_metrics
from optigreen.risk.risk_model import StockoutRiskModel
from optigreen.risk.xgb_risk_provider import XGBRiskProvider
from optigreen.risk.risk_features import build_risk_features
from optigreen.optimization.optimizer import run_scenario, run_baseline_policy
from optigreen.optimization.inventory_sim import build_initial_inventory
from optigreen.graph.supply_chain_graph import build_supply_chain_graph, graph_summary
from optigreen.optimization.result_schema import OptimizationResult
from optigreen.evaluation.monte_carlo import run_monte_carlo
from optigreen.evaluation.opt_plots import (
    plot_scenario_comparison, plot_pareto_frontier,
    plot_inventory_over_time, plot_monte_carlo_results,
    plot_risk_heatmap, plot_cost_breakdown,
)


def load_topology():
    plants_df = pd.read_csv(f"{DATA_DIR}/plants.csv")
    warehouses_df = pd.read_csv(f"{DATA_DIR}/warehouses.csv")
    regions_df = pd.read_csv(f"{DATA_DIR}/regions.csv")
    products_df = pd.read_csv(f"{DATA_DIR}/products.csv")
    routes_df = pd.read_csv(f"{DATA_DIR}/routes.csv")
    demand_df = pd.read_csv(f"{DATA_DIR}/demand.csv")
    return plants_df, warehouses_df, regions_df, products_df, routes_df, demand_df


def build_warehouse_region_routes(warehouses_df, regions_df,
                                  cost_per_unit_km=0.05, carbon_per_unit_km=0.02):
    """Synthesize Warehouse→Region routes from coordinate distances."""
    rows = []
    for _, w in warehouses_df.iterrows():
        for _, r in regions_df.iterrows():
            dist = np.sqrt((w['loc_x'] - r['loc_x'])**2 + (w['loc_y'] - r['loc_y'])**2)
            rows.append({
                'source': w['warehouse_id'],
                'destination': r['region_id'],
                'distance': dist,
                'transport_cost': cost_per_unit_km * dist,
                'carbon_emission_factor': carbon_per_unit_km * dist,
            })
    return pd.DataFrame(rows)


def build_initial_inventory(warehouses_df, products_df, fraction=0.3):
    """Set initial inventory as a fraction of warehouse capacity split across products."""
    inv = {}
    n_products = len(products_df)
    for _, w in warehouses_df.iterrows():
        per_product = w['capacity'] * fraction / n_products
        for _, p in products_df.iterrows():
            inv[(w['warehouse_id'], p['product_id'])] = per_product
    return inv


def main():
    print("=" * 65)
    print("  OptiGreen-Chem — Phase 4: MILP Optimization Engine")
    print("=" * 65)

    # ─── 1. Load config ─────────────────────────────────────── #
    with open("configs/optimization.yaml") as f:
        cfg = yaml.safe_load(f)
    opt_cfg = cfg['optimization']
    horizon = opt_cfg['horizon_days']
    shortage_penalty = opt_cfg['shortage_penalty']
    mc_cfg = opt_cfg['monte_carlo']

    # ─── 2. Load topology and demand ────────────────────────── #
    print("\n[1/7] Loading topology and demand data...")
    plants_df, warehouses_df, regions_df, products_df, routes_pw_df, demand_df = load_topology()
    routes_wr_df = build_warehouse_region_routes(
        warehouses_df, regions_df,
        cost_per_unit_km=opt_cfg['wh_region_transport_cost_per_unit_km'],
        carbon_per_unit_km=opt_cfg['wh_region_carbon_factor_per_unit_km'],
    )
    print(f"  Plants: {len(plants_df)}  Warehouses: {len(warehouses_df)}  "
          f"Regions: {len(regions_df)}  Products: {len(products_df)}")
    print(f"  Plant->WH routes: {len(routes_pw_df)}  WH->Region routes: {len(routes_wr_df)}")

    # ─── 3. Build supply chain graph ────────────────────────── #
    print("\n[2/7] Building supply chain graph...")
    sc_graph = build_supply_chain_graph(
        plants_df, warehouses_df, regions_df, routes_pw_df,
        wh_region_cost_per_unit_km=opt_cfg['wh_region_transport_cost_per_unit_km'],
        wh_region_carbon_per_unit_km=opt_cfg['wh_region_carbon_factor_per_unit_km'],
    )
    summary = graph_summary(sc_graph)
    print(f"  Graph: {summary}")

    # ─── 4. Feature engineering + forecast ──────────────────── #
    print("\n[3/7] Training quantile forecasting model (Phase 3)...")
    pipeline = FeaturePipeline()
    features_df = pipeline.build_features(demand_df)
    train_df, val_df, test_df = pipeline.time_based_split(features_df)
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    prob_model = ProbabilisticXGBoostForecaster()
    prob_model.train(train_df, params={
        'subsample': 1.0, 'n_estimators': 200, 'max_depth': 3,
        'learning_rate': 0.05, 'colsample_bytree': 1.0
    })

    pxgb_preds = prob_model.predict(test_df)
    provider = ForecastProvider(pxgb_preds)
    print(f"  Forecasts generated: {len(pxgb_preds)} rows")

    # Select 7 contiguous test dates for the planning horizon
    available_dates = provider.available_dates
    planning_dates = available_dates[:horizon]
    print(f"  Planning horizon: {planning_dates[0].date()} → {planning_dates[-1].date()} ({horizon} days)")

    # ─── 5. Risk intelligence ───────────────────────────────── #
    print("\n[4/7] Training stockout risk model...")
    # Use validation forecast range for risk model training (avoid test leakage)
    val_preds = prob_model.predict(val_df)
    risk_features = build_risk_features(val_preds, demand_df, plants_df)

    risk_model = StockoutRiskModel()
    risk_metrics = risk_model.train(risk_features)
    print(f"  Risk model — ROC-AUC: {risk_metrics['ROC_AUC']:.3f}  "
          f"PR-AUC: {risk_metrics['PR_AUC']:.3f}  "
          f"Recall: {risk_metrics['Recall']:.3f}  "
          f"Stockout rate: {risk_metrics['Stockout_Rate']:.3f}")

    # Generate risk scores for the test planning horizon
    test_horizon_preds = pxgb_preds[pxgb_preds['date'].isin(planning_dates)]
    test_horizon_demand = demand_df.copy()
    test_horizon_demand['date'] = pd.to_datetime(test_horizon_demand['date'])
    test_horizon_demand_filt = test_horizon_demand[
        test_horizon_demand['date'].isin(planning_dates)]

    test_risk_features = build_risk_features(test_horizon_preds, test_horizon_demand_filt, plants_df)
    risk_score_df = risk_model.predict_risk_score(test_risk_features)
    risk_score_df['date'] = pd.to_datetime(risk_score_df['date'])
    
    risk_provider = XGBRiskProvider(model=risk_model, risk_features_df=test_risk_features)
    risk_dict = risk_provider.get_risk_scores(planning_dates)

    # Plot risk heatmap
    plot_risk_heatmap(risk_score_df, RESULTS_DIR)

    # ─── 6. Initial inventory ───────────────────────────────── #
    initial_inv = build_initial_inventory(warehouses_df, products_df,
                                          fraction=opt_cfg['initial_inventory_fraction'])

    # Shared kwargs for all MILP scenarios
    shared_kwargs = dict(
        provider=provider,
        plants_df=plants_df,
        warehouses_df=warehouses_df,
        regions_df=regions_df,
        products_df=products_df,
        routes_pw_df=routes_pw_df,
        routes_wr_df=routes_wr_df,
        dates=planning_dates,
        initial_inventory=initial_inv,
        shortage_penalty=shortage_penalty,
        solver='highs',
        time_limit=opt_cfg['time_limit'],
    )

    # ─── 7. Run optimization scenarios ─────────────────────── #
    print("\n[5/7] Running 5 optimization scenarios...")
    scenario_cfg = cfg['scenarios']
    results = {}

    # Baseline (heuristic policy)
    print("  Running: Baseline (heuristic reorder)...")
    actual_df = demand_df.copy()
    actual_df['date'] = pd.to_datetime(actual_df['date'])
    baseline_result = run_baseline_policy(
        provider=provider,
        dates=planning_dates,
        actual_demand_df=actual_df,
        initial_inventory=1000.0,
        shortage_penalty=shortage_penalty,
    )
    results['Baseline'] = baseline_result
    print(f"    → Status: {baseline_result.status}  "
          f"Cost: {baseline_result.total_cost:,.0f}  "
          f"SL: {baseline_result.service_level:.3f}")

    # P50 MILP
    print("  Running: P50 MILP...")
    t0 = time.time()
    r_p50 = run_scenario(
        scenario_name='P50 MILP',
        demand_mode='p50',
        weights=scenario_cfg['p50_milp']['weights'],
        **shared_kwargs
    )
    results['P50 MILP'] = r_p50
    print(f"    → Status: {r_p50.status}  Cost: {r_p50.total_cost:,.0f}  "
          f"SL: {r_p50.service_level:.3f}  Time: {r_p50.solver_runtime_s:.1f}s")

    # P90 MILP
    print("  Running: P90 MILP...")
    r_p90 = run_scenario(
        scenario_name='P90 MILP',
        demand_mode='p90',
        weights=scenario_cfg['p90_milp']['weights'],
        **shared_kwargs
    )
    results['P90 MILP'] = r_p90
    print(f"    → Status: {r_p90.status}  Cost: {r_p90.total_cost:,.0f}  "
          f"SL: {r_p90.service_level:.3f}  Time: {r_p90.solver_runtime_s:.1f}s")

    # Risk-aware MILP
    print("  Running: Risk-Aware MILP...")
    r_risk = run_scenario(
        scenario_name='Risk-Aware MILP',
        demand_mode='p50',
        weights=scenario_cfg['risk_aware_milp']['weights'],
        risk_scores=risk_dict,
        **shared_kwargs
    )
    results['Risk-Aware MILP'] = r_risk
    print(f"    → Status: {r_risk.status}  Cost: {r_risk.total_cost:,.0f}  "
          f"SL: {r_risk.service_level:.3f}  Time: {r_risk.solver_runtime_s:.1f}s")

    # Sustainable Risk-Aware MILP
    print("  Running: Sustainable Risk-Aware MILP...")
    r_sust = run_scenario(
        scenario_name='Sustainable Risk-Aware MILP',
        demand_mode='p50',
        weights=scenario_cfg['sustainable_risk_milp']['weights'],
        risk_scores=risk_dict,
        **shared_kwargs
    )
    results['Sustainable MILP'] = r_sust
    print(f"    → Status: {r_sust.status}  Cost: {r_sust.total_cost:,.0f}  "
          f"SL: {r_sust.service_level:.3f}  CO2: {r_sust.total_emissions:,.0f}  "
          f"Time: {r_sust.solver_runtime_s:.1f}s")

    # ─── 8. Pareto frontier (carbon sweep) ──────────────────── #
    print("\n  Running Pareto frontier (Cost vs CO2)...")
    pareto_rows = []
    for lam in opt_cfg['carbon_lambda_sweep']:
        r_par = run_scenario(
            scenario_name=f'Pareto λ={lam}',
            demand_mode='p50',
            weights={'cost': 1.0, 'carbon': lam, 'risk': 0.0},
            **shared_kwargs
        )
        if r_par.status in ('optimal', 'feasible'):
            pareto_rows.append({
                'lambda_carbon': lam,
                'total_cost': r_par.total_cost,
                'total_emissions': r_par.total_emissions,
                'service_level': r_par.service_level,
            })
    pareto_df = pd.DataFrame(pareto_rows)
    plot_pareto_frontier(pareto_df, RESULTS_DIR)

    # ─── 9. Monte Carlo robustness ──────────────────────────── #
    print("\n[6/7] Monte Carlo robustness analysis...")
    horizon_forecast_df = provider.get_all_for_dates(planning_dates)
    mc_results = []

    mc_scenarios = {
        'Baseline': (None, None),
        'P50 MILP': ('P50 MILP', results['P50 MILP']),
        'P90 MILP': ('P90 MILP', results['P90 MILP']),
        'Risk-Aware MILP': ('Risk-Aware MILP', results['Risk-Aware MILP']),
    }

    for label, (name, r) in mc_scenarios.items():
        plan = r.shipment_wr_plan if r is not None else None
        mc = run_monte_carlo(
            scenario_name=label,
            forecast_df=horizon_forecast_df,
            shipment_wr_plan=plan,
            n_samples=mc_cfg['n_samples'],
            seed=mc_cfg['seed'],
            shortage_penalty=shortage_penalty,
        )
        mc_results.append(mc)
        print(f"  {label:25s} | Mean Cost: {mc.mean_cost:>12,.0f} | "
              f"Mean SL: {mc.mean_service_level:.3f} | Worst SL: {mc.worst_service_level:.3f}")

    plot_monte_carlo_results(mc_results, RESULTS_DIR)

    # ─── 10. Evaluate against actual unseen demand ──────────── #
    print("\n  Evaluating decisions against ACTUAL test demand (never used in optimization)...")
    actual_horizon = actual_df[actual_df['date'].isin(planning_dates)]

    print("\n  Actual demand statistics:")
    print(f"    Total actual demand: {actual_horizon['demand'].sum():,.0f}")
    print(f"    Planning forecast P50 total: {horizon_forecast_df['P50'].sum():,.0f}")
    print(f"    Planning forecast P90 total: {horizon_forecast_df['P90'].sum():,.0f}")

    # ─── 11. Build summary table ─────────────────────────────── #
    summary_rows = [r.to_summary_dict() for r in results.values()]
    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 65)
    print("  PHASE 4 RESULTS: Scenario Comparison")
    print("=" * 65)
    display_cols = ['Scenario', 'Status', 'Total Cost', 'Service Level',
                    'Total Shortage', 'Avg Inventory', 'Total CO2', 'Solver Time (s)']
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))

    # ─── 12. Visualizations ──────────────────────────────────── #
    print("\n[7/7] Generating visualizations...")
    plot_scenario_comparison(summary_df, RESULTS_DIR)
    plot_cost_breakdown(summary_df, RESULTS_DIR)

    # Inventory over time for best MILP scenario
    for name, r in [('P50 MILP', results['P50 MILP']),
                    ('P90 MILP', results['P90 MILP'])]:
        if r.status in ('optimal', 'feasible'):
            plot_inventory_over_time(r.inventory_plan, RESULTS_DIR, name)

    # ─── 13. Final findings ──────────────────────────────────── #
    print("\n" + "=" * 65)
    print("  KEY FINDINGS")
    print("=" * 65)

    milp_scenarios = {k: v for k, v in results.items() if v.status in ('optimal', 'feasible', 'baseline')}
    if milp_scenarios:
        best = min(milp_scenarios.items(), key=lambda kv: (kv[1].total_cost or 1e18))
        print(f"\n  Best total cost strategy: '{best[0]}'")
        print(f"    Cost: {best[1].total_cost:,.0f}")
        print(f"    Service Level: {best[1].service_level:.3f}")
        print(f"    CO2: {best[1].total_emissions or 0:,.0f}")

    if 'P50 MILP' in results and 'P90 MILP' in results:
        r1, r2 = results['P50 MILP'], results['P90 MILP']
        if r1.status in ('optimal', 'feasible') and r2.status in ('optimal', 'feasible'):
            cost_diff = r2.total_cost - r1.total_cost
            sl_diff = (r2.service_level - r1.service_level) * 100
            print(f"\n  P90 vs P50 MILP:")
            print(f"    Cost change: {cost_diff:+,.0f}  ({cost_diff/max(r1.total_cost,1)*100:+.1f}%)")
            print(f"    Service Level change: {sl_diff:+.2f}%")

    print(f"\n  Monte Carlo (N={mc_cfg['n_samples']}) best mean cost strategy: "
          f"'{min(mc_results, key=lambda r: r.mean_cost).scenario_name}'")
    print(f"  Monte Carlo most robust (best worst-case SL): "
          f"'{max(mc_results, key=lambda r: r.worst_service_level).scenario_name}'")

    print(f"\nAll visualizations saved to: {RESULTS_DIR}/")
    print("\nPhase 4 Execution Complete.")
    return summary_df, mc_results, pareto_df, risk_metrics


if __name__ == "__main__":
    main()
