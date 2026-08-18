"""
Phase 6: End-to-End Decision Value Integration.
Evaluates No Risk, XGB Risk, and GAT Risk across multiple scenarios.
"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm

from optigreen.forecasting.forecast_provider import ForecastProvider
from optigreen.forecasting.prob_xgboost import ProbabilisticXGBoostForecaster
from optigreen.risk.risk_model import StockoutRiskModel
from optigreen.risk.xgb_risk_provider import XGBRiskProvider
from optigreen.risk.gnn_risk_provider import GNNRiskProvider
from optigreen.risk.risk_features import build_risk_features
from optigreen.graph.gat_model import GATRiskModel
from optigreen.graph.graph_dataset import build_graph_dataset
from optigreen.optimization.optimizer import run_scenario
from optigreen.evaluation.monte_carlo import run_monte_carlo

def build_initial_inventory(warehouses_df, products_df, fraction=0.3):
    initial_inv = {}
    for _, w_row in warehouses_df.iterrows():
        wh_cap = w_row['capacity']
        for _, p_row in products_df.iterrows():
            initial_inv[(w_row['warehouse_id'], p_row['product_id'])] = (wh_cap * fraction) / len(products_df)
    return initial_inv



DATA_DIR = "data/synthetic"
PROC_DIR = "data/processed"
OUT_DIR = "data/phase6"
os.makedirs(OUT_DIR, exist_ok=True)


def load_models_and_providers(config: dict, 
                              demand_df: pd.DataFrame, plants_df: pd.DataFrame, 
                              wh_df: pd.DataFrame, regions_df: pd.DataFrame, 
                              routes_pw: pd.DataFrame, routes_wr: pd.DataFrame, 
                              pxgb_preds: pd.DataFrame):
    """Initializes forecasting and risk models."""
    # 1. Forecasting
    forecaster = ProbabilisticXGBoostForecaster()
    try:
        forecaster.load("models/pxgb_model.json")
    except:
        print("Warning: Could not load pxgb_model.json, creating dummy.")
    provider = ForecastProvider(pxgb_preds)
    
    # 2. XGB Risk
    risk_model = StockoutRiskModel()
    
    test_demand = demand_df.copy()
    test_demand['date'] = pd.to_datetime(test_demand['date'])
    features = build_risk_features(pxgb_preds, test_demand, plants_df)
    
    try:
        risk_model.load("models/xgb_risk_model.json")
    except Exception as e:
        print(f"Training XGBRiskModel (could not load models/xgb_risk_model.json: {e})")
        risk_model.train(features)
        risk_model.save("models/xgb_risk_model.json")
        
    xgb_provider = XGBRiskProvider(risk_model, features)
    
    # 3. GAT Risk
    print("Building temporal graph dataset to get metadata for GAT...")
    train_data, val_data, test_data, metadata = build_graph_dataset(
        demand_df, plants_df, wh_df, regions_df, routes_pw, pd.DataFrame(),
        window_weeks=config['gnn']['window_weeks'],
        train_frac=config['gnn']['train_frac'],
        val_frac=config['gnn']['val_frac'],
        seed=config['gnn']['seed']
    )
    
    gat = GATRiskModel(
        node_feature_dim=metadata['node_feature_dim'],
        hidden_dim=config['gnn']['hidden_dim'],
        heads=config['gnn']['heads'],
        out_dim=32,
        out_heads=config['gnn']['out_heads'],
        dropout=config['gnn']['dropout'],
        edge_dim=metadata['edge_feature_dim']
    )
    try:
        gat.load_state_dict(torch.load("models/gat_model.pt", weights_only=True))
    except:
        print("Warning: models/gat_model.pt not found. Using untrained GAT.")
        
    from optigreen.graph.graph_dataset import compute_weekly_features
    weekly_df = compute_weekly_features(demand_df)
    
    gat_provider = GNNRiskProvider(
        model=gat,
        demand_df=demand_df,
        plants_df=plants_df,
        warehouses_df=wh_df,
        regions_df=regions_df,
        metadata=metadata,
        edge_index=torch.tensor(metadata['edge_index'], dtype=torch.long),
        edge_attr=test_data[0].edge_attr if len(test_data) > 0 else torch.ones((metadata['n_edges'], metadata['edge_feature_dim'])),
        weekly_features_df=weekly_df
    )
    
    return provider, xgb_provider, gat_provider, features


def apply_disruption(scenario, p, w, r, d):
    p = p.copy()
    w = w.copy()
    r = r.copy()
    d = d.copy()
    
    if scenario == "plant_capacity_reduction":
        p['production_capacity'] *= 0.30  # Massive 70% cut across all plants
    elif scenario == "warehouse_reduction":
        w['capacity'] *= 0.30
    elif scenario == "route_disruption":
        r['capacity'] *= 0.30 # Massive transport cut
    elif scenario == "regional_spike":
        spikes = ['R1', 'R2', 'R3', 'R4', 'R5']
        d.loc[d['region_id'].isin(spikes), 'demand'] *= 3.0
    elif scenario == "combined":
        p['production_capacity'] *= 0.50
        spikes = ['R1', 'R2', 'R3', 'R4', 'R5']
        d.loc[d['region_id'].isin(spikes), 'demand'] *= 3.0
        
    return p, w, r, d


def run_strategy(name: str, risk_scores: dict, scenario_name: str, dates: list,
                 provider: ForecastProvider, p: pd.DataFrame, w: pd.DataFrame, 
                 r: pd.DataFrame, prods: pd.DataFrame, r_pw: pd.DataFrame, r_wr: pd.DataFrame,
                 initial_inv: dict, weights: dict, actual_demand: pd.DataFrame):
    
    # 1. Run MILP to get optimization decisions
    res = run_scenario(
        scenario_name=f"{scenario_name}_{name}",
        demand_mode="p50",
        weights=weights,
        provider=provider,
        plants_df=p,
        warehouses_df=w,
        regions_df=r,
        products_df=prods,
        routes_pw_df=r_pw,
        routes_wr_df=r_wr,
        dates=dates,
        risk_scores=risk_scores,
        initial_inventory=initial_inv,
        shortage_penalty=200.0,
        time_limit=30
    )
    
    print(f"MILP Summary for {scenario_name} ({name}):\n{res.to_summary_dict()}")
    
    # 2. Evaluate against ACTUAL demand
    from optigreen.evaluation.inventory_sim import simulate_inventory_forward
    sim_res = simulate_inventory_forward(
        actual_demand=actual_demand,
        dates=dates,
        regions=r['region_id'].tolist(),
        products=prods['product_id'].tolist(),
        milp_shipments=res.shipment_wr_plan,
        initial_inv=initial_inv,
        shortage_penalty=200.0,
        holding_cost=3.0
    )
    
    # Update MILP result with actual simulated metrics to avoid circular evaluation
    res.total_shortage = sim_res['total_shortage_qty']
    res.service_level = sim_res['service_level']
    res.shortage_cost = sim_res['total_shortage_cost']
    res.holding_cost = sim_res['total_holding_cost']
    res.total_cost = res.production_cost + res.transport_cost + res.holding_cost + res.shortage_cost
    
    return res, sim_res['region_shortages']


def main():
    print("=== OptiGreen-Chem Phase 6: Decision Intelligence Evaluation ===")
    
    with open("configs/optimization.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open("configs/gnn_config.yaml", "r") as f:
        gnn_config = yaml.safe_load(f)
        config.update(gnn_config)
        
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load Data
    print("\nLoading data...")
    demand_df = pd.read_csv(os.path.join(DATA_DIR, "demand.csv"))
    plants_df = pd.read_csv(os.path.join(DATA_DIR, "plants.csv"))
    wh_df = pd.read_csv(os.path.join(DATA_DIR, "warehouses.csv"))
    regions_df = pd.read_csv(os.path.join(DATA_DIR, "regions.csv"))
    products_df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    routes = pd.read_csv(os.path.join(DATA_DIR, "routes.csv"))
    routes_pw = routes[routes['source'].str.startswith('P')].copy()
    routes_wr = routes[routes['source'].str.startswith('W')].copy()
    
    try:
        from optigreen.features.pipeline import FeaturePipeline
        from optigreen.forecasting.prob_xgboost import ProbabilisticXGBoostForecaster
        
        print("Running Feature Pipeline...")
        pipeline = FeaturePipeline()
        features_df = pipeline.build_features(demand_df)
        
        _, _, test_df = pipeline.time_based_split(features_df)
        
        print("Generating P10/P50/P90 forecasts...")
        prob_model = ProbabilisticXGBoostForecaster()
        try:
            prob_model.load("models/pxgb_model.json")
        except:
            print("Failed to load pxgb_model.json. Training on the fly...")
            train_df = features_df[features_df['date'] < '2025-01-01']
            prob_model.train(train_df)
            os.makedirs("models", exist_ok=True)
            prob_model.save("models/pxgb_model.json")
            
        pxgb_preds = prob_model.predict(test_df)
    except Exception as e:
        print(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return
        
    provider, xgb_provider, gat_provider, risk_feat_df = load_models_and_providers(
        config, demand_df, plants_df, wh_df, regions_df, routes_pw, routes_wr, pxgb_preds
    )
    
    # Planning Horizon (1 week from test set)
    test_start = demand_df['date'].max() # actually, let's just pick a week
    planning_dates = sorted(list(pxgb_preds['date'].unique()))[:7]
    print(f"\nPlanning Horizon: {planning_dates[0]} to {planning_dates[-1]}")
    
    demand_df['date'] = pd.to_datetime(demand_df['date'])
    planning_dates = [pd.to_datetime(d) for d in planning_dates]
    actual_demand = demand_df[demand_df['date'].isin(planning_dates)].copy()
    
    # Precompute risk scores for the horizon
    print("Generating risk scores...")
    risk_xgb = xgb_provider.get_risk_scores(planning_dates)
    risk_gat = gat_provider.get_risk_scores(planning_dates)
    
    scenarios = [
        "baseline",
        "regional_spike",
        "plant_capacity_reduction",
        "warehouse_reduction",
        "route_disruption",
        "combined"
    ]
    
    results = []
    
    # Base configuration
    initial_inv = build_initial_inventory(wh_df, products_df, fraction=0.0)
    weights = {'cost': 1.0, 'carbon': 0.1, 'risk': 500.0}
    
    print("\n--- Running Scenarios ---")
    for scen in scenarios:
        print(f"\nScenario: {scen}")
        # Apply disruptions
        p, w, r_pw, d_actual = apply_disruption(scen, plants_df, wh_df, routes_pw, actual_demand)
        
        # In a real setup, forecast would change if demand spikes, but we assume forecast is fixed P50 
        # (disruptions are "unseen shocks"). For upstream cuts (plant/wh), forecast doesn't change anyway.
        
        for strat, r_scores in [('No_Risk', {}), ('XGB_Risk', risk_xgb), ('GAT_Risk', risk_gat)]:
            print(f"  Strategy: {strat}")
            res, _ = run_strategy(
                name=strat, risk_scores=r_scores, scenario_name=scen, dates=planning_dates,
                provider=provider, p=p, w=w, r=regions_df, prods=products_df, r_pw=r_pw, r_wr=routes_wr,
                initial_inv=initial_inv, weights=weights, actual_demand=d_actual
            )
            
            results.append({
                'Scenario': scen,
                'Strategy': strat,
                'Total Cost': res.total_cost,
                'Production Cost': res.production_cost,
                'Transport Cost': res.transport_cost,
                'Holding Cost': res.holding_cost,
                'Shortage Cost': res.shortage_cost,
                'Service Level': res.service_level,
                'Total Shortage': res.total_shortage,
                'CO2': res.total_emissions,
            })
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "scenario_results.csv"), index=False)
    print("\nFinal Scenario Results:")
    print(res_df.groupby(['Scenario', 'Strategy'])[['Total Cost', 'Service Level']].mean())
    
    print("\nGenerating Plots...")
    from optigreen.evaluation.phase6_plots import plot_disruption_strategy_comparison, plot_phase6_pareto_frontier, plot_monte_carlo_risk_comparison
    plot_disruption_strategy_comparison(res_df, OUT_DIR)
    plot_phase6_pareto_frontier(res_df, OUT_DIR)
    
    print("\nPhase 6 initial evaluation complete.")
if __name__ == "__main__":
    main()
