import streamlit as st
import pandas as pd
import numpy as np
import yaml
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import os
import sys

# Ensure src is in PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from optigreen.forecasting.forecast_provider import ForecastProvider
from optigreen.forecasting.prob_xgboost import ProbabilisticXGBoostForecaster
from optigreen.risk.risk_model import StockoutRiskModel
from optigreen.risk.xgb_risk_provider import XGBRiskProvider
from optigreen.risk.gnn_risk_provider import GNNRiskProvider
from optigreen.risk.risk_features import build_risk_features
from optigreen.graph.gat_model import GATRiskModel
from optigreen.graph.graph_dataset import build_graph_dataset, compute_weekly_features
from optigreen.optimization.optimizer import run_scenario

st.set_page_config(
    page_title="OptiGreen-Chem | AI Decision Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Data ---
@st.cache_data
def load_config():
    with open(ROOT_DIR / "configs" / "optimization.yaml") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_datasets():
    data_dir = ROOT_DIR / "data" / "synthetic"
    proc_dir = ROOT_DIR / "data" / "processed"
    
    demand = pd.read_csv(data_dir / "demand.csv")
    plants = pd.read_csv(data_dir / "plants.csv")
    wh = pd.read_csv(data_dir / "warehouses.csv")
    prods = pd.read_csv(data_dir / "products.csv")
    regions = pd.read_csv(data_dir / "regions.csv")
    routes = pd.read_csv(data_dir / "routes.csv")
    
    if not (proc_dir / "pxgb_preds.csv").exists():
        raise FileNotFoundError("Missing prediction artifact: data/processed/pxgb_preds.csv")
    preds = pd.read_csv(proc_dir / "pxgb_preds.csv")
        
    # Split routes based on prefix
    routes_pw = routes[routes['source'].str.startswith('P')]
    routes_wr = routes[routes['source'].str.startswith('W')]
    
    # Precompute features
    demand['date'] = pd.to_datetime(demand['date'])
    features = build_risk_features(preds, demand, plants)
    
    return demand, plants, wh, prods, regions, routes_pw, routes_wr, preds, features

# --- Caching Models ---
@st.cache_resource
def load_models_and_providers(_demand, _plants, _wh, _regions, _routes_pw, _routes_wr, _preds, _features, config):
    try:
        # 1. Forecasting
        forecaster = ProbabilisticXGBoostForecaster()
        model_path = ROOT_DIR / "models" / "pxgb_model_P50.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing artifact: {model_path}")
            
        provider = ForecastProvider(_preds)
        
        # 2. XGB Risk
        risk_model = StockoutRiskModel()
        xgb_path = ROOT_DIR / "models" / "xgb_risk_model.json"
        if not xgb_path.exists():
            raise FileNotFoundError(f"Missing artifact: {xgb_path}")
        risk_model.load(str(xgb_path))
        xgb_provider = XGBRiskProvider(risk_model, _features)
        
        # 3. GAT Risk
        # Build minimal graph to get metadata
        train_data, val_data, test_data, metadata = build_graph_dataset(
            _demand, _plants, _wh, _regions, _routes_pw, pd.DataFrame(),
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
        gat_path = ROOT_DIR / "models" / "gat_model.pt"
        if not gat_path.exists():
            raise FileNotFoundError(f"Missing artifact: {gat_path}")
            
        gat.load_state_dict(torch.load(gat_path, weights_only=True))
        weekly_df = compute_weekly_features(_demand)
        
        gat_provider = GNNRiskProvider(
            model=gat,
            demand_df=_demand,
            plants_df=_plants,
            warehouses_df=_wh,
            regions_df=_regions,
            metadata=metadata,
            edge_index=torch.tensor(metadata['edge_index'], dtype=torch.long),
            edge_attr=test_data[0].edge_attr if len(test_data) > 0 else torch.ones((metadata['n_edges'], metadata['edge_feature_dim'])),
            weekly_features_df=weekly_df
        )
        
        return provider, xgb_provider, gat_provider
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# --- Helper logic ---
def apply_disruption(scenario, p, w, r, d):
    p = p.copy()
    w = w.copy()
    r = r.copy()
    d = d.copy()
    
    if scenario == "plant_capacity_reduction":
        p['production_capacity'] *= 0.30
    elif scenario == "warehouse_reduction":
        w['capacity'] *= 0.30
    elif scenario == "route_disruption":
        r['capacity'] *= 0.30
    elif scenario == "regional_spike":
        spikes = ['R1', 'R2', 'R3', 'R4', 'R5']
        d.loc[d['region_id'].isin(spikes), 'demand'] *= 3.0
    elif scenario == "combined":
        p['production_capacity'] *= 0.50
        spikes = ['R1', 'R2', 'R3', 'R4', 'R5']
        d.loc[d['region_id'].isin(spikes), 'demand'] *= 3.0
        
    return p, w, r, d

# --- Load state ---
config = load_config()
demand_df, plants_df, wh_df, prods_df, regions_df, routes_pw, routes_wr, pxgb_preds, risk_features = load_datasets()
forecast_provider, xgb_provider, gat_provider = load_models_and_providers(
    demand_df, plants_df, wh_df, regions_df, routes_pw, routes_wr, pxgb_preds, risk_features, config
)

# --- UI Sidebar ---
st.sidebar.title("🌿 OptiGreen-Chem")
st.sidebar.markdown("### AI Decision Intelligence")
tab_selection = st.sidebar.radio(
    "Navigation", 
    ["Overview", "Demand Forecast", "Risk Intelligence", "Optimization", "Sustainability", "Robustness", "Model Comparison"]
)

# --- 1. OVERVIEW ---
if tab_selection == "Overview":
    st.title("OptiGreen-Chem: Overview")
    st.markdown("### AI-Driven Sustainable Chemical Supply Chain Decision Intelligence")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Plants", len(plants_df))
    col2.metric("Warehouses", len(wh_df))
    col3.metric("Regions", len(regions_df))
    col4.metric("Products", len(prods_df))
    col5.metric("Network Edges", len(routes_pw) + len(routes_wr))
    
    st.markdown("""
    **Architecture Flow:**
    1. **Probabilistic Demand Forecasting:** XGBoost predicts demand uncertainty (P10/P50/P90).
    2. **Risk Intelligence:** XGBoost and GAT (Graph Attention Network) models predict supply chain disruption risks.
    3. **MILP Optimization:** Pyomo uses these predictions to make optimal, risk-averse routing, production, and inventory decisions.
    4. **Sustainability:** The objective balances cost with CO2 emissions.
    """)
    
    with st.expander("Technical Details"):
        st.markdown("""
        - **Forecasting:** Probabilistic XGBoost (P10/P50/P90 Quantile regression)
        - **Risk:** XGBoost classifier and GAT graph neural network
        - **Optimization:** Pyomo with HiGHS MILP Solver
        - **Robustness:** Monte Carlo evaluation
        - **Graph:** 32 nodes, 192 edges, temporal network analysis
        """)

# --- 2. DEMAND FORECAST ---
elif tab_selection == "Demand Forecast":
    st.title("Probabilistic Demand Forecast")
    
    col1, col2 = st.columns(2)
    selected_region = col1.selectbox("Region", regions_df['region_id'].unique())
    selected_product = col2.selectbox("Product", prods_df['product_id'].unique())
    
    st.markdown("Comparing historical demand with P10, P50, and P90 uncertainty bands.")
    
    # Filter data
    d_subset = demand_df[(demand_df['region_id'] == selected_region) & (demand_df['product_id'] == selected_product)].copy()
    d_subset['date'] = pd.to_datetime(d_subset['date'])
    p_subset = pxgb_preds[(pxgb_preds['region_id'] == selected_region) & (pxgb_preds['product_id'] == selected_product)].copy()
    p_subset['date'] = pd.to_datetime(p_subset['date'])
    
    if not p_subset.empty:
        fig = go.Figure()
        
        # Uncertainty band
        fig.add_trace(go.Scatter(
            x=p_subset['date'].tolist() + p_subset['date'].tolist()[::-1],
            y=p_subset['P90'].tolist() + p_subset['P10'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='P10 - P90 Band'
        ))
        
        # P50 Forecast
        fig.add_trace(go.Scatter(
            x=p_subset['date'], y=p_subset['P50'],
            line=dict(color='blue', dash='dash'),
            name='P50 Forecast'
        ))
        
        # Actual Demand
        merged = pd.merge(p_subset, d_subset, on=['date', 'region_id', 'product_id'], how='inner')
        if not merged.empty:
            fig.add_trace(go.Scatter(
                x=merged['date'], y=merged['demand'],
                mode='lines+markers', line=dict(color='black'),
                name='Actual Demand'
            ))
            
        fig.update_layout(title=f"Forecast vs Actual: {selected_region} - {selected_product}", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No prediction data available for this selection.")

# --- 3. RISK INTELLIGENCE ---
elif tab_selection == "Risk Intelligence":
    st.title("Supply Chain Risk Intelligence")
    st.markdown("XGBoost performed better on pure prediction (PR-AUC 0.470 vs GAT 0.423). However, GAT provides topological propagation awareness.")
    
    date_val = pxgb_preds['date'].max()
    
    col1, col2 = st.columns(2)
    selected_product = col1.selectbox("Product", prods_df['product_id'].unique())
    scenario = col2.selectbox("Disruption Scenario", ["normal", "plant_capacity_reduction", "route_disruption", "regional_spike", "combined"])
    
    if st.button("Analyze Risk Profile"):
        p, w, r, d = apply_disruption(scenario, plants_df, wh_df, routes_pw, demand_df)
        
        xgb_scores = xgb_provider.get_risk_scores(date_val)
        gat_scores = gat_provider.get_risk_scores(date_val)
        
        x_sub = xgb_scores[(xgb_scores['date'] == date_val) & (xgb_scores['product_id'] == selected_product)]
        g_sub = gat_scores[(gat_scores['date'] == date_val) & (gat_scores['product_id'] == selected_product)]
        
        merged = pd.merge(x_sub[['region_id', 'risk_score']], g_sub[['region_id', 'risk_score']], on='region_id', suffixes=('_XGB', '_GAT'))
        merged = merged.sort_values('risk_score_XGB', ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=merged['region_id'], y=merged['risk_score_XGB'], name='XGB Risk'))
        fig.add_trace(go.Bar(x=merged['region_id'], y=merged['risk_score_GAT'], name='GAT Risk'))
        fig.update_layout(barmode='group', title=f"Risk Comparison for {selected_product} on {date_val}")
        st.plotly_chart(fig, use_container_width=True)

# --- 4. OPTIMIZATION ---
elif tab_selection == "Optimization":
    st.title("Decision Intelligence Optimizer")
    
    with st.form("opt_form"):
        col1, col2, col3 = st.columns(3)
        forecast_mode = col1.selectbox("Forecast Strategy", ["p50", "p90"])
        risk_strat = col2.selectbox("Risk Intelligence", ["No Risk", "XGB Risk", "GAT Risk"])
        scenario = col3.selectbox("Scenario", ["normal", "plant_capacity_reduction", "warehouse_reduction", "route_disruption", "regional_spike", "combined"])
        
        submit = st.form_submit_button("Run Optimization", type="primary")
        
    if submit:
        with st.spinner(f"Running Pyomo MILP optimization for {scenario}..."):
            try:
                p, w, r_pw, d = apply_disruption(scenario, plants_df, wh_df, routes_pw, demand_df)
                
                # Mock dates for the scenario evaluation (first 5 days)
                dates = sorted(d['date'].unique())[:5]
                date_val = dates[0]
                
                if risk_strat == "No Risk":
                    scores = None
                elif risk_strat == "XGB Risk":
                    scores_df = xgb_provider.get_risk_scores(date_val)
                    scores = {(row['region_id'], row['product_id']): row['risk_score'] for _, row in scores_df.iterrows()}
                else:
                    scores_df = gat_provider.get_risk_scores(date_val)
                    scores = {(row['region_id'], row['product_id']): row['risk_score'] for _, row in scores_df.iterrows()}
                    
                res = run_scenario(
                    scenario_name=f"{scenario}_{risk_strat}",
                    demand_mode=forecast_mode,
                    weights=config['optimization']['weights'],
                    provider=forecast_provider,
                    plants_df=p,
                    warehouses_df=w,
                    regions_df=regions_df,
                    products_df=prods_df,
                    routes_pw_df=r_pw,
                    routes_wr_df=routes_wr,
                    dates=dates,
                    initial_inventory={},
                    risk_scores=scores
                )
                
                st.session_state["opt_result"] = res
                st.success("Optimization Complete!")
            except Exception as e:
                st.error(f"Optimizer failed: {str(e)}")
                
    if "opt_result" in st.session_state:
        res = st.session_state["opt_result"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Cost", f"${res.total_cost:,.0f}")
        c2.metric("Service Level", f"{res.service_level*100:.1f}%")
        c3.metric("Total CO2", f"{res.total_carbon:,.0f} kg")
        c4.metric("Solver Time", f"{res.solver_time:.2f} s")
        
        with st.expander("Cost Breakdown"):
            b_df = pd.DataFrame([
                {"Category": "Production", "Cost": res.production_cost},
                {"Category": "Transport", "Cost": res.transport_cost},
                {"Category": "Holding", "Cost": res.holding_cost},
                {"Category": "Shortage", "Cost": res.shortage_cost}
            ])
            fig = px.pie(b_df, values='Cost', names='Category', title="Cost Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
        with st.expander("Shortage Summary"):
            if not res.shortages.empty:
                st.dataframe(res.shortages)
            else:
                st.info("No shortages observed in this scenario.")

# --- 5. SUSTAINABILITY ---
elif tab_selection == "Sustainability":
    st.title("Sustainability & Trade-offs")
    st.markdown("""
    Phase 4/6 findings revealed that the Sustainable MILP reduced CO2 emissions while introducing only a minimal cost increase. 
    However, under severe scarcity, higher service levels and lower CO2 require significant cost trade-offs.
    """)
    st.info("Pareto frontier visualisations from offline research would be displayed here (loaded from precomputed data).")
    
# --- 6. ROBUSTNESS ---
elif tab_selection == "Robustness":
    st.title("Monte Carlo Robustness")
    st.markdown("""
    Monte Carlo evaluation analyzes robustness under demand uncertainty.
    Instead of re-running expensive simulations (N=100), we summarize the project findings.
    """)
    st.info("Monte Carlo results (Mean Cost, Cost Std Dev, 5th/95th Percentile Service Levels) would be plotted here.")

# --- 7. MODEL COMPARISON ---
elif tab_selection == "Model Comparison":
    st.title("Offline Research Results")
    st.markdown("Honest representation of the experimental metrics.")
    
    st.subheader("1. Demand Forecasting (WAPE)")
    metrics = {
        "Model": ["Seasonal Naive", "Moving Average", "XGBoost", "Probabilistic XGBoost", "TFT-lite"],
        "WAPE": ["19.4%", "18.2%", "13.9%", "14.0%", "17.1%"]
    }
    st.table(pd.DataFrame(metrics))
    
    st.subheader("2. Risk Intelligence (PR-AUC)")
    risk_metrics = {
        "Model": ["Logistic Regression", "XGBoost", "GCN", "GAT"],
        "PR-AUC": ["0.312", "0.470", "0.410", "0.423"]
    }
    st.table(pd.DataFrame(risk_metrics))
    
    st.markdown("""
    **Conclusions:**
    1. XGBoost forecasting (13.9% WAPE) was stronger than the minimalist TFT.
    2. XGBoost Risk (PR-AUC 0.470) outperformed Graph Attention Networks (0.423) purely in classification.
    3. GAT proved useful strictly for tracking network-aware disruption propagation.
    """)
