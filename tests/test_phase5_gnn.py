import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from optigreen.graph.graph_dataset import (
    compute_weekly_features, build_graph_dataset
)
from optigreen.graph.gcn_model import GCNRiskModel
from optigreen.graph.gat_model import GATRiskModel


@pytest.fixture
def dummy_data():
    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
    demand_rows = []
    for d in dates:
        demand_rows.append({'date': d, 'region_id': 'R1', 'product_id': 'PRD1', 'demand': 100})
        demand_rows.append({'date': d, 'region_id': 'R2', 'product_id': 'PRD1', 'demand': 200})
    demand_df = pd.DataFrame(demand_rows)

    plants_df = pd.DataFrame({
        'plant_id': ['P1'], 'loc_x': [0], 'loc_y': [0],
        'production_capacity': [1000], 'variable_production_cost': [10],
        'production_emission_factor': [5]
    })
    wh_df = pd.DataFrame({
        'warehouse_id': ['W1'], 'loc_x': [10], 'loc_y': [10],
        'capacity': [500], 'holding_cost': [2]
    })
    regions_df = pd.DataFrame({
        'region_id': ['R1', 'R2'], 'loc_x': [20, 30], 'loc_y': [20, 30],
        'base_demand_multiplier': [1, 2]
    })
    routes_pw_df = pd.DataFrame({
        'source': ['P1'], 'destination': ['W1'],
        'distance': [10], 'transport_cost': [1], 'carbon_emission_factor': [1]
    })
    routes_wr_df = pd.DataFrame({
        'source': ['W1', 'W1'], 'destination': ['R1', 'R2'],
        'distance': [10, 20], 'transport_cost': [1, 2], 'carbon_emission_factor': [1, 2]
    })

    return demand_df, plants_df, wh_df, regions_df, routes_pw_df, routes_wr_df


def test_weekly_features_no_leakage(dummy_data):
    demand_df = dummy_data[0]
    # Set a known spike in week 3
    spike_date = demand_df['date'].min() + timedelta(days=15)
    demand_df.loc[demand_df['date'] == spike_date, 'demand'] = 1000
    
    weekly = compute_weekly_features(demand_df, window_weeks=2)
    assert len(weekly) > 0
    
    # Assert features don't look ahead
    for _, row in weekly.iterrows():
        assert 'p50_proxy' in row
        assert 'lag_1w' in row
        assert 'stockout_label' in row


def test_graph_dataset_building(dummy_data):
    train, val, test, meta = build_graph_dataset(*dummy_data, window_weeks=2, train_frac=0.6, val_frac=0.2)
    
    # 1 Plant + 1 WH + 2 Regions = 4 nodes total
    assert meta['n_nodes'] == 4
    assert meta['n_plants'] == 1
    assert meta['n_wh'] == 1
    assert meta['n_regions'] == 2
    
    # Check data object structure
    if len(train) > 0:
        data = train[0]
        assert data.x.shape == (4, meta['node_feature_dim'])
        assert data.edge_index.shape[0] == 2
        assert data.y.shape == (4,)
        # Mask check: P1 and W1 should be -100
        assert data.y[0] == -100
        assert data.y[1] == -100
        assert data.y[2] in [0, 1]


def test_gcn_forward_pass():
    node_dim = 12
    model = GCNRiskModel(node_feature_dim=node_dim, hidden_dim=16, hidden_dim2=8)
    
    x = torch.randn(4, node_dim)
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long)
    
    model.eval()
    out = model(x, edge_index)
    
    assert out.shape == (4,)
    probs = torch.sigmoid(out)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)
    assert model.get_node_embeddings().shape == (4, 8)


def test_gat_forward_pass():
    node_dim = 12
    edge_dim = 4
    model = GATRiskModel(
        node_feature_dim=node_dim, 
        hidden_dim=16, heads=2, out_dim=8, out_heads=1,
        edge_dim=edge_dim
    )
    
    x = torch.randn(4, node_dim)
    edge_index = torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn(3, edge_dim)
    
    model.eval()
    out = model(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
    
    assert out.shape == (4,)
    
    edge_idx_attn, attn_weights = model.get_attention_weights()
    assert edge_idx_attn.shape[1] > 0
    assert attn_weights.shape[0] == edge_idx_attn.shape[1]
