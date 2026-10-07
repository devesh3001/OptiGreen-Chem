"""
Graph Dataset: Converts the supply chain time-series into PyTorch Geometric
Data objects (one per weekly snapshot per product).

Learning Problem:
  Input (week w):  node features derived from demand history up to week w
  Target (week w+1): binary stockout label per region node

Temporal Leakage Prevention:
  All features use only information available AT week w (rolling stats with shift).
  Labels use actual demand from week w+1.
  Train/val/test split is strictly chronological.

Node layout (32 nodes per snapshot):
  [0 : n_plants]             → Plant nodes
  [n_plants : n_plants+n_wh] → Warehouse nodes
  [n_plants+n_wh : end]      → Region nodes  ← only these get labels

Node feature vector (12-dim):
  [0]  primary operational feature 1 (capacity / demand)
  [1]  primary operational feature 2 (cost / spread)
  [2]  primary operational feature 3 (emission / volatility)
  [3]  secondary feature (utilization / lag / 0)
  [4]  node_type: is_region
  [5]  node_type: is_warehouse
  [6]  node_type: is_plant
  [7-11] product one-hot (5 dims)

Edge feature vector (4-dim):
  [0] distance (normalized)
  [1] transport_cost (normalized)
  [2] carbon_emission_factor (normalized)
  [3] transit_time (normalized)
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler


# ─── Node index layout ─────────────────────────────────────── #
def get_node_layout(plants_df, warehouses_df, regions_df):
    """Returns ordered node IDs and type boundaries."""
    plant_ids = list(plants_df['plant_id'])
    wh_ids = list(warehouses_df['warehouse_id'])
    region_ids = list(regions_df['region_id'])
    all_ids = plant_ids + wh_ids + region_ids
    id_to_idx = {nid: i for i, nid in enumerate(all_ids)}
    return all_ids, id_to_idx, len(plant_ids), len(wh_ids), len(region_ids)


# ─── Edge index builder ─────────────────────────────────────── #
def build_edge_index_and_attr(routes_pw_df, routes_wr_df, id_to_idx,
                               edge_scaler: Optional[StandardScaler] = None,
                               fit_scaler: bool = False):
    """
    Builds edge_index [2, E] and edge_attr [E, 4] from route dataframes.
    routes_wr_df must have: source, destination, distance, transport_cost,
                            carbon_emission_factor, (transit_time optional)
    """
    rows = []
    # Plant → Warehouse
    for _, r in routes_pw_df.iterrows():
        if r['source'] in id_to_idx and r['destination'] in id_to_idx:
            rows.append({
                'src': id_to_idx[r['source']],
                'dst': id_to_idx[r['destination']],
                'distance': r['distance'],
                'transport_cost': r['transport_cost'],
                'carbon': r['carbon_emission_factor'],
                'transit': r.get('transport_time', r.get('transit_time', 1.0)),
            })
    # Warehouse → Region
    for _, r in routes_wr_df.iterrows():
        if r['source'] in id_to_idx and r['destination'] in id_to_idx:
            rows.append({
                'src': id_to_idx[r['source']],
                'dst': id_to_idx[r['destination']],
                'distance': r['distance'],
                'transport_cost': r['transport_cost'],
                'carbon': r['carbon_emission_factor'],
                'transit': r.get('transport_time', r.get('transit_time', 1.0)),
            })

    if not rows:
        return None, None, None

    edge_df = pd.DataFrame(rows)
    edge_index = torch.tensor(
        [edge_df['src'].values, edge_df['dst'].values], dtype=torch.long
    )
    raw_attr = edge_df[['distance', 'transport_cost', 'carbon', 'transit']].values.astype(np.float32)

    if fit_scaler:
        edge_scaler = StandardScaler()
        raw_attr = edge_scaler.fit_transform(raw_attr)
    elif edge_scaler is not None:
        raw_attr = edge_scaler.transform(raw_attr)

    edge_attr = torch.tensor(raw_attr, dtype=torch.float)
    return edge_index, edge_attr, edge_scaler


# ─── Weekly feature computation ─────────────────────────────── #
def compute_weekly_features(demand_df: pd.DataFrame,
                            window_weeks: int = 4) -> pd.DataFrame:
    """
    Aggregates daily demand into weekly snapshots with rolling features.
    All features use only past information (shift before rolling).

    Returns: DataFrame with columns:
        week_start, region_id, product_id,
        p50_proxy, p90_proxy, spread, demand_vol,
        lag_1w, stockout_label (actual demand next week > p90_proxy this week)
    """
    df = demand_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week_start'] = df['date'].dt.to_period('W').apply(lambda p: p.start_time)

    # Aggregate to weekly
    weekly = (
        df.groupby(['week_start', 'region_id', 'product_id'])['demand']
        .agg(['sum', 'std'])
        .reset_index()
        .rename(columns={'sum': 'weekly_demand', 'std': 'weekly_std'})
    )
    weekly['weekly_std'] = weekly['weekly_std'].fillna(0)
    weekly = weekly.sort_values(['region_id', 'product_id', 'week_start'])

    # Rolling stats (no leakage: shift by 1 week before rolling)
    grp = weekly.groupby(['region_id', 'product_id'])

    weekly['p50_proxy'] = grp['weekly_demand'].transform(
        lambda x: x.shift(1).rolling(window_weeks, min_periods=1).mean()
    )
    weekly['demand_vol'] = grp['weekly_demand'].transform(
        lambda x: x.shift(1).rolling(window_weeks, min_periods=1).std().fillna(0)
    )
    weekly['lag_1w'] = grp['weekly_demand'].transform(lambda x: x.shift(1))
    weekly['p90_proxy'] = weekly['p50_proxy'] + 1.28 * weekly['demand_vol']
    weekly['spread'] = weekly['p90_proxy'] - weekly['p50_proxy']

    # Label: next week's demand > p90_proxy this week
    weekly['next_demand'] = grp['weekly_demand'].transform(lambda x: x.shift(-1))
    weekly['stockout_label'] = (weekly['next_demand'] > weekly['p90_proxy']).astype(int)

    # Drop first few weeks (insufficient history) and last week (no label)
    weekly = weekly.dropna(subset=['p50_proxy', 'lag_1w', 'next_demand'])

    return weekly.reset_index(drop=True)


# ─── Snapshot builder ───────────────────────────────────────── #
def build_graph_snapshot(
    week_start,
    product_id: str,
    product_idx: int,
    n_products: int,
    weekly_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    warehouses_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    all_ids: List,
    n_plants: int,
    n_wh: int,
    n_regions: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_scaler: Optional[StandardScaler] = None,
    fit_node_scaler: bool = False,
    total_capacity: float = 1.0,
    total_weekly_demand: float = 1.0,
) -> Optional[Data]:
    """Builds a single PyG Data object for (week, product)."""

    week_data = weekly_df[
        (weekly_df['week_start'] == week_start) &
        (weekly_df['product_id'] == product_id)
    ].set_index('region_id')

    if len(week_data) == 0:
        return None

    # Product one-hot
    prod_onehot = np.zeros(n_products, dtype=np.float32)
    prod_onehot[product_idx] = 1.0

    # Utilization proxy: total weekly demand / total weekly capacity
    total_weekly_capacity = total_capacity * 7
    utilization = float(total_weekly_demand / max(total_weekly_capacity, 1.0))

    node_features = []

    # Plant nodes: [capacity_norm, prod_cost_norm, emission_norm, utilization, is_region, is_wh, is_plant] + product_onehot
    for _, p in plants_df.iterrows():
        feats = [
            float(p['production_capacity']),
            float(p['variable_production_cost']),
            float(p['production_emission_factor']),
            utilization,
            0.0, 0.0, 1.0,
        ]
        node_features.append(feats + list(prod_onehot))

    # Warehouse nodes: [capacity, holding_cost, 0, 0, is_region=0, is_wh=1, is_plant=0] + product_onehot
    for _, w in warehouses_df.iterrows():
        feats = [
            float(w['capacity']),
            float(w['holding_cost']),
            0.0,
            0.0,
            0.0, 1.0, 0.0,
        ]
        node_features.append(feats + list(prod_onehot))

    # Region nodes: [p50, spread, demand_vol, lag_1w, is_region=1, is_wh=0, is_plant=0] + product_onehot
    labels = []
    for _, r in regions_df.iterrows():
        rid = r['region_id']
        if rid in week_data.index:
            row = week_data.loc[rid]
            feats = [
                float(row['p50_proxy']),
                float(row['spread']),
                float(row['demand_vol']),
                float(row['lag_1w']),
                1.0, 0.0, 0.0,
            ]
            labels.append(int(row['stockout_label']))
        else:
            feats = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            labels.append(0)
        node_features.append(feats + list(prod_onehot))

    x = np.array(node_features, dtype=np.float32)

    # Build label tensor: -100 for plant/warehouse nodes (masked), 0/1 for regions
    y_full = np.full(len(all_ids), -100, dtype=np.int64)
    for i, label in enumerate(labels):
        y_full[n_plants + n_wh + i] = label

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y_full, dtype=torch.long),
        week_start=str(week_start),
        product_id=product_id,
        n_plants=n_plants,
        n_wh=n_wh,
        n_regions=n_regions,
    )
    return data


# ─── Dataset factory ────────────────────────────────────────── #
def build_graph_dataset(
    demand_df: pd.DataFrame,
    plants_df: pd.DataFrame,
    warehouses_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    routes_pw_df: pd.DataFrame,
    routes_wr_df: pd.DataFrame,
    window_weeks: int = 4,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Data], List[Data], List[Data], Dict]:
    """
    Full pipeline: demand CSV → train/val/test PyG Data lists.

    Returns: (train_data, val_data, test_data, metadata_dict)
    """
    print("  Computing weekly features...")
    weekly_df = compute_weekly_features(demand_df, window_weeks=window_weeks)

    all_ids, id_to_idx, n_plants, n_wh, n_regions = get_node_layout(
        plants_df, warehouses_df, regions_df
    )
    n_products = len(regions_df['region_id'].unique())  # will use actual products
    products = sorted(demand_df['product_id'].unique())
    n_products = len(products)
    product_to_idx = {p: i for i, p in enumerate(products)}

    # Build WH->Region route df from synthesized distances
    _wr_rows = []
    for _, w in warehouses_df.iterrows():
        for _, r in regions_df.iterrows():
            dist = float(np.sqrt((w['loc_x']-r['loc_x'])**2 + (w['loc_y']-r['loc_y'])**2))
            _wr_rows.append({
                'source': w['warehouse_id'], 'destination': r['region_id'],
                'distance': dist, 'transport_cost': 0.05 * dist,
                'carbon_emission_factor': 0.02 * dist, 'transit_time': max(1, int(dist/200))
            })
    _routes_wr_df = pd.DataFrame(_wr_rows)

    print("  Building edge index...")
    edge_index, edge_attr, edge_scaler = build_edge_index_and_attr(
        routes_pw_df, _routes_wr_df, id_to_idx, fit_scaler=True
    )

    weeks = sorted(weekly_df['week_start'].unique())
    n_weeks = len(weeks)
    n_train = int(n_weeks * train_frac)
    n_val = int(n_weeks * val_frac)

    train_weeks = weeks[:n_train]
    val_weeks = weeks[n_train:n_train + n_val]
    test_weeks = weeks[n_train + n_val:]

    total_capacity = float(plants_df['production_capacity'].sum())

    print(f"  Weeks: {n_weeks} total | {len(train_weeks)} train | {len(val_weeks)} val | {len(test_weeks)} test")
    print(f"  Products: {n_products} | Snapshots: ~{n_weeks * n_products}")

    # Build snapshots for all weeks, then split
    def _build_for_weeks(week_list):
        snapshots = []
        for week in week_list:
            week_total_demand = float(weekly_df[
                weekly_df['week_start'] == week
            ]['weekly_demand'].sum())
            for product_id in products:
                g = build_graph_snapshot(
                    week_start=week,
                    product_id=product_id,
                    product_idx=product_to_idx[product_id],
                    n_products=n_products,
                    weekly_df=weekly_df,
                    plants_df=plants_df,
                    warehouses_df=warehouses_df,
                    regions_df=regions_df,
                    all_ids=all_ids,
                    n_plants=n_plants,
                    n_wh=n_wh,
                    n_regions=n_regions,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    total_capacity=total_capacity,
                    total_weekly_demand=week_total_demand,
                )
                if g is not None:
                    snapshots.append(g)
        return snapshots

    print("  Building train snapshots...")
    train_data = _build_for_weeks(train_weeks)
    print("  Building val snapshots...")
    val_data = _build_for_weeks(val_weeks)
    print("  Building test snapshots...")
    test_data = _build_for_weeks(test_weeks)

    # Compute stockout rate
    all_labels = []
    for d in train_data + val_data + test_data:
        mask = d.y != -100
        all_labels.extend(d.y[mask].tolist())
    stockout_rate = float(np.mean(all_labels)) if all_labels else 0.0

    metadata = {
        'n_snapshots': len(train_data) + len(val_data) + len(test_data),
        'n_train': len(train_data),
        'n_val': len(val_data),
        'n_test': len(test_data),
        'n_nodes': len(all_ids),
        'n_edges': edge_index.shape[1],
        'node_feature_dim': train_data[0].x.shape[1] if train_data else 12,
        'edge_feature_dim': edge_attr.shape[1],
        'n_plants': n_plants,
        'n_wh': n_wh,
        'n_regions': n_regions,
        'n_products': n_products,
        'products': products,
        'stockout_rate': stockout_rate,
        'train_weeks': [str(w) for w in train_weeks],
        'val_weeks': [str(w) for w in val_weeks],
        'test_weeks': [str(w) for w in test_weeks],
        'edge_scaler': edge_scaler,
        'all_ids': all_ids,
        'id_to_idx': id_to_idx,
        'product_to_idx': product_to_idx,
        'edge_index': edge_index.numpy(),
    }

    print(f"  Dataset ready. Stockout rate: {stockout_rate:.3f}")
    return train_data, val_data, test_data, metadata
