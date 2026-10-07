"""
Risk feature engineering from forecast outputs.
All features are derived from P10/P50/P90 and supply chain topology.
No future information is used.
"""
import pandas as pd
import numpy as np


def build_risk_features(forecast_df: pd.DataFrame,
                        demand_df: pd.DataFrame,
                        plants_df: pd.DataFrame,
                        horizon_days: int = 7) -> pd.DataFrame:
    """
    Constructs risk features per (region_id, product_id, date).

    Features:
      - forecast_spread: P90 - P10 (absolute uncertainty)
      - relative_spread: (P90 - P10) / (P50 + 1) — normalised uncertainty
      - p50_level: median demand forecast
      - p90_level: upper demand forecast
      - demand_volatility: rolling std of actual demand (past 14 days)
      - safety_stock_coverage: how many days of P50 a fixed safety stock covers
      - total_plant_capacity: total daily production capacity (shared across products)
      - capacity_utilization: P50 total demand / total plant capacity
      - forecast_bias: P50 - P10 (asymmetry indicator)

    Target (for classification):
      - stockout_label: 1 if actual demand in next horizon_days > some threshold
        (Simulated: 1 if actual demand > P90 demand level, which implies a stockout
         is likely under median planning)
    """
    df = forecast_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Core forecast uncertainty features
    df['forecast_spread'] = df['P90'] - df['P10']
    df['relative_spread'] = df['forecast_spread'] / (df['P50'] + 1)
    df['p50_level'] = df['P50']
    df['p90_level'] = df['P90']
    df['forecast_bias'] = df['P50'] - df['P10']

    # Demand volatility from actual historical demand
    actual = demand_df.copy()
    actual['date'] = pd.to_datetime(actual['date'])
    actual_sorted = actual.sort_values(['region_id', 'product_id', 'date'])
    actual_sorted['demand_vol_14'] = (
        actual_sorted.groupby(['region_id', 'product_id'])['demand']
        .transform(lambda x: x.shift(1).rolling(14, min_periods=1).std())
        .fillna(0)
    )
    vol_df = actual_sorted[['date', 'region_id', 'product_id', 'demand_vol_14']]
    df = pd.merge(df, vol_df, on=['date', 'region_id', 'product_id'], how='left')
    df['demand_vol_14'] = df['demand_vol_14'].fillna(0)

    # Capacity utilization: ratio of total P50 demand to total plant capacity
    total_capacity = plants_df['production_capacity'].sum()
    daily_p50_total = df.groupby('date')['P50'].sum().reset_index(name='total_p50')
    df = pd.merge(df, daily_p50_total, on='date', how='left')
    df['capacity_utilization'] = df['total_p50'] / (total_capacity + 1)

    # Stockout risk label: 1 if actual demand > P90 (model missed the spike)
    # Merge from actual demand on the same date/region/product
    actual_demand = actual_sorted[['date', 'region_id', 'product_id', 'demand']].copy()
    actual_demand['date'] = pd.to_datetime(actual_demand['date'])
    df['date'] = pd.to_datetime(df['date'])
    # Drop any pre-existing demand column to avoid suffix conflicts
    df = df.drop(columns=['demand'], errors='ignore')
    df = pd.merge(df, actual_demand, on=['date', 'region_id', 'product_id'], how='left')
    df['stockout_label'] = (df['demand'] > df['P90']).fillna(0).astype(int)

    feature_cols = [
        'date', 'region_id', 'product_id',
        'forecast_spread', 'relative_spread',
        'p50_level', 'p90_level', 'forecast_bias',
        'demand_vol_14', 'capacity_utilization',
        'stockout_label'
    ]

    return df[feature_cols].dropna(subset=['p50_level'])
