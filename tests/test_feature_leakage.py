import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from optigreen.features.pipeline import FeaturePipeline

@pytest.fixture
def sample_demand_data():
    dates = pd.date_range('2024-01-01', '2024-03-31')
    n_days = len(dates)
    
    # 1 region, 1 product
    data = []
    base = 100
    for d in dates:
        data.append({
            'date': d,
            'region_id': 'R1',
            'product_id': 'P1',
            'demand': base
        })
        base += 10 # increasing trend
        
    return pd.DataFrame(data)

def test_feature_leakage(sample_demand_data):
    # We will introduce a massive shock at day 50 and verify that it doesn't affect
    # any features generated for day 50 or earlier.
    
    df1 = sample_demand_data.copy()
    df2 = sample_demand_data.copy()
    
    # Introduce shock at index 50 (which is day 51 since 0-indexed)
    shock_index = 50
    df2.loc[shock_index, 'demand'] = 999999
    
    pipeline = FeaturePipeline()
    feat1 = pipeline.build_features(df1)
    feat2 = pipeline.build_features(df2)
    
    # After build_features, the first 28 days are dropped due to lag_28.
    # The original index 50 becomes index 22 in the new dataframe.
    
    shock_date = df2.loc[shock_index, 'date']
    
    feat1_pre_shock = feat1[feat1['date'] <= shock_date].reset_index(drop=True)
    feat2_pre_shock = feat2[feat2['date'] <= shock_date].reset_index(drop=True)
    
    # Verify that all features (except 'demand' itself at the shock date) are identical
    cols_to_check = [c for c in feat1.columns if c != 'demand']
    
    pd.testing.assert_frame_equal(feat1_pre_shock[cols_to_check], feat2_pre_shock[cols_to_check])
    
    # Verify the shock actually influenced features AFTER the shock date
    feat1_post_shock = feat1[feat1['date'] > shock_date].reset_index(drop=True)
    feat2_post_shock = feat2[feat2['date'] > shock_date].reset_index(drop=True)
    
    # lag_1 for the day after the shock should be different
    assert feat1_post_shock.loc[0, 'lag_1'] != feat2_post_shock.loc[0, 'lag_1']
    assert feat2_post_shock.loc[0, 'lag_1'] == 999999

def test_time_based_split(sample_demand_data):
    pipeline = FeaturePipeline()
    feat = pipeline.build_features(sample_demand_data)
    
    train, val, test = pipeline.time_based_split(feat, train_ratio=0.7, val_ratio=0.15)
    
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    
    # Check strict chronological order boundaries
    assert train['date'].max() < val['date'].min()
    assert val['date'].max() < test['date'].min()
    
    # No overlap
    assert set(train['date']).isdisjoint(set(val['date']))
    assert set(val['date']).isdisjoint(set(test['date']))
