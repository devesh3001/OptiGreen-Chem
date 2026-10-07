import pytest
import pandas as pd
import numpy as np
from optigreen.evaluation.prob_metrics import pinball_loss, calculate_probabilistic_metrics
from optigreen.forecasting.prob_xgboost import ProbabilisticXGBoostForecaster

@pytest.fixture
def sample_test_data():
    dates = pd.date_range('2024-03-01', '2024-03-10')
    data = []
    for d in dates:
        data.append({
            'date': d,
            'region_id': 'R1',
            'product_id': 'P1',
            'demand': 100,
            'day': d.day,
            'day_of_week': d.dayofweek,
            'week': d.isocalendar().week,
            'month': d.month,
            'quarter': d.quarter,
            'year': d.year,
            'day_of_year': d.dayofyear,
            'is_weekend': int(d.dayofweek >= 5),
            'lag_1': 98,
            'lag_7': 90,
            'lag_14': 85,
            'lag_28': 80,
            'rolling_mean_7': 95,
            'rolling_mean_14': 92,
            'rolling_mean_28': 88,
            'rolling_std_7': 2.0,
            'rolling_std_28': 5.0,
            'recent_trend': 0.05,
            'region_encoded': 0,
            'product_encoded': 0
        })
    return pd.DataFrame(data)

def test_pinball_loss():
    y_true = np.array([100, 100])
    y_pred = np.array([110, 90])
    
    # q=0.5 (MAE / 2)
    # diffs = [-10, 10]
    # loss1 = (1-0.5)*10 = 5.0
    # loss2 = 0.5*10 = 5.0
    # mean = 5.0
    assert np.isclose(pinball_loss(y_true, y_pred, 0.5), 5.0)
    
    # q=0.9
    # loss1 (y < y_pred) = (1-0.9)*10 = 1.0
    # loss2 (y > y_pred) = 0.9*10 = 9.0
    # mean = 5.0
    assert np.isclose(pinball_loss(y_true, y_pred, 0.9), 5.0)
    
    # q=0.1
    # loss1 = (1-0.1)*10 = 9.0
    # loss2 = 0.1*10 = 1.0
    # mean = 5.0
    assert np.isclose(pinball_loss(y_true, y_pred, 0.1), 5.0)

def test_calculate_probabilistic_metrics():
    y = np.array([100, 200, 300])
    p10 = np.array([90, 150, 290])
    p50 = np.array([100, 200, 300])
    p90 = np.array([110, 250, 310])
    
    metrics = calculate_probabilistic_metrics(y, p10, p50, p90)
    
    # Coverage: [90,110] covers 100, [150,250] covers 200, [290,310] covers 300 -> 1.0
    assert metrics['PICP_80'] == 1.0
    # Calibration error: |1.0 - 0.8| = 0.2
    assert np.isclose(metrics['Calibration_Error'], 0.2)
    
    # MPIW: (20 + 100 + 20) / 3 = 140 / 3 = 46.666...
    assert np.isclose(metrics['MPIW'], 140.0 / 3.0)

def test_quantile_xgboost(sample_test_data):
    train_df = sample_test_data.iloc[:5].copy()
    test_df = sample_test_data.iloc[5:].copy()
    
    model = ProbabilisticXGBoostForecaster()
    # Use very few estimators for fast testing
    params = {'n_estimators': 2, 'max_depth': 2, 'learning_rate': 0.1}
    model.train(train_df, params)
    
    preds = model.predict(test_df)
    
    assert 'P10' in preds.columns
    assert 'P50' in preds.columns
    assert 'P90' in preds.columns
    
    # Quantile ordering test
    assert all(preds['P10'] <= preds['P50'])
    assert all(preds['P50'] <= preds['P90'])
    
    # Non-negative test
    assert all(preds['P10'] >= 0)
