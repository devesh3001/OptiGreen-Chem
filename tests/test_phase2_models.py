import pytest
import pandas as pd
import numpy as np
from datetime import timedelta
from optigreen.forecasting.baselines import SeasonalNaiveBaseline, MovingAverageBaseline
from optigreen.evaluation.forecast_metrics import calculate_metrics, evaluate_forecast
from optigreen.forecasting.xgboost_model import XGBoostForecaster

@pytest.fixture
def sample_test_data():
    dates = pd.date_range('2024-03-01', '2024-03-10')
    n_days = len(dates)
    data = []
    
    for d in dates:
        data.append({
            'date': d,
            'region_id': 'R1',
            'product_id': 'P1',
            'demand': 100,
            'lag_7': 90,
            'rolling_mean_7': 95,
            # XGBoost features
            'day': d.day,
            'day_of_week': d.dayofweek,
            'week': d.isocalendar().week,
            'month': d.month,
            'quarter': d.quarter,
            'year': d.year,
            'day_of_year': d.dayofyear,
            'is_weekend': int(d.dayofweek >= 5),
            'lag_1': 98,
            'lag_14': 85,
            'lag_28': 80,
            'rolling_mean_14': 92,
            'rolling_mean_28': 88,
            'rolling_std_7': 2.0,
            'rolling_std_28': 5.0,
            'recent_trend': 0.05,
            'region_encoded': 0,
            'product_encoded': 0
        })
    return pd.DataFrame(data)

def test_metrics():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 300])
    
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics['MAE'] == 20 / 3
    
    rmse = np.sqrt(200 / 3)
    assert np.isclose(metrics['RMSE'], rmse)
    
    # sum_abs_err = 20, sum_true = 600
    assert metrics['WAPE'] == 20 / 600

def test_seasonal_naive(sample_test_data):
    model = SeasonalNaiveBaseline(season_length=7)
    preds = model.predict(sample_test_data)
    assert all(preds['prediction'] == 90)

def test_moving_average(sample_test_data):
    model = MovingAverageBaseline(window=7)
    preds = model.predict(sample_test_data)
    assert all(preds['prediction'] == 95)

def test_xgboost_training(sample_test_data):
    train_df = sample_test_data.iloc[:5].copy()
    val_df = sample_test_data.iloc[5:8].copy()
    test_df = sample_test_data.iloc[8:].copy()
    
    model = XGBoostForecaster()
    # Fast tune for testing
    model.train_with_validation(train_df, val_df, n_iter=1)
    
    preds = model.predict(test_df)
    
    assert len(preds) == len(test_df)
    assert 'prediction' in preds.columns
    assert all(preds['prediction'] >= 0)
