import pytest
import torch
import pandas as pd
from optigreen.forecasting.transformer_dataset import SequentialDemandDataset, create_dataloaders
from optigreen.forecasting.transformer_model import TimeSeriesTransformer, PinballLoss

@pytest.fixture
def sample_panel_data():
    dates = pd.date_range('2024-01-01', '2024-02-15') # 46 days
    data = []
    
    for r in ['R1', 'R2']:
        for d in dates:
            data.append({
                'date': d,
                'region_id': r,
                'product_id': 'P1',
                'demand': 100,
                'day': d.day,
                'day_of_week': d.dayofweek,
                'week': d.isocalendar().week,
                'month': d.month,
                'is_weekend': int(d.dayofweek >= 5),
                'rolling_mean_7': 95,
                'rolling_std_7': 2.0,
                'recent_trend': 0.05,
                'region_encoded': 0 if r == 'R1' else 1,
                'product_encoded': 0
            })
    return pd.DataFrame(data)

def test_sequential_dataset(sample_panel_data):
    # 46 days total. context=28, horizon=7.
    # We should get 46 - 28 - 7 + 1 = 12 samples per region/product
    # Total = 12 * 2 regions = 24 samples
    dataset = SequentialDemandDataset(sample_panel_data, context_length=28, horizon=7)
    
    assert len(dataset) == 24
    
    x, y = dataset[0]
    # num_features = 10 in the feature_cols list
    assert x.shape == (28, 10)
    assert y.shape == (7,)

def test_pinball_loss():
    loss_fn = PinballLoss(quantiles=[0.1, 0.5, 0.9])
    
    # Batch = 2, Horizon = 1
    # preds: (2, 1, 3)
    preds = torch.tensor([[[90., 100., 110.]], [[190., 200., 210.]]])
    # target: (2, 1)
    target = torch.tensor([[100.], [200.]])
    
    # For exactly correct p50, loss is 0. 
    # For p10 = 90, target = 100. diff = 10. q=0.1. loss = 0.1 * 10 = 1.0
    # For p90 = 110, target = 100. diff = -10. q=0.9. loss = (1-0.9)*10 = 1.0
    # Average across quantiles: (1+0+1)/3 = 2/3
    loss = loss_fn(preds, target)
    assert torch.isclose(loss, torch.tensor(2.0/3.0))

def test_transformer_forward():
    model = TimeSeriesTransformer(num_features=10, d_model=16, nhead=2, num_layers=1, horizon=7)
    
    # batch = 2, context = 28, features = 10
    x = torch.randn(2, 28, 10)
    out = model(x)
    
    # Output should be (batch, horizon, len(quantiles))
    assert out.shape == (2, 7, 3)
