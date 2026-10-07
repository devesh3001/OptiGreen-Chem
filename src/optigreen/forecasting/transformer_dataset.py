import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SequentialDemandDataset(Dataset):
    def __init__(self, df: pd.DataFrame, context_length: int = 28, horizon: int = 7):
        """
        Creates sequences of length `context_length` to predict the next `horizon` days.
        """
        self.context_length = context_length
        self.horizon = horizon
        self.data = []
        
        # Sort and group by region and product
        df_sorted = df.sort_values(by=['region_id', 'product_id', 'date']).reset_index(drop=True)
        
        # Features to use as sequential input
        # Note: we exclude target 'demand' from the last `horizon` steps
        self.feature_cols = [
            'day', 'day_of_week', 'week', 'month', 'is_weekend',
            'rolling_mean_7', 'rolling_std_7', 'recent_trend',
            'region_encoded', 'product_encoded'
        ]
        
        for (region, product), group in df_sorted.groupby(['region_id', 'product_id']):
            group_vals = group[self.feature_cols].values
            target_vals = group['demand'].values
            
            n_samples = len(group) - self.context_length - self.horizon + 1
            if n_samples > 0:
                for i in range(n_samples):
                    x = group_vals[i:i+self.context_length]
                    y = target_vals[i+self.context_length:i+self.context_length+self.horizon]
                    self.data.append((x, y))
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                      context_length: int = 28, horizon: int = 7, batch_size: int = 64):
    """
    Helper to create PyTorch DataLoaders.
    Note: For strict time-series without overlap leakage across train/val boundaries, 
    we treat each dataframe separately.
    """
    train_dataset = SequentialDemandDataset(train_df, context_length, horizon)
    val_dataset = SequentialDemandDataset(val_df, context_length, horizon)
    test_dataset = SequentialDemandDataset(test_df, context_length, horizon)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
