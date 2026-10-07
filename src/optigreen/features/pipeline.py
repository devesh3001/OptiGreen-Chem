import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder

class FeaturePipeline:
    def __init__(self):
        self.region_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()

    def build_features(self, demand_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build temporal, lag, rolling, and trend features for demand forecasting.
        Ensures strict leakage prevention by shifting data appropriately.
        """
        df = demand_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort to ensure chronological order before shifting
        df = df.sort_values(by=['region_id', 'product_id', 'date']).reset_index(drop=True)
        
        # 1. Temporal Features
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        
        # Encodings
        df['region_encoded'] = self.region_encoder.fit_transform(df['region_id'])
        df['product_encoded'] = self.product_encoder.fit_transform(df['product_id'])
        
        # We need to compute lags and rolling features per region & product
        # STRICT LEAKAGE PREVENTION: We must shift before applying rolling or use closed='left'
        grouped = df.groupby(['region_id', 'product_id'])['demand']
        
        # 2. Lag Features
        df['lag_1'] = grouped.shift(1)
        df['lag_7'] = grouped.shift(7)
        df['lag_14'] = grouped.shift(14)
        df['lag_28'] = grouped.shift(28)
        
        # 3. Rolling Features
        # Using min_periods=1 so we don't drop too many rows, though early rows will be less accurate
        # We MUST shift by 1 to avoid leakage (rolling mean at time t should only use up to t-1)
        shifted_demand = grouped.shift(1)
        shifted_grouped = df.assign(shifted_demand=shifted_demand).groupby(['region_id', 'product_id'])['shifted_demand']
        
        df['rolling_mean_7'] = shifted_grouped.rolling(window=7, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        df['rolling_mean_14'] = shifted_grouped.rolling(window=14, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        df['rolling_mean_28'] = shifted_grouped.rolling(window=28, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        
        df['rolling_std_7'] = shifted_grouped.rolling(window=7, min_periods=1).std().reset_index(level=[0,1], drop=True).fillna(0)
        df['rolling_std_28'] = shifted_grouped.rolling(window=28, min_periods=1).std().reset_index(level=[0,1], drop=True).fillna(0)
        
        # 4. Trend Features
        # deviation from 28-day seasonal average (immediate shock indicator)
        df['recent_trend'] = (df['lag_1'] - df['rolling_mean_28']) / (df['rolling_mean_28'] + 1e-5)
        
        # Drop initial rows with NaNs caused by the longest lag (28 days)
        # Actually, let's keep them and let the model handle or explicitly drop in split
        df = df.dropna(subset=['lag_28']).reset_index(drop=True)
        
        return df
        
    def time_based_split(self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological split of the dataset.
        Returns: Train, Validation, Test
        """
        # Ensure chronological order
        df = df.sort_values(by='date')
        
        dates = np.sort(df['date'].unique())
        
        n_days = len(dates)
        train_end_idx = int(n_days * train_ratio)
        val_end_idx = int(n_days * (train_ratio + val_ratio))
        
        train_end_date = dates[train_end_idx]
        val_end_date = dates[val_end_idx]
        
        train_df = df[df['date'] < train_end_date].copy()
        val_df = df[(df['date'] >= train_end_date) & (df['date'] < val_end_date)].copy()
        test_df = df[df['date'] >= val_end_date].copy()
        
        return train_df, val_df, test_df
