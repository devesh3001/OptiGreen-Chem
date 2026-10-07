import pandas as pd
import numpy as np

class SeasonalNaiveBaseline:
    """
    Predicts demand using the demand from the previous season (e.g. 7 days ago).
    """
    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a dataframe with features, including lag_{season_length}
        """
        out_df = df[['date', 'region_id', 'product_id', 'demand']].copy()
        
        # If lag feature is present in the df, we use it directly
        lag_col = f'lag_{self.season_length}'
        if lag_col in df.columns:
            out_df['prediction'] = df[lag_col]
        else:
            # Fallback if lag is not pre-computed (not recommended due to performance/leakage risks)
            raise ValueError(f"{lag_col} must be present in the dataframe to use SeasonalNaiveBaseline.")
            
        out_df['prediction'] = out_df['prediction'].fillna(0)
        return out_df

class MovingAverageBaseline:
    """
    Predicts demand using the moving average over the past k days.
    """
    def __init__(self, window: int = 7):
        self.window = window
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a dataframe with features, including rolling_mean_{window}
        """
        out_df = df[['date', 'region_id', 'product_id', 'demand']].copy()
        
        mean_col = f'rolling_mean_{self.window}'
        if mean_col in df.columns:
            out_df['prediction'] = df[mean_col]
        else:
            raise ValueError(f"{mean_col} must be present in the dataframe to use MovingAverageBaseline.")
            
        out_df['prediction'] = out_df['prediction'].fillna(0)
        return out_df
