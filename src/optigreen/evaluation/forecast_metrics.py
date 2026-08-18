import numpy as np
import pandas as pd
from typing import Dict, Any

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate MAE, RMSE, and WAPE.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    sum_abs_err = np.sum(np.abs(y_true - y_pred))
    sum_true = np.sum(np.abs(y_true))
    wape = sum_abs_err / sum_true if sum_true != 0 else np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'WAPE': wape
    }

def evaluate_forecast(results_df: pd.DataFrame, true_col: str = 'demand', pred_col: str = 'prediction') -> Dict[str, Any]:
    """
    Calculate global and granular metrics given a dataframe containing true and predicted values.
    results_df must contain: region_id, product_id, true_col, pred_col
    """
    metrics = {}
    
    # Global
    global_metrics = calculate_metrics(results_df[true_col].values, results_df[pred_col].values)
    metrics['global'] = global_metrics
    
    # Per Region
    metrics['per_region'] = {}
    for region, group in results_df.groupby('region_id'):
        metrics['per_region'][region] = calculate_metrics(group[true_col].values, group[pred_col].values)
        
    # Per Product
    metrics['per_product'] = {}
    for product, group in results_df.groupby('product_id'):
        metrics['per_product'][product] = calculate_metrics(group[true_col].values, group[pred_col].values)
        
    return metrics
