import numpy as np
import pandas as pd
from typing import Dict, Any

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """
    Calculates the pinball loss for quantile q.
    L_q(y, y_hat) = q*(y - y_hat) if y >= y_hat else (1-q)*(y_hat - y)
    """
    diff = y_true - y_pred
    loss = np.where(diff >= 0, q * diff, (1 - q) * (-diff))
    return np.mean(loss)

def calculate_probabilistic_metrics(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Dict[str, float]:
    """
    Calculate Pinball Loss, PICP (Coverage), and MPIW (Width).
    """
    # Pinball Losses
    pl_10 = pinball_loss(y_true, p10, 0.10)
    pl_50 = pinball_loss(y_true, p50, 0.50)
    pl_90 = pinball_loss(y_true, p90, 0.90)
    
    # Prediction Interval Coverage Probability (Target is 80% for P10-P90)
    coverage = np.mean((y_true >= p10) & (y_true <= p90))
    
    # Mean Prediction Interval Width
    mpiw = np.mean(p90 - p10)
    
    # Calibration metric: How close is coverage to 80%
    calibration_error = np.abs(coverage - 0.80)
    
    return {
        'Pinball_10': pl_10,
        'Pinball_50': pl_50,
        'Pinball_90': pl_90,
        'Mean_Pinball': (pl_10 + pl_50 + pl_90) / 3.0,
        'PICP_80': coverage,
        'MPIW': mpiw,
        'Calibration_Error': calibration_error
    }

def evaluate_probabilistic_forecast(df: pd.DataFrame, true_col='demand') -> Dict[str, Any]:
    """
    Evaluates probabilistic forecast dataframe containing P10, P50, P90.
    """
    metrics = {}
    
    y = df[true_col].values
    p10 = df['P10'].values
    p50 = df['P50'].values
    p90 = df['P90'].values
    
    metrics['global'] = calculate_probabilistic_metrics(y, p10, p50, p90)
    
    # Per Region
    metrics['per_region'] = {}
    for region, group in df.groupby('region_id'):
        metrics['per_region'][region] = calculate_probabilistic_metrics(
            group[true_col].values, group['P10'].values, group['P50'].values, group['P90'].values
        )
        
    return metrics
