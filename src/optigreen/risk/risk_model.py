"""
Stockout Risk Model: XGBoost classifier predicting P(stockout) per (region, product, t).
The resulting probability is used as a risk signal in the MILP objective.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             classification_report)
from typing import Dict, Tuple


FEATURE_COLS = [
    'forecast_spread', 'relative_spread',
    'p50_level', 'p90_level', 'forecast_bias',
    'demand_vol_14', 'capacity_utilization',
]


class StockoutRiskModel:
    """
    Binary XGBoost classifier: predicts whether actual demand will exceed P90
    (i.e., a likely stockout if planning was done on median demand).

    The probability output is directly used as a risk_score in the MILP.
    """

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,   # Handle class imbalance: stockouts are rare
            use_label_encoder=False,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
        )
        self._trained = False

    def train(self, risk_features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train on a risk features dataframe (output of build_risk_features).
        Returns evaluation metrics on a held-out validation split.
        """
        df = risk_features_df.dropna(subset=FEATURE_COLS + ['stockout_label'])
        X = df[FEATURE_COLS].values
        y = df['stockout_label'].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        self._trained = True

        y_prob = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            'ROC_AUC': roc_auc_score(y_val, y_prob),
            'PR_AUC': average_precision_score(y_val, y_prob),
            'F1': f1_score(y_val, y_pred, zero_division=0),
            'Precision': precision_score(y_val, y_pred, zero_division=0),
            'Recall': recall_score(y_val, y_pred, zero_division=0),
            'Stockout_Rate': float(y_val.mean()),
        }
        return metrics

    def predict_risk_score(self, risk_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with (date, region_id, product_id, risk_score).
        risk_score is in [0, 1]: probability of stockout.
        """
        if not self._trained:
            raise ValueError("Model not trained. Call .train() first.")

        df = risk_features_df.copy()
        X = df[FEATURE_COLS].fillna(0).values
        probs = self.model.predict_proba(X)[:, 1]

        result = df[['date', 'region_id', 'product_id']].copy()
        result['risk_score'] = probs
        return result

    def compute_statistical_risk_score(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback statistical risk score (no classifier required).
        risk_score = relative_spread = (P90 - P10) / (P50 + 1), clipped to [0, 1].
        Used if the classifier has insufficient training data.
        """
        df = forecast_df.copy()
        df['risk_score'] = ((df['P90'] - df['P10']) / (df['P50'] + 1)).clip(0, 1)
        return df[['date', 'region_id', 'product_id', 'risk_score']]

    def save(self, filepath: str):
        """Save the XGBoost model to a JSON file."""
        if not self._trained:
            raise ValueError("Model not trained. Call .train() first.")
        self.model.save_model(filepath)
        
    def load(self, filepath: str):
        """Load the XGBoost model from a JSON file."""
        self.model.load_model(filepath)
        self._trained = True
