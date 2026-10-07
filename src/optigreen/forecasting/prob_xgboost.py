import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple

class ProbabilisticXGBoostForecaster:
    def __init__(self):
        self.models = {}
        self.region_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.features = [
            'day', 'day_of_week', 'week', 'month', 'quarter', 'year', 'day_of_year', 'is_weekend',
            'lag_1', 'lag_7', 'lag_14', 'lag_28',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
            'rolling_std_7', 'rolling_std_28',
            'recent_trend',
            'region_encoded', 'product_encoded'
        ]

    def _prepare_data(self, df: pd.DataFrame, is_train: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        processed = df.copy()
        
        # Encodings are now handled in the FeaturePipeline
            
        processed.fillna(0, inplace=True)
        
        X = processed[self.features]
        y = processed['demand']
        
        return X, y

    def train(self, train_df: pd.DataFrame, params: Dict = None):
        """
        Trains three separate quantile regressors for P10, P50, and P90.
        Uses objective='reg:quantileerror'.
        """
        if params is None:
            # Default to best params from Phase 2
            params = {'subsample': 1.0, 'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 1.0}
            
        X_train, y_train = self._prepare_data(train_df, is_train=True)
        
        quantiles = [0.10, 0.50, 0.90]
        
        for q in quantiles:
            model_params = params.copy()
            # In XGBoost >= 2.0, objective can be reg:quantileerror and quantile_alpha sets the quantile
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_jobs=-1,
                random_state=42,
                **model_params
            )
            model.fit(X_train, y_train)
            self.models[f'P{int(q*100)}'] = model

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise ValueError("Models are not trained yet.")
            
        X_test, _ = self._prepare_data(test_df, is_train=False)
        
        out_df = test_df[['date', 'region_id', 'product_id', 'demand']].copy()
        
        p10 = self.models['P10'].predict(X_test)
        p50 = self.models['P50'].predict(X_test)
        p90 = self.models['P90'].predict(X_test)
        
        # Predictions cannot be negative
        p10 = np.maximum(0, p10)
        p50 = np.maximum(0, p50)
        p90 = np.maximum(0, p90)
        
        # Enforce Quantile Ordering to fix crossing quantiles
        out_df['P10'] = np.minimum(p10, p50)
        out_df['P50'] = p50
        out_df['P90'] = np.maximum(p50, p90)
        # Ensure P10 <= P90 just in case
        out_df['P10'] = np.minimum(out_df['P10'], out_df['P90'])
        
        return out_df

    def save(self, filepath: str):
        """Save models to JSON format."""
        import os
        base, ext = os.path.splitext(filepath)
        for q, model in self.models.items():
            model.save_model(f"{base}_{q}{ext}")
            
    def load(self, filepath: str):
        """Load models from JSON format."""
        import os
        base, ext = os.path.splitext(filepath)
        for q in ['P10', 'P50', 'P90']:
            model = xgb.XGBRegressor()
            model.load_model(f"{base}_{q}{ext}")
            self.models[q] = model
