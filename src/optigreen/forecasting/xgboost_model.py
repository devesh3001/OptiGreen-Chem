import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple

class XGBoostForecaster:
    def __init__(self):
        self.model = None
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
            
        # Fill any remaining NaNs (e.g. early rolling std)
        processed.fillna(0, inplace=True)
        
        X = processed[self.features]
        y = processed['demand']
        
        return X, y

    def train_with_validation(self, train_df: pd.DataFrame, val_df: pd.DataFrame, n_iter: int = 5):
        """
        Trains the XGBoost model using RandomizedSearchCV on the validation set.
        Does not touch the test set.
        """
        X_train, y_train = self._prepare_data(train_df, is_train=True)
        X_val, y_val = self._prepare_data(val_df, is_train=False)
        
        # Combine train and val for GridSearchCV but use PredefinedSplit to only validate on val_df
        X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
        
        # Predefined split: -1 for train, 0 for validation
        test_fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
        ps = PredefinedSplit(test_fold)
        
        base_estimator = xgb.XGBRegressor(
            objective='reg:squarederror', 
            n_jobs=-1,
            random_state=42
        )
        
        param_distributions = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='neg_mean_absolute_error',
            cv=ps,
            verbose=1,
            random_state=42
        )
        
        search.fit(X_combined, y_combined)
        
        self.model = search.best_estimator_
        return search.best_params_

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained yet.")
            
        X_test, _ = self._prepare_data(test_df, is_train=False)
        preds = self.model.predict(X_test)
        
        out_df = test_df[['date', 'region_id', 'product_id', 'demand']].copy()
        # Predictions cannot be negative
        out_df['prediction'] = np.maximum(0, preds)
        
        return out_df
