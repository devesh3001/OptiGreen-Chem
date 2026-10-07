"""
Classical Baselines (Logistic Regression & XGBoost) for Phase 5.
Used to evaluate whether graph structure provides additional value
over flat tabular features.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, brier_score_loss
)


class BaselineModels:
    def __init__(self, random_state: int = 42, pos_weight: float = 10.0):
        self.lr = LogisticRegression(
            class_weight={0: 1.0, 1: pos_weight},
            random_state=random_state,
            max_iter=1000
        )
        self.xgb = XGBClassifier(
            scale_pos_weight=pos_weight,
            random_state=random_state,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    def _extract_tabular_data(self, data_list):
        """
        Extract flat features from PyG Data objects.
        Only keeps region nodes (where y != -100).
        """
        X_all, y_all = [], []
        for data in data_list:
            mask = data.y != -100
            X_all.append(data.x[mask].numpy())
            y_all.append(data.y[mask].numpy())
        
        return np.vstack(X_all), np.concatenate(y_all)

    def train(self, train_data, val_data=None):
        """Train baseline models on flat features."""
        print("    Extracting tabular data for baselines...")
        X_train, y_train = self._extract_tabular_data(train_data)
        print(f"    Tabular train shape: {X_train.shape}, positives: {y_train.sum()}")

        print("    Training Logistic Regression...")
        self.lr.fit(X_train, y_train)

        print("    Training XGBoost...")
        if val_data:
            X_val, y_val = self._extract_tabular_data(val_data)
            self.xgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.xgb.fit(X_train, y_train)

    def evaluate(self, test_data) -> Dict[str, Dict[str, float]]:
        """Evaluate baselines and return standard metrics."""
        X_test, y_test = self._extract_tabular_data(test_data)
        
        # Predictions
        lr_probs = self.lr.predict_proba(X_test)[:, 1]
        xgb_probs = self.xgb.predict_proba(X_test)[:, 1]

        lr_preds = (lr_probs >= 0.5).astype(int)
        xgb_preds = (xgb_probs >= 0.5).astype(int)

        def _metrics(y_true, y_prob, y_pred):
            if len(np.unique(y_true)) < 2:
                return {'ROC_AUC': 0.0, 'PR_AUC': 0.0, 'F1': 0.0, 
                        'Recall': 0.0, 'Precision': 0.0, 'Brier': 0.0}
            return {
                'ROC_AUC': roc_auc_score(y_true, y_prob),
                'PR_AUC': average_precision_score(y_true, y_prob),
                'F1': f1_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Brier': brier_score_loss(y_true, y_prob)
            }

        return {
            'Logistic': _metrics(y_test, lr_probs, lr_preds),
            'XGBoost': _metrics(y_test, xgb_probs, xgb_preds)
        }
