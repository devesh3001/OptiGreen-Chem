"""
Training loop and evaluation suite for Phase 5 GNN/GAT models.
Handles class imbalance, early stopping, and metric computation.
"""
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, brier_score_loss
)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0


class GNNTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0005)
        )
        
        # Binary Cross Entropy with positive class weight
        pos_weight = torch.tensor([config.get('pos_weight', 10.0)]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward
            out = self.model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None), batch.batch)
            
            # Mask out non-region nodes (-100 label)
            mask = batch.y != -100
            if not mask.any():
                continue
                
            loss = self.criterion(out[mask], batch.y[mask].float())
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        
        all_probs = []
        all_labels = []
        
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None), batch.batch)
            
            mask = batch.y != -100
            if not mask.any():
                continue
                
            loss = self.criterion(out[mask], batch.y[mask].float())
            total_loss += loss.item() * batch.num_graphs
            
            probs = torch.sigmoid(out[mask]).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch.y[mask].cpu().numpy())
            
        avg_loss = total_loss / len(loader.dataset)
        metrics = self._compute_metrics(np.array(all_labels), np.array(all_probs))
        return avg_loss, metrics

    def _compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        if len(np.unique(y_true)) < 2:
            return {'ROC_AUC': 0.0, 'PR_AUC': 0.0, 'F1': 0.0, 'Recall': 0.0, 'Brier': 0.0}
            
        y_pred = (y_prob >= 0.5).astype(int)
        return {
            'ROC_AUC': roc_auc_score(y_true, y_prob),
            'PR_AUC': average_precision_score(y_true, y_prob),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Brier': brier_score_loss(y_true, y_prob)
        }

    def train(self, train_data: list, val_data: list) -> Dict[str, float]:
        """Full training loop with early stopping."""
        batch_size = self.config.get('batch_size', 32)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        epochs = self.config.get('epochs', 100)
        early_stopping = EarlyStopping(patience=self.config.get('patience', 15))
        
        print(f"    Starting {self.model.name} training...")
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._val_epoch(val_loader)
            
            early_stopping(val_loss, self.model)
            
            if (epoch + 1) % 10 == 0 or early_stopping.early_stop:
                print(f"      Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val PR-AUC: {val_metrics['PR_AUC']:.3f}")
                
            if early_stopping.early_stop:
                print(f"    Early stopping at epoch {epoch+1}.")
                break
                
        # Load best weights
        if early_stopping.best_state is not None:
            self.model.load_state_dict(early_stopping.best_state)
            
        # Final validation eval
        _, final_val_metrics = self._val_epoch(val_loader)
        return final_val_metrics

    @torch.no_grad()
    def evaluate(self, test_data: list) -> Dict[str, float]:
        """Evaluate on test set."""
        loader = DataLoader(test_data, batch_size=self.config.get('batch_size', 32), shuffle=False)
        _, metrics = self._val_epoch(loader)
        return metrics
