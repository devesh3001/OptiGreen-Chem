"""
GCN Model for supply chain node-level stockout risk prediction.

Architecture:
  NodeFeatures → GCNConv(hidden=64) → ReLU → BatchNorm
               → GCNConv(hidden=32) → ReLU → BatchNorm
               → Dropout → MLP → Sigmoid

Only region nodes receive predictions; plant/warehouse nodes are masked during loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNRiskModel(nn.Module):
    """
    Two-layer GCN for node-level binary classification.
    Predicts per-region stockout probability.
    """
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        hidden_dim2: int = 32,
        dropout: float = 0.3,
        improved: bool = True,
    ):
        super().__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim, improved=improved)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim2, improved=improved)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Args:
            x: [N, F] node features
            edge_index: [2, E]
            edge_attr: not used by GCN (for API compatibility)
            batch: not used in node-level inference
        Returns:
            logits: [N] raw logits for all nodes (mask externally)
        """
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)

        # Save embeddings for downstream analysis
        self.node_embeddings = h.detach()

        logits = self.classifier(h).squeeze(-1)   # [N]
        return logits

    def get_node_embeddings(self) -> torch.Tensor:
        """Returns the pre-classifier node embeddings from last forward pass."""
        return self.node_embeddings

    @property
    def name(self):
        return "GCN"
