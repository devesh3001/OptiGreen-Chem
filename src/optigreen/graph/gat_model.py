"""
GAT Model for supply chain node-level stockout risk prediction.

Architecture:
  NodeFeatures
  → GATConv(hidden=64, heads=4, concat=True)  → ELU → Dropout → BatchNorm
  → GATConv(hidden=32, heads=1, concat=False) → ELU → Dropout → BatchNorm
  → MLP → Sigmoid

Attention weights are stored from the final layer and can be visualized.

IMPORTANT: Attention weights are *model attention* — learned importance weights
for information aggregation. They should NOT be interpreted as causal importance
or feature importance without further analysis (e.g., GNNExplainer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple


class GATRiskModel(nn.Module):
    """
    Two-layer GAT for node-level binary classification.
    Predicts per-region stockout probability.
    """
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        out_dim: int = 32,
        out_heads: int = 1,
        dropout: float = 0.3,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        # Layer 1: multi-head attention, concatenated output → hidden_dim * heads
        self.conv1 = GATConv(
            node_feature_dim, hidden_dim, heads=heads, concat=True,
            dropout=dropout, edge_dim=edge_dim,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        # Layer 2: single-head attention, averaged output → out_dim
        self.conv2 = GATConv(
            hidden_dim * heads, out_dim, heads=out_heads, concat=False,
            dropout=dropout, edge_dim=edge_dim,
        )
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.ELU(),
            nn.Linear(16, 1),
        )
        self._init_weights()

        # Storage for attention coefficients (last forward pass)
        self._attention_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._node_embeddings: Optional[torch.Tensor] = None

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None, batch=None,
                return_attention_weights: bool = False):
        """
        Args:
            x:             [N, F] node features
            edge_index:    [2, E]
            edge_attr:     [E, edge_dim] optional edge features
            return_attention_weights: if True, store attention for visualization
        Returns:
            logits: [N] raw scores for all nodes
        """
        kw = {}
        if edge_attr is not None and self.conv1.edge_dim is not None:
            kw['edge_attr'] = edge_attr

        # Layer 1
        if return_attention_weights:
            h, (edge_idx, attn1) = self.conv1(
                x, edge_index, return_attention_weights=True, **kw)
        else:
            h = self.conv1(x, edge_index, **kw)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)

        # Layer 2
        if return_attention_weights:
            h, (edge_idx2, attn2) = self.conv2(
                h, edge_index, return_attention_weights=True, **kw)
            self._attention_weights = (edge_idx2, attn2.detach())
        else:
            h = self.conv2(h, edge_index, **kw)
        h = self.bn2(h)
        h = F.elu(h)
        h = self.dropout(h)

        self._node_embeddings = h.detach()
        logits = self.classifier(h).squeeze(-1)   # [N]
        return logits

    def get_attention_weights(self):
        """Returns (edge_index, attention_coefficients) from last forward pass."""
        return self._attention_weights

    def get_node_embeddings(self) -> Optional[torch.Tensor]:
        return self._node_embeddings

    @property
    def name(self):
        return "GAT"
