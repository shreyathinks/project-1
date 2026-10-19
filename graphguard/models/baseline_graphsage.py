"""
GraphGuard — Baseline GraphSAGE classifier.
Works with PyG's SAGEConv. Falls back to a pure-PyTorch
manual neighbourhood-averaging implementation if PyG is unavailable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch fallback SAGEConv (mean aggregation)
# ─────────────────────────────────────────────────────────────────────────────
class _ManualSAGEConv(nn.Module):
    """Minimal mean-aggregation GraphSAGE convolution (no PyG dependency)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index  # shape [2, E]
        N = x.size(0)
        # Aggregate neighbours (mean) for every destination node
        agg = torch.zeros(N, x.size(1), device=x.device)
        count = torch.zeros(N, 1, device=x.device)
        agg.index_add_(0, dst, x[src])
        count.index_add_(0, dst, torch.ones(dst.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count
        # Concatenate self + aggregated neighbour features
        out = torch.cat([x, agg], dim=-1)
        return self.lin(out)


def _get_conv(in_channels: int, out_channels: int):
    if _HAS_PYG:
        return SAGEConv(in_channels, out_channels)
    return _ManualSAGEConv(in_channels, out_channels)


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE classifier with Ego-Feature Residual Skip Connections
# ─────────────────────────────────────────────────────────────────────────────
class GraphSAGEClassifier(nn.Module):
    """
    Two-layer GraphSAGE with Ego-Feature Residual Skip Connection.
    Concatenates raw node features x_v, Layer 1 embedding h1, and Layer 2
    embedding h2 before classification to prevent oversmoothing sharp local features.

    Args:
        in_channels:  Number of input node features (165 for Elliptic).
        hidden_channels: Hidden dimension (default 256).
        out_channels: Number of output classes (1 for binary logit, 2 for softmax).
        dropout: Dropout probability applied between layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = _get_conv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = _get_conv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Skip connection: raw features + h1 + h2
        cat_dim = in_channels + hidden_channels + hidden_channels
        self.norm = nn.LayerNorm(cat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cat_dim, hidden_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h1 = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), 0.2)
        h1_drop = F.dropout(h1, p=self.dropout, training=self.training)

        h2 = F.leaky_relu(self.bn2(self.conv2(h1_drop, edge_index)), 0.2)

        # Concatenate raw local features x, 1st hop h1, and 2nd hop h2
        h_cat = torch.cat([x, h1, h2], dim=-1)
        h_norm = F.dropout(self.norm(h_cat), p=self.dropout, training=self.training)

        return self.classifier(h_norm)

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return probabilities (sigmoid for binary, softmax for multi-class)."""
        logits = self.forward(x, edge_index)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

