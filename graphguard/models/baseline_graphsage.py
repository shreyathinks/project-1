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
# 3-layer GraphSAGE classifier
# ─────────────────────────────────────────────────────────────────────────────
class GraphSAGEClassifier(nn.Module):
    """
    Two-layer GraphSAGE for node classification.

    Args:
        in_channels:  Number of input node features (166 for Elliptic).
        hidden_channels: Hidden dimension (default 256).
        out_channels: Number of output classes (2 for binary fraud detection).
        dropout: Dropout probability applied between layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = _get_conv(in_channels, hidden_channels)
        self.conv2 = _get_conv(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Removed third layer and dedicated linear classifier, conv2 now outputs directly to out_channels

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.bn1(self.conv1(x, edge_index)), 0.2)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        
        return h

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.forward(x, edge_index), dim=-1)
