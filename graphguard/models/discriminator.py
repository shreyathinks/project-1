"""
GraphGuard — Dual-Head Discriminator.

Head A — Node realism   : MLP(node_features) → real/fake
Head B — Structural realism : mini-GraphSAGE(local subgraph) → real/fake

Both heads are trained adversarially; their gradients flow back through
the edge predictor and generator — THIS is what distinguishes GraphGuard
from THG-OAFN's fixed post-hoc structural filter.
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
# Fallback manual SAGE conv (reused from baseline_graphsage)
# ─────────────────────────────────────────────────────────────────────────────
class _ManualSAGEConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.lin = nn.Linear(in_c * 2, out_c)

    def forward(self, x, edge_index):
        src, dst = edge_index
        N = x.size(0)
        agg = torch.zeros(N, x.size(1), device=x.device)
        cnt = torch.zeros(N, 1, device=x.device)
        agg.index_add_(0, dst, x[src])
        cnt.index_add_(0, dst, torch.ones(dst.size(0), 1, device=x.device))
        agg = agg / cnt.clamp(min=1)
        return self.lin(torch.cat([x, agg], dim=-1))


def _conv(in_c, out_c):
    return SAGEConv(in_c, out_c) if _HAS_PYG else _ManualSAGEConv(in_c, out_c)


# ─────────────────────────────────────────────────────────────────────────────
# Head A — Node-level discriminator
# ─────────────────────────────────────────────────────────────────────────────
class NodeDiscriminator(nn.Module):
    """
    Standard MLP discriminator operating on individual node features.
    Uses spectral normalisation for training stability.
    """

    def __init__(self, feat_dim: int = 166, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim // 2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape [B, 1]."""
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Head B — Structural discriminator (1-2 layer mini-GraphSAGE)
# ─────────────────────────────────────────────────────────────────────────────
class StructuralDiscriminator(nn.Module):
    """
    Encodes a local subgraph (center node + its neighbours) with a
    2-layer GraphSAGE, then classifies whether the subgraph is real or fake.

    Additionally takes hand-crafted structural stats as auxiliary input:
      [degree, clustering_coeff, homophily_ratio, ego_density]  → 4 scalars

    The GNN encoder is updated adversarially, which forces the generator
    to produce structurally realistic subgraphs, not just feature-realistic nodes.
    """

    def __init__(
        self,
        feat_dim: int = 166,
        struct_dim: int = 4,   # degree, clustering, homophily, ego_density
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.conv1 = _conv(feat_dim, hidden_dim)
        self.conv2 = _conv(hidden_dim, hidden_dim // 2)

        # Classifier on graph embedding + structural stats
        graph_emb_dim = hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(graph_emb_dim + struct_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, 1)),
        )

    def encode_subgraph(
        self,
        subgraph_x: torch.Tensor,       # [n_nodes, feat_dim] local node features
        subgraph_edge_index: torch.Tensor,  # [2, n_edges] local edge index
        center_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the embedding of the center nodes after 2 SAGE layers."""
        h = F.leaky_relu(self.conv1(subgraph_x, subgraph_edge_index), 0.2)
        h = F.leaky_relu(self.conv2(h, subgraph_edge_index), 0.2)
        if center_indices is not None:
            return h[center_indices]
        return h[0]  # fallback embedding of center node

    def forward(
        self,
        subgraph_x: torch.Tensor,           # [n_nodes, feat_dim]
        subgraph_edge_index: torch.Tensor,  # [2, n_edges]
        struct_stats: torch.Tensor,          # [B, 4] or [4]
        center_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns logits."""
        emb = self.encode_subgraph(subgraph_x, subgraph_edge_index, center_indices)
        if emb.dim() == 1 and struct_stats.dim() == 1:
            emb = emb.unsqueeze(0)
            struct_stats = struct_stats.unsqueeze(0)
        combined = torch.cat([emb, struct_stats], dim=-1)
        return self.classifier(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Combined discriminator wrapper
# ─────────────────────────────────────────────────────────────────────────────
class GraphGuardDiscriminator(nn.Module):
    """
    Wraps both discriminator heads. During training:
      - Head A loss + Head B loss are summed → update D weights
      - Head A + B gradients flow back through G + EdgePredictor
    """

    def __init__(
        self,
        feat_dim: int = 166,
        struct_dim: int = 4,
        hidden_a: int = 256,
        hidden_b: int = 128,
    ):
        super().__init__()
        self.head_a = NodeDiscriminator(feat_dim, hidden_a)
        self.head_b = StructuralDiscriminator(feat_dim, struct_dim, hidden_b)

    def discriminate_node(self, x: torch.Tensor) -> torch.Tensor:
        """Head A: node-level. Returns logit [B, 1]."""
        return self.head_a(x)

    def discriminate_subgraph(
        self,
        subgraph_x: torch.Tensor,
        subgraph_edge_index: torch.Tensor,
        struct_stats: torch.Tensor,
        center_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """Head B: subgraph-level. Returns logits."""
        return self.head_b(subgraph_x, subgraph_edge_index, struct_stats, center_indices)
