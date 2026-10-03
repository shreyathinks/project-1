"""
GraphGuard — Temporal-Conditioned Generator.

Architecture:
  conditioning_vector c = [node_features | neighbourhood_embedding | sinusoidal_PE(timestep)]
  noise z ~ N(0, I)
  output = MLP([z ; c]) → synthetic_node_features (same dim as real features)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional encoding for timestep
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    """
    Maps a scalar timestep t ∈ {1, …, T_max} to a d_model-dimensional
    sinusoidal embedding (same idea as Transformer positional encoding).
    Avoids the sparse-gradient problem of a learned embedding table.
    """

    def __init__(self, d_model: int = 64, T_max: int = 49):
        super().__init__()
        self.d_model = d_model
        # Pre-compute the embedding matrix and register as a buffer (not a param)
        pe = torch.zeros(T_max + 1, d_model)  # index 0 unused; steps 1-49
        position = torch.arange(0, T_max + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: LongTensor of shape [B] with values in {1, …, T_max}
        Returns:
            Tensor of shape [B, d_model]
        """
        return self.pe[timesteps]  # shape [B, d_model]


# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood aggregator (lightweight, no PyG dependency)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_neighbourhood(
    node_feats: torch.Tensor,
    center_idx: torch.Tensor,
    edge_index: torch.Tensor,
    k_hops: int = 2,
) -> torch.Tensor:
    """
    For each seed node in center_idx, gather all k-hop neighbours and
    return their mean-pooled feature vector.

    Args:
        node_feats:   [N, F] all node features
        center_idx:   [B] indices of the conditioning (real illicit) nodes
        edge_index:   [2, E] graph edges (src → dst)
        k_hops:       how many hops to expand
    Returns:
        [B, F] mean-pooled neighbourhood embeddings
    """
    src, dst = edge_index
    N, F = node_feats.shape
    B = center_idx.size(0)

    # Build adjacency as a set for fast lookup
    # For large graphs we use sparse gather; here we use a loop over B (fine for batch sizes used in GAN training)
    result = []
    for idx in center_idx:
        idx = idx.item()
        neighbours = {idx}
        frontier = {idx}
        for _ in range(k_hops):
            new_frontier = set()
            for n in frontier:
                mask = src == n
                new_frontier.update(dst[mask].tolist())
                mask2 = dst == n
                new_frontier.update(src[mask2].tolist())
            frontier = new_frontier - neighbours
            neighbours.update(frontier)
        nb_idx = torch.tensor(list(neighbours), dtype=torch.long, device=node_feats.device)
        result.append(node_feats[nb_idx].mean(dim=0))

    return torch.stack(result, dim=0)  # [B, F]


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────
class TemporalGenerator(nn.Module):
    """
    Temporal-conditioned generator for synthetic fraud nodes.

    Input  : noise z [B, noise_dim], conditioning vector c [B, cond_dim]
    Output : synthetic node features [B, feat_dim]

    cond_dim = feat_dim (node features) + feat_dim (neighbourhood mean) + pe_dim (sinusoidal PE)
    """

    def __init__(
        self,
        feat_dim: int = 166,
        noise_dim: int = 64,
        pe_dim: int = 64,
        hidden_dim: int = 512,
        T_max: int = 49,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.noise_dim = noise_dim
        self.pe_dim = pe_dim

        self.pe = SinusoidalPE(d_model=pe_dim, T_max=T_max)

        # Conditioning encoder: compress [feat | neigh_feat | PE] → hidden_dim
        cond_raw_dim = feat_dim + feat_dim + pe_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Main generator MLP
        in_dim = noise_dim + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, feat_dim),
        )

    # ------------------------------------------------------------------
    def build_conditioning(
        self,
        real_feats: torch.Tensor,       # [B, feat_dim] — real illicit node features
        neigh_feats: torch.Tensor,      # [B, feat_dim] — mean-pooled k-hop neighbourhood
        timesteps: torch.Tensor,        # [B] — integer timestep {1..49}
    ) -> torch.Tensor:
        pe_embed = self.pe(timesteps)   # [B, pe_dim]
        raw_c = torch.cat([real_feats, neigh_feats, pe_embed], dim=-1)
        return self.cond_encoder(raw_c)  # [B, hidden_dim]

    def forward(
        self,
        z: torch.Tensor,            # [B, noise_dim] sampled noise
        conditioning: torch.Tensor,  # [B, hidden_dim] output of build_conditioning
    ) -> torch.Tensor:
        inp = torch.cat([z, conditioning], dim=-1)
        return self.net(inp)  # [B, feat_dim]

    @torch.no_grad()
    def sample(
        self,
        real_feats: torch.Tensor,
        neigh_feats: torch.Tensor,
        timesteps: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Convenience wrapper for inference-time generation."""
        self.eval()
        B = real_feats.size(0)
        device = real_feats.device
        z = torch.randn(B * n_samples, self.noise_dim, device=device)
        cond = self.build_conditioning(
            real_feats.repeat_interleave(n_samples, dim=0),
            neigh_feats.repeat_interleave(n_samples, dim=0),
            timesteps.repeat_interleave(n_samples),
        )
        return self.forward(z, cond)
