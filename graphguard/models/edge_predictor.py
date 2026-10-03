"""
GraphGuard — Edge Predictor.

Given a synthetic node's feature vector and a set of candidate existing
nodes (restricted to time <= t_gen), predicts which edges to form.

Two modes:
  1. Differentiable (Gumbel-softmax)  ← default, used during training
  2. Top-k hard selection             ← used at inference / as fallback

Score(synthetic, candidate) = MLP([synthetic_feat ; candidate_feat ; |Δt|])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePredictor(nn.Module):
    """
    MLP-based edge scorer with Gumbel-softmax differentiable sampling.

    Args:
        feat_dim:       Node feature dimension (166 for Elliptic).
        hidden_dim:     MLP hidden size.
        k:              Number of edges to select per synthetic node.
        tau_init:       Initial Gumbel-softmax temperature (annealed during training).
    """

    def __init__(
        self,
        feat_dim: int = 166,
        hidden_dim: int = 256,
        k: int = 5,
        tau_init: float = 1.0,
    ):
        super().__init__()
        self.k = k
        self.tau = tau_init  # can be annealed externally

        # Input: [synth_feat | cand_feat | |Δt|] → score
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim + feat_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    def score_candidates(
        self,
        synth_feat: torch.Tensor,   # [feat_dim]  single synthetic node
        cand_feats: torch.Tensor,   # [C, feat_dim] candidate nodes
        synth_t: int,               # integer timestep of synthetic node
        cand_t: torch.Tensor,       # [C] timesteps of candidates
    ) -> torch.Tensor:
        """Return raw logit scores for all candidates. Shape [C]."""
        C = cand_feats.size(0)
        s_exp = synth_feat.unsqueeze(0).expand(C, -1)   # [C, feat_dim]
        delta_t = (synth_t - cand_t).float().abs().unsqueeze(-1)  # [C, 1]
        inp = torch.cat([s_exp, cand_feats, delta_t], dim=-1)     # [C, 2*F+1]
        return self.scorer(inp).squeeze(-1)                         # [C]

    # ------------------------------------------------------------------
    def forward_gumbel(
        self,
        synth_feats: torch.Tensor,      # [B, feat_dim]
        cand_feats: torch.Tensor,       # [C, feat_dim] shared candidate pool
        synth_ts: torch.Tensor,         # [B] integer timesteps
        cand_ts: torch.Tensor,          # [C] integer timesteps
        temporal_mask: torch.Tensor,    # [C] bool — cand_t <= synth_t (pre-computed)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gumbel-softmax edge selection (differentiable).

        Returns:
            soft_weights : [B, C] soft adjacency weights (∑ per row ≈ k)
            hard_edges   : [B, C] hard {0,1} selection (straight-through gradient)
        """
        B, C = synth_feats.size(0), cand_feats.size(0)
        # Mask future candidates (set their logits to -inf)
        t_mask = temporal_mask.unsqueeze(0).expand(B, -1)  # [B, C]

        all_logits = []
        for i in range(B):
            logits = self.score_candidates(
                synth_feats[i], cand_feats, synth_ts[i].item(), cand_ts
            )
            all_logits.append(logits)
        logits = torch.stack(all_logits, dim=0)  # [B, C]

        # Apply temporal mask
        logits = logits.masked_fill(~t_mask, float("-inf"))

        # Gumbel-softmax: sample k edges using top-k trick
        # We apply Gumbel noise then take top-k
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        perturbed = (logits + gumbel_noise) / max(self.tau, 1e-3)

        # Soft weights via softmax over all candidates (sum-to-1)
        soft_weights = F.softmax(perturbed, dim=-1)  # [B, C]

        # Hard selection: top-k per row (straight-through)
        topk_vals, topk_idx = perturbed.topk(min(self.k, C), dim=-1)  # [B, k]
        hard = torch.zeros_like(soft_weights)
        hard.scatter_(1, topk_idx, 1.0)
        # Straight-through gradient
        hard_st = (hard - soft_weights).detach() + soft_weights

        return soft_weights, hard_st

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_topk(
        self,
        synth_feats: torch.Tensor,
        cand_feats: torch.Tensor,
        synth_ts: torch.Tensor,
        cand_ts: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Hard top-k selection (no gradients). Returns list of B LongTensors
        containing candidate indices to connect to each synthetic node.
        """
        B = synth_feats.size(0)
        result = []
        for i in range(B):
            logits = self.score_candidates(
                synth_feats[i], cand_feats, synth_ts[i].item(), cand_ts
            )
            logits[~temporal_mask] = float("-inf")
            k = min(self.k, (temporal_mask).sum().item())
            _, idx = logits.topk(k)
            result.append(idx)
        return result

    def anneal_temperature(self, epoch: int, anneal_rate: float = 0.01, tau_min: float = 0.1):
        """Call after each epoch to anneal Gumbel temperature."""
        self.tau = max(tau_min, self.tau * (1 - anneal_rate))
