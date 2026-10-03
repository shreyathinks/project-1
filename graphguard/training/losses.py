"""
GraphGuard — Loss functions for GAN training.

Losses:
  - Adversarial: BCE (with label smoothing option)
  - Temporal violation penalty: penalises edges to future-timestep nodes
  - Feature matching: ||E[real_feats] - E[fake_feats]||² (stabilises G training)
  - Optional InfoNCE contrastive (Phase 3.5 enhancement)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial losses
# ─────────────────────────────────────────────────────────────────────────────
def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Standard BCE discriminator loss with one-sided label smoothing.
    real → label (1 - smoothing), fake → label 0.
    """
    real_label = 1.0 - label_smoothing
    loss_real = F.binary_cross_entropy_with_logits(
        real_logits, torch.full_like(real_logits, real_label)
    )
    loss_fake = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    return (loss_real + loss_fake) * 0.5


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: G wants D to output 1 for fakes."""
    return F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temporal violation penalty
# ─────────────────────────────────────────────────────────────────────────────
def temporal_violation_penalty(
    edge_weights: torch.Tensor,     # [B, C] soft weights from EdgePredictor
    synth_ts: torch.Tensor,         # [B] synthetic node timesteps
    cand_ts: torch.Tensor,          # [C] candidate node timesteps
) -> torch.Tensor:
    """
    Penalises assigning positive weight to candidates whose timestep
    is LATER than the synthetic node's timestep.

    The penalty is the sum of soft weights assigned to future candidates,
    scaled by the number of violations — this should trend to ~0 during training.
    """
    B, C = edge_weights.shape
    # future_mask[b, c] = 1 if cand_ts[c] > synth_ts[b]
    future_mask = (cand_ts.unsqueeze(0) > synth_ts.unsqueeze(1)).float()  # [B, C]
    penalty = (edge_weights * future_mask).sum() / (B * C)
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# Feature matching loss (stabiliser)
# ─────────────────────────────────────────────────────────────────────────────
def feature_matching_loss(
    real_feats: torch.Tensor,   # [B_real, F]
    fake_feats: torch.Tensor,   # [B_fake, F]
) -> torch.Tensor:
    """
    ||mean(real_feats) - mean(fake_feats)||² 
    Encourages the generator to match the first-order statistics of real samples.
    Use as a secondary G loss term with a small weight (e.g. 0.1).
    """
    return F.mse_loss(fake_feats.mean(0), real_feats.mean(0))


# ─────────────────────────────────────────────────────────────────────────────
# InfoNCE contrastive loss (Phase 3.5 optional enhancement)
# ─────────────────────────────────────────────────────────────────────────────
def infonce_loss(
    real_embeds: torch.Tensor,   # [B, D] real subgraph embeddings from D_struct
    fake_embeds: torch.Tensor,   # [B, D] fake subgraph embeddings from D_struct
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE: encourage real embeddings to be close together (positive pairs)
    and far from fake embeddings (negative pairs).
    Works on the subgraph embedding output of Head B's GNN encoder.
    """
    # L2-normalise
    real_n = F.normalize(real_embeds, dim=-1)  # [B, D]
    fake_n = F.normalize(fake_embeds, dim=-1)  # [B, D]

    # Positive: real-real similarity (upper-triangular pairs)
    sim_rr = torch.mm(real_n, real_n.T) / temperature  # [B, B]
    # Negative: real-fake similarity
    sim_rf = torch.mm(real_n, fake_n.T) / temperature  # [B, B]

    B = real_n.size(0)
    labels = torch.arange(B, device=real_n.device)

    # Treat each real sample as the query; its positive is itself,
    # negatives are all fake samples
    logits = torch.cat([sim_rr, sim_rf], dim=-1)  # [B, 2B]
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Combined G loss
# ─────────────────────────────────────────────────────────────────────────────
def total_generator_loss(
    fake_node_logits: torch.Tensor,       # Head A output for fakes
    fake_struct_logits: torch.Tensor,     # Head B output for fakes (scalar per sample)
    edge_weights: torch.Tensor,           # [B, C] for temporal penalty
    synth_ts: torch.Tensor,
    cand_ts: torch.Tensor,
    real_feats: torch.Tensor,
    fake_feats: torch.Tensor,
    lambda_temporal: float = 1.0,
    lambda_fm: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    Combines all generator losses into a dict for easy logging.
    Returns total and individual component losses.
    """
    adv_a = generator_loss(fake_node_logits)
    adv_b = generator_loss(fake_struct_logits)
    temp_pen = temporal_violation_penalty(edge_weights, synth_ts, cand_ts)
    feat_m = feature_matching_loss(real_feats, fake_feats)

    total = adv_a + adv_b + lambda_temporal * temp_pen + lambda_fm * feat_m

    return {
        "total": total,
        "adv_node": adv_a,
        "adv_struct": adv_b,
        "temporal_penalty": temp_pen,
        "feature_matching": feat_m,
    }
