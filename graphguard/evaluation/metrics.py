"""
GraphGuard — Evaluation metrics.

Covers:
  - Classification metrics (illicit-class focused): F1, Recall, Precision, AUC-ROC, AUC-PR, Macro-F1
  - Structural validity metrics: Wasserstein / KL-div on degree & clustering distributions
  - Temporal violation rate tracker
"""
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from scipy.stats import wasserstein_distance, entropy as kl_entropy


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_class: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Compute all primary classification metrics focused on the illicit class.

    Args:
        y_true:       Ground-truth labels (0=licit, 1=illicit). Unknown (-1) must be filtered out beforehand.
        y_pred:       Predicted class labels.
        y_prob:       Predicted probability for class 1 (illicit). Shape [N].
        target_class: The minority class index (1 for illicit).
        verbose:      If True, prints sklearn classification report.

    Returns:
        dict with keys: illicit_f1, illicit_recall, illicit_precision,
                        auc_roc, auc_pr, macro_f1
    """
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_prob = y_prob[mask]

    illicit_f1      = f1_score(y_true, y_pred, pos_label=target_class, zero_division=0)
    illicit_recall  = recall_score(y_true, y_pred, pos_label=target_class, zero_division=0)
    illicit_prec    = precision_score(y_true, y_pred, pos_label=target_class, zero_division=0)
    macro_f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUC scores require at least one positive and one negative sample
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr  = average_precision_score(y_true, y_prob)
    except ValueError:
        auc_roc = float("nan")
        auc_pr  = float("nan")

    if verbose:
        print(classification_report(y_true, y_pred, target_names=["licit", "illicit"], zero_division=0))

    return {
        "illicit_f1":        illicit_f1,
        "illicit_recall":    illicit_recall,
        "illicit_precision": illicit_prec,
        "auc_roc":           auc_roc,
        "auc_pr":            auc_pr,
        "macro_f1":          macro_f1,
    }


def evaluate_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Run model inference and return classification metrics for masked nodes.
    Filters out unknown-label nodes (y == -1) automatically.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device), edge_index.to(device))
        probs  = torch.softmax(logits, dim=-1)[:, 1]  # P(illicit)
        preds  = logits.argmax(dim=-1)

    y_true = y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()
    y_prob = probs[mask].cpu().numpy()

    # Remove unknowns
    known = y_true != -1
    return compute_classification_metrics(y_true[known], y_pred[known], y_prob[known])


# ─────────────────────────────────────────────────────────────────────────────
# Structural validity metrics
# ─────────────────────────────────────────────────────────────────────────────
def wasserstein_distribution_gap(
    real_values: np.ndarray,
    fake_values: np.ndarray,
) -> float:
    """Earth-mover (Wasserstein-1) distance between two 1-D distributions."""
    return wasserstein_distance(real_values, fake_values)


def kl_distribution_gap(
    real_values: np.ndarray,
    fake_values: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    KL divergence between histograms of two distributions.
    Adds small epsilon to avoid log(0).
    """
    lo = min(real_values.min(), fake_values.min())
    hi = max(real_values.max(), fake_values.max()) + 1e-9
    bins = np.linspace(lo, hi, n_bins + 1)

    p, _ = np.histogram(real_values, bins=bins, density=True)
    q, _ = np.histogram(fake_values, bins=bins, density=True)

    eps = 1e-10
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()

    return float(kl_entropy(p, q))  # KL(P || Q)


def compute_structural_validity(
    real_degrees: np.ndarray,
    fake_degrees: np.ndarray,
    real_clustering: np.ndarray,
    fake_clustering: np.ndarray,
) -> dict:
    """
    Compute all structural validity metrics comparing real fraud-cluster
    statistics to synthetic fraud-cluster statistics.

    Returns dict with:
        degree_wasserstein, degree_kl,
        clustering_wasserstein, clustering_kl
    """
    return {
        "degree_wasserstein":      wasserstein_distribution_gap(real_degrees, fake_degrees),
        "degree_kl":               kl_distribution_gap(real_degrees, fake_degrees),
        "clustering_wasserstein":  wasserstein_distribution_gap(real_clustering, fake_clustering),
        "clustering_kl":           kl_distribution_gap(real_clustering, fake_clustering),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Temporal violation tracker
# ─────────────────────────────────────────────────────────────────────────────
def temporal_violation_rate(
    synth_ts: np.ndarray,   # [B] timesteps of synthetic nodes
    edge_list: list[tuple], # list of (synth_idx, cand_idx) pairs
    cand_ts: np.ndarray,    # [N_cand] timesteps of all candidates
) -> float:
    """
    Fraction of edges where the candidate node's timestep > synthetic node's timestep.
    Should trend toward 0% during training (Novelty Claim #1 quantitative proof).
    """
    if len(edge_list) == 0:
        return 0.0
    violations = sum(
        1 for (s_i, c_i) in edge_list if cand_ts[c_i] > synth_ts[s_i]
    )
    return violations / len(edge_list)
