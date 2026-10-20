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
    fbeta_score,
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
        dict with keys: illicit_f1, illicit_f05, illicit_recall, illicit_precision,
                        auc_roc, auc_pr, macro_f1
    """
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_prob = y_prob[mask]

    illicit_f1      = f1_score(y_true, y_pred, pos_label=target_class, zero_division=0)
    illicit_f2      = fbeta_score(y_true, y_pred, beta=2.0, pos_label=target_class, zero_division=0)
    illicit_f05     = fbeta_score(y_true, y_pred, beta=0.5, pos_label=target_class, zero_division=0)
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
        "illicit_f2":        illicit_f2,
        "illicit_f05":       illicit_f05,
        "illicit_recall":    illicit_recall,
        "illicit_precision": illicit_prec,
        "auc_roc":           auc_roc,
        "auc_pr":            auc_pr,
        "macro_f1":          macro_f1,
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    beta: float = 1.0,
    num_thresholds: int = 100,
) -> tuple[float, float]:
    """
    Tune the decision threshold strictly on validation probabilities.
    metric: 'f1' (Balanced, default), 'f2' (Recall priority),
            'f0.5' (Precision priority), 'precision', or 'recall'.
    Returns: (best_threshold, best_score)
    """
    mask = y_true != -1
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric in ("f0.5", "f2"):
            b = 0.5 if metric == "f0.5" else 2.0
            score = fbeta_score(y_true, preds, beta=b, pos_label=1, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, preds, pos_label=1, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, pos_label=1, zero_division=0)
        else:
            score = f1_score(y_true, preds, pos_label=1, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    return float(best_thresh), float(best_score)


def find_all_operating_thresholds(y_val: np.ndarray, val_probs: np.ndarray, min_recall: float = 0.65) -> dict:
    """
    Calibrate four strategic operating thresholds strictly on validation data:
      1. 'f1': Balanced Operating Point (Validation Max F1)
      2. 'f2': AML Recall-Priority Point (Validation Max F2, beta=2.0)
      3. 'constrained': Sweet-Spot Point (Validation Max F1 subject to Val Recall >= 0.65)
      4. 'f0.5': Ultra-High Precision Point (Validation Max F0.5, beta=0.5)
    """
    mask = y_val != -1
    y_val = y_val[mask]
    val_probs = val_probs[mask]

    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1, t_f1 = -1.0, 0.5
    best_f2, t_f2 = -1.0, 0.5
    best_f05, t_f05 = -1.0, 0.5
    best_const_f1, t_const = -1.0, 0.5
    max_rec_found, t_max_rec = -1.0, 0.5

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        p = precision_score(y_val, preds, pos_label=1, zero_division=0)
        r = recall_score(y_val, preds, pos_label=1, zero_division=0)
        f1 = (2 * p * r) / (p + r + 1e-8)
        f2 = (5 * p * r) / (4 * p + r + 1e-8)
        f05 = (1.25 * p * r) / (0.25 * p + r + 1e-8)

        if f1 > best_f1:
            best_f1, t_f1 = f1, t
        if f2 > best_f2:
            best_f2, t_f2 = f2, t
        if f05 > best_f05:
            best_f05, t_f05 = f05, t
        if r >= min_recall and f1 > best_const_f1:
            best_const_f1, t_const = f1, t
        if r > max_rec_found:
            max_rec_found, t_max_rec = r, t

    if best_const_f1 < 0:
        t_const = t_max_rec  # fallback to threshold that gave highest recall

    return {
        "f1": float(t_f1),
        "f2": float(t_f2),
        "constrained": float(t_const),
        "f0.5": float(t_f05),
    }


def evaluate_operating_point(y_test: np.ndarray, test_probs: np.ndarray, threshold: float) -> dict:
    """
    Evaluate binary classification predictions on test set given a specific decision threshold.
    Returns full confusion matrix (TP, FP, FN, TN), rates, and metrics.
    """
    mask = y_test != -1
    y_true = y_test[mask]
    probs = test_probs[mask]
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    p = precision_score(y_true, preds, pos_label=1, zero_division=0)
    r = recall_score(y_true, preds, pos_label=1, zero_division=0)
    f1 = (2 * p * r) / (p + r + 1e-8)
    f2 = (5 * p * r) / (4 * p + r + 1e-8)
    f05 = (1.25 * p * r) / (0.25 * p + r + 1e-8)

    try:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "f2": float(f2),
        "f05": float(f05),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
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
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= 0.5).long()
        else:
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
