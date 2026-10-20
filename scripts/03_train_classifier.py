import os
import sys
import argparse
import copy
import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, fbeta_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphguard.models.generator import TemporalGenerator, aggregate_neighbourhood
from graphguard.models.edge_predictor import EdgePredictor
from graphguard.models.baseline_graphsage import GraphSAGEClassifier
from graphguard.training.losses import FocalLoss
from graphguard.evaluation.metrics import (
    optimize_threshold,
    find_all_operating_thresholds,
    evaluate_operating_point,
    compute_classification_metrics,
)

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Downstream GraphSAGE Classifier on GraphGuard Augmented Graph")
    parser.add_argument("--aug_ratio", type=float, default=1.50, help="Augmentation ratio relative to training illicit class (default 1.50 = +5193 nodes)")
    parser.add_argument("--top_k", type=int, default=3, help="Directed degree K for synthetic edges (default 3)")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "bce"], help="Loss criterion (focal or bce)")
    parser.add_argument("--threshold_metric", type=str, default="f1", choices=["f1", "f2", "f0.5", "constrained"], help="Metric to optimize on validation set (default f1)")
    parser.add_argument("--focal_alpha", type=float, default=0.75, help="Alpha parameter for Focal Loss (default 0.75, penalizes missed fraud 3x)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss (default 2.0)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default 200)")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate (default 5e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (default 1e-5)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for GraphSAGE (default 256)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default 0.3)")
    args, _ = parser.parse_known_args()
    return args

def train_classifier(args=None):
    if args is None:
        args = parse_args()

    print("=" * 90)
    print("[GraphGuard] Downstream Classifier Training Pipeline (High-Recall & High-Precision Calibration)")
    print(f"Config: Aug Ratio={args.aug_ratio*100:.1f}%, Top-K={args.top_k}, Loss={args.loss} (alpha={args.focal_alpha}, gamma={args.focal_gamma}), Epochs={args.epochs}")
    print("=" * 90)

    print("Loading preprocessed .pt data...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'graphguard', 'data', 'processed')
    
    X = torch.load(os.path.join(data_dir, 'node_features.pt'))
    y = torch.load(os.path.join(data_dir, 'labels.pt'))
    timesteps = torch.load(os.path.join(data_dir, 'timestep.pt'))
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'))
    train_mask = torch.load(os.path.join(data_dir, 'train_mask.pt'))
    val_mask = torch.load(os.path.join(data_dir, 'val_mask.pt'))
    test_mask = torch.load(os.path.join(data_dir, 'test_mask.pt'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    y = y.to(device).float()
    timesteps = timesteps.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    
    # Filter out label -1 (unknown) for supervised training, validation, and testing
    real_train_idx = torch.nonzero(train_mask & (y != -1)).squeeze().to(device)
    val_idx = torch.nonzero(val_mask & (y != -1)).squeeze().to(device)
    test_idx = torch.nonzero(test_mask & (y != -1)).squeeze().to(device)
    
    # Symmetrize real edges from the original graph
    real_edges_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    print("Loading Generator and Edge Predictor for Oversampling...")
    generator = TemporalGenerator(feat_dim=165).to(device)
    edge_predictor = EdgePredictor(feat_dim=165, k=args.top_k).to(device)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, "graphguard", "models")
    gen_path = os.path.join(models_dir, "graphguard_generator.pt")
    ep_path = os.path.join(models_dir, "graphguard_edge_predictor.pt")
    
    num_illicit = (y[real_train_idx] == 1).sum().item()
    # 150% Augmentation Budget: +5,193 synthetic illicit nodes (~25% of labeled training set)
    num_to_generate = int(num_illicit * args.aug_ratio)
    
    if os.path.exists(gen_path) and os.path.exists(ep_path) and num_to_generate > 0:
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        generator.eval()
        edge_predictor.load_state_dict(torch.load(ep_path, map_location=device))
        edge_predictor.eval()
        
        illicit_mask = y[real_train_idx] == 1
        real_illicit_idx = real_train_idx[illicit_mask]
        real_illicit_feats = X[real_illicit_idx]
        
        torch.manual_seed(42)
        idx = torch.randint(0, len(real_illicit_feats), (num_to_generate,), device=device)
        seed_feats = real_illicit_feats[idx]
        seed_timesteps = timesteps[real_illicit_idx][idx]
        seed_center_idx = real_illicit_idx[idx]
        
        print(f"Generating {num_to_generate} synthetic node features ({args.aug_ratio*100:.0f}% augmentation budget)...")
        neigh_feats = aggregate_neighbourhood(X, seed_center_idx, edge_index, k_hops=1)
        
        with torch.no_grad():
            synthetic_feats = generator.sample(seed_feats, neigh_feats, seed_timesteps, n_samples=1)
            
        synthetic_labels = torch.ones(num_to_generate, dtype=torch.float32, device=device)
        
        X = torch.cat([X, synthetic_feats], dim=0)
        y = torch.cat([y, synthetic_labels], dim=0)
        
        synth_idx = torch.arange(len(X) - num_to_generate, len(X), device=device)
        extended_train_idx = torch.cat([real_train_idx, synth_idx], dim=0)
        print(f"Oversampled {num_to_generate} synthetic illicit nodes (total training nodes: {len(extended_train_idx)}).")
        
        # --- Class-Conditional 'Goldilocks' Edge Wiring ---
        print("Injecting Class-Conditional 'Goldilocks' synthetic edges via EdgePredictor...")
        print("  - Edges to Real Illicit candidates (y == 1): BIDIRECTIONAL (reinforces sparse fraud clusters)")
        print("  - Edges to Licit/Unlabeled candidates: STRICTLY UNIDIRECTIONAL (candidate -> synth, zero feature bleed)")
        
        cand_mask = train_mask & (y[:len(X)-num_to_generate] != 0)
        cand_idx = torch.nonzero(cand_mask).squeeze().to(device)
        cand_feats = X[cand_idx]
        cand_ts = timesteps[cand_idx]
        
        new_src = []
        new_dst = []
        
        lin1 = edge_predictor.scorer[0]
        W_s = lin1.weight[:, :165]
        W_c = lin1.weight[:, 165:330]
        W_t = lin1.weight[:, 330:]
        b = lin1.bias
        
        term2 = cand_feats @ W_c.T  # [C, hidden_dim]
        
        bidirectional_count = 0
        unidirectional_count = 0
        
        with torch.no_grad():
            for i in range(num_to_generate):
                if (i + 1) % 1000 == 0 or (i + 1) == num_to_generate:
                    print(f"  Wired edges for {i+1}/{num_to_generate} synthetic nodes...")
                
                s_feat = synthetic_feats[i]
                s_ts = seed_timesteps[i]
                
                min_ts_i = max(1, s_ts.item() - 2)
                t_mask = (cand_ts <= s_ts) & (cand_ts >= min_ts_i)
                if t_mask.sum() == 0:
                    t_mask = (cand_ts <= s_ts)
                if t_mask.sum() == 0:
                    t_mask = torch.ones_like(cand_ts, dtype=torch.bool)
                
                sub_term2 = term2[t_mask]
                sub_cand_idx = cand_idx[t_mask]
                sub_cand_ts = cand_ts[t_mask]
                
                term1 = s_feat @ W_s.T
                delta_t = (s_ts - sub_cand_ts).float().abs().unsqueeze(-1)
                term3 = delta_t @ W_t.T
                
                pre = term1.unsqueeze(0) + sub_term2 + term3 + b.unsqueeze(0)
                logits = edge_predictor.scorer[1:](pre).squeeze(-1)
                
                k = min(args.top_k, logits.size(0))
                _, neighbors = logits.topk(k)
                
                synth_node_id = synth_idx[i].item()
                for neighbor_local in neighbors:
                    neighbor_global = sub_cand_idx[neighbor_local.item()].item()
                    
                    # 1. Candidate -> Synthetic node (always present)
                    new_src.append(neighbor_global)
                    new_dst.append(synth_node_id)
                    
                    # 2. Class-Conditional Bidirectional check:
                    # If candidate is a Real Illicit node (y == 1), add reverse edge (synth -> candidate)
                    if y[neighbor_global] == 1:
                        new_src.append(synth_node_id)
                        new_dst.append(neighbor_global)
                        bidirectional_count += 1
                    else:
                        unidirectional_count += 1
                        
        synth_edges = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        edge_index = torch.cat([real_edges_sym, synth_edges], dim=1)
        print(f"Injected {synth_edges.size(1)} Goldilocks edges ({bidirectional_count} bidirectional fraud-reinforcing, {unidirectional_count} clean unidirectional).")
        print(f"Total graph edges for GNN training: {edge_index.size(1)}")
        
    else:
        print("Generator/EdgePredictor not found or no balancing needed. Training on real data only.")
        extended_train_idx = real_train_idx
        edge_index = real_edges_sym
        
    # Hybrid Mean + Max GraphSAGE Classifier with Ego-Feature Residual Skip Connections
    model = GraphSAGEClassifier(
        in_channels=165,
        hidden_channels=args.hidden_dim,
        out_channels=1,
        dropout=args.dropout,
    ).to(device)
    
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Recall-Prioritized Focal Loss: alpha=0.75 (3x penalty on missed fraud)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Loss criterion: Binary FocalLoss (alpha={args.focal_alpha} [favors Recall 3x], gamma={args.focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss criterion: BCEWithLogitsLoss")
    
    epochs = args.epochs
    
    # Setup Results Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "results", "classifier_gnn", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    train_loss_hist = []
    val_epochs = []
    val_loss_hist = []
    val_f1_hist = []
    
    print(f"\nTraining Hybrid Mean+Max GraphSAGE Classifier for {epochs} epochs (Full Batch)...", flush=True)
    best_val_score = -1.0
    best_thresh = 0.5
    best_model_weights = None
    
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        
        logits = model(X, edge_index).squeeze()
        loss = criterion(logits[extended_train_idx], y[extended_train_idx])
        loss.backward()
        opt.step()
        
        train_loss_hist.append(loss.item())
        
        # Validation evaluation cadence (every 5 epochs, early epochs, and final epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs or (epoch + 1) <= 5:
            model.eval()
            with torch.no_grad():
                eval_logits = model(X, edge_index).squeeze()
                val_logits = eval_logits[val_idx]
                val_loss = criterion(val_logits, y[val_idx]).item()
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                y_val_np = y[val_idx].cpu().numpy()
                
                thresh, val_score = optimize_threshold(
                    y_val_np,
                    val_probs,
                    metric=args.threshold_metric,
                    beta=1.0 if args.threshold_metric == "f1" else 2.0,
                )
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_thresh = thresh
                    best_model_weights = copy.deepcopy(model.state_dict())
                
            val_epochs.append(epoch + 1)
            val_loss_hist.append(val_loss)
            val_f1_hist.append(val_score)
        
        if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch+1:4d}/{epochs} | Train Loss: {loss.item():.4f} | Val {args.threshold_metric.upper()}: {best_val_score:.4f} @ T={best_thresh:.2f}", flush=True)

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\nRestored best model checkpoint with Val {args.threshold_metric.upper()}={best_val_score:.4f} at Threshold={best_thresh:.2f}")

    print("\n" + "=" * 125)
    print("[RESULTS] RIGOROUS MULTI-POINT EVALUATION ON UNTOUCHED REAL TEST SET (TIMESTEPS 40-49)")
    print("=" * 125)
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index).squeeze()
        val_logits = logits[val_idx]
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        y_val_np = y[val_idx].cpu().numpy()

        test_logits = logits[test_idx]
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        y_test_np = y[test_idx].cpu().numpy()
        
        # Calibrate all 4 operating thresholds strictly on Validation Set
        thresholds_dict = find_all_operating_thresholds(y_val_np, val_probs, min_recall=0.65)
        
        points = [
            ("1. Balanced Operating Point (Max F1)", thresholds_dict["f1"]),
            ("2. AML Recall-Priority Point (Max F2)", thresholds_dict["f2"]),
            ("3. Sweet-Spot Point (Val Rec >= 0.65)", thresholds_dict["constrained"]),
            ("4. Ultra-High Precision Point (Max F0.5)", thresholds_dict["f0.5"]),
        ]
        
        results_by_point = {}
        for label, t in points:
            results_by_point[label] = evaluate_operating_point(y_test_np, test_probs, t)
            
        # Full threshold sweep across T in [0.05, 0.95]
        threshold_sweep = []
        for t_sweep in np.arange(0.05, 0.96, 0.01):
            threshold_sweep.append(evaluate_operating_point(y_test_np, test_probs, float(t_sweep)))
            
    # Print formatted 4-point comparison table
    print(f"\n{'Operating Point':<38} | {'Thresh':<6} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'TN':<6} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'F2':<7} | {'ROC-AUC':<7} | {'PR-AUC':<7}")
    print("-" * 125)
    for label, res in results_by_point.items():
        print(f"{label:<38} | {res['threshold']:<6.2f} | {res['tp']:<5d} | {res['fp']:<5d} | {res['fn']:<5d} | {res['tn']:<6d} | {res['precision']:<7.4f} | {res['recall']:<7.4f} | {res['f1']:<7.4f} | {res['f2']:<7.4f} | {res['roc_auc']:<7.4f} | {res['pr_auc']:<7.4f}")
    print("=" * 125)
    
    metrics_summary = {
        "operating_points": results_by_point,
        "primary_point": results_by_point["1. Balanced Operating Point (Max F1)"],
        "threshold_sweep": threshold_sweep,
        "aug_ratio": args.aug_ratio,
        "top_k": args.top_k,
        "loss": args.loss,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "epochs": epochs
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "graphguard_classifier.pt"))
    torch.save(model.state_dict(), os.path.join(models_dir, "graphsage.pt"))
    torch.save(model.state_dict(), os.path.join(run_dir, "graphsage.pt"))
    print("Classifier checkpoint and metrics saved successfully.")

    print(f"\nSaving plots locally to {run_dir}...")
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), train_loss_hist, label='Train Loss')
        if val_epochs:
            plt.plot(val_epochs, val_loss_hist, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Hybrid Mean+Max GNN Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
        plt.close()
        
        plt.figure(figsize=(10, 5))
        if val_epochs:
            plt.plot(val_epochs, val_f1_hist, label=f'Val {args.threshold_metric.upper()}', color='green')
        plt.xlabel('Epoch')
        plt.ylabel(f'{args.threshold_metric.upper()} Score')
        plt.title(f'Validation {args.threshold_metric.upper()} over Epochs')
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'val_metric_curve.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to save plots: {e}")

    return metrics_summary

if __name__ == "__main__":
    train_classifier()
