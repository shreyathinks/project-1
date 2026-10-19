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
from graphguard.evaluation.metrics import optimize_threshold, compute_classification_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train Downstream GraphSAGE Classifier on GraphGuard Augmented Graph")
    parser.add_argument("--aug_ratio", type=float, default=0.30, help="Augmentation ratio relative to training illicit class (default 0.30 = +1038 nodes)")
    parser.add_argument("--top_k", type=int, default=2, help="Directed degree K for synthetic edges (default 2)")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "bce"], help="Loss criterion (focal or bce)")
    parser.add_argument("--threshold_metric", type=str, default="f0.5", choices=["f0.5", "f1"], help="Metric to optimize on validation set (default f0.5)")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Alpha parameter for Focal Loss (default 0.25)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss (default 2.0)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default 1000)")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate (default 5e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (default 1e-5)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for GraphSAGE (default 256)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate (default 0.3)")
    args, _ = parser.parse_known_args()
    return args

def train_classifier(args=None):
    if args is None:
        args = parse_args()

    print("=" * 80)
    print("🛡️ GraphGuard Downstream Classifier Training Pipeline")
    print(f"Config: Aug Ratio={args.aug_ratio*100:.1f}%, Top-K={args.top_k}, Loss={args.loss}, Threshold Metric={args.threshold_metric}")
    print("=" * 80)

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
    
    # Fix 1: Symmetrize ONLY the real edges from the original graph
    real_edges_sym = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    print("Loading Generator and Edge Predictor for Oversampling...")
    generator = TemporalGenerator(feat_dim=165).to(device)
    edge_predictor = EdgePredictor(feat_dim=165, k=args.top_k).to(device)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, "graphguard", "models")
    gen_path = os.path.join(models_dir, "graphguard_generator.pt")
    ep_path = os.path.join(models_dir, "graphguard_edge_predictor.pt")
    
    num_illicit = (y[real_train_idx] == 1).sum().item()
    # Fix 2: Calibrated Augmentation Budget (30% ratio by default)
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
        
        print(f"Generating {num_to_generate} synthetic node features (30% ratio)...")
        neigh_feats = aggregate_neighbourhood(X, seed_center_idx, edge_index, k_hops=1)
        
        with torch.no_grad():
            synthetic_feats = generator.sample(seed_feats, neigh_feats, seed_timesteps, n_samples=1)
            
        synthetic_labels = torch.ones(num_to_generate, dtype=torch.float32, device=device)
        
        X = torch.cat([X, synthetic_feats], dim=0)
        y = torch.cat([y, synthetic_labels], dim=0)
        
        synth_idx = torch.arange(len(X) - num_to_generate, len(X), device=device)
        extended_train_idx = torch.cat([real_train_idx, synth_idx], dim=0)
        print(f"Oversampled {num_to_generate} synthetic illicit nodes.")
        
        # --- Fix 1 & Fix 2: Unidirectional Edge Injection with Restricted Candidate Pool ---
        print("Injecting unidirectional edges for synthetic nodes via trained EdgePredictor...")
        
        # Restrict candidate pool to known illicit (y == 1) or unlabeled (y == -1) in train_mask
        # Exclude confirmed licit nodes (y == 0) from being wired as direct 1-hop neighbors
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
        
        with torch.no_grad():
            for i in range(num_to_generate):
                if (i + 1) % 500 == 0 or (i + 1) == num_to_generate:
                    print(f"  Injected edges for {i+1}/{num_to_generate} nodes...")
                
                s_feat = synthetic_feats[i]
                s_ts = seed_timesteps[i]
                
                term1 = s_feat @ W_s.T
                delta_t = (s_ts - cand_ts).float().abs().unsqueeze(-1)
                term3 = delta_t @ W_t.T
                
                pre = term1.unsqueeze(0) + term2 + term3 + b.unsqueeze(0)
                
                h = edge_predictor.scorer[1](pre)
                h = edge_predictor.scorer[2](h)
                h = edge_predictor.scorer[3](h)
                h = edge_predictor.scorer[4](h)
                h = edge_predictor.scorer[5](h)
                logits = h.squeeze(-1)
                
                # Tight local temporal window: max(1, s_ts - 2) <= cand_ts <= s_ts
                min_ts_i = max(1, s_ts.item() - 2)
                temporal_mask_i = (cand_ts <= s_ts) & (cand_ts >= min_ts_i)
                
                # Safe fallback if tight window has 0 candidates
                if temporal_mask_i.sum() == 0:
                    temporal_mask_i = (cand_ts <= s_ts)
                if temporal_mask_i.sum() == 0:
                    temporal_mask_i = torch.ones_like(temporal_mask_i, dtype=torch.bool)
                
                logits[~temporal_mask_i] = float("-inf")
                
                k = min(args.top_k, temporal_mask_i.sum().item())
                _, neighbors = logits.topk(k)
                
                synth_node_id = synth_idx[i].item()
                for neighbor_local in neighbors:
                    neighbor_global = cand_idx[neighbor_local.item()].item()
                    # Fix 1: Unidirectional edge from candidate (source) -> synthetic node (target)
                    new_src.append(neighbor_global)
                    new_dst.append(synth_node_id)
                        
        synth_edges_directed = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        edge_index = torch.cat([real_edges_sym, synth_edges_directed], dim=1)
        print(f"Injected {synth_edges_directed.size(1)} strictly unidirectional synthetic edges (candidate -> synth).")
        print(f"Total graph edges for training: {edge_index.size(1)}")
        
    else:
        print("Generator/EdgePredictor not found or no balancing needed. Training on real data only.")
        extended_train_idx = real_train_idx
        edge_index = real_edges_sym
        
    # Fix 3: Initialize GraphSAGE with Ego-Feature Residual Skip Connections
    model = GraphSAGEClassifier(
        in_channels=165,
        hidden_channels=args.hidden_dim,
        out_channels=1,
        dropout=args.dropout,
    ).to(device)
    
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Fix 5: Focal Loss Criterion
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Loss function: Binary FocalLoss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss function: BCEWithLogitsLoss")
    
    epochs = args.epochs
    
    # Setup Results Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "results", "classifier_gnn", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    train_loss_hist = []
    val_loss_hist = []
    val_metric_hist = []
    
    print(f"\nTraining Downstream GraphSAGE Classifier for {epochs} epochs (Full Batch)...")
    best_val_metric = 0.0
    best_thresh = 0.5
    best_model_weights = None
    
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        
        logits = model(X, edge_index).squeeze()
        loss = criterion(logits[extended_train_idx], y[extended_train_idx])
        loss.backward()
        opt.step()
        
        # Validation evaluation & threshold calibration
        model.eval()
        with torch.no_grad():
            val_logits = logits[val_idx]
            val_loss = criterion(val_logits, y[val_idx]).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            y_val_np = y[val_idx].cpu().numpy()
            
            # Fix 5: Validation Threshold Calibration (F0.5 or F1)
            thresh, val_score = optimize_threshold(
                y_val_np,
                val_probs,
                metric=args.threshold_metric,
                beta=0.5 if args.threshold_metric == "f0.5" else 1.0,
            )
            
            if val_score > best_val_metric:
                best_val_metric = val_score
                best_thresh = thresh
                best_model_weights = copy.deepcopy(model.state_dict())
            
        train_loss_hist.append(loss.item())
        val_loss_hist.append(val_loss)
        val_metric_hist.append(val_score)
        
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch+1:4d}/{epochs} | Train Loss: {loss.item():.4f} | Val {args.threshold_metric.upper()}: {val_score:.4f} @ T={thresh:.2f}")

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\nRestored best model checkpoint with Val {args.threshold_metric.upper()}={best_val_metric:.4f} at Threshold={best_thresh:.2f}")

    print("\n" + "=" * 80)
    print("Evaluating on Untouched Real Test Set (Timesteps 40-49)...")
    print("=" * 80)
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index).squeeze()
        test_logits = logits[test_idx]
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        y_test_np = y[test_idx].cpu().numpy()
        
        test_preds = (test_probs >= best_thresh).astype(int)
        metrics_dict = compute_classification_metrics(y_test_np, test_preds, test_probs, verbose=True)
        
    print(f"\n🏆 Final Test Performance Summary (Threshold T={best_thresh:.2f}):")
    print(f"  • Illicit Precision : {metrics_dict['illicit_precision']:.4f}")
    print(f"  • Illicit Recall    : {metrics_dict['illicit_recall']:.4f}")
    print(f"  • Illicit F1-Score  : {metrics_dict['illicit_f1']:.4f}")
    print(f"  • Illicit F0.5-Score: {metrics_dict['illicit_f05']:.4f}")
    print(f"  • PR-AUC            : {metrics_dict['auc_pr']:.4f}")
    print(f"  • ROC-AUC           : {metrics_dict['auc_roc']:.4f}")
    
    print(f"\nSaving plots and metrics locally to {run_dir}...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_loss_hist, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss_hist, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GNN Classifier Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), val_metric_hist, label=f'Val {args.threshold_metric.upper()}', color='green')
    plt.xlabel('Epoch')
    plt.ylabel(f'{args.threshold_metric.upper()} Score')
    plt.title(f'Validation {args.threshold_metric.upper()} over Epochs')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'val_metric_curve.png'))
    plt.close()
    
    metrics_summary = {
        "test_f1": metrics_dict['illicit_f1'],
        "test_f05": metrics_dict['illicit_f05'],
        "test_precision": metrics_dict['illicit_precision'],
        "test_recall": metrics_dict['illicit_recall'],
        "test_roc_auc": metrics_dict['auc_roc'],
        "test_pr_auc": metrics_dict['auc_pr'],
        "best_val_threshold": best_thresh,
        "best_val_metric": best_val_metric,
        "aug_ratio": args.aug_ratio,
        "top_k": args.top_k,
        "loss": args.loss,
        "threshold_metric": args.threshold_metric,
        "epochs": epochs
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "graphguard_classifier.pt"))
    torch.save(model.state_dict(), os.path.join(models_dir, "graphsage.pt"))
    torch.save(model.state_dict(), os.path.join(run_dir, "graphsage.pt"))
    print("Classifier checkpoint saved successfully.")

    return metrics_summary

if __name__ == "__main__":
    train_classifier()
