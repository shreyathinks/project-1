import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphguard.models.generator import TemporalGenerator
from graphguard.models.baseline_graphsage import GraphSAGEClassifier

def optimize_threshold(y_true, y_prob):
    """
    Fix for the Thresholding Trap:
    Instead of using 0.5 (or argmax on logits), we find the threshold that maximizes F1.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = 0.0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def train_classifier():
    print("Loading preprocessed data...")
    data_dir = "e:/GraphGuard/project-1/dataset/processed"
    df_nodes = pd.read_csv(os.path.join(data_dir, 'processed_nodes.csv'))
    df_edges = pd.read_csv(os.path.join(data_dir, 'processed_edges.csv'))
    
    # Sort by time_step
    df_nodes = df_nodes.sort_values('time_step').reset_index(drop=True)
    
    # Build tx_to_idx mapping
    tx_to_idx = {tx: i for i, tx in enumerate(df_nodes['txId'])}
    
    # Build edge_index
    valid_edges = df_edges[df_edges['txId1'].isin(tx_to_idx) & df_edges['txId2'].isin(tx_to_idx)]
    src = torch.tensor([tx_to_idx[tx] for tx in valid_edges['txId1']], dtype=torch.long)
    dst = torch.tensor([tx_to_idx[tx] for tx in valid_edges['txId2']], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    
    # Make edges undirected (bidirectional) so GNN can aggregate from both parents and children
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Base Graph
    X = torch.tensor(df_nodes[[f'f{i}' for i in range(1, 166)]].values, dtype=torch.float32)
    y = torch.tensor(df_nodes['class_label'].values, dtype=torch.float32)
    timesteps = torch.tensor(df_nodes['time_step'].values, dtype=torch.long)
    
    # STRICT METHODOLOGY (Phase 1.4): Temporal Splits
    # Train: 1-29, Val: 30-34, Test: 35-49
    known_mask = (y != -1)
    train_mask = known_mask & (timesteps <= 29)
    val_mask = known_mask & (timesteps >= 30) & (timesteps <= 34)
    test_mask = known_mask & (timesteps >= 35)
    
    real_train_idx = torch.nonzero(train_mask).squeeze()
    val_idx = torch.nonzero(val_mask).squeeze()
    test_idx = torch.nonzero(test_mask).squeeze()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = X.to(device)
    y = y.to(device)
    timesteps = timesteps.to(device)
    edge_index = edge_index.to(device)
    
    real_train_idx = real_train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    
    print("Loading Generator for Oversampling...")
    generator = TemporalGenerator(feat_dim=165).to(device)
    gen_path = "e:/GraphGuard/project-1/graphguard/models/trained/generator.pt"
    
    num_illicit = (y[real_train_idx] == 1).sum().item()
    num_licit = (y[real_train_idx] == 0).sum().item()
    num_to_generate = num_licit - num_illicit
    
    # Synthetic nodes will be appended to X and y, and their indices added to real_train_idx
    if os.path.exists(gen_path) and num_to_generate > 0:
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        generator.eval()
        
        illicit_mask = y[real_train_idx] == 1
        real_illicit_feats = X[real_train_idx][illicit_mask]
        
        idx = torch.randint(0, len(real_illicit_feats), (num_to_generate,), device=device)
        seed_feats = real_illicit_feats[idx]
        
        # STRICT METHODOLOGY: Use actual timesteps of the seed nodes, not dummy ones!
        seed_timesteps = timesteps[real_train_idx][illicit_mask][idx]
        
        dummy_neigh = torch.zeros_like(seed_feats)
        
        with torch.no_grad():
            synthetic_feats = generator.sample(seed_feats, dummy_neigh, seed_timesteps, n_samples=1)
            
        synthetic_labels = torch.ones(num_to_generate, dtype=torch.float32, device=device)
        
        X = torch.cat([X, synthetic_feats], dim=0)
        y = torch.cat([y, synthetic_labels], dim=0)
        
        # New nodes have indices from len(X)-num_to_generate to len(X)-1
        synth_idx = torch.arange(len(X) - num_to_generate, len(X), device=device)
        extended_train_idx = torch.cat([real_train_idx, synth_idx], dim=0)
        print(f"Oversampled {num_to_generate} synthetic illicit nodes.")
        
        # --- Edge Injection (KNN Fallback from Implementation Plan) ---
        print("Injecting edges for synthetic nodes via K-NN...")
        import torch.nn.functional as F
        
        real_illicit_norm = F.normalize(real_illicit_feats, p=2, dim=1)
        synth_norm = F.normalize(synthetic_feats, p=2, dim=1)
        
        # Memory-efficient similarity and top-K computation
        K = 3
        new_src = []
        new_dst = []
        
        global_real_illicit_idx = real_train_idx[illicit_mask]
        real_illicit_timesteps = timesteps[global_real_illicit_idx]
        
        # Process in batches to avoid OOM on similarity matrix
        batch_size_knn = 1000
        for i in range(0, num_to_generate, batch_size_knn):
            end_idx = min(i + batch_size_knn, num_to_generate)
            sim_batch = torch.matmul(synth_norm[i:end_idx], real_illicit_norm.t())
            
            # STRICT METHODOLOGY: Mask out future nodes (time_candidate > time_synthetic)
            for local_i in range(end_idx - i):
                synth_t = seed_timesteps[i + local_i]
                # Candidates must be <= synth_t
                valid_candidates_mask = (real_illicit_timesteps <= synth_t)
                
                # If no valid candidates (rare), fallback to all illicit nodes
                if not valid_candidates_mask.any():
                    valid_candidates_mask = torch.ones_like(valid_candidates_mask, dtype=torch.bool)
                
                # Apply mask to similarity scores (-inf to invalid)
                masked_sim = sim_batch[local_i].clone()
                masked_sim[~valid_candidates_mask] = -float('inf')
                
                # Ensure we don't ask for more neighbors than valid candidates
                actual_k = min(K, valid_candidates_mask.sum().item())
                if actual_k == 0:
                    continue
                    
                _, topk_idx = torch.topk(masked_sim, k=actual_k)
                
                synth_node_id = synth_idx[i + local_i].item()
                for k in range(actual_k):
                    neighbor_local = topk_idx[k].item()
                    neighbor_global = global_real_illicit_idx[neighbor_local].item()
                    new_src.extend([synth_node_id, neighbor_global])
                    new_dst.extend([neighbor_global, synth_node_id])
                    
        new_edges = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        print(f"Injected {new_edges.size(1)} synthetic edges into the graph.")
        
    else:
        extended_train_idx = real_train_idx
        
    # Initialize GraphSAGE
    # We change out_channels to 1 for BCEWithLogitsLoss
    model = GraphSAGEClassifier(in_channels=165, hidden_channels=256, out_channels=1).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 1000
    
    # Setup Results Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"e:/GraphGuard/project-1/results/classifier_gnn/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    train_loss_hist = []
    val_loss_hist = []
    val_f1_hist = []
    
    print("Training Downstream GraphSAGE Classifier (Full Batch)...")
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        
        # Full batch forward pass
        logits = model(X, edge_index).squeeze()
        
        # Compute loss only on extended_train_idx
        loss = criterion(logits[extended_train_idx], y[extended_train_idx])
        loss.backward()
        opt.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = logits[val_idx]
            val_loss = criterion(val_logits, y[val_idx]).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            y_val_np = y[val_idx].cpu().numpy()
            
            best_thresh, val_f1 = optimize_threshold(y_val_np, val_probs)
            
        train_loss_hist.append(loss.item())
        val_loss_hist.append(val_loss)
        val_f1_hist.append(val_f1)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} @ T={best_thresh:.2f}")

    print("Evaluating on Test Set...")
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index).squeeze()
        test_logits = logits[test_idx]
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        y_test_np = y[test_idx].cpu().numpy()
        
        test_preds = (test_probs >= best_thresh).astype(int)
        test_f1 = f1_score(y_test_np, test_preds, zero_division=0)
        test_prec = precision_score(y_test_np, test_preds, zero_division=0)
        test_rec = recall_score(y_test_np, test_preds, zero_division=0)
        test_auc = roc_auc_score(y_test_np, test_probs)
        
    print(f"Test Results (Threshold={best_thresh:.2f}): F1={test_f1:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}, AUC={test_auc:.4f}")
    
    print(f"Saving plots and metrics locally to {run_dir}...")
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
    plt.plot(range(1, epochs + 1), val_f1_hist, label='Val F1', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 over Epochs')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'val_f1_curve.png'))
    plt.close()
    
    metrics = {
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_auc": test_auc,
        "best_val_threshold": best_thresh,
        "epochs": epochs
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    os.makedirs("e:/GraphGuard/project-1/graphguard/models/trained", exist_ok=True)
    torch.save(model.state_dict(), "e:/GraphGuard/project-1/graphguard/models/trained/graphsage.pt")
    torch.save(model.state_dict(), os.path.join(run_dir, "graphsage.pt"))
    print("Classifier saved locally.")

if __name__ == "__main__":
    train_classifier()
