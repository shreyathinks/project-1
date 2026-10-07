import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphguard.models.generator import TemporalGenerator, aggregate_neighbourhood
from graphguard.models.edge_predictor import EdgePredictor
from graphguard.models.baseline_graphsage import GraphSAGEClassifier

def optimize_threshold(y_true, y_prob):
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
    print("Loading preprocessed .pt data...")
    # Fix Issue 1: Correct paths to local processed data
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
    
    real_train_idx = torch.nonzero(train_mask).squeeze().to(device)
    val_idx = torch.nonzero(val_mask).squeeze().to(device)
    test_idx = torch.nonzero(test_mask).squeeze().to(device)
    
    print("Loading Generator and Edge Predictor for Oversampling...")
    generator = TemporalGenerator(feat_dim=166).to(device)
    edge_predictor = EdgePredictor(feat_dim=166).to(device)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, "graphguard", "models", "trained")
    gen_path = os.path.join(models_dir, "generator.pt")
    ep_path = os.path.join(models_dir, "edge_predictor.pt")
    
    num_illicit = (y[real_train_idx] == 1).sum().item()
    num_licit = (y[real_train_idx] == 0).sum().item()
    num_to_generate = num_licit - num_illicit
    
    if os.path.exists(gen_path) and os.path.exists(ep_path) and num_to_generate > 0:
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        generator.eval()
        edge_predictor.load_state_dict(torch.load(ep_path, map_location=device))
        edge_predictor.eval()
        
        illicit_mask = y[real_train_idx] == 1
        real_illicit_idx = real_train_idx[illicit_mask]
        real_illicit_feats = X[real_illicit_idx]
        
        idx = torch.randint(0, len(real_illicit_feats), (num_to_generate,), device=device)
        seed_feats = real_illicit_feats[idx]
        seed_timesteps = timesteps[real_illicit_idx][idx]
        seed_center_idx = real_illicit_idx[idx]
        
        print("Generating synthetic node features...")
        neigh_feats = aggregate_neighbourhood(X, seed_center_idx, edge_index, k_hops=1)
        
        with torch.no_grad():
            synthetic_feats = generator.sample(seed_feats, neigh_feats, seed_timesteps, n_samples=1)
            
        synthetic_labels = torch.ones(num_to_generate, dtype=torch.float32, device=device)
        
        X = torch.cat([X, synthetic_feats], dim=0)
        y = torch.cat([y, synthetic_labels], dim=0)
        
        synth_idx = torch.arange(len(X) - num_to_generate, len(X), device=device)
        extended_train_idx = torch.cat([real_train_idx, synth_idx], dim=0)
        print(f"Oversampled {num_to_generate} synthetic illicit nodes.")
        
        # --- Fix Issue 4 & 6: Use EdgePredictor instead of K-NN for broad topology ---
        print("Injecting edges for synthetic nodes via trained EdgePredictor...")
        
        cand_feats = X[:len(X)-num_to_generate]
        cand_ts = timesteps[:len(timesteps)-num_to_generate]
        cand_idx_tensor = torch.arange(len(cand_feats), device=device)
        
        new_src = []
        new_dst = []
        
        batch_size = 500
        with torch.no_grad():
            for i in range(0, num_to_generate, batch_size):
                end = min(i + batch_size, num_to_generate)
                batch_synth_feats = synthetic_feats[i:end]
                batch_synth_ts = seed_timesteps[i:end]
                
                temporal_mask = (cand_ts.unsqueeze(0) <= batch_synth_ts.unsqueeze(1))
                
                topk_indices_list = edge_predictor.select_topk(
                    batch_synth_feats, cand_feats, batch_synth_ts, cand_ts, temporal_mask
                )
                
                for local_i, neighbors in enumerate(topk_indices_list):
                    synth_node_id = synth_idx[i + local_i].item()
                    for neighbor_local in neighbors:
                        neighbor_global = cand_idx_tensor[neighbor_local.item()].item()
                        new_src.extend([synth_node_id, neighbor_global])
                        new_dst.extend([neighbor_global, synth_node_id])
                        
        new_edges = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        print(f"Injected {new_edges.size(1)} synthetic edges into the graph.")
        
    else:
        print("Generator/EdgePredictor not found or no balancing needed. Training on real data only.")
        extended_train_idx = real_train_idx
        
    # Make edges undirected so GNN can aggregate from both parents and children
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Initialize GraphSAGE
    model = GraphSAGEClassifier(in_channels=166, hidden_channels=256, out_channels=1).to(device)
    opt = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 1000
    
    # Setup Results Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "results", "classifier_gnn", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    train_loss_hist = []
    val_loss_hist = []
    val_f1_hist = []
    
    print("Training Downstream GraphSAGE Classifier (Full Batch)...")
    best_val_f1 = 0
    best_thresh = 0.5
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        
        logits = model(X, edge_index).squeeze()
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
            
            thresh, val_f1 = optimize_threshold(y_val_np, val_probs)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_thresh = thresh
            
        train_loss_hist.append(loss.item())
        val_loss_hist.append(val_loss)
        val_f1_hist.append(val_f1)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} @ T={thresh:.2f}")

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
    
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "graphsage.pt"))
    torch.save(model.state_dict(), os.path.join(run_dir, "graphsage.pt"))
    print("Classifier saved locally.")

if __name__ == "__main__":
    train_classifier()
