import os
import sys
import copy
import json
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphguard.models.generator import TemporalGenerator, aggregate_neighbourhood

def smote_oversample(X_illicit, num_samples, k_neighbors=5, random_state=42):
    """
    Lightweight, robust pure-numpy/sklearn SMOTE implementation.
    Does not require external heavy dependencies.
    """
    np.random.seed(random_state)
    n_illicit = len(X_illicit)
    k = min(k_neighbors, n_illicit - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X_illicit)
    _, indices = nn.kneighbors(X_illicit)
    
    synthetic = []
    for _ in range(num_samples):
        i = np.random.randint(0, n_illicit)
        nn_idx = indices[i, np.random.randint(1, k + 1)]
        diff = X_illicit[nn_idx] - X_illicit[i]
        alpha = np.random.rand()
        synthetic.append(X_illicit[i] + alpha * diff)
    return np.array(synthetic, dtype=np.float32)

def evaluate_classifier(clf, X_val, y_val, X_test, y_test, model_name="Model"):
    """
    Validation-calibrated thresholding as requested:
    Threshold is tuned strictly on the validation set, then applied to the untouched real test set.
    """
    # 1. Validation predictions & threshold tuning
    val_probs = clf.predict_proba(X_val)[:, 1]
    best_thresh, best_val_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.90, 86):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = t
            
    # 2. Test evaluation on untouched test set
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_thresh).astype(int)
    
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    
    print(f"[{model_name}] Val-Calibrated T={best_thresh:.2f} (Val F1: {best_val_f1:.4f}) -> Test F1: {f1:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_thresh": best_thresh,
        "val_f1": best_val_f1
    }

def run_rigorous_validation():
    print("=" * 80)
    print("🛡️ GraphGuard Scientific Validation Protocol")
    print("Experiment: Multi-Control Independent Baseline (Random Forest)")
    print("=" * 80)
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'graphguard', 'data', 'processed')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading preprocessed tensors...")
    X_t = torch.load(os.path.join(data_dir, 'node_features.pt'))
    y_t = torch.load(os.path.join(data_dir, 'labels.pt'))
    timesteps_t = torch.load(os.path.join(data_dir, 'timestep.pt'))
    edge_index_t = torch.load(os.path.join(data_dir, 'edge_index.pt'))
    train_mask = torch.load(os.path.join(data_dir, 'train_mask.pt'))
    val_mask = torch.load(os.path.join(data_dir, 'val_mask.pt'))
    test_mask = torch.load(os.path.join(data_dir, 'test_mask.pt'))
    
    # Strict temporal isolation: Only evaluate on known labels (exclude unknown -1)
    train_idx = torch.nonzero(train_mask & (y_t != -1)).squeeze()
    val_idx = torch.nonzero(val_mask & (y_t != -1)).squeeze()
    test_idx = torch.nonzero(test_mask & (y_t != -1)).squeeze()
    
    X_train_real = X_t[train_idx].numpy()
    y_train_real = y_t[train_idx].numpy()
    
    X_val = X_t[val_idx].numpy()
    y_val = y_t[val_idx].numpy()
    
    X_test = X_t[test_idx].numpy()
    y_test = y_t[test_idx].numpy()
    
    print(f"\nData Splits (Known Ground-Truth Labels Only):")
    print(f"  Train Set (Timesteps <= 34) : {len(y_train_real)} nodes (Licit: {(y_train_real==0).sum()}, Illicit: {(y_train_real==1).sum()})")
    print(f"  Val Set   (Timesteps 35-39): {len(y_val)} nodes (Licit: {(y_val==0).sum()}, Illicit: {(y_val==1).sum()})")
    print(f"  Test Set  (Timesteps 40-49): {len(y_test)} nodes (Licit: {(y_test==0).sum()}, Illicit: {(y_test==1).sum()})")
    
    # Determine augmentation budget (e.g. 30% augmentation of real illicit count)
    real_illicit_idx = np.where(y_train_real == 1)[0]
    X_illicit_train = X_train_real[real_illicit_idx]
    num_to_augment = int(len(X_illicit_train) * 0.30)
    print(f"\nMinority Augmentation Budget: +{num_to_augment} synthetic illicit nodes (30% ratio)")
    
    rf_params = {
        "n_estimators": 100,
        "max_depth": 18,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # -------------------------------------------------------------------------
    # Experiment A: Real Training Data Only (Baseline)
    # -------------------------------------------------------------------------
    print("\n--- 1. Training Experiment A: Real Data Only ---")
    clf_a = RandomForestClassifier(**rf_params)
    clf_a.fit(X_train_real, y_train_real)
    res_a = evaluate_classifier(clf_a, X_val, y_val, X_test, y_test, model_name="A: Real Only")
    
    # -------------------------------------------------------------------------
    # Experiment B: GraphGuard (TempoStruct-GAN Generated Nodes)
    # -------------------------------------------------------------------------
    print("\n--- 2. Training Experiment B: Real + GraphGuard (TempoStruct-GAN) ---")
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'graphguard', 'models')
    gen_path = os.path.join(models_dir, 'graphguard_generator.pt')
    
    generator = TemporalGenerator(feat_dim=165).to(device)
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    generator.eval()
    
    # Sample from training era illicit seeds on CPU, then send to device
    real_illicit_tensor_idx = train_idx[y_t[train_idx] == 1]
    torch.manual_seed(42)
    sample_idx = torch.randint(0, len(real_illicit_tensor_idx), (num_to_augment,))
    seed_node_idx = real_illicit_tensor_idx[sample_idx]
    
    seed_feats = X_t[seed_node_idx].to(device)
    seed_timesteps = timesteps_t[seed_node_idx].to(device)
    
    # Neighborhood aggregation strictly on training graph
    neigh_feats = aggregate_neighbourhood(X_t.to(device), seed_node_idx.to(device), edge_index_t.to(device), k_hops=1)
    with torch.no_grad():
        gg_synthetic_feats = generator.sample(seed_feats, neigh_feats, seed_timesteps, n_samples=1).cpu().numpy()
        
    X_train_b = np.vstack([X_train_real, gg_synthetic_feats])
    y_train_b = np.concatenate([y_train_real, np.ones(num_to_augment, dtype=np.int64)])
    
    clf_b = RandomForestClassifier(**rf_params)
    clf_b.fit(X_train_b, y_train_b)
    res_b = evaluate_classifier(clf_b, X_val, y_val, X_test, y_test, model_name="B: GraphGuard")
    
    # -------------------------------------------------------------------------
    # Experiment C: Random Oversampling Control
    # -------------------------------------------------------------------------
    print("\n--- 3. Training Experiment C: Random Oversampling Control ---")
    np.random.seed(42)
    ros_idx = np.random.choice(real_illicit_idx, size=num_to_augment, replace=True)
    X_ros = X_train_real[ros_idx]
    
    X_train_c = np.vstack([X_train_real, X_ros])
    y_train_c = np.concatenate([y_train_real, np.ones(num_to_augment, dtype=np.int64)])
    
    clf_c = RandomForestClassifier(**rf_params)
    clf_c.fit(X_train_c, y_train_c)
    res_c = evaluate_classifier(clf_c, X_val, y_val, X_test, y_test, model_name="C: Random Oversampling")
    
    # -------------------------------------------------------------------------
    # Experiment D: SMOTE Control
    # -------------------------------------------------------------------------
    print("\n--- 4. Training Experiment D: SMOTE Feature Interpolation Control ---")
    X_smote = smote_oversample(X_illicit_train, num_samples=num_to_augment, k_neighbors=5, random_state=42)
    
    X_train_d = np.vstack([X_train_real, X_smote])
    y_train_d = np.concatenate([y_train_real, np.ones(num_to_augment, dtype=np.int64)])
    
    clf_d = RandomForestClassifier(**rf_params)
    clf_d.fit(X_train_d, y_train_d)
    res_d = evaluate_classifier(clf_d, X_val, y_val, X_test, y_test, model_name="D: SMOTE")
    
    # -------------------------------------------------------------------------
    # Generate Final Results Table (Exact format requested)
    # -------------------------------------------------------------------------
    metrics_list = [
        ("Illicit precision", "precision"),
        ("Illicit recall", "recall"),
        ("Illicit F1", "f1"),
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc")
    ]
    
    print("\n" + "=" * 80)
    print("🏆 FINAL SCIENTIFIC VALIDATION TABLE")
    print("=" * 80)
    print(f"{'Metric':<20} | {'A: Real only':<14} | {'B: GraphGuard':<14} | {'C: Random OS':<14} | {'D: SMOTE':<14}")
    print("-" * 80)
    for label, key in metrics_list:
        print(f"{label:<20} | {res_a[key]:<14.4f} | {res_b[key]:<14.4f} | {res_c[key]:<14.4f} | {res_d[key]:<14.4f}")
    print("=" * 80)
    
    print("\nMarkdown Table for Project Report / Viva Defense:\n")
    print("| Metric | A: Real only | B: GraphGuard | C: Random oversampling | D: SMOTE |")
    print("|---|---|---|---|---|")
    for label, key in metrics_list:
        print(f"| {label} | {res_a[key]:.4f} | {res_b[key]:.4f} | {res_c[key]:.4f} | {res_d[key]:.4f} |")
        
    print("\n📈 Delta Analysis (GraphGuard vs Baselines):")
    delta_pr = (res_b['pr_auc'] - res_a['pr_auc']) / res_a['pr_auc'] * 100
    delta_f1 = (res_b['f1'] - res_a['f1']) / res_a['f1'] * 100
    print(f"  • Relative PR-AUC Improvement over Real Only: {delta_pr:+.2f}%")
    print(f"  • Relative Illicit-F1 Improvement over Real Only: {delta_f1:+.2f}%")

if __name__ == "__main__":
    run_rigorous_validation()
