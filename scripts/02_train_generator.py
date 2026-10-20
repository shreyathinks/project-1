import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import networkx as nx

def get_k_hop_subgraph(node_idx, num_hops, edge_index):
    device = edge_index.device
    num_nodes = edge_index.max().item() + 1
    
    subset = node_idx.clone()
    for _ in range(num_hops):
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_mask[subset] = True
        
        edge_mask = node_mask[edge_index[0]] | node_mask[edge_index[1]]
        neighbors = torch.cat([edge_index[0, edge_mask], edge_index[1, edge_mask]])
        subset = torch.unique(torch.cat([subset, neighbors]))
        
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_mask[subset] = True
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    sub_edge_index = edge_index[:, edge_mask]
    
    # relabel nodes
    n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
    n_idx[subset] = torch.arange(subset.size(0), device=device)
    sub_edge_index = n_idx[sub_edge_index]
    mapping = n_idx[node_idx]
    
    return subset, sub_edge_index, mapping, edge_mask

# Add graphguard to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphguard.models.generator import TemporalGenerator, aggregate_neighbourhood
from graphguard.models.discriminator import GraphGuardDiscriminator
from graphguard.models.edge_predictor import EdgePredictor
from graphguard.training.losses import (
    temporal_violation_penalty,
    generator_loss,
    feature_matching_loss,
    local_aggregated_consistency_loss,
)

def compute_real_stats(subgraph_edge_index, center_nodes_rel, num_nodes, labels_sub=None):
    device = subgraph_edge_index.device
    edges = subgraph_edge_index.t().cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    
    centers = center_nodes_rel.cpu().numpy().tolist()
    # Clustering coefficient for all center nodes
    clustering = nx.clustering(G, nodes=centers)
    
    stats = []
    for c in centers:
        deg = G.degree(c) / 100.0  # Normalized degree
        clust = clustering[c]
        
        # ego density
        ego = nx.ego_graph(G, c, radius=1)
        dens = nx.density(ego)
        
        # homophily in ego network
        homo = 0.5
        if labels_sub is not None:
            ego_edges = list(ego.edges())
            if len(ego_edges) > 0:
                match = 0
                valid = 0
                for u, v in ego_edges:
                    l_u = labels_sub[u].item()
                    l_v = labels_sub[v].item()
                    if l_u != -1 and l_v != -1:
                        if l_u == l_v:
                            match += 1
                        valid += 1
                if valid > 0:
                    homo = match / valid
                    
        stats.append([deg, clust, homo, dens])
        
    return torch.tensor(stats, dtype=torch.float32, device=device)

def train_gan(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Train GraphGuard Generator and EdgePredictor")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--top_k", type=int, default=2)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--lambda_fm", type=float, default=0.1)
        parser.add_argument("--lambda_temp", type=float, default=1.0)
        parser.add_argument("--lambda_cons", type=float, default=0.1)
        args, _ = parser.parse_known_args()

    print("Loading preprocessed .pt data...")
    # Fix Issue 1: Correct paths to local processed data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'graphguard', 'data', 'processed')
    
    features = torch.load(os.path.join(data_dir, 'node_features.pt'))
    labels = torch.load(os.path.join(data_dir, 'labels.pt'))
    timesteps = torch.load(os.path.join(data_dir, 'timestep.pt'))
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'))
    train_mask = torch.load(os.path.join(data_dir, 'train_mask.pt'))
    
    print(f"Data ready. {features.size(0)} nodes, {edge_index.size(1)} edges.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    labels = labels.to(device)
    timesteps = timesteps.to(device)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    
    # Models
    generator = TemporalGenerator(feat_dim=165).to(device)
    discriminator = GraphGuardDiscriminator(feat_dim=165).to(device)
    edge_predictor = EdgePredictor(feat_dim=165, k=args.top_k).to(device) # Calibrated top-K degree (K=2)
    
    opt_G = optim.Adam(list(generator.parameters()) + list(edge_predictor.parameters()), lr=args.lr)
    opt_D = optim.Adam(discriminator.parameters(), lr=args.lr)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Only train on illicit nodes in the train set
    illicit_train_mask = (labels == 1) & train_mask
    illicit_idx = torch.nonzero(illicit_train_mask).squeeze()
    
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Determine the results directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "results", "generator", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    d_loss_history = []
    g_loss_history = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        edge_predictor.train()
        
        perm = torch.randperm(illicit_idx.size(0))
        illicit_idx = illicit_idx[perm]
        
        total_d_loss = 0
        total_g_loss = 0
        
        for i in range(0, len(illicit_idx), batch_size):
            batch_nodes = illicit_idx[i:i+batch_size]
            B = len(batch_nodes)
            
            if B == 0: continue
                
            real_feats = features[batch_nodes]
            batch_timesteps = timesteps[batch_nodes]
            
            # --- Fix Issue 5: Extract K-hop subgraph instead of passing full graph ---
            subset_real, edge_index_real, mapping_real, _ = get_k_hop_subgraph(
                batch_nodes, 1, edge_index
            )
            
            sub_feats_real = features[subset_real]
            sub_labels_real = labels[subset_real]
            
            # --- Fix Issue 2: Compute real structural stats ---
            struct_stats_real = compute_real_stats(edge_index_real, mapping_real, subset_real.size(0), sub_labels_real)
            
            # 1. Train Discriminator
            opt_D.zero_grad()
            
            d_real_node = discriminator.discriminate_node(real_feats)
            d_real_subgraph = discriminator.discriminate_subgraph(sub_feats_real, edge_index_real, struct_stats_real, mapping_real)
            
            loss_d_real = criterion(d_real_node, torch.ones_like(d_real_node) * 0.9) + \
                          criterion(d_real_subgraph, torch.ones_like(d_real_subgraph) * 0.9)
                          
            # Fake generation
            neigh_feats = aggregate_neighbourhood(features, batch_nodes, edge_index, k_hops=1)
            fake_feats = generator.sample(real_feats, neigh_feats, batch_timesteps, n_samples=1)
            
            # Fix 2: Restrict candidate pool to illicit (y == 1) or unlabeled (y == -1) in train_mask
            # Exclude confirmed licit nodes (y == 0) from being wired as direct 1-hop neighbors
            cand_mask = train_mask & (labels != 0)
            cand_idx = torch.nonzero(cand_mask).squeeze()
            
            # --- Subsample candidates if needed to prevent OOM ---
            if cand_idx.size(0) > 2000:
                perm = torch.randperm(cand_idx.size(0))[:2000]
                cand_idx = cand_idx[perm]
                
            cand_feats = features[cand_idx]
            cand_ts = timesteps[cand_idx]
            
            # Tight local temporal window: max(1, t_synth - 2) <= t_cand <= t_synth
            min_ts = (batch_timesteps.unsqueeze(1) - 2).clamp(min=1)
            temporal_mask = (cand_ts.unsqueeze(0) <= batch_timesteps.unsqueeze(1)) & (cand_ts.unsqueeze(0) >= min_ts)
            
            # Safe per-row fallback if local temporal window has 0 candidates
            fallback_mask = (cand_ts.unsqueeze(0) <= batch_timesteps.unsqueeze(1))
            no_cands = (temporal_mask.sum(dim=-1) == 0)
            if no_cands.any():
                temporal_mask[no_cands] = fallback_mask[no_cands]
            
            # Fallback if entire candidate set is in future (e.g. at t=1 with small pool)
            still_no_cands = (temporal_mask.sum(dim=-1) == 0)
            if still_no_cands.any():
                temporal_mask[still_no_cands] = True
            
            soft_weights, hard_edges = edge_predictor.forward_gumbel(
                fake_feats, cand_feats, batch_timesteps, cand_ts, temporal_mask
            )
            
            d_fake_node = discriminator.discriminate_node(fake_feats.detach())
            
            # Build fake subgraph using hard edges (detached for Discriminator training)
            features_with_fake = sub_feats_real.clone()
            features_with_fake[mapping_real] = fake_feats.detach()
            
            d_fake_subgraph = discriminator.discriminate_subgraph(features_with_fake, edge_index_real, struct_stats_real, mapping_real)
            
            loss_d_fake = criterion(d_fake_node, torch.zeros_like(d_fake_node)) + \
                          criterion(d_fake_subgraph, torch.zeros_like(d_fake_subgraph))
                          
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_D.step()
            
            # 2. Train Generator & Edge Predictor
            opt_G.zero_grad()
            
            d_fake_node_g = discriminator.discriminate_node(fake_feats)
            features_with_fake_g = sub_feats_real.clone()
            features_with_fake_g[mapping_real] = fake_feats
            d_fake_subgraph_g = discriminator.discriminate_subgraph(features_with_fake_g, edge_index_real, struct_stats_real, mapping_real)
            
            loss_g_node = criterion(d_fake_node_g, torch.ones_like(d_fake_node_g))
            loss_g_struct = criterion(d_fake_subgraph_g, torch.ones_like(d_fake_subgraph_g))
            loss_fm = feature_matching_loss(real_feats, fake_feats)
            loss_temporal = temporal_violation_penalty(soft_weights, batch_timesteps, cand_ts)
            # Fix 4: Local-to-Aggregated Feature Consistency Loss
            loss_cons = local_aggregated_consistency_loss(fake_feats, cand_feats, soft_weights)
            
            loss_g = loss_g_node + loss_g_struct + (args.lambda_fm * loss_fm) + (args.lambda_temp * loss_temporal) + (args.lambda_cons * loss_cons)
                     
            loss_g.backward()
            opt_G.step()
            
            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()
            
        avg_d_loss = total_d_loss / len(illicit_idx)
        avg_g_loss = total_g_loss / len(illicit_idx)
        d_loss_history.append(avg_d_loss)
        g_loss_history.append(avg_g_loss)
        
        edge_predictor.anneal_temperature(epoch)
        print(f"Epoch {epoch+1}/{epochs} - D Loss: {avg_d_loss:.4f} - G Loss: {avg_g_loss:.4f}")
        
        # Save checkpoints every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(generator.state_dict(), os.path.join(run_dir, f"generator_epoch_{epoch+1}.pt"))
            torch.save(edge_predictor.state_dict(), os.path.join(run_dir, f"edge_predictor_epoch_{epoch+1}.pt"))
        
    print(f"Training complete. Saving results to {run_dir}...")
        
    # Save Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), d_loss_history, label='D Loss')
    plt.plot(range(1, epochs + 1), g_loss_history, label='G Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator vs Discriminator Loss')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
    plt.close()
    
    # Save Metrics
    metrics = {
        "final_d_loss": d_loss_history[-1],
        "final_g_loss": g_loss_history[-1],
        "epochs": epochs,
        "batch_size": batch_size
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    models_dir = os.path.join(project_root, "graphguard", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    torch.save(generator.state_dict(), os.path.join(models_dir, "graphguard_generator.pt"))
    torch.save(edge_predictor.state_dict(), os.path.join(models_dir, "graphguard_edge_predictor.pt"))
    torch.save(generator.state_dict(), os.path.join(run_dir, "graphguard_generator.pt"))
    print("Generator and Edge Predictor saved locally.")

if __name__ == "__main__":
    train_gan()
