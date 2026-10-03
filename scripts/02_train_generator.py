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

# Add graphguard to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graphguard.models.generator import TemporalGenerator, aggregate_neighbourhood
from graphguard.models.discriminator import GraphGuardDiscriminator

def compute_structural_stats(edge_index, center_nodes, num_nodes):
    device = edge_index.device
    src, dst = edge_index
    
    # Degree of center nodes
    degrees = torch.zeros(num_nodes, device=device)
    degrees.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
    degrees.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    
    center_degrees = degrees[center_nodes]
    
    stats = torch.zeros((len(center_nodes), 4), device=device)
    stats[:, 0] = center_degrees / 100.0  # Normalized degree
    
    stats[:, 1] = torch.rand(len(center_nodes), device=device) * 0.5
    stats[:, 2] = torch.rand(len(center_nodes), device=device) * 0.5
    stats[:, 3] = torch.rand(len(center_nodes), device=device) * 0.5
    
    return stats

def train_gan():
    print("Loading preprocessed data...")
    data_dir = "e:/GraphGuard/project-1/dataset/processed"
    df_nodes = pd.read_csv(os.path.join(data_dir, 'processed_nodes.csv'))
    df_edges = pd.read_csv(os.path.join(data_dir, 'processed_edges.csv'))
    
    df_nodes = df_nodes.sort_values('time_step')
    
    features = torch.tensor(df_nodes[[f'f{i}' for i in range(1, 166)]].values, dtype=torch.float32)
    labels = torch.tensor(df_nodes['class_label'].values, dtype=torch.long)
    timesteps = torch.tensor(df_nodes['time_step'].values, dtype=torch.long)
    
    tx_to_idx = {tx: i for i, tx in enumerate(df_nodes['txId'])}
    
    df_edges_filtered = df_edges[df_edges['txId1'].isin(tx_to_idx) & df_edges['txId2'].isin(tx_to_idx)]
    
    src = torch.tensor([tx_to_idx[tx] for tx in df_edges_filtered['txId1']], dtype=torch.long)
    dst = torch.tensor([tx_to_idx[tx] for tx in df_edges_filtered['txId2']], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    
    print(f"Data ready. {len(features)} nodes, {edge_index.size(1)} edges.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    labels = labels.to(device)
    timesteps = timesteps.to(device)
    edge_index = edge_index.to(device)
    
    generator = TemporalGenerator(feat_dim=165).to(device)
    discriminator = GraphGuardDiscriminator(feat_dim=165).to(device)
    
    opt_G = optim.Adam(generator.parameters(), lr=1e-4)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    criterion = nn.BCEWithLogitsLoss()
    
    illicit_idx = torch.nonzero(labels == 1).squeeze()
    
    epochs = 20
    batch_size = 128
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"e:/GraphGuard/project-1/results/generator/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    d_loss_history = []
    g_loss_history = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        perm = torch.randperm(illicit_idx.size(0))
        illicit_idx = illicit_idx[perm]
        
        total_d_loss = 0
        total_g_loss = 0
        
        for i in range(0, len(illicit_idx), batch_size):
            batch_nodes = illicit_idx[i:i+batch_size]
            B = len(batch_nodes)
            
            if B == 0:
                continue
                
            real_feats = features[batch_nodes]
            batch_timesteps = timesteps[batch_nodes]
            
            # 1. Train Discriminator
            opt_D.zero_grad()
            
            d_real_node = discriminator.discriminate_node(real_feats)
            struct_stats_real = compute_structural_stats(edge_index, batch_nodes, features.size(0))
            d_real_subgraph = discriminator.discriminate_subgraph(features, edge_index, struct_stats_real, batch_nodes)
            
            loss_d_real = criterion(d_real_node, torch.ones_like(d_real_node)) + \
                          criterion(d_real_subgraph, torch.ones_like(d_real_subgraph))
                          
            # Fake
            neigh_feats = aggregate_neighbourhood(features, batch_nodes, edge_index, k_hops=1)
            fake_feats = generator.sample(real_feats, neigh_feats, batch_timesteps, n_samples=1)
            
            d_fake_node = discriminator.discriminate_node(fake_feats.detach())
            
            features_with_fake = features.clone()
            features_with_fake[batch_nodes] = fake_feats.detach()
            d_fake_subgraph = discriminator.discriminate_subgraph(features_with_fake, edge_index, struct_stats_real, batch_nodes)
            
            loss_d_fake = criterion(d_fake_node, torch.zeros_like(d_fake_node)) + \
                          criterion(d_fake_subgraph, torch.zeros_like(d_fake_subgraph))
                          
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_D.step()
            
            # 2. Train Generator
            opt_G.zero_grad()
            
            d_fake_node_g = discriminator.discriminate_node(fake_feats)
            features_with_fake_g = features.clone()
            features_with_fake_g[batch_nodes] = fake_feats
            d_fake_subgraph_g = discriminator.discriminate_subgraph(features_with_fake_g, edge_index, struct_stats_real, batch_nodes)
            
            loss_g = criterion(d_fake_node_g, torch.ones_like(d_fake_node_g)) + \
                     criterion(d_fake_subgraph_g, torch.ones_like(d_fake_subgraph_g))
                     
            loss_g.backward()
            opt_G.step()
            
            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()
            
        avg_d_loss = total_d_loss / len(illicit_idx)
        avg_g_loss = total_g_loss / len(illicit_idx)
        d_loss_history.append(avg_d_loss)
        g_loss_history.append(avg_g_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - D Loss: {avg_d_loss:.4f} - G Loss: {avg_g_loss:.4f}")
        
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
        
    os.makedirs("e:/GraphGuard/project-1/graphguard/models/trained", exist_ok=True)
    torch.save(generator.state_dict(), "e:/GraphGuard/project-1/graphguard/models/trained/generator.pt")
    torch.save(generator.state_dict(), os.path.join(run_dir, "generator.pt"))
    print("Generator saved locally.")

if __name__ == "__main__":
    train_gan()
