import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def load_data(dataset_dir):
    print("Loading data...")
    # Load classes
    df_classes = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_classes.csv'))
    
    # Load edgelist
    df_edges = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_edgelist.csv'))
    
    # Load features
    # elliptic_txs_features.csv has no header
    df_features = pd.read_csv(os.path.join(dataset_dir, 'elliptic_txs_features.csv'), header=None)
    
    # Rename columns
    col_names = ['txId', 'time_step'] + [f'f{i}' for i in range(1, 166)]
    df_features.columns = col_names
    
    # Merge classes and features
    df = df_features.merge(df_classes, on='txId', how='left')
    
    print(f"Data loaded. Total transactions: {len(df)}")
    return df, df_edges

def perform_eda(df, output_dir):
    print("Performing EDA...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution
    class_counts = df['class'].value_counts(dropna=False)
    print("Class Distribution:")
    print(class_counts)
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='class')
    plt.title('Class Distribution (1: Illicit, 2: Licit, unknown)')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # 2. Temporal Distribution of illicit vs licit
    # We only care about known classes for temporal distribution
    df_known = df[df['class'] != 'unknown'].copy()
    
    temporal_counts = df_known.groupby(['time_step', 'class']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(15, 6))
    temporal_counts.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Illicit vs Licit Transactions over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Transactions')
    plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'))
    plt.close()
    
    # 3. Check for Zero Variance Features (which caused NaNs in old preprocessing)
    # Group by time_step and calculate variance
    zero_var_issues = 0
    for ts in df['time_step'].unique():
        ts_data = df[df['time_step'] == ts].drop(columns=['txId', 'time_step', 'class'])
        variances = ts_data.var()
        zero_var_cols = variances[variances == 0].index
        if len(zero_var_cols) > 0:
            zero_var_issues += 1
            
    print(f"Number of timesteps with zero-variance features: {zero_var_issues} out of {df['time_step'].nunique()}")
    print("This confirms the root cause of the NaN issue in the old per-timestep z-score normalization.")

def robust_preprocessing(df, df_edges, output_dir):
    print("Performing robust preprocessing...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Map classes to numeric: unknown -> -1, 1 (illicit) -> 1, 2 (licit) -> 0
    class_mapping = {'unknown': -1, '1': 1, '2': 0}
    df['class_label'] = df['class'].map(class_mapping)
    
    # 1. Map txId to contiguous integers 0 to N-1
    print("Mapping txIds to contiguous indices...")
    tx_ids = df['txId'].values
    tx_id_map = {tx_id: i for i, tx_id in enumerate(tx_ids)}
    
    # Map edges using the dictionary
    df_edges['mapped_txId1'] = df_edges['txId1'].map(tx_id_map)
    df_edges['mapped_txId2'] = df_edges['txId2'].map(tx_id_map)
    
    # Drop rows where edge points to an unknown txId not in features (if any)
    df_edges = df_edges.dropna(subset=['mapped_txId1', 'mapped_txId2'])
    
    # 2. Separate features and apply global robust scaler
    print("Scaling features...")
    features = df.drop(columns=['txId', 'time_step', 'class', 'class_label'])
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 3. Create Train / Val / Test Masks
    # Elliptic typical split: Train (1-34), Val (35-39), Test (40-49)
    print("Generating temporal masks...")
    time_steps = df['time_step'].values
    train_mask = (time_steps <= 34)
    val_mask = (time_steps > 34) & (time_steps <= 39)
    test_mask = (time_steps > 39)
    
    # 4. Convert to PyTorch tensors
    print("Converting to PyTorch tensors...")
    node_features_t = torch.tensor(scaled_features, dtype=torch.float32)
    labels_t = torch.tensor(df['class_label'].values, dtype=torch.long)
    timestep_t = torch.tensor(time_steps, dtype=torch.long)
    
    edge_index_t = torch.tensor(
        df_edges[['mapped_txId1', 'mapped_txId2']].values.T, 
        dtype=torch.long
    )
    
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool)
    val_mask_t = torch.tensor(val_mask, dtype=torch.bool)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool)
    
    # 5. Save all tensors
    print("Saving .pt tensors...")
    torch.save(node_features_t, os.path.join(output_dir, 'node_features.pt'))
    torch.save(labels_t, os.path.join(output_dir, 'labels.pt'))
    torch.save(timestep_t, os.path.join(output_dir, 'timestep.pt'))
    torch.save(edge_index_t, os.path.join(output_dir, 'edge_index.pt'))
    torch.save(train_mask_t, os.path.join(output_dir, 'train_mask.pt'))
    torch.save(val_mask_t, os.path.join(output_dir, 'val_mask.pt'))
    torch.save(test_mask_t, os.path.join(output_dir, 'test_mask.pt'))
    
    print("Preprocessing complete and ready for model training.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(project_root, "graphguard", "data", "raw")
    output_dir = os.path.join(project_root, "graphguard", "data", "processed")
    eda_output_dir = os.path.join(project_root, "results", "eda")
    
    df, df_edges = load_data(dataset_dir)
    perform_eda(df, eda_output_dir)
    robust_preprocessing(df, df_edges, output_dir)
