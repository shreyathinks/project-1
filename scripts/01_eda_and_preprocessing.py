import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # Separate features
    features = df.drop(columns=['txId', 'time_step', 'class', 'class_label'])
    
    # Instead of per-timestep z-score, we apply a global robust scaler
    # This prevents NaN issues when a feature has 0 variance in a specific timestep
    from sklearn.preprocessing import RobustScaler
    
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Assign back
    df_scaled = df.copy()
    df_scaled[features.columns] = scaled_features
    
    print(f"Features scaled successfully. Any NaNs? {np.isnan(scaled_features).any()}")
    
    # Save processed data
    # We will save node features and edges for model consumption
    print("Saving preprocessed data...")
    df_scaled.to_csv(os.path.join(output_dir, 'processed_nodes.csv'), index=False)
    df_edges.to_csv(os.path.join(output_dir, 'processed_edges.csv'), index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    dataset_dir = "e:/GraphGuard/project-1/dataset/elliptic_bitcoin_dataset"
    output_dir = "e:/GraphGuard/project-1/dataset/processed"
    eda_output_dir = "e:/GraphGuard/project-1/dataset/eda"
    
    df, df_edges = load_data(dataset_dir)
    perform_eda(df, eda_output_dir)
    robust_preprocessing(df, df_edges, output_dir)
