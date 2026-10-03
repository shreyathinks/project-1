# GraphGuard: TempoStruct-GAN for Graph Anomaly Detection

**GraphGuard** is a framework for mitigating extreme class imbalance in temporal graph datasets (specifically the Elliptic Bitcoin dataset). It introduces **TempoStruct-GAN**, a novel dual-head Generative Adversarial Network that synthesizes structurally realistic and temporally valid illicit nodes to improve Graph Neural Network (GNN) performance in fraud detection.

## 🚀 Key Features

* **TempoStruct-GAN**: A custom GAN architecture designed specifically for dynamic graphs.
* **Dual-Head Discriminator**: Simultaneously evaluates node-level feature realism (Head A) and localized subgraph structural realism (Head B).
* **Gumbel-Softmax Edge Predictor**: Wires synthetic nodes into the existing graph topology dynamically, allowing gradients to flow backwards through the edge-wiring process.
* **Strict Temporal Validity**: Enforces a temporal penalty and causal masking so the model cannot wire edges "backwards in time", maintaining the chronological integrity of the graph.
* **Sinusoidal Positional Encodings**: Uses transformer-style time encodings to help the generator smoothly interpolate across time steps.

## 📁 Repository Structure

```
.
├── 01_preprocessing.ipynb       # Data parsing, temporal normalisation, EDA
├── 02_training.ipynb            # Baseline GraphSAGE, TempoStruct-GAN training, Augmented Classifier
├── 03_validation.ipynb          # Final evaluations, baselines, and ROC/PR plots
├── graphguard/                  # Core modules
│   ├── data/                    # Raw & processed graph datasets
│   ├── models/                  # PyTorch neural network architectures
│   ├── training/                # Custom adversarial & graph loss functions
│   ├── evaluation/              # Custom evaluation metrics
│   └── results/                 # Auto-generated plots, CSVs, and model checkponts
└── memory.md                    # Detailed developer log & configuration docs
```

## 📊 Experimental Results

By augmenting the training graph with 30% synthetic illicit nodes, GraphGuard successfully increased illicit recall compared to a standard GraphSAGE baseline.

| Method | Illicit-F1 | Recall | Precision | AUC-ROC | AUC-PR | Macro-F1 |
|---|---|---|---|---|---|---|
| Plain GraphSAGE | **0.3970** | 0.7350 | **0.2720** | **0.8972** | **0.6471** | **0.6573** |
| GraphGuard (10% aug) | 0.3627 | 0.7442 | 0.2398 | 0.8904 | 0.5849 | 0.6324 |
| GraphGuard (20% aug) | 0.3683 | 0.7378 | 0.2454 | 0.8857 | 0.4807 | 0.6369 |
| GraphGuard (30% aug) | 0.3652 | **0.7461** | 0.2418 | 0.8871 | 0.4980 | 0.6340 |

*Note: While GraphGuard successfully increased illicit recall (finding more hidden fraud), the injection of synthetic features caused a drop in precision (more false positives). In real-world AML (Anti-Money Laundering) compliance, higher recall is heavily prioritized.*

## ⚙️ Quick Start Setup

1. **Install Dependencies:**
   ```bash
   pip install torch pandas numpy scikit-learn networkx matplotlib seaborn
   ```
2. **Download Dataset:**
   Download the [Elliptic Data Set from Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). Extract the 3 CSV files into `graphguard/data/raw/`.
3. **Run Pipeline:**
   Execute the three Jupyter Notebooks sequentially.

---
*Developed as a capstone machine learning project focused on advanced GNN anomaly detection.*
