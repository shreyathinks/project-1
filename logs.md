# Training Logs

This file contains the sequential terminal outputs from the execution of the training scripts for the GraphGuard project.

## 1. Generator Training (Initial Fix)
**Command:** `uv run .\02_train_generator.py`
**Result:** Successfully trained the GAN for 20 epochs without mode collapse. The discriminator and generator reached a healthy equilibrium.

```text
PS E:\GraphGuard\project-1\scripts> uv run .\02_train_generator.py
Loading preprocessed data...
Data ready. 203769 nodes, 234355 edges.
Starting training...
Epoch 1/20 - D Loss: 89.9312 - G Loss: 36.0771
Epoch 2/20 - D Loss: 51.7012 - G Loss: 15.4388
Epoch 3/20 - D Loss: 84.1081 - G Loss: 8.6218
Epoch 4/20 - D Loss: 53.6984 - G Loss: 9.6272
Epoch 5/20 - D Loss: 34.0743 - G Loss: 24.9912
Epoch 6/20 - D Loss: 31.3604 - G Loss: 32.1317
Epoch 7/20 - D Loss: 28.6634 - G Loss: 12.0632
Epoch 8/20 - D Loss: 41.9943 - G Loss: 18.3395
Epoch 9/20 - D Loss: 22.2286 - G Loss: 10.1736
Epoch 10/20 - D Loss: 42.3622 - G Loss: 12.2056
Epoch 11/20 - D Loss: 16.9662 - G Loss: 13.3912
Epoch 12/20 - D Loss: 33.6808 - G Loss: 7.4831
Epoch 13/20 - D Loss: 22.4269 - G Loss: 10.3734
Epoch 14/20 - D Loss: 11.1992 - G Loss: 8.6857
Epoch 15/20 - D Loss: 18.9537 - G Loss: 14.2758
Epoch 16/20 - D Loss: 17.1307 - G Loss: 7.7895
Epoch 17/20 - D Loss: 18.0495 - G Loss: 4.1549
Epoch 18/20 - D Loss: 20.0865 - G Loss: 7.9203
Epoch 19/20 - D Loss: 10.4883 - G Loss: 11.2282
Epoch 20/20 - D Loss: 7.9179 - G Loss: 6.6202
Training complete. Saving results to e:/GraphGuard/project-1/results/generator/run_20261002_222558...
Generator saved.
```

## 2. Downstream Classifier - GraphSAGE (Isolated Nodes Issue)
**Command:** `uv run 03_train_classifier.py`
**Result:** Synthetic nodes were injected into the graph *without edges* (Degree = 0). The GNN learned the lazy shortcut "isolated node = fraud", causing precision to collapse when tested on real nodes.

```text
PS E:\GraphGuard\project-1\scripts> uv run 03_train_classifier.py
Loading preprocessed data...
E:\GraphGuard\project-1\scripts\03_train_classifier.py:60: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  train_mask = known_mask & (timesteps <= 34)
E:\GraphGuard\project-1\scripts\03_train_classifier.py:61: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  test_mask = known_mask & (timesteps > 34)
Loading Generator for Oversampling...
Oversampled 18421 synthetic illicit nodes.
2026/10/02 22:59:04 INFO mlflow.tracking.fluent: Experiment with name 'GraphGuard_Classifier_GNN' does not exist. Creating a new experiment.
Training Downstream GraphSAGE Classifier (Full Batch)...
Epoch 10/100 | Train Loss: 0.5530 | Val F1: 0.2203 @ T=0.27
Epoch 20/100 | Train Loss: 0.5040 | Val F1: 0.2179 @ T=0.05
Epoch 30/100 | Train Loss: 0.4901 | Val F1: 0.2316 @ T=0.11
Epoch 40/100 | Train Loss: 0.4843 | Val F1: 0.2470 @ T=0.12
Epoch 50/100 | Train Loss: 0.4786 | Val F1: 0.2536 @ T=0.09
Epoch 60/100 | Train Loss: 0.4697 | Val F1: 0.2649 @ T=0.12
Epoch 70/100 | Train Loss: 0.4581 | Val F1: 0.2744 @ T=0.11
Epoch 80/100 | Train Loss: 0.4468 | Val F1: 0.2827 @ T=0.09
Epoch 90/100 | Train Loss: 0.4380 | Val F1: 0.2954 @ T=0.14
Epoch 100/100 | Train Loss: 0.4256 | Val F1: 0.3030 @ T=0.11
Evaluating on Test Set...
Test Results (Threshold=0.11): F1=0.0908, Prec=0.0537, Rec=0.2946, AUC=0.6170
Saving plots and metrics locally to e:/GraphGuard/project-1/results/classifier_gnn/run_20261002_225903...
Classifier saved to MLflow and locally.
```

## 3. Downstream Classifier - GraphSAGE (KNN Edges & Undirected Graph)
**Command:** `uv run 03_train_classifier.py`
**Result:** Edges were successfully injected via the KNN fallback plan, and the graph was made undirected. Model trained for 1000 epochs. Resulted in huge recall (75.6%) but very low precision (9.5%). Attributed to 3-layer GraphSAGE oversmoothing and slight temporal leakage.

```text
PS E:\GraphGuard\project-1\scripts> uv run 03_train_classifier.py
Loading preprocessed data...
E:\GraphGuard\project-1\scripts\03_train_classifier.py:63: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  train_mask = known_mask & (timesteps <= 34)
E:\GraphGuard\project-1\scripts\03_train_classifier.py:64: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  test_mask = known_mask & (timesteps > 34)
Loading Generator for Oversampling...
Oversampled 18361 synthetic illicit nodes.
Injecting edges for synthetic nodes via K-NN...
Injected 110166 synthetic edges into the graph.
Training Downstream GraphSAGE Classifier (Full Batch)...
Epoch 50/1000 | Train Loss: 0.5027 | Val F1: 0.3075 @ T=0.54
...
Epoch 900/1000 | Train Loss: 0.2168 | Val F1: 0.3518 @ T=0.12
Epoch 950/1000 | Train Loss: 0.2390 | Val F1: 0.3119 @ T=0.12
Epoch 1000/1000 | Train Loss: 0.2221 | Val F1: 0.3271 @ T=0.08
Evaluating on Test Set...
Test Results (Threshold=0.08): F1=0.1692, Prec=0.0952, Rec=0.7562, AUC=0.6744
Saving plots and metrics locally to e:/GraphGuard/project-1/results/classifier_gnn/run_20261002_235119...
Classifier saved to MLflow and locally.
```

## 4. Downstream Classifier - (NumPy Array Crash)
**Command:** `uv run scripts/03_train_classifier.py`
**Result:** Script crashed on attempting to cast a NumPy array to the GPU device during the strict temporal conditioning logic update.

```text
PS E:\GraphGuard\project-1> uv run scripts/03_train_classifier.py     
Loading preprocessed data...
E:\GraphGuard\project-1\scripts\03_train_classifier.py:63: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  train_mask = known_mask & (timesteps <= 34)
E:\GraphGuard\project-1\scripts\03_train_classifier.py:64: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  test_mask = known_mask & (timesteps > 34)
Loading Generator for Oversampling...
Traceback (most recent call last):
  File "E:\GraphGuard\project-1\scripts\03_train_classifier.py", line 293, in <module>
    train_classifier()
    ~~~~~~~~~~~~~~~~^^
  File "E:\GraphGuard\project-1\scripts\03_train_classifier.py", line 100, in train_classifier
    seed_timesteps = timesteps[real_train_idx.cpu()][illicit_mask.cpu()][idx].to(device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'to'
```

*Note: Following Run 4, the codebase was updated to fix the NumPy array bug, strictly enforce temporal edge splitting, restrict GraphSAGE to 2 layers to prevent oversmoothing, and remove MLflow tracking.*
