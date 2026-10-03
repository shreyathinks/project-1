# GraphGuard / TempoStruct-GAN — End-to-End Implementation Plan

**Project:** Adversarial, Time-Aware, Subgraph-Realism-Checked Oversampling for Minority-Class Fraud Detection
**Dataset anchor:** Elliptic Bitcoin Dataset (+ optional Elliptic++ / IEEE-CIS)
**Core novelty being implemented:** Temporal conditioning + an adversarially-learned structural-realism discriminator, jointly — the exact combination your lit review shows nobody else has done.

This plan is organized so each phase produces something demoable at your next review, and maps cleanly onto the TRL ladder in your prep notes (TRL 3 → 4 → 5 → 6).

---

## Phase 0 — Environment & Repo Setup

**Stack:**
- Python 3.10+, PyTorch 2.x, **PyTorch Geometric (PyG)** — this is the right library since it has native `SAGEConv`, `GATConv`, mini-batch neighbor sampling, and utilities for edge prediction.
- `scikit-learn` (metrics, baselines), `pandas`/`numpy` (preprocessing), `networkx` (structural stats: clustering coefficient, degree distributions — needed for your discriminator's structural checks), `matplotlib`/`seaborn` (result plots), `wandb` or `tensorboard` (experiment tracking — you'll want this once you're running 6+ baseline comparisons).

**Repo structure (recommend this now, not later — reviewers like seeing organized code):**
```
graphguard/
├── data/
│   ├── raw/                  # untouched downloaded files
│   ├── processed/            # cleaned tensors, splits
│   └── preprocessing.py
├── models/
│   ├── baseline_graphsage.py
│   ├── generator.py           # temporal-conditioned generator
│   ├── edge_predictor.py
│   ├── discriminator.py       # subgraph/structural discriminator
│   └── classifier.py          # downstream GraphSAGE/GAT
├── training/
│   ├── train_baseline.py
│   ├── train_gan.py
│   ├── train_augmented.py
│   └── losses.py
├── evaluation/
│   ├── metrics.py
│   ├── baselines_reimpl/      # GraphSMOTE, GAT-COBO, GGA, LAGA, THG-OAFN
│   └── run_comparison.py
├── configs/
├── notebooks/                 # exploration only, not for final results
└── results/
```

---

## Phase 1 — Data Collection & Preprocessing

### 1.1 Acquire the data
- **Elliptic Bitcoin Dataset** (Kaggle, "Elliptic Data Set"): three CSVs —
  - `elliptic_txs_features.csv` — 203,769 nodes × 166 features (first 94 are "local" transaction features, remaining 72 are "aggregated" features from 1-hop neighbors — **note this in your report**, it matters for feature engineering later).
  - `elliptic_txs_classes.csv` — labels: `1` = illicit, `2` = licit, `unknown` = unlabeled (~79% of nodes).
  - `elliptic_txs_edgelist.csv` — directed edges (txId1 → txId2).
- **Optional secondary:** Elliptic++ (actor-level wallet graph) — only pull this in Phase 5+ if time permits; treat it as a generalization test, not core path.
- **Optional cross-domain:** IEEE-CIS Fraud Detection (Kaggle) — tabular, not graph-native; only use if you want to show the classifier (not the GAN) generalizes to non-graph fraud, which is a stretch goal, not core.

### 1.2 Clean & structure
1. Map `txId` → contiguous integer node IDs (0…N-1); keep a lookup table.
2. Recode labels: illicit → `1`, licit → `0`, unknown → `-1` (mask out of loss, but **keep in the graph** — unknown nodes still carry structural information other nodes connect to).
3. Attach the `time step` column (1–49) as a node attribute — this is the backbone of your temporal conditioning, so verify it end-to-end with a sanity plot (fraud rate per time step — there's a known spike around timestep 43 in this dataset from a real dark-market shutdown; expect ~2-3% fraud in most steps, spiking higher there).
4. Build a **temporal edge-validity mask**: for every edge, confirm `time(src) <= time(dst)` or handle same-timestep edges explicitly — this matters because your generator later must not create edges to future nodes, and you need to verify the raw graph itself respects (or violates) this before you enforce it synthetically.

### 1.3 Feature engineering
- Standardize/normalize the 166 features (z-score per time step, not globally — feature distributions drift across time steps, and cross-time normalization would leak temporal information into the "local" features).
- Compute **structural features per node** you'll need for the discriminator later, using `networkx` or PyG utilities:
  - degree (in/out)
  - local clustering coefficient
  - k-hop neighborhood label composition (fraction of neighbors labeled illicit — a strong graph signal in this dataset)
  - ego-graph density
- Store these separately as `structural_features.pt` — the discriminator will consume both raw features AND these structural stats.

### 1.4 Splits (temporal, not random — this is important and often gotten wrong)
- Standard Elliptic protocol: **train on time steps 1–34, test on time steps 35–49** (this matches most published baselines you're comparing against, including GraphSMOTE/GGA/LAGA papers where reproducible — makes your baseline table defensible).
- Within train, hold out steps 30–34 as validation.
- **Critical for your GAN**: the generator/discriminator must only ever be trained on the train-split time range — synthetic nodes for evaluation-period timesteps would be data leakage.

### 1.5 Deliverable for Phase 1
- `data/processed/{node_features.pt, edge_index.pt, labels.pt, timestep.pt, structural_features.pt, train_mask.pt, val_mask.pt, test_mask.pt}`
- A short EDA notebook/report: class imbalance chart, fraud-rate-per-timestep chart, degree distribution, feature correlation heatmap — useful both for your own sanity check and as Review-1/2 slides.

---

## Phase 2 — Baseline (establishes the "before" numbers)

Train a **plain GraphSAGE** (2-3 layers, mean/pool aggregator) directly on the imbalanced graph, standard cross-entropy loss, no augmentation.

Expected result pattern (this is what you want to *see*, to prove the problem is real): high accuracy (~95-97%), poor recall/F1 on illicit class (this is the number you're setting out to fix — published Elliptic baselines typically land illicit-F1 somewhere in the 0.55-0.65 range for plain GCN/GraphSAGE without augmentation).

This baseline is also your **first re-implementation deliverable** for the comparison table (Phase 5), so build it well once.

---

## Phase 3 — The Novel Component: Graph-Conditioned, Temporally-Aware GAN

This is the heart of the project and where your two research-gap claims get implemented. Build it in three sub-modules, matching your architecture diagram exactly.

### 3.1 Conditioning Embedding
For each real illicit node, extract:
- its own feature vector
- an aggregated embedding of its k-hop neighborhood (mean or GraphSAGE-style pooling)
- its **time step**, embedded via a small learned embedding table or sinusoidal positional encoding (sinusoidal is easier to justify to reviewers as "standard practice from Transformers," and avoids needing 49 separate learned vectors with sparse training signal per step)

Concatenate these into a single conditioning vector `c`.

### 3.2 Generator — `Noise + c → synthetic node features`
- Simple MLP (3-4 layers, LeakyReLU, batch norm) mapping `[noise_vector ; c] → synthetic_feature_vector`.
- **Temporal constraint enforcement**: the generator also outputs (or is given) a target timestep `t_gen`. This is what operationalizes your Novelty Claim #1 — pass `t_gen` explicitly and make sure downstream edge prediction is restricted to nodes with `time <= t_gen`.

### 3.3 Edge Predictor — decides which existing nodes to connect to
- Score candidate existing nodes (restricted to those with `time <= t_gen`, and biased toward nodes within the same or nearby illicit cluster — sample candidates from a k-hop neighborhood around the conditioning node rather than scoring the whole graph, for tractability) using a bilinear or MLP link-prediction head: `score(synthetic_node, candidate) = MLP([synthetic_feat ; candidate_feat ; |t_synthetic - t_candidate|])`.
- Use top-k or Gumbel-softmax sampling to select actual edges (Gumbel-softmax is preferable — keeps edge selection differentiable so gradients flow back to the generator, which matters for genuinely adversarial training rather than a two-stage disconnected pipeline).

### 3.4 Discriminator — **this is Novelty Claim #2, be careful here**
Two heads, trained jointly, both feeding gradients back to the generator+edge predictor:

**Head A — Node realism** (standard, most prior work stops here):
`D_node(node_features) → real/fake`

**Head B — Structural realism (your actual novelty)**:
Takes the synthetic node's freshly-attached local subgraph (the node + its predicted edges + those neighbors' existing features) and compares it against real fraud-cluster subgraphs on:
- degree of the new node
- local clustering coefficient of the resulting subgraph
- neighbor label homophily (fraction of neighbors that are illicit)
- possibly a small GNN encoder (1-2 layer GraphSAGE) that embeds the whole local subgraph into a vector, then a classifier head on that embedding — this is the "learns and improves" part that distinguishes you from THG-OAFN's fixed post-hoc rule.

`D_struct(subgraph_embedding) → real/fake`

**Why this must be trained adversarially, not as a fixed filter** (this is literally the sentence that separates you from THG-OAFN — say this explicitly in your writeup): the structural discriminator's own weights are updated via backprop from its classification loss on real vs. fake subgraphs, and its gradients flow back through the edge predictor and generator. A fixed rule-based filter (like THG-OAFN's approach) can only reject bad samples after the fact; yours actively reshapes what the generator learns to produce.

**Optional enhancement from your notes**: add a contrastive loss term in the discriminator (real subgraph vs. fake subgraph pairs, InfoNCE-style) to sharpen real/fake substructure separation — worth adding once the base adversarial loop is stable, treat as a Phase 3.5 ablation rather than core-path risk.

### 3.5 Adversarial training loop (standard GAN alternation, adapted)
```
for epoch in range(num_epochs):
    # --- Train Discriminator ---
    sample real illicit-node subgraphs (positive examples)
    generate fake nodes + edges via G (current weights), form fake subgraphs (negative examples)
    loss_D = BCE(D_node(real), 1) + BCE(D_node(fake.detach()), 0)
           + BCE(D_struct(real_subgraph), 1) + BCE(D_struct(fake_subgraph.detach()), 0)
    update D

    # --- Train Generator + Edge Predictor ---
    generate fake nodes + edges again (with grad)
    loss_G = BCE(D_node(fake), 1) + BCE(D_struct(fake_subgraph), 1)
           + temporal_violation_penalty   # penalize any edge to a future timestep
    update G, EdgePredictor
```
Track: generator/discriminator loss curves, and periodically (every N epochs) a held-out check of generated-subgraph structural stats vs. real fraud-cluster stats (degree/clustering distributions) — this becomes a very convincing Review-2 slide ("our fake fraud clusters statistically converge toward real ones over training").

### 3.6 Deliverable for Phase 3
- Trained G, EdgePredictor, D checkpoints.
- Plot: real vs. synthetic subgraph structural statistic distributions (degree, clustering) at epoch 0 vs. final epoch — directly demonstrates Novelty Claim #2 working, not just claimed.

---

## Phase 4 — Augmented Training (Downstream Classifier)

1. Use the trained generator to produce N synthetic illicit nodes + their edges (choose N to roughly balance the class ratio — don't over-balance to 50/50 blindly; sweep 10%/20%/30% synthetic-augmentation ratios as an ablation, since over-augmenting can also hurt precision).
2. Merge into the real graph (only within/before the training time window — Phase 1.4 leakage rule still applies).
3. Retrain **GraphSAGE (and optionally GAT)** on this balanced graph — same architecture as your Phase 2 baseline, so the comparison is apples-to-apples and any improvement is attributable to the augmentation, not a stronger classifier.

---

## Phase 5 — Baselines to Re-implement (from your literature review table)

Since none have published results on Elliptic itself, you re-implement all of them on the same splits for a fair table:

| Baseline | Re-implementation effort | Notes |
|---|---|---|
| Plain GCN/GraphSAGE (no augmentation) | Low — already have this from Phase 2 | Your "before" number |
| GraphSMOTE | Medium — public repo exists, adapt to Elliptic's features | Non-adversarial oversampling comparison |
| GAT-COBO (cost-sensitive) | Medium | No new data generated — reweighting-only baseline |
| GGA (CIKM 2023) | High — no time-awareness, standard classifier discriminator, closest architecture to yours minus your two novelties | Most important comparison — this proves your delta |
| LAGA (CogMI 2023) | High — same authors, improved edge placement | Second-most important comparison |
| THG-OAFN (PLOS ONE 2025) | High — has time + structural check but non-adversarial | Proves adversarial-vs-fixed-rule matters |

**Practical tip:** if GGA/LAGA/THG-OAFN have no public code, re-implement only their described core mechanism faithfully (don't over-engineer beyond what their papers specify) and document exactly what you replicated vs. approximated — reviewers accept "faithful re-implementation, here's what we approximated and why" far better than silence on the gap.

---

## Phase 6 — Evaluation & Final Metrics

**Primary metrics (all reported on the illicit class specifically, not overall accuracy — say this explicitly, it's the whole point of the project):**
- F1-score (illicit class)
- Recall (illicit class) — "did we actually catch the fraud"
- Precision (illicit class) — "how many false alarms"
- AUC-ROC and AUC-PR (PR curve matters more than ROC under heavy imbalance — include both, but foreground PR-AUC)
- Macro-F1 (secondary, shows overall balance)

**Structural validity metrics (unique to your project, use these to defend Novelty Claim #2 quantitatively, not just qualitatively):**
- KL-divergence or Wasserstein distance between real-fraud-cluster and synthetic-fraud-cluster degree distributions
- Same for clustering-coefficient distributions
- % of synthetic edges that violate the temporal constraint (should trend to ~0% — a clean quantitative proof of Novelty Claim #1)

**Final comparison table format** (this is your single most important results artifact):

| Method | Illicit-F1 | Recall | Precision | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| Plain GraphSAGE | | | | | |
| GraphSMOTE | | | | | |
| GAT-COBO | | | | | |
| GGA (re-impl) | | | | | |
| LAGA (re-impl) | | | | | |
| THG-OAFN (re-impl) | | | | | |
| **GraphGuard (ours)** | | | | | |

**Ablations to run** (these make the paper/report much stronger and directly map to your novelty claims):
1. Ours w/o temporal conditioning (proves Claim #1 matters)
2. Ours w/o structural discriminator (Head B only removed → node-level discriminator only, proves Claim #2 matters)
3. Ours with fixed-rule structural filter instead of adversarial Head B (directly recreates THG-OAFN's approach inside your own pipeline — the single most convincing ablation, isolates "adversarial vs. fixed rule" as the only variable)
4. Sweep synthetic-augmentation ratio (10/20/30/50%) vs. F1

### Final system output
The deliverable is: a trained downstream classifier + a reusable generator that can be re-run to produce additional synthetic illicit-transaction data on demand, packaged as:
- `graphguard_classifier.pt` — deployable fraud classifier
- `graphguard_generator.pt` + `edge_predictor.pt` — reusable synthetic-data generator
- Results notebook/report with the comparison table, ablations, and structural-validity plots above
- A short inference script: given a new (unseen) transaction subgraph slice, output illicit-probability per node

---

## Timeline vs. TRL (mapping to your Review stages)

| Stage | TRL | What's done |
|---|---|---|
| Review 1 (now) | TRL 3 | Literature/patent gap analysis done, architecture designed on paper — **you are here** |
| +2-3 weeks | TRL 4 | Phase 1 (data) + Phase 2 (baseline) + Generator/Discriminator built and unit-tested in isolation |
| Review 2 target | TRL 4→5 | Phase 3 (adversarial training converging) + Phase 4 (augmented classifier) integrated end-to-end |
| Review 3 target | TRL 5→6 | Phase 5 (all baselines re-implemented) + Phase 6 (full comparison table + ablations) — results stable and benchmarked |

---

## Risk / honesty notes for your own planning
- GAN training instability is the single biggest technical risk here — budget real time for mode collapse debugging (generator producing near-identical synthetic nodes). Standard mitigations: spectral normalization on D, feature matching loss, or WGAN-GP loss instead of vanilla BCE if vanilla GAN training is unstable.
- Gumbel-softmax edge sampling can be finicky to tune (temperature annealing). Have a fallback: train edge predictor as a separate non-differentiable top-k step first (like GGA does) to get *something* working end-to-end before attempting the fully differentiable version — you can honestly report which version you shipped.
- Re-implementing 3 papers with no public code (GGA, LAGA, THG-OAFN) is real engineering effort — don't underestimate this in your timeline; it's often bigger than building your own model.
