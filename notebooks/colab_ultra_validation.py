#!/usr/bin/env python3
"""
ULTRA Foundation Model - Coverage Blind Spot Validation (Colab Version)

Copy this script to Google Colab with GPU runtime for fast full evaluation.

Expected runtime: ~5-10 minutes on Colab GPU (vs 30+ minutes on CPU)
"""

# ============================================================
# CELL 1: Setup
# ============================================================
# !git clone https://github.com/DeepGraphLearning/ULTRA
# !pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
# !pip install torch-geometric
# !wget https://zenodo.org/record/8278563/files/ultra_3g.pth -O ULTRA/ckpts/ultra_3g.pth

import sys
sys.path.insert(0, 'ULTRA')

import os
os.chdir('ULTRA')

import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
import time

from ultra.models import Ultra
from ultra import datasets as ultra_datasets

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CELL 2: Load Model
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = Ultra(
    rel_model_cfg={
        'class': 'RelNBFNet',
        'input_dim': 64,
        'hidden_dims': [64, 64, 64, 64, 64, 64],
        'message_func': 'distmult',
        'aggregate_func': 'sum',
        'short_cut': True,
        'layer_norm': True
    },
    entity_model_cfg={
        'class': 'EntityNBFNet',
        'input_dim': 64,
        'hidden_dims': [64, 64, 64, 64, 64, 64],
        'message_func': 'distmult',
        'aggregate_func': 'sum',
        'short_cut': True,
        'layer_norm': True
    }
)

state = torch.load('ckpts/ultra_3g.pth', map_location='cpu')
model.load_state_dict(state['model'])
model = model.to(device)
model.eval()
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# CELL 3: Load Dataset
# ============================================================
# Choose dataset
DATASET = 'fb15k237'  # or 'wn18rr'

if DATASET == 'fb15k237':
    dataset = ultra_datasets.FB15k237('kg-datasets')
else:
    dataset = ultra_datasets.WN18RR('kg-datasets')

train_data = dataset[0]
test_data = dataset[2]

num_entities = train_data.num_nodes
num_relations = train_data.num_relations
print(f"Dataset: {DATASET}")
print(f"Entities: {num_entities:,}, Relations: {num_relations}")
print(f"Test triples: {test_data.target_edge_index.shape[1]:,}")

# ============================================================
# CELL 4: Build Coverage Matrix
# ============================================================
def build_coverage_matrix(edge_index, edge_type, num_entities, num_relations):
    coverage = np.zeros((num_entities, num_relations), dtype=np.float32)
    for i in range(edge_index.shape[1]):
        h = edge_index[0, i].item()
        t = edge_index[1, i].item()
        r = edge_type[i].item()
        if r >= num_relations // 2:
            r = r - num_relations // 2
        coverage[h, r] = 1.0
        coverage[t, r] = 1.0
    return coverage

coverage = build_coverage_matrix(
    train_data.edge_index,
    train_data.edge_type,
    num_entities,
    num_relations
)
print(f"Coverage rate: {coverage.sum() / coverage.size * 100:.1f}%")

# ============================================================
# CELL 5: Categorize Test Triples
# ============================================================
# Compute entity frequencies
freq = defaultdict(int)
for i in range(train_data.edge_index.shape[1]):
    h = train_data.edge_index[0, i].item()
    t = train_data.edge_index[1, i].item()
    freq[h] += 1
    freq[t] += 1

thresh = np.percentile(list(freq.values()), 25)

emerging_idx = []
novel_ctx_idx = []
id_idx = []
num_base_relations = num_relations // 2

for i in range(test_data.target_edge_index.shape[1]):
    h = test_data.target_edge_index[0, i].item()
    t = test_data.target_edge_index[1, i].item()
    r = test_data.target_edge_type[i].item()

    r_base = r - num_base_relations if r >= num_base_relations else r
    h_freq = freq.get(h, 0)
    t_freq = freq.get(t, 0)

    if h_freq <= thresh or t_freq <= thresh:
        emerging_idx.append(i)
    elif coverage[h, r_base] == 0 or coverage[t, r_base] == 0:
        novel_ctx_idx.append(i)
    else:
        id_idx.append(i)

print(f"Emerging: {len(emerging_idx):,}")
print(f"Novel context: {len(novel_ctx_idx):,}")
print(f"In-distribution: {len(id_idx):,}")

# ============================================================
# CELL 6: Score Triples
# ============================================================
def score_triples(model, data, indices, batch_size=128):
    """Score test triples using ULTRA."""
    model.eval()
    data = data.to(device)

    h = test_data.target_edge_index[0][indices]
    t = test_data.target_edge_index[1][indices]
    r = test_data.target_edge_type[indices]

    scores = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            end_idx = min(i + batch_size, len(indices))
            batch_h = h[i:end_idx].to(device)
            batch_t = t[i:end_idx].to(device)
            batch_r = r[i:end_idx].to(device)

            batch = torch.stack([batch_h, batch_t, batch_r], dim=-1).unsqueeze(1)
            score = model(data, batch).squeeze(-1).cpu()
            scores.append(score)

            if (i // batch_size) % 10 == 0:
                print(f"  Progress: {i}/{len(indices)}")

    return torch.cat(scores).numpy()

print("\nScoring all test triples...")
all_idx = emerging_idx + novel_ctx_idx + id_idx
t0 = time.time()
all_scores = score_triples(model, train_data, all_idx)
print(f"Scoring took {time.time() - t0:.1f}s")

# Remap indices
n_e = len(emerging_idx)
n_n = len(novel_ctx_idx)
n_i = len(id_idx)
emerging_new = list(range(0, n_e))
novel_new = list(range(n_e, n_e + n_n))
id_new = list(range(n_e + n_n, n_e + n_n + n_i))

uncertainties = -all_scores  # negative score = uncertainty

# ============================================================
# CELL 7: Compute AUROC
# ============================================================
def compute_auroc(unc, ood_idx, id_idx):
    ood_unc = unc[ood_idx]
    id_unc = unc[id_idx]
    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])
    return roc_auc_score(labels, scores)

# Novel Context AUROC (THE KEY METRIC)
novel_ctx_auroc = compute_auroc(uncertainties, novel_new, id_new)
emerging_auroc = compute_auroc(uncertainties, emerging_new, id_new)
overall_auroc = compute_auroc(uncertainties, emerging_new + novel_new, id_new)

print("\n" + "=" * 60)
print("ULTRA COVERAGE BLIND SPOT VALIDATION RESULTS")
print("=" * 60)
print(f"\nDataset: {DATASET}")
print(f"Test triples: {len(all_idx):,}")
print(f"  Emerging: {n_e:,}")
print(f"  Novel context: {n_n:,}")
print(f"  In-distribution: {n_i:,}")

print(f"\nAUROC Results:")
print(f"  Overall AUROC:     {overall_auroc:.4f}")
print(f"  Emerging AUROC:    {emerging_auroc:.4f}")
print(f"  Novel Ctx AUROC:   {novel_ctx_auroc:.4f}  <- KEY METRIC")

print("\n" + "-" * 60)
print("INTERPRETATION")
print("-" * 60)
if novel_ctx_auroc < 0.55:
    print(f"""
Novel Context AUROC = {novel_ctx_auroc:.3f} (near-random)

This CONFIRMS the coverage blind spot hypothesis:
- ULTRA cannot distinguish queries with unseen (entity, relation) pairs
- Its NBFNet architecture uses relation-agnostic entity embeddings
- High-connectivity entities appear "confident" regardless of relation

Implication: Scale does not fix the coverage blind spot.
Solution: Explicit (entity, relation) coverage tracking is required.
""")
else:
    print(f"""
Novel Context AUROC = {novel_ctx_auroc:.3f} (above random)

This result needs investigation:
- If AUROC > 0.6: ULTRA may have some novel-context awareness
- Check if the categorization is correct
- Consider re-running with different random seeds
""")

# ============================================================
# CELL 8: Save Results
# ============================================================
results = {
    'dataset': DATASET,
    'n_emerging': n_e,
    'n_novel_ctx': n_n,
    'n_id': n_i,
    'overall_auroc': float(overall_auroc),
    'emerging_auroc': float(emerging_auroc),
    'novel_ctx_auroc': float(novel_ctx_auroc),
}

import json
with open('ultra_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to ultra_validation_results.json")
