#!/usr/bin/env python3
"""
ULTRA Foundation Model Evaluation for OOD Detection

Tests whether ULTRA (a foundation KG model) suffers from the same coverage blind spot
as traditional KGE methods.

Hypothesis: ULTRA's predictions should be confident on novel contexts (because it uses
entity embeddings that don't encode coverage). If AUROC ~ 0.5 on novel contexts,
this confirms "the blind spot is fundamental, not fixable by scale."
"""

import sys
import os
from pathlib import Path

# Add ninja to PATH before importing anything else
os.environ['PATH'] = '/Users/i767700/Library/Python/3.9/bin:' + os.environ.get('PATH', '')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path.home() / 'Github' / 'ultra_test'))

import argparse
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
import time

# ULTRA imports
from ultra.models import Ultra
from ultra import datasets as ultra_datasets
from ultra.tasks import build_relation_graph

# Our data loaders
from src.data.loaders import load_fb15k237, load_wn18rr


def setup_device():
    # ULTRA's rspmm extension only supports CUDA or CPU, not MPS
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Force CPU for ULTRA as MPS is not supported
    return torch.device('cpu')


def load_ultra_model(checkpoint_path, device):
    """Load pretrained ULTRA model."""
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

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model'])
    model = model.to(device)
    model.eval()
    return model


def score_triples_ultra(model, data, triples, device, batch_size=64):
    """
    Score triples using ULTRA model.
    Returns logit scores (higher = more likely to be true).
    """
    model.eval()
    scores = []

    # Prepare data for ULTRA
    data = data.to(device)

    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            batch_triples = triples[i:i+batch_size]
            h = torch.tensor(batch_triples[:, 0], device=device)
            r = torch.tensor(batch_triples[:, 1], device=device)
            t = torch.tensor(batch_triples[:, 2], device=device)

            # ULTRA expects batch of shape (bs, 1+num_negs, 3)
            # For scoring single triples, we use num_negs=0
            batch = torch.stack([h, t, r], dim=-1).unsqueeze(1)  # (bs, 1, 3)

            try:
                score = model(data, batch)  # (bs, 1)
                scores.append(score.squeeze(-1).cpu())
            except Exception as e:
                print(f"Error scoring batch {i}: {e}")
                scores.append(torch.zeros(len(batch_triples)))

    return torch.cat(scores).numpy()


def get_ultra_uncertainty(scores):
    """
    Convert ULTRA scores to uncertainty.
    Higher score = more confident = lower uncertainty.
    """
    return -scores  # Negative score as uncertainty (like Energy-based)


def load_ultra_fb15k237(root_path):
    """Load FB15k-237 in ULTRA format."""
    # FB15k237 function already builds relation graphs
    dataset = ultra_datasets.FB15k237(root_path)
    return dataset


def load_ultra_wn18rr(root_path):
    """Load WN18RR in ULTRA format."""
    # WN18RR function already builds relation graphs
    dataset = ultra_datasets.WN18RR(root_path)
    return dataset


def build_coverage_matrix(train_triples, num_entities, num_relations):
    """Build coverage matrix from training triples."""
    coverage = np.zeros((num_entities, num_relations))
    for h, r, t in train_triples:
        coverage[h, r] = 1.0
        coverage[t, r] = 1.0
    return coverage


def evaluate_temporal_ood(
    model,
    ultra_data,
    train_triples,
    test_triples,
    coverage,
    device,
    emerging_operator='leq'
):
    """
    Evaluate ULTRA on temporal OOD detection.
    Returns AUROC for emerging, novel_ctx, and overall.
    """
    # Entity frequencies from training
    freq = defaultdict(int)
    for h, r, t in train_triples:
        freq[h] += 1
        freq[t] += 1

    thresh = np.percentile(list(freq.values()), 25)

    # Categorize test triples
    new_entity_idx, new_pair_idx, id_idx = [], [], []
    for i, (h, r, t) in enumerate(test_triples):
        h_freq = freq.get(h, 0)
        t_freq = freq.get(t, 0)

        if emerging_operator == 'leq':
            is_emerging = h_freq <= thresh or t_freq <= thresh
        else:
            is_emerging = h_freq < thresh or t_freq < thresh

        if is_emerging:
            new_entity_idx.append(i)
        elif coverage[h, r] == 0 or coverage[t, r] == 0:
            new_pair_idx.append(i)
        else:
            id_idx.append(i)

    print(f"  Split: emerging={len(new_entity_idx)}, novel_ctx={len(new_pair_idx)}, id={len(id_idx)}")

    results = {
        'n_emerging': len(new_entity_idx),
        'n_novel_ctx': len(new_pair_idx),
        'n_id': len(id_idx),
        'threshold': float(thresh),
    }

    # Score all test triples
    print("  Scoring test triples...")
    t0 = time.time()
    all_scores = score_triples_ultra(model, ultra_data, test_triples, device, batch_size=32)
    print(f"  Scoring took {time.time() - t0:.1f}s")

    all_uncertainties = get_ultra_uncertainty(all_scores)

    # Overall temporal OOD: emerging + novel_ctx vs ID
    ood_idx = new_entity_idx + new_pair_idx
    if len(ood_idx) > 50 and len(id_idx) > 50:
        ood_unc = all_uncertainties[ood_idx]
        id_unc = all_uncertainties[id_idx]

        labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
        scores = np.concatenate([id_unc, ood_unc])

        try:
            results['overall_auroc'] = float(roc_auc_score(labels, scores))
            results['overall_aupr'] = float(average_precision_score(labels, scores))
        except:
            results['overall_auroc'] = 0.5
            results['overall_aupr'] = 0.5

    # Emerging vs ID
    if len(new_entity_idx) > 50 and len(id_idx) > 50:
        e_unc = all_uncertainties[new_entity_idx]
        i_unc = all_uncertainties[id_idx]

        labels = np.concatenate([np.zeros(len(i_unc)), np.ones(len(e_unc))])
        scores = np.concatenate([i_unc, e_unc])

        try:
            results['emerging_auroc'] = float(roc_auc_score(labels, scores))
        except:
            results['emerging_auroc'] = 0.5

    # Novel context vs ID
    if len(new_pair_idx) > 50 and len(id_idx) > 50:
        n_unc = all_uncertainties[new_pair_idx]
        i_unc = all_uncertainties[id_idx]

        labels = np.concatenate([np.zeros(len(i_unc)), np.ones(len(n_unc))])
        scores = np.concatenate([i_unc, n_unc])

        try:
            results['novel_ctx_auroc'] = float(roc_auc_score(labels, scores))
        except:
            results['novel_ctx_auroc'] = 0.5

    return results


def main():
    parser = argparse.ArgumentParser(description="ULTRA Foundation Model OOD Evaluation")
    parser.add_argument('--dataset', type=str, default='fb15k237', choices=['fb15k237', 'wn18rr'])
    parser.add_argument('--checkpoint', type=str, default=str(Path.home() / 'Github' / 'ultra_test' / 'ckpts' / 'ultra_3g.pth'))
    parser.add_argument('--data-root', type=str, default=str(Path.home() / 'Github' / 'ultra_test' / 'kg-datasets'))
    args = parser.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    # Load ULTRA model
    print(f"\nLoading ULTRA model from {args.checkpoint}")
    model = load_ultra_model(args.checkpoint, device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())}")

    # Load dataset in ULTRA format
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == 'fb15k237':
        ultra_dataset = load_ultra_fb15k237(args.data_root)
        our_train, _, our_test = load_fb15k237()
    else:
        ultra_dataset = load_ultra_wn18rr(args.data_root)
        our_train, _, our_test = load_wn18rr()

    train_data = ultra_dataset[0]
    test_data = ultra_dataset[2]

    print(f"  ULTRA: {train_data.num_nodes} entities, {train_data.num_relations} relations")
    print(f"  Our data: {our_train.num_entities} entities, {our_train.num_relations} relations")

    # Build coverage matrix from our training data
    coverage = build_coverage_matrix(
        our_train.triples,
        our_train.num_entities,
        our_train.num_relations
    )

    # Evaluate
    print("\nEvaluating ULTRA on temporal OOD...")
    results = evaluate_temporal_ood(
        model,
        test_data,  # ULTRA test data (with graph structure)
        our_train.triples,
        our_test.triples,
        coverage,
        device
    )

    print("\n" + "=" * 60)
    print("ULTRA RESULTS")
    print("=" * 60)
    print(f"  Overall AUROC:    {results.get('overall_auroc', 'N/A'):.4f}")
    print(f"  Emerging AUROC:   {results.get('emerging_auroc', 'N/A'):.4f}")
    print(f"  Novel Ctx AUROC:  {results.get('novel_ctx_auroc', 'N/A'):.4f}")
    print(f"\n  Split sizes:")
    print(f"    Emerging:    {results['n_emerging']}")
    print(f"    Novel Ctx:   {results['n_novel_ctx']}")
    print(f"    ID:          {results['n_id']}")


if __name__ == "__main__":
    main()
