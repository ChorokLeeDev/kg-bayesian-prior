#!/usr/bin/env python3
"""Test fixed CAGP on standard OOD (random corruptions) + compute AUPR."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scripts.run_wn18rr_temporal import (
    CAGP, CoverageOnly, GPOnly, train_model, setup_device,
)
from src.data.loaders import load_wn18rr, load_fb15k237
from collections import defaultdict

SEEDS = [42, 123, 456]

def evaluate_standard_ood(model, test_triples, n_ent, device):
    model.eval()
    with torch.no_grad():
        h = torch.tensor(test_triples[:, 0]).to(device)
        r = torch.tensor(test_triples[:, 1]).to(device)
        t = torch.tensor(test_triples[:, 2]).to(device)
        id_unc = model.get_uncertainty(h, r, t).cpu().numpy()
        t_corrupt = torch.randint(0, n_ent, (len(test_triples),)).to(device)
        ood_unc = model.get_uncertainty(h, r, t_corrupt).cpu().numpy()
    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    return auroc, aupr

def evaluate_temporal_aupr(model, train, test, n_ent, device):
    model.eval()
    freq = defaultdict(int)
    for i in range(len(train)):
        freq[train[i, 0]] += 1
        freq[train[i, 2]] += 1
    threshold = np.percentile([freq[e] for e in freq], 25)
    coverage_set = set()
    for i in range(len(train)):
        coverage_set.add((train[i, 0], train[i, 1]))
        coverage_set.add((train[i, 2], train[i, 1]))
    with torch.no_grad():
        h = torch.tensor(test[:, 0]).to(device)
        r = torch.tensor(test[:, 1]).to(device)
        t = torch.tensor(test[:, 2]).to(device)
        unc = model.get_uncertainty(h, r, t).cpu().numpy()
    labels = []
    for i in range(len(test)):
        hi, ri, ti = int(test[i,0]), int(test[i,1]), int(test[i,2])
        min_freq = min(freq.get(hi, 0), freq.get(ti, 0))
        if min_freq <= threshold:
            labels.append(1)
        elif (hi, ri) not in coverage_set or (ti, ri) not in coverage_set:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    auroc = roc_auc_score(labels, unc)
    aupr = average_precision_score(labels, unc)
    return auroc, aupr

def run_dataset(name, loader, device):
    sep = "=" * 60
    print("")
    print(sep)
    print("  %s -- Standard OOD + AUPR -- 3 seeds" % name)
    print(sep)
    train_ds, _, test_ds = loader()
    train = train_ds.triples
    test = test_ds.triples
    n_ent, n_rel = train_ds.num_entities, train_ds.num_relations
    print("Entities: %d, Relations: %d" % (n_ent, n_rel))
    for model_name, ModelClass in [("CAGP", CAGP), ("CoverageOnly", CoverageOnly), ("GPOnly", GPOnly)]:
        std_aurocs, std_auprs, temp_auprs = [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = ModelClass(n_ent, n_rel)
            m.precompute_coverage(train)
            m = train_model(m, train, device, epochs=30)
            if hasattr(m, "calibrate_normalization"):
                m.calibrate_normalization(train, device)
            auroc, aupr = evaluate_standard_ood(m, test, n_ent, device)
            std_aurocs.append(auroc)
            std_auprs.append(aupr)
            t_auroc, t_aupr = evaluate_temporal_aupr(m, train, test, n_ent, device)
            temp_auprs.append(t_aupr)
        print("  %-15s: std_AUROC=%.3f+/-%.3f  std_AUPR=%.3f+/-%.3f  temp_AUPR=%.3f+/-%.3f" % (
            model_name, np.mean(std_aurocs), np.std(std_aurocs),
            np.mean(std_auprs), np.std(std_auprs),
            np.mean(temp_auprs), np.std(temp_auprs)))
    old_std = {
        "WN18RR": {"CAGP": 0.66, "CoverageOnly": 0.66, "GPOnly": 0.68},
        "FB15k-237": {"CAGP": 0.82, "CoverageOnly": 0.82, "GPOnly": 0.76},
    }
    old_aupr = {
        "WN18RR": {"CAGP": 0.898, "CoverageOnly": 0.898, "GPOnly": 0.789},
        "FB15k-237": {"CAGP": 0.917, "CoverageOnly": 0.917, "GPOnly": 0.476},
    }
    if name in old_std:
        print("")
        print("  OLD standard OOD: " + str(old_std[name]))
        print("  OLD temporal AUPR: " + str(old_aupr[name]))

def main():
    device = setup_device()
    for name, loader in [("WN18RR", load_wn18rr), ("FB15k-237", load_fb15k237)]:
        run_dataset(name, loader, device)

if __name__ == "__main__":
    main()
