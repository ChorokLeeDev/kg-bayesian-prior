#!/usr/bin/env python3
"""
Empirically test Definition 2.3's CMI condition: I(phi(e); c(e,r) | freq(e)) ≈ 0

This script tests whether trained DistMult embeddings encode coverage information
beyond what frequency already reveals. The key insight:

- If CMI ≈ 0: Embeddings are "frequency-equivalent" for coverage prediction
- If CMI > 0: Embeddings encode coverage beyond frequency (violates Definition 2.3)

Test methodology:
1. Load trained DistMult embeddings from FB15k-237
2. Stratify entities by frequency quintiles
3. Within each stratum, train logistic regression to predict c(e,r) from phi(e)
4. If CMI ≈ 0, AUC should be near 0.5 (random) within each stratum
5. Compare to baseline: predict c(e,r) from freq(e) alone

Key insight: If trained embeddings achieve AUC >> 0.5 within frequency strata,
then embeddings DO encode coverage beyond frequency, violating Definition 2.3.
If AUC ≈ 0.5 within strata, then CMI ≈ 0 and Definition 2.3 holds.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data.loaders import load_fb15k237
from src.models.distmult import DistMult


def compute_entity_frequency(triples: np.ndarray, num_entities: int) -> np.ndarray:
    """Compute entity frequency (number of triples each entity appears in)."""
    freq = np.zeros(num_entities, dtype=np.float32)
    for h, r, t in triples:
        freq[h] += 1
        freq[t] += 1
    return freq


def compute_coverage_matrix(triples: np.ndarray, num_entities: int, num_relations: int) -> np.ndarray:
    """Compute coverage matrix: C[e, r] = 1 if entity e appears with relation r."""
    coverage = np.zeros((num_entities, num_relations), dtype=np.float32)
    for h, r, t in triples:
        coverage[h, r] = 1
        coverage[t, r] = 1
    return coverage


def train_distmult(train_dataset, device, epochs=100, lr=0.001, embedding_dim=100):
    """Train a DistMult model on the training data."""
    model = DistMult(
        num_entities=train_dataset.num_entities,
        num_relations=train_dataset.num_relations,
        embedding_dim=embedding_dim,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    triples = torch.tensor(train_dataset.triples, device=device)
    batch_size = 1024
    n_triples = len(triples)

    print(f"Training DistMult for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        perm = torch.randperm(n_triples, device=device)

        for i in range(0, n_triples, batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = triples[batch_idx]

            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # Generate negative samples (corrupt tail)
            neg_t = torch.randint(0, train_dataset.num_entities, (len(batch),), device=device)

            pos_scores = model.score_triple(h, r, t)
            neg_scores = model.score_triple(h, r, neg_t)

            # BCE loss
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            # Add regularization
            loss += 0.001 * model.regularization_loss(lambda_reg=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


def get_frequency_quintiles(freq: np.ndarray, active_mask: np.ndarray):
    """Assign entities to frequency quintiles (1-5)."""
    active_freq = freq[active_mask]
    quintile_thresholds = np.percentile(active_freq, [20, 40, 60, 80])

    quintiles = np.zeros(len(freq), dtype=int)
    for i, f in enumerate(freq):
        if not active_mask[i]:
            quintiles[i] = 0  # Inactive
        elif f <= quintile_thresholds[0]:
            quintiles[i] = 1
        elif f <= quintile_thresholds[1]:
            quintiles[i] = 2
        elif f <= quintile_thresholds[2]:
            quintiles[i] = 3
        elif f <= quintile_thresholds[3]:
            quintiles[i] = 4
        else:
            quintiles[i] = 5

    return quintiles, quintile_thresholds


def test_cmi_within_strata(embeddings, freq, coverage, quintiles, num_relations, output_lines):
    """
    Core CMI test: Within each frequency stratum, test if embeddings predict coverage.

    If I(phi(e); c(e,r) | freq(e)) = 0, then WITHIN each frequency stratum,
    phi(e) should provide no information about c(e,r), so AUC ≈ 0.5.
    """
    results_by_quintile = {}

    for q in range(1, 6):
        quintile_mask = quintiles == q
        n_entities_q = quintile_mask.sum()

        if n_entities_q < 100:
            msg = f"  Quintile {q}: Too few entities ({n_entities_q}), skipping"
            print(msg)
            output_lines.append(msg)
            continue

        # Get entities in this quintile
        entities_q = np.where(quintile_mask)[0]
        emb_q = embeddings[entities_q]
        coverage_q = coverage[entities_q]
        freq_q = freq[entities_q]

        # Scale embeddings
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb_q)

        # Test on multiple relations
        relation_positives = coverage_q.sum(axis=0)
        testable_rels = np.where(
            (relation_positives >= 20) &
            (relation_positives <= len(entities_q) - 20)
        )[0]

        if len(testable_rels) < 5:
            msg = f"  Quintile {q}: Not enough testable relations ({len(testable_rels)}), skipping"
            print(msg)
            output_lines.append(msg)
            continue

        # Sample relations for efficiency
        if len(testable_rels) > 30:
            np.random.seed(42 + q)
            testable_rels = np.random.choice(testable_rels, 30, replace=False)

        aucs_emb = []
        aucs_random = []

        # Generate random embeddings for control
        np.random.seed(42)
        random_emb = np.random.randn(*emb_scaled.shape).astype(np.float32)

        for r in testable_rels:
            y = coverage_q[:, r]

            if y.sum() < 10 or (1 - y).sum() < 10:
                continue

            # Split data
            try:
                X_emb_tr, X_emb_te, y_tr, y_te = train_test_split(
                    emb_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
                X_rand_tr, X_rand_te, _, _ = train_test_split(
                    random_emb, y, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError:
                continue

            # Train classifiers
            lr_emb = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
            lr_rand = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')

            try:
                lr_emb.fit(X_emb_tr, y_tr)
                lr_rand.fit(X_rand_tr, y_tr)

                pred_emb = lr_emb.predict_proba(X_emb_te)[:, 1]
                pred_rand = lr_rand.predict_proba(X_rand_te)[:, 1]

                auc_emb = roc_auc_score(y_te, pred_emb)
                auc_rand = roc_auc_score(y_te, pred_rand)

                aucs_emb.append(auc_emb)
                aucs_random.append(auc_rand)
            except Exception:
                continue

        if len(aucs_emb) < 5:
            msg = f"  Quintile {q}: Not enough successful tests ({len(aucs_emb)})"
            print(msg)
            output_lines.append(msg)
            continue

        mean_auc_emb = np.mean(aucs_emb)
        std_auc_emb = np.std(aucs_emb)
        mean_auc_rand = np.mean(aucs_random)
        std_auc_rand = np.std(aucs_random)

        freq_range = f"[{freq_q.min():.0f}, {freq_q.max():.0f}]"

        results_by_quintile[q] = {
            'n_entities': n_entities_q,
            'n_relations_tested': len(aucs_emb),
            'freq_range': freq_range,
            'auc_emb_mean': mean_auc_emb,
            'auc_emb_std': std_auc_emb,
            'auc_random_mean': mean_auc_rand,
            'auc_random_std': std_auc_rand,
            'delta': mean_auc_emb - 0.5,
            'emb_vs_random': mean_auc_emb - mean_auc_rand,
        }

        msg = f"  Quintile {q} (freq {freq_range}, n={n_entities_q}):"
        print(msg)
        output_lines.append(msg)
        msg = f"    phi(e) AUC: {mean_auc_emb:.4f} +/- {std_auc_emb:.4f}"
        print(msg)
        output_lines.append(msg)
        msg = f"    Random AUC: {mean_auc_rand:.4f} +/- {std_auc_rand:.4f}"
        print(msg)
        output_lines.append(msg)
        msg = f"    Delta from 0.5: {mean_auc_emb - 0.5:+.4f}"
        print(msg)
        output_lines.append(msg)

    return results_by_quintile


def test_frequency_baseline(freq, coverage, active_entities, num_relations, output_lines):
    """
    Baseline: Predict c(e,r) from freq(e) alone (no stratification).
    This shows how much predictive power frequency has across all entities.
    """
    msg = "\nBaseline: freq(e) alone predicts c(e,r) (no stratification)"
    print(msg)
    output_lines.append(msg)

    freq_active = freq[active_entities].reshape(-1, 1)
    coverage_active = coverage[active_entities]

    scaler = StandardScaler()
    freq_scaled = scaler.fit_transform(freq_active)

    relation_positives = coverage_active.sum(axis=0)
    testable_rels = np.where(
        (relation_positives >= 50) &
        (relation_positives <= len(active_entities) - 50)
    )[0]

    if len(testable_rels) > 50:
        np.random.seed(42)
        testable_rels = np.random.choice(testable_rels, 50, replace=False)

    aucs = []
    for r in testable_rels:
        y = coverage_active[:, r]

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                freq_scaled, y, test_size=0.3, random_state=42, stratify=y
            )

            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_tr, y_tr)

            pred = lr.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, pred))
        except Exception:
            continue

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    msg = f"  freq(e) AUC: {mean_auc:.4f} +/- {std_auc:.4f} (over {len(aucs)} relations)"
    print(msg)
    output_lines.append(msg)

    return mean_auc, std_auc


def main():
    output_lines = []

    header = "=" * 70
    title = "CMI Empirical Test: I(phi(e); c(e,r) | freq(e)) ≈ 0"
    timestamp = f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    print(header)
    print(title)
    print(timestamp)
    print(header)

    output_lines.extend([header, title, timestamp, header, ""])

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    msg = f"Using device: {device}"
    print(msg)
    output_lines.append(msg)

    # Load FB15k-237
    msg = "\nLoading FB15k-237..."
    print(msg)
    output_lines.append(msg)

    train_ds, valid_ds, test_ds = load_fb15k237()

    msg = f"  Train: {len(train_ds)} triples"
    print(msg)
    output_lines.append(msg)
    msg = f"  Entities: {train_ds.num_entities}"
    print(msg)
    output_lines.append(msg)
    msg = f"  Relations: {train_ds.num_relations}"
    print(msg)
    output_lines.append(msg)

    # Train DistMult
    output_lines.append("")
    model = train_distmult(train_ds, device, epochs=100, embedding_dim=100)

    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.entity_embeddings.weight.cpu().numpy()

    # Compute frequency and coverage
    freq = compute_entity_frequency(train_ds.triples, train_ds.num_entities)
    coverage = compute_coverage_matrix(train_ds.triples, train_ds.num_entities, train_ds.num_relations)

    # Filter active entities
    active_mask = freq > 0
    active_entities = np.where(active_mask)[0]

    msg = f"\nActive entities (freq > 0): {len(active_entities)}"
    print(msg)
    output_lines.append(msg)

    # Assign frequency quintiles
    quintiles, thresholds = get_frequency_quintiles(freq, active_mask)

    msg = f"Frequency quintile thresholds: {thresholds}"
    print(msg)
    output_lines.append(msg)

    # ================================================================
    # MAIN TEST: CMI within frequency strata
    # ================================================================
    msg = "\n" + "=" * 70
    print(msg)
    output_lines.append(msg)
    msg = "CMI TEST: Within each frequency stratum, predict c(e,r) from phi(e)"
    print(msg)
    output_lines.append(msg)
    msg = "=" * 70
    print(msg)
    output_lines.append(msg)
    msg = "\nIf I(phi; c | freq) ≈ 0, then AUC ≈ 0.5 within each stratum."
    print(msg)
    output_lines.append(msg)
    msg = "If AUC >> 0.5, embeddings encode coverage beyond frequency.\n"
    print(msg)
    output_lines.append(msg)

    results_by_quintile = test_cmi_within_strata(
        embeddings, freq, coverage, quintiles, train_ds.num_relations, output_lines
    )

    # ================================================================
    # BASELINE: Frequency alone (unstratified)
    # ================================================================
    msg = "\n" + "=" * 70
    print(msg)
    output_lines.append(msg)
    msg = "BASELINE: freq(e) alone predicts c(e,r)"
    print(msg)
    output_lines.append(msg)
    msg = "=" * 70
    print(msg)
    output_lines.append(msg)

    freq_auc, freq_std = test_frequency_baseline(
        freq, coverage, active_entities, train_ds.num_relations, output_lines
    )

    # ================================================================
    # SUMMARY
    # ================================================================
    msg = "\n" + "=" * 70
    print(msg)
    output_lines.append(msg)
    msg = "SUMMARY: CMI Empirical Test Results"
    print(msg)
    output_lines.append(msg)
    msg = "=" * 70
    print(msg)
    output_lines.append(msg)

    msg = "\nTable: AUC for predicting c(e,r) from phi(e) within frequency strata"
    print(msg)
    output_lines.append(msg)
    msg = "-" * 70
    print(msg)
    output_lines.append(msg)
    msg = f"{'Quintile':<10} {'Freq Range':<15} {'N Entities':<12} {'phi(e) AUC':<15} {'Random AUC':<15} {'Delta':<10}"
    print(msg)
    output_lines.append(msg)
    msg = "-" * 70
    print(msg)
    output_lines.append(msg)

    overall_aucs = []
    for q in sorted(results_by_quintile.keys()):
        r = results_by_quintile[q]
        msg = f"{q:<10} {r['freq_range']:<15} {r['n_entities']:<12} {r['auc_emb_mean']:.3f} +/- {r['auc_emb_std']:.3f}  {r['auc_random_mean']:.3f} +/- {r['auc_random_std']:.3f}  {r['delta']:+.3f}"
        print(msg)
        output_lines.append(msg)
        overall_aucs.append(r['auc_emb_mean'])

    msg = "-" * 70
    print(msg)
    output_lines.append(msg)

    if overall_aucs:
        overall_mean = np.mean(overall_aucs)
        msg = f"\nOverall mean AUC within strata: {overall_mean:.4f}"
        print(msg)
        output_lines.append(msg)

        msg = f"Baseline (freq alone, unstratified): {freq_auc:.4f}"
        print(msg)
        output_lines.append(msg)

    # ================================================================
    # INTERPRETATION
    # ================================================================
    msg = "\n" + "=" * 70
    print(msg)
    output_lines.append(msg)
    msg = "INTERPRETATION"
    print(msg)
    output_lines.append(msg)
    msg = "=" * 70
    print(msg)
    output_lines.append(msg)

    if overall_aucs and overall_mean < 0.55:
        interpretation = """
DEFINITION 2.3 CONFIRMED: I(phi(e); c(e,r) | freq(e)) ≈ 0

Within each frequency stratum, embeddings achieve AUC close to 0.5 (random),
meaning they provide essentially no information about coverage beyond what
frequency already reveals.

Key findings:
1. Embeddings DO NOT encode relation-specific coverage patterns
2. The predictive power from freq alone (unstratified) comes entirely
   from the frequency-coverage correlation
3. Once we control for frequency, embeddings are uninformative

This confirms that standard KG embeddings are "relation-agnostic" in the
sense of Definition 2.3: the only coverage information they contain is
what can be inferred from entity frequency.
"""
    elif overall_aucs and overall_mean < 0.60:
        interpretation = """
DEFINITION 2.3 APPROXIMATELY HOLDS

Within frequency strata, embeddings achieve AUC slightly above 0.5,
suggesting minimal coverage information beyond frequency.

The small improvement (< 0.1 AUC) may be due to:
1. Residual correlation between embedding geometry and graph structure
2. Imperfect stratification (quintiles are coarse bins)
3. Relation-specific entity clustering effects

For practical purposes, frequency remains the dominant predictor.
"""
    else:
        interpretation = """
WARNING: Embeddings show non-trivial predictive power within frequency strata.

This suggests embeddings may encode some coverage information beyond
frequency, potentially providing evidence against Definition 2.3.

However, this could also indicate:
1. Correlation between embedding geometry and local graph structure
2. Need for finer-grained frequency stratification
3. Entity-specific patterns that correlate with coverage
"""

    print(interpretation)
    output_lines.append(interpretation)

    # Final verdict
    msg = "\n" + "=" * 70
    print(msg)
    output_lines.append(msg)

    if overall_aucs:
        if overall_mean < 0.55:
            verdict = "VERDICT: Definition 2.3 CONFIRMED - CMI ≈ 0"
        elif overall_mean < 0.60:
            verdict = "VERDICT: Definition 2.3 APPROXIMATELY HOLDS"
        else:
            verdict = "VERDICT: Definition 2.3 requires further investigation"
    else:
        verdict = "VERDICT: Insufficient data to conclude"

    print(verdict)
    output_lines.append(verdict)
    msg = "=" * 70
    print(msg)
    output_lines.append(msg)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "cmi_empirical_test.txt"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    msg = f"\nResults saved to: {output_path}"
    print(msg)


if __name__ == "__main__":
    main()
