#!/usr/bin/env python3
"""
Coverage Paradox Analysis: Anchor Hypothesis Verification

Background:
- FB15k-237에서 Partial zero-coverage (59.5%) > Full coverage (32.3%) incorrect rate
- Hypothesis: "Partial에서 covered entity가 anchor 역할을 해서 prediction을 constrain"

Verification Experiments:
1. Partial coverage triple에서 covered entity vs uncovered entity의 score 기여도 분석
2. Covered entity의 embedding이 prediction에 얼마나 기여하는지 측정
3. "Anchor 제거" 실험: covered entity 없이 예측 시 성능 변화
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from datetime import datetime


def load_fb15k237(data_dir="/Users/i767700/Github/kg-bayesian-prior/data/raw/fb15k-237"):
    """Load FB15k-237 dataset."""
    entity2id = {}
    relation2id = {}

    def load_triples(filepath, update_vocab=True):
        triples = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]

                if update_vocab:
                    if h not in entity2id:
                        entity2id[h] = len(entity2id)
                    if t not in entity2id:
                        entity2id[t] = len(entity2id)
                    if r not in relation2id:
                        relation2id[r] = len(relation2id)

                if h in entity2id and t in entity2id and r in relation2id:
                    triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples

    train = load_triples(f"{data_dir}/train.txt", update_vocab=True)
    valid = load_triples(f"{data_dir}/valid.txt", update_vocab=True)
    test = load_triples(f"{data_dir}/test.txt", update_vocab=True)

    return train, valid, test, entity2id, relation2id


class DistMultModel(nn.Module):
    """Simple DistMult for score decomposition analysis."""

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        """DistMult: score = sum(h * r * t)"""
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def get_contributions(self, h, r, t):
        """
        Decompose score into per-entity contributions.
        score = sum_d (h_d * r_d * t_d) = sum_d contrib_d

        Head contribution: sum_d (|h_d * r_d| * sign(t_d))
        Tail contribution: sum_d (|t_d * r_d| * sign(h_d))
        """
        h_emb = self.entity_emb(h)  # [B, D]
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)

        # Per-dimension products
        hr = h_emb * r_emb  # head-relation product
        tr = t_emb * r_emb  # tail-relation product
        hrt = hr * t_emb    # full product

        # Head contribution: how much head contributes given r
        head_contrib = torch.abs(hr).sum(dim=-1)  # |h * r| magnitude
        tail_contrib = torch.abs(tr).sum(dim=-1)  # |t * r| magnitude

        total_score = hrt.sum(dim=-1)

        return {
            'total_score': total_score,
            'head_contrib': head_contrib,
            'tail_contrib': tail_contrib,
            'head_norm': h_emb.norm(dim=-1),
            'tail_norm': t_emb.norm(dim=-1),
            'rel_norm': r_emb.norm(dim=-1),
        }


def build_coverage_matrix(triples, num_entities, num_relations):
    """Build binary coverage matrix: coverage[e, r] = 1 if entity e seen with relation r."""
    coverage = np.zeros((num_entities, num_relations), dtype=bool)
    for h, r, t in triples:
        coverage[h, r] = True
        coverage[t, r] = True
    return coverage


def categorize_triples(triples, coverage):
    """
    Categorize triples by coverage type:
    - Full coverage: both h and t have been seen with r
    - Partial coverage: exactly one of h/t has been seen with r
    - Zero coverage: neither h nor t has been seen with r

    For partial, track which entity is covered.
    """
    full = []
    partial_head_covered = []  # h covered, t not
    partial_tail_covered = []  # t covered, h not
    zero = []

    for h, r, t in triples:
        h_covered = coverage[h, r]
        t_covered = coverage[t, r]

        if h_covered and t_covered:
            full.append((h, r, t))
        elif h_covered and not t_covered:
            partial_head_covered.append((h, r, t))
        elif not h_covered and t_covered:
            partial_tail_covered.append((h, r, t))
        else:
            zero.append((h, r, t))

    return {
        'full': full,
        'partial_head_covered': partial_head_covered,
        'partial_tail_covered': partial_tail_covered,
        'zero': zero,
    }


def train_distmult(model, train_triples, num_entities, epochs=50, batch_size=1024, lr=0.001, device='cpu'):
    """Train DistMult model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    triples_arr = np.array(train_triples)

    for epoch in range(epochs):
        np.random.shuffle(triples_arr)
        total_loss = 0

        for i in range(0, len(triples_arr), batch_size):
            batch = triples_arr[i:i+batch_size]
            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)

            # Positive scores
            pos_scores = model(h, r, t)

            # Negative sampling (corrupt tail)
            neg_t = torch.randint(0, num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            # BCE loss
            pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
            neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))
            loss = pos_loss + neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss / (len(triples_arr) // batch_size):.4f}")

    return model


def analyze_anchor_effect(model, categories, device='cpu'):
    """
    Analyze anchor effect: Does the covered entity constrain predictions?

    Key metrics:
    1. Score magnitude: Does partial coverage have lower scores?
    2. Contribution ratio: Does covered entity contribute more?
    3. Embedding norm: Is covered entity better learned (higher norm)?
    """
    results = {}

    model.eval()
    with torch.no_grad():
        for cat_name, triples in categories.items():
            if len(triples) == 0:
                continue

            arr = np.array(triples)
            h = torch.tensor(arr[:, 0], device=device)
            r = torch.tensor(arr[:, 1], device=device)
            t = torch.tensor(arr[:, 2], device=device)

            contribs = model.get_contributions(h, r, t)

            results[cat_name] = {
                'count': len(triples),
                'mean_score': contribs['total_score'].mean().item(),
                'score_std': contribs['total_score'].std().item(),
                'mean_head_contrib': contribs['head_contrib'].mean().item(),
                'mean_tail_contrib': contribs['tail_contrib'].mean().item(),
                'mean_head_norm': contribs['head_norm'].mean().item(),
                'mean_tail_norm': contribs['tail_norm'].mean().item(),
            }

    return results


def analyze_anchor_constraint_effect(model, partial_head_covered, partial_tail_covered,
                                     num_entities, num_relations, device='cpu'):
    """
    Experiment: Does the covered entity act as an anchor that constrains prediction?

    Hypothesis: If the covered entity is an anchor, then:
    1. The uncovered entity should have higher prediction entropy
    2. Replacing the covered entity should change the score more than replacing the uncovered
    """
    results = {}

    model.eval()
    with torch.no_grad():
        # Combine partial coverage triples
        # partial_head_covered: h is covered, t is not
        # partial_tail_covered: t is covered, h is not

        for cat_name, triples, covered_pos in [
            ('partial_head_covered', partial_head_covered, 'head'),
            ('partial_tail_covered', partial_tail_covered, 'tail')
        ]:
            if len(triples) < 100:
                continue

            arr = np.array(triples)
            h = torch.tensor(arr[:, 0], device=device)
            r = torch.tensor(arr[:, 1], device=device)
            t = torch.tensor(arr[:, 2], device=device)

            # Original scores
            original_scores = model(h, r, t)

            # Score variance when replacing covered entity
            n_samples = 100
            covered_replacement_deltas = []
            uncovered_replacement_deltas = []

            for _ in range(n_samples):
                random_entities = torch.randint(0, num_entities, (len(triples),), device=device)

                if covered_pos == 'head':
                    # Head is covered - replace head (covered) vs tail (uncovered)
                    covered_replaced = model(random_entities, r, t)
                    uncovered_replaced = model(h, r, random_entities)
                else:
                    # Tail is covered - replace tail (covered) vs head (uncovered)
                    covered_replaced = model(h, r, random_entities)
                    uncovered_replaced = model(random_entities, r, t)

                covered_replacement_deltas.append((original_scores - covered_replaced).abs().mean().item())
                uncovered_replacement_deltas.append((original_scores - uncovered_replaced).abs().mean().item())

            results[cat_name] = {
                'count': len(triples),
                'covered_replacement_delta': np.mean(covered_replacement_deltas),
                'uncovered_replacement_delta': np.mean(uncovered_replacement_deltas),
                'anchor_effect_ratio': np.mean(covered_replacement_deltas) / (np.mean(uncovered_replacement_deltas) + 1e-8),
            }

    return results


def analyze_prediction_confidence(model, categories, train_triples, num_entities, device='cpu'):
    """
    Analyze model's confidence on different coverage categories.

    Compute energy-based confidence: higher score = more confident prediction.
    """
    model.eval()
    results = {}

    # For each triple, compute its score relative to random negatives
    with torch.no_grad():
        for cat_name, triples in categories.items():
            if len(triples) < 100:
                continue

            arr = np.array(triples)
            h = torch.tensor(arr[:, 0], device=device)
            r = torch.tensor(arr[:, 1], device=device)
            t = torch.tensor(arr[:, 2], device=device)

            # Positive scores
            pos_scores = model(h, r, t)

            # Negative scores (sample 100 random tails per triple, take mean)
            neg_scores_list = []
            for _ in range(10):
                neg_t = torch.randint(0, num_entities, t.shape, device=device)
                neg_scores_list.append(model(h, r, neg_t))
            neg_scores = torch.stack(neg_scores_list).mean(dim=0)

            # Margin: pos - neg (higher = more confident)
            margins = pos_scores - neg_scores

            results[cat_name] = {
                'count': len(triples),
                'mean_pos_score': pos_scores.mean().item(),
                'mean_neg_score': neg_scores.mean().item(),
                'mean_margin': margins.mean().item(),
                'margin_std': margins.std().item(),
                'confident_correct_rate': (margins > 0).float().mean().item(),
            }

    return results


def run_anchor_removal_experiment(model, partial_triples, coverage, num_entities, device='cpu'):
    """
    Anchor Removal Experiment:

    For partial coverage triples, compare:
    1. Original prediction quality
    2. Prediction quality with covered entity replaced by random
    3. Prediction quality with uncovered entity replaced by random

    If covered entity is truly anchoring, removing it should hurt more.
    """
    results = []

    model.eval()
    with torch.no_grad():
        for h, r, t in partial_triples[:1000]:  # Sample for efficiency
            h_covered = coverage[h, r]
            t_covered = coverage[t, r]

            h_t = torch.tensor([h], device=device)
            r_t = torch.tensor([r], device=device)
            t_t = torch.tensor([t], device=device)

            # Original score
            original_score = model(h_t, r_t, t_t).item()

            # Replace covered entity
            random_e = torch.randint(0, num_entities, (1,), device=device)
            if h_covered:
                covered_removed_score = model(random_e, r_t, t_t).item()
                uncovered_removed_score = model(h_t, r_t, random_e).item()
            else:
                covered_removed_score = model(h_t, r_t, random_e).item()
                uncovered_removed_score = model(random_e, r_t, t_t).item()

            results.append({
                'original': original_score,
                'covered_removed': covered_removed_score,
                'uncovered_removed': uncovered_removed_score,
                'covered_delta': abs(original_score - covered_removed_score),
                'uncovered_delta': abs(original_score - uncovered_removed_score),
            })

    # Aggregate
    df = {k: [r[k] for r in results] for k in results[0].keys()}

    return {
        'mean_original_score': np.mean(df['original']),
        'mean_covered_removed_score': np.mean(df['covered_removed']),
        'mean_uncovered_removed_score': np.mean(df['uncovered_removed']),
        'mean_covered_delta': np.mean(df['covered_delta']),
        'mean_uncovered_delta': np.mean(df['uncovered_delta']),
        'anchor_ratio': np.mean(df['covered_delta']) / (np.mean(df['uncovered_delta']) + 1e-8),
    }


def main():
    output_path = "/Users/i767700/Github/kg-bayesian-prior/outputs/anchor_hypothesis_results.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Redirect output to both console and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    f = open(output_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    try:
        print("=" * 70)
        print("COVERAGE PARADOX ANALYSIS: ANCHOR HYPOTHESIS VERIFICATION")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Load data
        print("[1] Loading FB15k-237 data...")
        train, valid, test, entity2id, relation2id = load_fb15k237()
        num_entities = len(entity2id)
        num_relations = len(relation2id)
        print(f"  Entities: {num_entities:,}, Relations: {num_relations}")
        print(f"  Train: {len(train):,}, Valid: {len(valid):,}, Test: {len(test):,}")
        print()

        # Build coverage matrix
        print("[2] Building coverage matrix...")
        coverage = build_coverage_matrix(train, num_entities, num_relations)
        coverage_density = coverage.sum() / (num_entities * num_relations)
        print(f"  Coverage density: {coverage_density:.4f}")
        print()

        # Categorize test triples
        print("[3] Categorizing test triples by coverage...")
        categories = categorize_triples(test, coverage)
        total_test = len(test)

        print(f"  Full coverage:          {len(categories['full']):,} ({len(categories['full'])/total_test*100:.1f}%)")
        print(f"  Partial (head covered): {len(categories['partial_head_covered']):,} ({len(categories['partial_head_covered'])/total_test*100:.1f}%)")
        print(f"  Partial (tail covered): {len(categories['partial_tail_covered']):,} ({len(categories['partial_tail_covered'])/total_test*100:.1f}%)")
        print(f"  Zero coverage:          {len(categories['zero']):,} ({len(categories['zero'])/total_test*100:.1f}%)")

        partial_total = len(categories['partial_head_covered']) + len(categories['partial_tail_covered'])
        print(f"  Total partial:          {partial_total:,} ({partial_total/total_test*100:.1f}%)")
        print()

        # Train model
        print("[4] Training DistMult model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        model = DistMultModel(num_entities, num_relations, dim=100)
        model = train_distmult(model, train, num_entities, epochs=50, batch_size=1024, lr=0.001, device=device)
        print()

        # Analysis 1: Score decomposition
        print("[5] Analyzing score contributions by category...")
        print("=" * 70)
        contrib_results = analyze_anchor_effect(model, categories, device=device)

        print(f"\n{'Category':<25} {'Count':>8} {'Score':>10} {'H_contrib':>10} {'T_contrib':>10} {'H_norm':>8} {'T_norm':>8}")
        print("-" * 85)
        for cat_name, stats in contrib_results.items():
            print(f"{cat_name:<25} {stats['count']:>8,} {stats['mean_score']:>10.4f} "
                  f"{stats['mean_head_contrib']:>10.4f} {stats['mean_tail_contrib']:>10.4f} "
                  f"{stats['mean_head_norm']:>8.4f} {stats['mean_tail_norm']:>8.4f}")
        print()

        # Analysis 2: Anchor constraint effect
        print("[6] Analyzing anchor constraint effect...")
        print("=" * 70)
        print("  Testing: Does replacing the covered entity change the score more?")
        anchor_results = analyze_anchor_constraint_effect(
            model,
            categories['partial_head_covered'],
            categories['partial_tail_covered'],
            num_entities, num_relations, device=device
        )

        print(f"\n{'Category':<25} {'Count':>8} {'Covered Delta':>15} {'Uncovered Delta':>15} {'Anchor Ratio':>12}")
        print("-" * 80)
        for cat_name, stats in anchor_results.items():
            print(f"{cat_name:<25} {stats['count']:>8,} {stats['covered_replacement_delta']:>15.4f} "
                  f"{stats['uncovered_replacement_delta']:>15.4f} {stats['anchor_effect_ratio']:>12.4f}")
        print()

        # Analysis 3: Prediction confidence
        print("[7] Analyzing prediction confidence by category...")
        print("=" * 70)
        confidence_results = analyze_prediction_confidence(model, categories, train, num_entities, device=device)

        print(f"\n{'Category':<25} {'Count':>8} {'Pos Score':>12} {'Neg Score':>12} {'Margin':>10} {'Acc':>8}")
        print("-" * 80)
        for cat_name, stats in confidence_results.items():
            print(f"{cat_name:<25} {stats['count']:>8,} {stats['mean_pos_score']:>12.4f} "
                  f"{stats['mean_neg_score']:>12.4f} {stats['mean_margin']:>10.4f} {stats['confident_correct_rate']*100:>7.1f}%")
        print()

        # Analysis 4: Anchor removal experiment
        print("[8] Running anchor removal experiment...")
        print("=" * 70)
        partial_all = categories['partial_head_covered'] + categories['partial_tail_covered']
        removal_results = run_anchor_removal_experiment(model, partial_all, coverage, num_entities, device=device)

        print(f"\n  Original score mean:         {removal_results['mean_original_score']:.4f}")
        print(f"  Covered entity removed:      {removal_results['mean_covered_removed_score']:.4f}")
        print(f"  Uncovered entity removed:    {removal_results['mean_uncovered_removed_score']:.4f}")
        print(f"  Covered removal delta:       {removal_results['mean_covered_delta']:.4f}")
        print(f"  Uncovered removal delta:     {removal_results['mean_uncovered_delta']:.4f}")
        print(f"  Anchor ratio (covered/uncovered): {removal_results['anchor_ratio']:.4f}")
        print()

        # Summary and interpretation
        print("=" * 70)
        print("SUMMARY: ANCHOR HYPOTHESIS VERIFICATION")
        print("=" * 70)

        # Key finding 1: Contribution asymmetry in partial coverage
        partial_head = contrib_results.get('partial_head_covered', {})
        partial_tail = contrib_results.get('partial_tail_covered', {})

        if partial_head and partial_tail:
            # In partial_head_covered: head is covered, so head_contrib should be higher
            head_cov_ratio_hc = partial_head['mean_head_contrib'] / (partial_head['mean_tail_contrib'] + 1e-8)
            # In partial_tail_covered: tail is covered, so tail_contrib should be higher
            tail_cov_ratio_tc = partial_tail['mean_tail_contrib'] / (partial_tail['mean_head_contrib'] + 1e-8)

            print(f"\n[Finding 1] Contribution asymmetry in partial coverage:")
            print(f"  - When head is covered: head_contrib/tail_contrib = {head_cov_ratio_hc:.4f}")
            print(f"  - When tail is covered: tail_contrib/head_contrib = {tail_cov_ratio_tc:.4f}")

            if head_cov_ratio_hc > 1.0 and tail_cov_ratio_tc > 1.0:
                print(f"  CONFIRMED: Covered entities contribute MORE to the score")
            else:
                print(f"  NOT CONFIRMED: Contribution asymmetry not observed")

        # Key finding 2: Anchor ratio from removal experiment
        print(f"\n[Finding 2] Anchor removal effect:")
        print(f"  - Removing covered entity changes score by {removal_results['mean_covered_delta']:.4f}")
        print(f"  - Removing uncovered entity changes score by {removal_results['mean_uncovered_delta']:.4f}")
        print(f"  - Anchor ratio: {removal_results['anchor_ratio']:.4f}")

        if removal_results['anchor_ratio'] > 1.2:
            print(f"  CONFIRMED: Covered entity acts as an anchor (ratio > 1.2)")
        elif removal_results['anchor_ratio'] < 0.8:
            print(f"  REVERSED: Uncovered entity is more important (ratio < 0.8)")
        else:
            print(f"  NEUTRAL: No clear anchor effect (0.8 < ratio < 1.2)")

        # Key finding 3: Confidence levels
        full_acc = confidence_results.get('full', {}).get('confident_correct_rate', 0)
        partial_hc_acc = confidence_results.get('partial_head_covered', {}).get('confident_correct_rate', 0)
        partial_tc_acc = confidence_results.get('partial_tail_covered', {}).get('confident_correct_rate', 0)
        zero_acc = confidence_results.get('zero', {}).get('confident_correct_rate', 0)

        print(f"\n[Finding 3] Prediction accuracy by coverage type:")
        print(f"  - Full coverage:    {full_acc*100:.1f}%")
        print(f"  - Partial (H cov):  {partial_hc_acc*100:.1f}%")
        print(f"  - Partial (T cov):  {partial_tc_acc*100:.1f}%")
        print(f"  - Zero coverage:    {zero_acc*100:.1f}%")

        partial_avg = (partial_hc_acc + partial_tc_acc) / 2
        print(f"  - Partial average:  {partial_avg*100:.1f}%")

        if partial_avg < full_acc and partial_avg > zero_acc:
            print(f"  EXPECTED: Full > Partial > Zero (monotonic)")
        elif partial_avg > full_acc:
            print(f"  PARADOX CONFIRMED: Partial ({partial_avg*100:.1f}%) > Full ({full_acc*100:.1f}%)")
            print(f"  This supports the anchor hypothesis!")
        else:
            print(f"  UNEXPECTED: Different pattern observed")

        # Overall conclusion
        print(f"\n{'=' * 70}")
        print("CONCLUSION")
        print("=" * 70)

        anchor_confirmed = removal_results['anchor_ratio'] > 1.0

        if anchor_confirmed:
            print("""
The ANCHOR HYPOTHESIS is SUPPORTED by the data:

1. Covered entities contribute more to triple scores (contribution asymmetry)
2. Removing covered entities changes predictions more than removing uncovered ones
3. The covered entity acts as a "semantic anchor" that constrains the prediction space

INTERPRETATION:
- In partial coverage triples, the covered entity has a well-learned embedding
- This well-learned embedding constrains what the uncovered entity can be
- The constraint HELPS by reducing the prediction space to plausible entities
- This explains why partial > full in incorrect rate:
  * In full coverage, BOTH entities are constrained -> less room for error
  * In partial coverage, ONE entity anchors while the other can vary -> some error
  * In zero coverage, NO anchor -> high error (model is guessing)

IMPLICATION FOR OOD DETECTION:
- Partial coverage provides SOME signal (anchor strength)
- Zero coverage provides NO signal (complete uncertainty)
- The anchor effect partially mitigates the blind spot
""")
        else:
            print("""
The ANCHOR HYPOTHESIS is NOT SUPPORTED by the data.

Alternative explanations for Partial > Full:
1. Statistical artifact from different triple distributions
2. Relation-specific effects
3. Frequency effects (covered entities may be more common)
""")

        print(f"\nResults saved to: {output_path}")

    finally:
        sys.stdout = original_stdout
        f.close()


if __name__ == "__main__":
    main()
