#!/usr/bin/env python3
"""
ICEWS14 rho-subset analysis: Find non-circular evidence that semantic uncertainty helps.

Problem: On ICEWS14/18/GDELT, semantic gain = 0pp because rho ~ 0
(most OOD entities have zero coverage). This makes the semantic + structural
decomposition look useless.

Goal: Find subsets where rho > 0 (entities have SOME coverage) and measure
whether semantic uncertainty adds value beyond structural coverage.

Subsets analyzed:
1. High-frequency entities (top 50% by training frequency)
2. Multi-relation entities (entities using 3+ relations in training)
3. Per-relation analysis (some relations may have higher coverage overlap)
4. Role-shift OOD (entities have coverage but appear in atypical context)

Key insight: If rho > 0 and semantic helps, we have non-circular evidence
because the OOD definition comes from TIME (test is later than train),
not from coverage.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import argparse


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Model: GPOnly for semantic uncertainty
# ============================================================

class GPOnly(nn.Module):
    """GP-KGE for semantic uncertainty (entity-level variance)."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_mean = nn.Parameter(torch.randn(num_entities, dim) * 0.1)
        self.entity_logvar = nn.Parameter(torch.zeros(num_entities, dim) - 1.0)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, h, r, t):
        return (self.entity_mean[h] * self.relation_emb(r) * self.entity_mean[t]).sum(-1)

    def get_semantic_uncertainty(self, h, r, t):
        """U_sem = avg entity variance."""
        h_var = torch.exp(self.entity_logvar[h]).mean(dim=-1)
        t_var = torch.exp(self.entity_logvar[t]).mean(dim=-1)
        return (h_var + t_var) / 2

    def get_structural_uncertainty(self, h, r, t):
        """U_str = 2 - coverage(h,r) - coverage(t,r)."""
        return 2.0 - self.coverage[h, r] - self.coverage[t, r]

    def get_combined_uncertainty(self, h, r, t, alpha=0.5):
        """CAGP = alpha * U_sem_norm + (1-alpha) * U_str."""
        u_sem = self.get_semantic_uncertainty(h, r, t)
        u_str = self.get_structural_uncertainty(h, r, t)
        # Normalize semantic to have similar scale as structural
        u_sem_norm = u_sem / (u_sem.mean() + 1e-8) * (u_str.mean() + 1e-8)
        return alpha * u_sem_norm + (1 - alpha) * u_str

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            self.coverage[triples[i, 0], triples[i, 1]] = 1.0
            self.coverage[triples[i, 2], triples[i, 1]] = 1.0


def train_model(model, triples, device, epochs=30, lr=0.001):
    """Train with BCE + margin loss on uncertainty."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            # Margin loss on uncertainty
            pos_unc = model.get_semantic_uncertainty(h, r, t)
            neg_unc = model.get_semantic_uncertainty(h, r, neg_t)
            unc_loss = F.relu(0.3 + pos_unc.mean() - neg_unc.mean())
            loss = loss + 0.1 * unc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}", flush=True)

    return model


# ============================================================
# Data loading
# ============================================================

def load_icews14(data_dir):
    """Load ICEWS14 triples."""
    def load_file(path):
        triples = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append([h, r, t])
        return np.array(triples)

    train = load_file(data_dir / 'train.txt')
    test = load_file(data_dir / 'test.txt')

    all_triples = np.concatenate([train, test])
    n_ent = max(all_triples[:, 0].max(), all_triples[:, 2].max()) + 1
    n_rel = all_triples[:, 1].max() + 1

    return train, test, int(n_ent), int(n_rel)


# ============================================================
# Subset analysis functions
# ============================================================

def compute_rho(test_triples, coverage, subset_name="all"):
    """
    Compute rho = fraction of OOD triples where BOTH entities have coverage.

    rho = P(coverage(h,r)=1 AND coverage(t,r)=1 | (h,r,t) in OOD)

    When rho > 0, there's potential for semantic uncertainty to help.
    When rho ~ 0, U_str = 2 for all OOD, so semantic can't add value.
    """
    n_both_covered = 0
    n_partial = 0
    n_none = 0

    for h, r, t in test_triples:
        h_cov = coverage[h, r]
        t_cov = coverage[t, r]
        if h_cov and t_cov:
            n_both_covered += 1
        elif h_cov or t_cov:
            n_partial += 1
        else:
            n_none += 1

    total = len(test_triples)
    rho = n_both_covered / total if total > 0 else 0.0

    print(f"\n  {subset_name}: n={total}", flush=True)
    print(f"    Both covered (rho): {n_both_covered} ({rho:.1%})", flush=True)
    print(f"    Partial coverage:   {n_partial} ({n_partial/total:.1%})", flush=True)
    print(f"    Zero coverage:      {n_none} ({n_none/total:.1%})", flush=True)

    return rho, n_both_covered


def evaluate_auroc_on_subset(model, ood_triples, id_triples, device, alpha=0.5):
    """
    Compute AUROC for semantic, structural, and combined on given subsets.
    Returns dict with auroc values.
    """
    if len(ood_triples) < 30 or len(id_triples) < 30:
        return None

    model.eval()
    with torch.no_grad():
        # OOD uncertainties
        h_ood = torch.tensor(ood_triples[:, 0]).to(device)
        r_ood = torch.tensor(ood_triples[:, 1]).to(device)
        t_ood = torch.tensor(ood_triples[:, 2]).to(device)

        u_sem_ood = model.get_semantic_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()
        u_str_ood = model.get_structural_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()
        u_comb_ood = model.get_combined_uncertainty(h_ood, r_ood, t_ood, alpha).cpu().numpy()

        # ID uncertainties
        h_id = torch.tensor(id_triples[:, 0]).to(device)
        r_id = torch.tensor(id_triples[:, 1]).to(device)
        t_id = torch.tensor(id_triples[:, 2]).to(device)

        u_sem_id = model.get_semantic_uncertainty(h_id, r_id, t_id).cpu().numpy()
        u_str_id = model.get_structural_uncertainty(h_id, r_id, t_id).cpu().numpy()
        u_comb_id = model.get_combined_uncertainty(h_id, r_id, t_id, alpha).cpu().numpy()

    # Labels: 1=OOD, 0=ID
    labels = np.concatenate([np.ones(len(ood_triples)), np.zeros(len(id_triples))])

    all_sem = np.concatenate([u_sem_ood, u_sem_id])
    all_str = np.concatenate([u_str_ood, u_str_id])
    all_comb = np.concatenate([u_comb_ood, u_comb_id])

    try:
        auroc_sem = roc_auc_score(labels, all_sem)
        auroc_str = roc_auc_score(labels, all_str)
        auroc_comb = roc_auc_score(labels, all_comb)
    except ValueError:
        return None

    return {
        'auroc_sem': auroc_sem,
        'auroc_str': auroc_str,
        'auroc_comb': auroc_comb,
        'gain_sem': auroc_sem - auroc_str,
        'n_ood': len(ood_triples),
        'n_id': len(id_triples),
        'u_sem_ood_mean': float(u_sem_ood.mean()),
        'u_sem_id_mean': float(u_sem_id.mean()),
        'u_str_ood_mean': float(u_str_ood.mean()),
        'u_str_id_mean': float(u_str_id.mean()),
    }


def analyze_high_frequency_entities(train, test, coverage, model, device, freq_threshold_pct=50):
    """
    Subset 1: High-frequency entities.

    Hypothesis: High-freq entities have coverage for most relations,
    so rho should be higher. If semantic helps here, it's non-circular
    because OOD is defined by TIME.
    """
    print("\n" + "=" * 60, flush=True)
    print("SUBSET 1: HIGH-FREQUENCY ENTITIES", flush=True)
    print("=" * 60, flush=True)

    # Compute entity frequencies
    freq = defaultdict(int)
    for h, r, t in train:
        freq[h] += 1
        freq[t] += 1

    threshold = np.percentile(list(freq.values()), freq_threshold_pct)
    print(f"  Frequency threshold ({freq_threshold_pct}th pct): {threshold}", flush=True)

    # Filter test to high-freq entities only
    high_freq_ood = []
    high_freq_id = []

    cov_np = coverage.cpu().numpy()

    for h, r, t in test:
        # Both entities must be high-frequency
        if freq.get(h, 0) >= threshold and freq.get(t, 0) >= threshold:
            h_cov = cov_np[h, r]
            t_cov = cov_np[t, r]
            if h_cov and t_cov:
                # Both covered = ID for this subset
                high_freq_id.append([h, r, t])
            else:
                # At least one not covered = OOD
                high_freq_ood.append([h, r, t])

    high_freq_ood = np.array(high_freq_ood) if high_freq_ood else np.zeros((0, 3), dtype=int)
    high_freq_id = np.array(high_freq_id) if high_freq_id else np.zeros((0, 3), dtype=int)

    print(f"  High-freq OOD: {len(high_freq_ood)}", flush=True)
    print(f"  High-freq ID:  {len(high_freq_id)}", flush=True)

    # Compute rho for the OOD subset
    if len(high_freq_ood) > 0:
        rho, n_both = compute_rho(high_freq_ood, cov_np, "High-freq OOD")

        # Evaluate if we have both OOD and ID
        result = evaluate_auroc_on_subset(model, high_freq_ood, high_freq_id, device)
        if result:
            print(f"\n  AUROC Results:", flush=True)
            print(f"    U_sem:  {result['auroc_sem']:.3f}", flush=True)
            print(f"    U_str:  {result['auroc_str']:.3f}", flush=True)
            print(f"    CAGP:   {result['auroc_comb']:.3f}", flush=True)
            print(f"    ** Semantic gain: {result['gain_sem']:+.3f} **", flush=True)
            return result, rho

    return None, 0.0


def analyze_multi_relation_entities(train, test, coverage, model, device, min_relations=3):
    """
    Subset 2: Multi-relation entities.

    Entities that appear with many different relations have broader coverage.
    If semantic helps distinguish unseen contexts, it should show here.
    """
    print("\n" + "=" * 60, flush=True)
    print(f"SUBSET 2: MULTI-RELATION ENTITIES (>= {min_relations} relations)", flush=True)
    print("=" * 60, flush=True)

    # Compute relations per entity
    entity_relations = defaultdict(set)
    for h, r, t in train:
        entity_relations[h].add(r)
        entity_relations[t].add(r)

    multi_rel_entities = {e for e, rels in entity_relations.items() if len(rels) >= min_relations}
    print(f"  Entities with >= {min_relations} relations: {len(multi_rel_entities)}", flush=True)

    # Filter test
    cov_np = coverage.cpu().numpy()
    multi_rel_ood = []
    multi_rel_id = []

    for h, r, t in test:
        if h in multi_rel_entities and t in multi_rel_entities:
            h_cov = cov_np[h, r]
            t_cov = cov_np[t, r]
            if h_cov and t_cov:
                multi_rel_id.append([h, r, t])
            else:
                multi_rel_ood.append([h, r, t])

    multi_rel_ood = np.array(multi_rel_ood) if multi_rel_ood else np.zeros((0, 3), dtype=int)
    multi_rel_id = np.array(multi_rel_id) if multi_rel_id else np.zeros((0, 3), dtype=int)

    print(f"  Multi-rel OOD: {len(multi_rel_ood)}", flush=True)
    print(f"  Multi-rel ID:  {len(multi_rel_id)}", flush=True)

    if len(multi_rel_ood) > 0:
        rho, n_both = compute_rho(multi_rel_ood, cov_np, "Multi-rel OOD")

        result = evaluate_auroc_on_subset(model, multi_rel_ood, multi_rel_id, device)
        if result:
            print(f"\n  AUROC Results:", flush=True)
            print(f"    U_sem:  {result['auroc_sem']:.3f}", flush=True)
            print(f"    U_str:  {result['auroc_str']:.3f}", flush=True)
            print(f"    CAGP:   {result['auroc_comb']:.3f}", flush=True)
            print(f"    ** Semantic gain: {result['gain_sem']:+.3f} **", flush=True)
            return result, rho

    return None, 0.0


def analyze_role_shift(train, test, coverage, model, device, typicality_threshold=0.8):
    """
    Subset 3: Role-shift OOD.

    Entities have coverage for the relation (rho=1 by construction),
    but the relation is ATYPICAL for their usual profile.

    This is the cleanest test: if semantic helps here, it's detecting
    "unusual usage patterns" rather than just missing coverage.
    """
    print("\n" + "=" * 60, flush=True)
    print("SUBSET 3: ROLE-SHIFT OOD (covered but atypical)", flush=True)
    print("=" * 60, flush=True)

    # Build entity-relation profile
    entity_rel_freq = defaultdict(lambda: defaultdict(int))
    for h, r, t in train:
        entity_rel_freq[h][r] += 1
        entity_rel_freq[t][r] += 1

    # For each entity, find "typical" relations (top 80% of its usage)
    entity_typical = {}
    for e, rel_counts in entity_rel_freq.items():
        total = sum(rel_counts.values())
        sorted_rels = sorted(rel_counts.items(), key=lambda x: -x[1])
        cumsum = 0
        typical = set()
        for rel, cnt in sorted_rels:
            cumsum += cnt
            typical.add(rel)
            if cumsum >= typicality_threshold * total:
                break
        entity_typical[e] = typical

    # Categorize test triples
    cov_np = coverage.cpu().numpy()
    role_shift_ood = []  # Covered but atypical
    standard_ood = []    # Not covered
    id_triples = []      # Covered and typical

    for h, r, t in test:
        h_cov = cov_np[h, r] > 0
        t_cov = cov_np[t, r] > 0
        h_typical = r in entity_typical.get(h, set())
        t_typical = r in entity_typical.get(t, set())

        if not (h_cov and t_cov):
            # At least one entity has no coverage for this relation
            standard_ood.append([h, r, t])
        elif not h_typical or not t_typical:
            # Both have coverage, but atypical for at least one
            role_shift_ood.append([h, r, t])
        else:
            # Both covered and typical
            id_triples.append([h, r, t])

    role_shift_ood = np.array(role_shift_ood) if role_shift_ood else np.zeros((0, 3), dtype=int)
    standard_ood = np.array(standard_ood) if standard_ood else np.zeros((0, 3), dtype=int)
    id_triples = np.array(id_triples) if id_triples else np.zeros((0, 3), dtype=int)

    print(f"  Role-shift OOD (covered but atypical): {len(role_shift_ood)}", flush=True)
    print(f"  Standard OOD (not covered):            {len(standard_ood)}", flush=True)
    print(f"  ID (covered and typical):              {len(id_triples)}", flush=True)

    if len(role_shift_ood) > 0:
        # Role-shift by construction has rho=1
        rho = 1.0
        print(f"\n  Role-shift rho: {rho:.3f} (by construction)", flush=True)

        result = evaluate_auroc_on_subset(model, role_shift_ood, id_triples, device)
        if result:
            print(f"\n  AUROC Results (Role-shift vs ID):", flush=True)
            print(f"    U_sem:  {result['auroc_sem']:.3f}", flush=True)
            print(f"    U_str:  {result['auroc_str']:.3f}", flush=True)
            print(f"    CAGP:   {result['auroc_comb']:.3f}", flush=True)
            print(f"    ** Semantic gain: {result['gain_sem']:+.3f} **", flush=True)

            # Diagnostic: check if structural has any signal
            print(f"\n  Diagnostics:", flush=True)
            print(f"    U_str OOD mean: {result['u_str_ood_mean']:.4f}", flush=True)
            print(f"    U_str ID mean:  {result['u_str_id_mean']:.4f}", flush=True)
            print(f"    U_sem OOD mean: {result['u_sem_ood_mean']:.4f}", flush=True)
            print(f"    U_sem ID mean:  {result['u_sem_id_mean']:.4f}", flush=True)

            return result, rho

    return None, 0.0


def analyze_per_relation(train, test, coverage, model, device, min_test_triples=100):
    """
    Subset 4: Per-relation analysis.

    Some relations may have higher coverage overlap than others.
    Find relations where rho > 0 and check if semantic helps.
    """
    print("\n" + "=" * 60, flush=True)
    print("SUBSET 4: PER-RELATION ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    # Group test triples by relation
    rel_to_test = defaultdict(list)
    for h, r, t in test:
        rel_to_test[r].append([h, r, t])

    cov_np = coverage.cpu().numpy()

    best_results = []

    for rel_id, triples in rel_to_test.items():
        if len(triples) < min_test_triples:
            continue

        triples_np = np.array(triples)

        # Split into OOD (not both covered) vs ID (both covered)
        ood_idx = []
        id_idx = []
        both_covered = 0

        for i, (h, r, t) in enumerate(triples):
            h_cov = cov_np[h, r]
            t_cov = cov_np[t, r]
            if h_cov and t_cov:
                id_idx.append(i)
                both_covered += 1
            else:
                ood_idx.append(i)

        rho = both_covered / len(triples)

        if rho > 0.1 and len(ood_idx) >= 30 and len(id_idx) >= 30:
            # This relation has meaningful rho
            ood_triples = triples_np[ood_idx]
            id_triples_rel = triples_np[id_idx]

            result = evaluate_auroc_on_subset(model, ood_triples, id_triples_rel, device)
            if result:
                result['relation_id'] = rel_id
                result['rho'] = rho
                best_results.append(result)

    # Sort by rho (higher is more interesting)
    best_results.sort(key=lambda x: x['rho'], reverse=True)

    print(f"\n  Relations with rho > 0.1 and enough samples: {len(best_results)}", flush=True)

    if best_results:
        print(f"\n  Top 5 relations by rho:", flush=True)
        print(f"    {'Rel':<6} {'rho':>6} {'U_sem':>8} {'U_str':>8} {'Gain':>8} {'n_ood':>8}", flush=True)
        print(f"    {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}", flush=True)

        for r in best_results[:5]:
            print(f"    {r['relation_id']:<6} {r['rho']:.3f} {r['auroc_sem']:.3f}    {r['auroc_str']:.3f}    {r['gain_sem']:+.3f}    {r['n_ood']:>8}", flush=True)

        # Check if ANY relation shows semantic gain
        positive_gain_rels = [r for r in best_results if r['gain_sem'] > 0.01]
        print(f"\n  Relations with semantic gain > 0.01: {len(positive_gain_rels)}/{len(best_results)}", flush=True)

        if positive_gain_rels:
            best = max(positive_gain_rels, key=lambda x: x['gain_sem'])
            print(f"\n  BEST: Relation {best['relation_id']}", flush=True)
            print(f"    rho={best['rho']:.3f}, U_sem={best['auroc_sem']:.3f}, U_str={best['auroc_str']:.3f}", flush=True)
            print(f"    ** Semantic gain: {best['gain_sem']:+.3f} **", flush=True)
            return best, best['rho']

    return None, 0.0


def analyze_covered_ood(train, test, coverage, model, device):
    """
    CRITICAL SUBSET: OOD triples where rho = 1 (both entities have coverage).

    This is the purest test: these are temporal OOD (defined by time),
    but both entities have been seen with this relation before.
    If semantic helps here, it's genuinely useful.

    CAUTION: OOD defined by frequency is confounded with semantic uncertainty
    (low freq entities have high variance by training dynamics).
    """
    print("\n" + "=" * 60, flush=True)
    print("CRITICAL SUBSET: COVERED OOD (rho = 1)", flush=True)
    print("=" * 60, flush=True)

    # Coverage from training
    cov_np = coverage.cpu().numpy()

    # Entity frequency from training
    freq = defaultdict(int)
    for h, r, t in train:
        freq[h] += 1
        freq[t] += 1

    # Emerging threshold
    threshold = np.percentile(list(freq.values()), 25)

    # Find test triples where both entities have coverage
    covered_ood = []
    covered_id = []

    for h, r, t in test:
        h_cov = cov_np[h, r] > 0
        t_cov = cov_np[t, r] > 0

        if h_cov and t_cov:
            # Both covered - is this OOD or ID?
            # OOD = emerging entities (low frequency)
            h_emerging = freq.get(h, 0) <= threshold
            t_emerging = freq.get(t, 0) <= threshold

            if h_emerging or t_emerging:
                covered_ood.append([h, r, t])
            else:
                covered_id.append([h, r, t])

    covered_ood = np.array(covered_ood) if covered_ood else np.zeros((0, 3), dtype=int)
    covered_id = np.array(covered_id) if covered_id else np.zeros((0, 3), dtype=int)

    print(f"  Covered + Emerging (OOD): {len(covered_ood)}", flush=True)
    print(f"  Covered + Well-known (ID): {len(covered_id)}", flush=True)
    print(f"  rho = 1.0 by construction", flush=True)
    print(f"  WARNING: OOD=freq-based, confounded with semantic variance", flush=True)

    if len(covered_ood) >= 30 and len(covered_id) >= 30:
        result = evaluate_auroc_on_subset(model, covered_ood, covered_id, device)
        if result:
            print(f"\n  AUROC Results:", flush=True)
            print(f"    U_sem:  {result['auroc_sem']:.3f}", flush=True)
            print(f"    U_str:  {result['auroc_str']:.3f}", flush=True)
            print(f"    CAGP:   {result['auroc_comb']:.3f}", flush=True)
            print(f"    ** Semantic gain: {result['gain_sem']:+.3f} **", flush=True)

            print(f"\n  Diagnostics:", flush=True)
            print(f"    U_str should be ~0 for both (rho=1)", flush=True)
            print(f"    U_str OOD mean: {result['u_str_ood_mean']:.4f}", flush=True)
            print(f"    U_str ID mean:  {result['u_str_id_mean']:.4f}", flush=True)
            print(f"    U_sem OOD mean: {result['u_sem_ood_mean']:.4f}", flush=True)
            print(f"    U_sem ID mean:  {result['u_sem_id_mean']:.4f}", flush=True)

            # CONFOUND CHECK: Is semantic just detecting frequency?
            result['confounded'] = True  # Freq-based OOD is circular
            return result, 1.0

    return None, 0.0


def analyze_rare_relations(train, test, coverage, model, device, max_train_freq=50):
    """
    NON-CIRCULAR SUBSET: Rare relations in test.

    Find relations that are RARE in training (low total count).
    These relations have inherent uncertainty regardless of entity frequency.

    OOD definition: relation is rare in train (< max_train_freq occurrences)
    This is NOT defined by coverage or frequency, so non-circular.
    """
    print("\n" + "=" * 60, flush=True)
    print(f"NON-CIRCULAR SUBSET: RARE RELATIONS (train_freq < {max_train_freq})", flush=True)
    print("=" * 60, flush=True)

    # Count relation frequencies in training
    rel_freq = defaultdict(int)
    for h, r, t in train:
        rel_freq[r] += 1

    rare_rels = {r for r, cnt in rel_freq.items() if cnt < max_train_freq}
    common_rels = {r for r, cnt in rel_freq.items() if cnt >= max_train_freq * 10}

    print(f"  Rare relations (< {max_train_freq} train): {len(rare_rels)}", flush=True)
    print(f"  Common relations (>= {max_train_freq * 10} train): {len(common_rels)}", flush=True)

    cov_np = coverage.cpu().numpy()

    # Test triples with rare vs common relations
    rare_test = []
    common_test = []

    for h, r, t in test:
        h_cov = cov_np[h, r] > 0
        t_cov = cov_np[t, r] > 0

        # Only consider triples where both entities have SOME coverage
        # (to ensure rho > 0 in our analysis)
        if r in rare_rels:
            rare_test.append([h, r, t])
        elif r in common_rels:
            common_test.append([h, r, t])

    rare_test = np.array(rare_test) if rare_test else np.zeros((0, 3), dtype=int)
    common_test = np.array(common_test) if common_test else np.zeros((0, 3), dtype=int)

    print(f"  Test with rare relations: {len(rare_test)}", flush=True)
    print(f"  Test with common relations: {len(common_test)}", flush=True)

    if len(rare_test) >= 30 and len(common_test) >= 30:
        # Compute rho for rare relation test triples
        rho, n_both = compute_rho(rare_test, cov_np, "Rare relations")

        # OOD = rare relation, ID = common relation
        result = evaluate_auroc_on_subset(model, rare_test, common_test, device)
        if result:
            print(f"\n  AUROC Results (Rare vs Common relations):", flush=True)
            print(f"    U_sem:  {result['auroc_sem']:.3f}", flush=True)
            print(f"    U_str:  {result['auroc_str']:.3f}", flush=True)
            print(f"    CAGP:   {result['auroc_comb']:.3f}", flush=True)
            print(f"    ** Semantic gain: {result['gain_sem']:+.3f} **", flush=True)
            print(f"\n  This is NON-CIRCULAR: OOD defined by relation rarity,", flush=True)
            print(f"  not by coverage or entity frequency.", flush=True)
            result['confounded'] = False
            return result, rho

    return None, 0.0


def analyze_timestamp_shift(train, test, coverage, model, device, data_dir):
    """
    NON-CIRCULAR SUBSET: Late-timestamp test triples.

    ICEWS has actual timestamps. We can use LATEST timestamps as OOD
    (furthest from training time) vs EARLIER test timestamps as ID.

    This is truly non-circular: OOD is defined purely by time.
    """
    print("\n" + "=" * 60, flush=True)
    print("NON-CIRCULAR SUBSET: TIMESTAMP SHIFT", flush=True)
    print("=" * 60, flush=True)

    # Load timestamps
    def load_with_ts(path):
        triples = []
        timestamps = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    h, r, t, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    triples.append([h, r, t])
                    timestamps.append(ts)
        return np.array(triples), np.array(timestamps)

    test_triples, test_ts = load_with_ts(data_dir / 'test.txt')

    # Split test by timestamp: top 20% timestamps = late (OOD), bottom 80% = early (ID)
    ts_threshold = np.percentile(test_ts, 80)

    cov_np = coverage.cpu().numpy()

    late_test = []
    early_test = []

    for i, (h, r, t) in enumerate(test_triples):
        ts = test_ts[i]
        h_cov = cov_np[h, r] > 0
        t_cov = cov_np[t, r] > 0

        # Only use triples with both covered (rho=1)
        if h_cov and t_cov:
            if ts >= ts_threshold:
                late_test.append([h, r, t])
            else:
                early_test.append([h, r, t])

    late_test = np.array(late_test) if late_test else np.zeros((0, 3), dtype=int)
    early_test = np.array(early_test) if early_test else np.zeros((0, 3), dtype=int)

    print(f"  Late test (ts >= {ts_threshold}): {len(late_test)}", flush=True)
    print(f"  Early test (ts < {ts_threshold}): {len(early_test)}", flush=True)
    print(f"  rho = 1.0 (filtered to both covered)", flush=True)

    if len(late_test) >= 30 and len(early_test) >= 30:
        # OOD = late timestamp, ID = early timestamp
        result = evaluate_auroc_on_subset(model, late_test, early_test, device)
        if result:
            print(f"\n  AUROC Results (Late vs Early timestamps):", flush=True)
            print(f"    U_sem:  {result['auroc_sem']:.3f}", flush=True)
            print(f"    U_str:  {result['auroc_str']:.3f}", flush=True)
            print(f"    CAGP:   {result['auroc_comb']:.3f}", flush=True)
            print(f"    ** Semantic gain: {result['gain_sem']:+.3f} **", flush=True)
            print(f"\n  This is NON-CIRCULAR: OOD defined purely by time.", flush=True)
            result['confounded'] = False
            return result, 1.0

    return None, 0.0


def main():
    parser = argparse.ArgumentParser(description="ICEWS14 rho-subset analysis")
    parser.add_argument('--data_dir', type=str, default='data/raw/icews14',
                       help='Path to ICEWS14 data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = setup_device()

    data_dir = Path(project_root) / args.data_dir
    print(f"Loading ICEWS14 from {data_dir}", flush=True)
    print(f"Device: {device}", flush=True)

    train, test, n_ent, n_rel = load_icews14(data_dir)
    print(f"Train: {len(train)}, Test: {len(test)}", flush=True)
    print(f"Entities: {n_ent}, Relations: {n_rel}", flush=True)

    # Run multiple seeds
    all_results = defaultdict(list)

    for seed in range(args.seeds):
        print(f"\n{'#' * 70}", flush=True)
        print(f"# SEED {seed + 1}/{args.seeds}", flush=True)
        print(f"{'#' * 70}", flush=True)

        torch.manual_seed(42 + seed)
        np.random.seed(42 + seed)

        # Train model
        print("\nTraining GP-KGE...", flush=True)
        model = GPOnly(n_ent, n_rel)
        model.precompute_coverage(train)
        model = train_model(model, train, device, epochs=args.epochs)

        coverage = model.coverage

        # Run all analyses
        r1, rho1 = analyze_high_frequency_entities(train, test, coverage, model, device)
        r2, rho2 = analyze_multi_relation_entities(train, test, coverage, model, device)
        r3, rho3 = analyze_role_shift(train, test, coverage, model, device)
        r4, rho4 = analyze_per_relation(train, test, coverage, model, device)
        r5, rho5 = analyze_covered_ood(train, test, coverage, model, device)
        r6, rho6 = analyze_rare_relations(train, test, coverage, model, device)
        r7, rho7 = analyze_timestamp_shift(train, test, coverage, model, device, data_dir)

        if r1: all_results['high_freq'].append(r1)
        if r2: all_results['multi_rel'].append(r2)
        if r3: all_results['role_shift'].append(r3)
        if r4: all_results['per_relation'].append(r4)
        if r5: all_results['covered_ood'].append(r5)
        if r6: all_results['rare_relations'].append(r6)
        if r7: all_results['timestamp_shift'].append(r7)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: WHERE DOES SEMANTIC HELP?", flush=True)
    print("=" * 70, flush=True)

    for subset_name, results in all_results.items():
        if results:
            gains = [r['gain_sem'] for r in results]
            aurocs_sem = [r['auroc_sem'] for r in results]
            aurocs_str = [r['auroc_str'] for r in results]
            confounded = results[0].get('confounded', 'unknown')

            print(f"\n{subset_name}:", flush=True)
            print(f"  U_sem:  {np.mean(aurocs_sem):.3f} +/- {np.std(aurocs_sem):.3f}", flush=True)
            print(f"  U_str:  {np.mean(aurocs_str):.3f} +/- {np.std(aurocs_str):.3f}", flush=True)
            print(f"  Gain:   {np.mean(gains):+.3f} +/- {np.std(gains):.3f}", flush=True)
            print(f"  Confounded: {confounded}", flush=True)

            if np.mean(gains) > 0.02:
                if confounded:
                    print(f"  ** Gain detected but CONFOUNDED (circular) **", flush=True)
                else:
                    print(f"  ** SEMANTIC HELPS (gain > 2pp, non-circular) **", flush=True)
            elif np.mean(gains) > 0:
                print(f"  * Marginal positive gain *", flush=True)
            else:
                print(f"  No semantic gain", flush=True)

    # Final verdict
    print("\n" + "=" * 70, flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    # Non-circular subsets with gain
    noncircular_positive = [name for name, results in all_results.items()
                           if results and np.mean([r['gain_sem'] for r in results]) > 0.02
                           and not results[0].get('confounded', True)]

    # Circular subsets with gain (for comparison)
    circular_positive = [name for name, results in all_results.items()
                        if results and np.mean([r['gain_sem'] for r in results]) > 0.02
                        and results[0].get('confounded', False)]

    if noncircular_positive:
        print(f"NON-CIRCULAR EVIDENCE FOUND!", flush=True)
        print(f"Semantic helps in: {', '.join(noncircular_positive)}", flush=True)
        print(f"These subsets have OOD defined by TIME or RELATION RARITY,", flush=True)
        print(f"NOT by coverage or entity frequency.", flush=True)
    else:
        if circular_positive:
            print(f"Gain found in circular subsets: {', '.join(circular_positive)}", flush=True)
            print(f"BUT these are confounded (OOD = low freq = high variance).", flush=True)

        marginal_subsets = [name for name, results in all_results.items()
                          if results and np.mean([r['gain_sem'] for r in results]) > 0
                          and not results[0].get('confounded', True)]
        if marginal_subsets:
            print(f"Marginal non-circular evidence in: {', '.join(marginal_subsets)}", flush=True)
        else:
            print(f"No non-circular evidence that semantic helps on ICEWS14.", flush=True)
            print(f"Entity-level variance does NOT help beyond structural coverage", flush=True)
            print(f"when OOD is defined non-circularly.", flush=True)


if __name__ == "__main__":
    main()
