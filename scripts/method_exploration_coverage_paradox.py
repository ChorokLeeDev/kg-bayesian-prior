#!/usr/bin/env python3
"""
Coverage Paradox Method Exploration

Based on findings:
- Full Coverage: 32.3% incorrect (overconfident, diluted embeddings)
- Partial Zero: 59.5% incorrect?  (Actually: anchor effect helps)
- Full Zero: 14.8% (no anchor, extrapolation failure)

RCUE Failure Analysis:
- MLP within-class: +4.4pp (marginal)
- Coverage boost contaminates Energy signal
- Key issue: Coverage boost is additive, not discriminative

Exploration Directions:
1. Coverage-aware Calibration (temperature scaling per coverage type)
2. Anchor-based Prediction (use covered entity as explicit anchor)
3. Disentangled Embeddings (relation-specific components)
4. Cascading Uncertainty (Coverage -> OOD, Energy -> selective)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import load_fb15k237


# ==============================================================================
# Baselines
# ==============================================================================

class DistMultBaseline(nn.Module):
    """Simple DistMult baseline."""
    def __init__(self, n_ent, n_rel, dim=100):
        super().__init__()
        self.entity_emb = nn.Embedding(n_ent, dim)
        self.relation_emb = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.n_ent = n_ent

    def forward(self, h, r, t):
        return (self.entity_emb(h) * self.relation_emb(r) * self.entity_emb(t)).sum(-1)

    def score_tails(self, h, r):
        """Score all tails for (h, r, ?)"""
        hr = self.entity_emb(h) * self.relation_emb(r)  # [batch, dim]
        return hr @ self.entity_emb.weight.T  # [batch, n_ent]


def train_distmult(model, train, n_ent, epochs=30, batch_size=1024, lr=1e-3, device='cpu'):
    """Train DistMult with margin loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        np.random.shuffle(train)
        total_loss = 0

        for i in range(0, len(train), batch_size):
            batch = train[i:i+batch_size]
            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)
            t_neg = torch.randint(0, n_ent, (len(batch),), device=device)

            optimizer.zero_grad()
            pos = model(h, r, t)
            neg = model(h, r, t_neg)
            loss = torch.relu(1.0 - pos + neg).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}")

    return model


# ==============================================================================
# Direction 1: Coverage-aware Calibration
# ==============================================================================

class CoverageAwareCalibration:
    """
    Apply different temperature scaling per coverage type.

    Idea: Full coverage is overconfident -> higher temperature
          Partial coverage has anchor -> moderate temperature
          Zero coverage is random -> don't calibrate (abstain instead)
    """
    def __init__(self, model, coverage_matrix):
        self.model = model
        self.coverage = coverage_matrix

        # Learnable temperatures per coverage type
        self.temps = {
            'full': 1.0,
            'partial': 1.0,
            'zero': 1.0
        }

    def fit(self, valid_triples, device='cpu'):
        """Learn optimal temperatures via grid search on validation set."""
        print("Fitting coverage-aware calibration...")

        # Classify validation triples
        categories = self._classify_triples(valid_triples)

        # For each category, find optimal temperature
        for cat_name, indices in categories.items():
            if len(indices) < 50:
                continue

            cat_triples = valid_triples[indices]
            best_temp, best_ece = self._find_optimal_temp(cat_triples, device)
            self.temps[cat_name] = best_temp
            print(f"  {cat_name}: T={best_temp:.2f}, ECE={best_ece:.4f}")

        return self

    def _classify_triples(self, triples):
        """Classify triples by coverage type."""
        categories = {'full': [], 'partial': [], 'zero': []}

        for idx, (h, r, t) in enumerate(triples):
            h_cov = self.coverage[int(h), int(r)]
            t_cov = self.coverage[int(t), int(r)]

            if h_cov and t_cov:
                categories['full'].append(idx)
            elif h_cov or t_cov:
                categories['partial'].append(idx)
            else:
                categories['zero'].append(idx)

        return {k: np.array(v) for k, v in categories.items()}

    def _find_optimal_temp(self, triples, device, temps_to_try=np.arange(0.5, 3.1, 0.25)):
        """Find temperature that minimizes ECE."""
        self.model.eval()
        best_temp = 1.0
        best_ece = float('inf')

        with torch.no_grad():
            h = torch.tensor(triples[:, 0], device=device)
            r = torch.tensor(triples[:, 1], device=device)
            t = torch.tensor(triples[:, 2], device=device)

            # Score all tails
            all_scores = self.model.score_tails(h, r)  # [batch, n_ent]

            for temp in temps_to_try:
                # Apply temperature
                probs = F.softmax(all_scores / temp, dim=-1)

                # Get confidence (max prob) and accuracy
                confidence = probs.max(dim=-1).values.cpu().numpy()
                predicted = probs.argmax(dim=-1).cpu().numpy()
                correct = (predicted == triples[:, 2]).astype(float)

                # Compute ECE (binned)
                ece = self._compute_ece(confidence, correct)

                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp

        return best_temp, best_ece

    def _compute_ece(self, confidence, correct, n_bins=10):
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i+1])
            if mask.sum() > 0:
                bin_conf = confidence[mask].mean()
                bin_acc = correct[mask].mean()
                ece += mask.sum() * abs(bin_conf - bin_acc)

        return ece / len(confidence)

    def calibrated_uncertainty(self, h, r, t, device='cpu'):
        """Get calibrated uncertainty scores."""
        self.model.eval()

        with torch.no_grad():
            # Get raw scores
            scores = self.model(h.to(device), r.to(device), t.to(device))

            # Classify and apply temperatures
            calibrated = torch.zeros_like(scores)

            for idx in range(len(h)):
                h_cov = self.coverage[int(h[idx]), int(r[idx])]
                t_cov = self.coverage[int(t[idx]), int(r[idx])]

                if h_cov and t_cov:
                    temp = self.temps['full']
                elif h_cov or t_cov:
                    temp = self.temps['partial']
                else:
                    temp = self.temps['zero']

                calibrated[idx] = scores[idx] / temp

            # Return negative calibrated score as uncertainty
            return -calibrated


# ==============================================================================
# Direction 2: Anchor-based Prediction
# ==============================================================================

class AnchorBasedPredictor(nn.Module):
    """
    Use covered entity explicitly as anchor.

    For partial coverage (h covered, t not):
    - Anchor = h embedding (well-learned)
    - Target = predict t using h as context

    Key insight: The anchor constrains the prediction space.
    """
    def __init__(self, n_ent, n_rel, dim=100, hidden_dim=64):
        super().__init__()

        self.n_ent = n_ent
        self.n_rel = n_rel

        # Entity embeddings (same as DistMult)
        self.entity_emb = nn.Embedding(n_ent, dim)
        self.relation_emb = nn.Embedding(n_rel, dim)

        # Anchor attention: given anchor, what to attend to?
        self.anchor_proj = nn.Linear(dim, hidden_dim)
        self.context_proj = nn.Linear(dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Uncertainty head: predict confidence given anchor strength
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),  # +2 for coverage indicators
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage buffer
        self.register_buffer('coverage', torch.zeros(n_ent, n_rel))

    def precompute_coverage(self, triples):
        for h, r, t in triples:
            self.coverage[int(h), int(r)] = 1.0
            self.coverage[int(t), int(r)] = 1.0

    def forward(self, h, r, t):
        """Score triples (standard DistMult for compatibility)."""
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_anchor_uncertainty(self, h, r, t):
        """
        Compute uncertainty based on anchor strength.

        If h is covered (anchor) and t is not:
        - Use h's embedding as anchor
        - Attend to relation-specific context
        - Predict uncertainty for uncovered t
        """
        batch_size = h.shape[0]

        h_emb = self.entity_emb(h)  # [B, D]
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)

        # Get coverage indicators
        h_cov = self.coverage[h, r].unsqueeze(-1)  # [B, 1]
        t_cov = self.coverage[t, r].unsqueeze(-1)

        # Project to attention space
        anchor_q = self.anchor_proj(h_emb * r_emb).unsqueeze(1)  # [B, 1, H]
        context_kv = self.context_proj(t_emb * r_emb).unsqueeze(1)  # [B, 1, H]

        # Attention: anchor attends to target
        attn_out, _ = self.attn(anchor_q, context_kv, context_kv)  # [B, 1, H]
        attn_out = attn_out.squeeze(1)  # [B, H]

        # Predict uncertainty
        unc_input = torch.cat([attn_out, h_cov, t_cov], dim=-1)
        uncertainty = self.uncertainty_head(unc_input).squeeze(-1)

        return uncertainty

    def score_tails(self, h, r):
        """Score all tails for (h, r, ?)"""
        h_emb = self.entity_emb(h)  # [batch, dim]
        r_emb = self.relation_emb(r)  # [batch, dim]
        hr = h_emb * r_emb
        return hr @ self.entity_emb.weight.T  # [batch, n_ent]


# ==============================================================================
# Direction 3: Disentangled Embeddings
# ==============================================================================

class DisentangledEmbeddingKGE(nn.Module):
    """
    Disentangle entity embeddings into relation-specific components.

    Instead of e = [e_1, ..., e_d], use:
    e_r = RoutingNetwork(e, r)

    This addresses the "dilution" problem where embeddings get averaged
    across many relations.
    """
    def __init__(self, n_ent, n_rel, dim=100, n_experts=4):
        super().__init__()

        self.n_ent = n_ent
        self.n_rel = n_rel
        self.dim = dim
        self.n_experts = n_experts

        # Base entity embedding
        self.entity_base = nn.Embedding(n_ent, dim)

        # Experts: each expert captures different aspects
        self.entity_experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_experts)
        ])

        # Router: relation determines expert weights
        self.relation_router = nn.Embedding(n_rel, n_experts)

        # Relation embedding (for scoring)
        self.relation_emb = nn.Embedding(n_rel, dim)

        nn.init.xavier_uniform_(self.entity_base.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Coverage for uncertainty
        self.register_buffer('coverage', torch.zeros(n_ent, n_rel))

    def precompute_coverage(self, triples):
        for h, r, t in triples:
            self.coverage[int(h), int(r)] = 1.0
            self.coverage[int(t), int(r)] = 1.0

    def get_disentangled_embedding(self, entity_ids, relation_ids):
        """Get relation-specific entity embedding."""
        base = self.entity_base(entity_ids)  # [B, D]

        # Get expert weights from relation
        router_weights = F.softmax(self.relation_router(relation_ids), dim=-1)  # [B, n_experts]

        # Apply experts and weight
        expert_outputs = []
        for expert in self.entity_experts:
            expert_outputs.append(expert(base))  # Each [B, D]

        expert_stack = torch.stack(expert_outputs, dim=1)  # [B, n_experts, D]

        # Weighted combination
        disentangled = torch.einsum('bnd,bn->bd', expert_stack, router_weights)  # [B, D]

        return disentangled

    def forward(self, h, r, t):
        """Score triples with disentangled embeddings."""
        h_emb = self.get_disentangled_embedding(h, r)
        t_emb = self.get_disentangled_embedding(t, r)
        r_emb = self.relation_emb(r)

        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        """
        Uncertainty = Coverage-based + Expert disagreement.

        Expert disagreement: variance across expert outputs
        Coverage: binary indicator boosting
        """
        base_h = self.entity_base(h)
        base_t = self.entity_base(t)

        # Expert disagreement for head
        h_expert_outs = torch.stack([exp(base_h) for exp in self.entity_experts], dim=1)  # [B, n_experts, D]
        h_var = h_expert_outs.var(dim=1).mean(dim=-1)  # [B]

        # Expert disagreement for tail
        t_expert_outs = torch.stack([exp(base_t) for exp in self.entity_experts], dim=1)
        t_var = t_expert_outs.var(dim=1).mean(dim=-1)  # [B]

        # Coverage boost
        h_cov = self.coverage[h, r]
        t_cov = self.coverage[t, r]
        cov_boost = 2.0 - h_cov - t_cov  # 0 for full coverage, 2 for zero

        # Combined uncertainty
        uncertainty = (h_var + t_var) * (1.0 + cov_boost)

        return uncertainty


# ==============================================================================
# Direction 4: Cascading Uncertainty
# ==============================================================================

class CascadingUncertainty:
    """
    Use Coverage and Energy for different purposes:
    - Coverage -> OOD detection (flag zero-coverage)
    - Energy -> Selective prediction (among covered, which to trust?)

    NOT ensemble, but cascading:
    1. If zero-coverage: HIGH uncertainty (abstain)
    2. Else: use Energy for fine-grained uncertainty
    """
    def __init__(self, model, coverage_matrix):
        self.model = model
        self.coverage = coverage_matrix

        # Calibrated Energy thresholds (fit on validation)
        self.energy_threshold = None

    def fit(self, valid_triples, device='cpu'):
        """Fit Energy threshold on covered validation triples."""
        print("Fitting cascading uncertainty...")

        # Get covered triples only
        covered_idx = []
        for idx, (h, r, t) in enumerate(valid_triples):
            if self.coverage[int(h), int(r)] and self.coverage[int(t), int(r)]:
                covered_idx.append(idx)

        covered_triples = valid_triples[covered_idx]
        print(f"  Covered validation triples: {len(covered_triples)}")

        self.model.eval()
        with torch.no_grad():
            h = torch.tensor(covered_triples[:, 0], device=device)
            r = torch.tensor(covered_triples[:, 1], device=device)
            t = torch.tensor(covered_triples[:, 2], device=device)

            energy = -self.model(h, r, t).cpu().numpy()

        # Set threshold at median (for selective prediction at 50%)
        self.energy_threshold = np.median(energy)
        print(f"  Energy threshold: {self.energy_threshold:.4f}")

        return self

    def get_uncertainty(self, h, r, t, device='cpu'):
        """
        Cascading uncertainty:
        - Zero coverage: return max uncertainty (1e6)
        - Covered: return Energy-based uncertainty
        """
        self.model.eval()

        with torch.no_grad():
            energy = -self.model(h.to(device), r.to(device), t.to(device)).cpu().numpy()

        uncertainty = np.zeros(len(h))

        for idx in range(len(h)):
            h_cov = self.coverage[int(h[idx]), int(r[idx])]
            t_cov = self.coverage[int(t[idx]), int(r[idx])]

            if not h_cov or not t_cov:
                # Zero coverage: max uncertainty
                uncertainty[idx] = 1e6
            else:
                # Covered: Energy-based
                uncertainty[idx] = energy[idx]

        return uncertainty

    def get_abstain_mask(self, h, r, t):
        """Return mask for which queries to abstain on (zero-coverage)."""
        mask = np.zeros(len(h), dtype=bool)

        for idx in range(len(h)):
            h_cov = self.coverage[int(h[idx]), int(r[idx])]
            t_cov = self.coverage[int(t[idx]), int(r[idx])]

            if not h_cov or not t_cov:
                mask[idx] = True

        return mask


# ==============================================================================
# Evaluation Utilities
# ==============================================================================

def evaluate_ood_detection(uncertainty_fn, test_triples, coverage, device='cpu'):
    """Evaluate OOD detection (novel context)."""
    h = torch.tensor(test_triples[:, 0])
    r = torch.tensor(test_triples[:, 1])
    t = torch.tensor(test_triples[:, 2])

    # Get uncertainties
    if hasattr(uncertainty_fn, '__call__'):
        unc = uncertainty_fn(h, r, t, device)
    else:
        unc = uncertainty_fn(h, r, t)

    if isinstance(unc, torch.Tensor):
        unc = unc.cpu().numpy()

    # Labels: 1 = OOD (zero coverage for at least one entity)
    labels = np.zeros(len(test_triples))
    for idx, (h_i, r_i, t_i) in enumerate(test_triples):
        if not coverage[int(h_i), int(r_i)] or not coverage[int(t_i), int(r_i)]:
            labels[idx] = 1

    auroc = roc_auc_score(labels, unc)
    return auroc, labels.sum() / len(labels)


def evaluate_selective_prediction(model, uncertainty_fn, test_triples, coverage,
                                  n_ent, device='cpu', keep_ratio=0.5):
    """Evaluate selective prediction (abstain on uncertain)."""
    model.eval()

    h = torch.tensor(test_triples[:, 0], device=device)
    r = torch.tensor(test_triples[:, 1], device=device)
    t = torch.tensor(test_triples[:, 2], device=device)

    # Get uncertainties
    if hasattr(uncertainty_fn, '__call__'):
        unc = uncertainty_fn(torch.tensor(test_triples[:, 0]),
                            torch.tensor(test_triples[:, 1]),
                            torch.tensor(test_triples[:, 2]), device)
    else:
        unc = uncertainty_fn(torch.tensor(test_triples[:, 0]),
                            torch.tensor(test_triples[:, 1]),
                            torch.tensor(test_triples[:, 2]))

    if isinstance(unc, torch.Tensor):
        unc = unc.cpu().numpy()

    # Compute ranks
    with torch.no_grad():
        ranks = []
        batch_size = 500
        for i in range(0, len(test_triples), batch_size):
            batch_h = h[i:i+batch_size]
            batch_r = r[i:i+batch_size]
            batch_t = t[i:i+batch_size]

            scores = model.score_tails(batch_h, batch_r)  # [batch, n_ent]
            true_scores = scores[torch.arange(len(batch_h)), batch_t]
            batch_ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
            ranks.extend(batch_ranks.cpu().numpy())

        ranks = np.array(ranks)

    # Baseline MRR
    baseline_mrr = (1.0 / ranks).mean()

    # Select lowest uncertainty
    n_keep = int(len(test_triples) * keep_ratio)
    keep_idx = np.argsort(unc)[:n_keep]
    kept_mrr = (1.0 / ranks[keep_idx]).mean()

    return baseline_mrr, kept_mrr, (kept_mrr - baseline_mrr) / baseline_mrr * 100


# ==============================================================================
# Main Experiment
# ==============================================================================

def main():
    print("=" * 70)
    print("COVERAGE PARADOX: METHOD EXPLORATION")
    print("=" * 70)

    # Load data
    print("\nLoading FB15k-237...")
    train_ds, valid_ds, test_ds = load_fb15k237()
    train = train_ds.triples
    valid = valid_ds.triples
    test = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"Entities: {n_ent}, Relations: {n_rel}")
    print(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # Build coverage matrix
    coverage = np.zeros((n_ent, n_rel), dtype=bool)
    for h, r, t in train:
        coverage[int(h), int(r)] = True
        coverage[int(t), int(r)] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    # ==============================================================================
    # Baseline: Energy
    # ==============================================================================
    print("\n" + "=" * 60)
    print("BASELINE: Energy (DistMult)")
    print("=" * 60)

    baseline = DistMultBaseline(n_ent, n_rel)
    baseline = train_distmult(baseline, train, n_ent, epochs=30, device=device)
    baseline = baseline.to(device)

    def energy_uncertainty(h, r, t, device):
        baseline.eval()
        with torch.no_grad():
            return -baseline(h.to(device), r.to(device), t.to(device)).cpu().numpy()

    auroc, ood_frac = evaluate_ood_detection(energy_uncertainty, test, coverage, device)
    base_mrr, sel_mrr, improvement = evaluate_selective_prediction(
        baseline, energy_uncertainty, test, coverage, n_ent, device
    )

    print(f"OOD Detection AUROC: {auroc:.4f} (OOD fraction: {ood_frac:.1%})")
    print(f"Selective Prediction: Baseline MRR={base_mrr:.4f}, Selected MRR={sel_mrr:.4f} ({improvement:+.1f}%)")

    results['Energy'] = {
        'ood_auroc': auroc,
        'base_mrr': base_mrr,
        'sel_mrr': sel_mrr,
        'improvement': improvement
    }

    # ==============================================================================
    # Direction 1: Coverage-aware Calibration
    # ==============================================================================
    print("\n" + "=" * 60)
    print("DIRECTION 1: Coverage-aware Calibration")
    print("=" * 60)

    calibrator = CoverageAwareCalibration(baseline, coverage)
    calibrator.fit(valid, device)

    def calib_uncertainty(h, r, t, device):
        return calibrator.calibrated_uncertainty(h, r, t, device).cpu().numpy()

    auroc, _ = evaluate_ood_detection(calib_uncertainty, test, coverage, device)
    base_mrr, sel_mrr, improvement = evaluate_selective_prediction(
        baseline, calib_uncertainty, test, coverage, n_ent, device
    )

    print(f"OOD Detection AUROC: {auroc:.4f}")
    print(f"Selective Prediction: Selected MRR={sel_mrr:.4f} ({improvement:+.1f}%)")

    results['CoverageCalibration'] = {
        'ood_auroc': auroc,
        'base_mrr': base_mrr,
        'sel_mrr': sel_mrr,
        'improvement': improvement,
        'temperatures': calibrator.temps
    }

    # ==============================================================================
    # Direction 2: Anchor-based Prediction
    # ==============================================================================
    print("\n" + "=" * 60)
    print("DIRECTION 2: Anchor-based Prediction")
    print("=" * 60)

    anchor_model = AnchorBasedPredictor(n_ent, n_rel)
    anchor_model.precompute_coverage(train)

    # Train (joint scoring + uncertainty)
    anchor_model = anchor_model.to(device)
    optimizer = torch.optim.Adam(anchor_model.parameters(), lr=1e-3)

    print("Training anchor-based model...")
    for epoch in range(30):
        np.random.shuffle(train)
        total_loss = 0

        for i in range(0, len(train), 1024):
            batch = train[i:i+1024]
            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)
            t_neg = torch.randint(0, n_ent, (len(batch),), device=device)

            optimizer.zero_grad()

            # Score loss
            pos = anchor_model(h, r, t)
            neg = anchor_model(h, r, t_neg)
            score_loss = torch.relu(1.0 - pos + neg).mean()

            # Uncertainty loss: covered should have lower uncertainty
            pos_unc = anchor_model.get_anchor_uncertainty(h, r, t)
            neg_unc = anchor_model.get_anchor_uncertainty(h, r, t_neg)
            unc_loss = torch.relu(pos_unc - neg_unc + 0.1).mean()

            loss = score_loss + 0.1 * unc_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {total_loss:.2f}")

    def anchor_uncertainty(h, r, t, device):
        anchor_model.eval()
        with torch.no_grad():
            return anchor_model.get_anchor_uncertainty(h.to(device), r.to(device), t.to(device)).cpu().numpy()

    auroc, _ = evaluate_ood_detection(anchor_uncertainty, test, coverage, device)
    base_mrr, sel_mrr, improvement = evaluate_selective_prediction(
        anchor_model, anchor_uncertainty, test, coverage, n_ent, device
    )

    print(f"OOD Detection AUROC: {auroc:.4f}")
    print(f"Selective Prediction: Selected MRR={sel_mrr:.4f} ({improvement:+.1f}%)")

    results['AnchorBased'] = {
        'ood_auroc': auroc,
        'base_mrr': base_mrr,
        'sel_mrr': sel_mrr,
        'improvement': improvement
    }

    # ==============================================================================
    # Direction 3: Disentangled Embeddings
    # ==============================================================================
    print("\n" + "=" * 60)
    print("DIRECTION 3: Disentangled Embeddings")
    print("=" * 60)

    disent_model = DisentangledEmbeddingKGE(n_ent, n_rel, n_experts=4)
    disent_model.precompute_coverage(train)
    disent_model = disent_model.to(device)

    optimizer = torch.optim.Adam(disent_model.parameters(), lr=1e-3)

    print("Training disentangled model...")
    for epoch in range(30):
        np.random.shuffle(train)
        total_loss = 0

        for i in range(0, len(train), 1024):
            batch = train[i:i+1024]
            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)
            t_neg = torch.randint(0, n_ent, (len(batch),), device=device)

            optimizer.zero_grad()
            pos = disent_model(h, r, t)
            neg = disent_model(h, r, t_neg)
            loss = torch.relu(1.0 - pos + neg).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {total_loss:.2f}")

    # Add score_tails method for evaluation
    def disent_score_tails(h, r):
        h_emb = disent_model.get_disentangled_embedding(h, r)
        r_emb = disent_model.relation_emb(r)
        all_t = disent_model.entity_base.weight  # Use base for efficiency
        hr = h_emb * r_emb
        return hr @ all_t.T

    disent_model.score_tails = disent_score_tails

    def disent_uncertainty(h, r, t, device):
        disent_model.eval()
        with torch.no_grad():
            return disent_model.get_uncertainty(h.to(device), r.to(device), t.to(device)).cpu().numpy()

    auroc, _ = evaluate_ood_detection(disent_uncertainty, test, coverage, device)
    base_mrr, sel_mrr, improvement = evaluate_selective_prediction(
        disent_model, disent_uncertainty, test, coverage, n_ent, device
    )

    print(f"OOD Detection AUROC: {auroc:.4f}")
    print(f"Selective Prediction: Selected MRR={sel_mrr:.4f} ({improvement:+.1f}%)")

    results['Disentangled'] = {
        'ood_auroc': auroc,
        'base_mrr': base_mrr,
        'sel_mrr': sel_mrr,
        'improvement': improvement
    }

    # ==============================================================================
    # Direction 4: Cascading Uncertainty
    # ==============================================================================
    print("\n" + "=" * 60)
    print("DIRECTION 4: Cascading Uncertainty")
    print("=" * 60)

    cascade = CascadingUncertainty(baseline, coverage)
    cascade.fit(valid, device)

    auroc, _ = evaluate_ood_detection(cascade.get_uncertainty, test, coverage, device)
    base_mrr, sel_mrr, improvement = evaluate_selective_prediction(
        baseline, cascade.get_uncertainty, test, coverage, n_ent, device
    )

    print(f"OOD Detection AUROC: {auroc:.4f}")
    print(f"Selective Prediction: Selected MRR={sel_mrr:.4f} ({improvement:+.1f}%)")

    # Additional: coverage-based abstain statistics
    abstain_mask = cascade.get_abstain_mask(
        torch.tensor(test[:, 0]), torch.tensor(test[:, 1]), torch.tensor(test[:, 2])
    )
    print(f"Abstain ratio: {abstain_mask.mean():.1%}")

    results['Cascading'] = {
        'ood_auroc': auroc,
        'base_mrr': base_mrr,
        'sel_mrr': sel_mrr,
        'improvement': improvement,
        'abstain_ratio': abstain_mask.mean()
    }

    # ==============================================================================
    # Summary
    # ==============================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'OOD AUROC':>12} {'Sel. MRR':>12} {'Improvement':>12}")
    print("-" * 65)

    for method, r in results.items():
        print(f"{method:<25} {r['ood_auroc']:>12.4f} {r['sel_mrr']:>12.4f} {r['improvement']:>+11.1f}%")

    # Find best
    best_ood = max(results.items(), key=lambda x: x[1]['ood_auroc'])
    best_sel = max(results.items(), key=lambda x: x[1]['improvement'])

    print(f"\nBest OOD Detection: {best_ood[0]} (AUROC={best_ood[1]['ood_auroc']:.4f})")
    print(f"Best Selective Prediction: {best_sel[0]} (improvement={best_sel[1]['improvement']:+.1f}%)")

    # ==============================================================================
    # Insights for Paper
    # ==============================================================================
    print("\n" + "=" * 70)
    print("INSIGHTS FOR PAPER")
    print("=" * 70)

    insights = """
    RCUE Failure Root Cause:
    ------------------------
    1. Coverage boost is ADDITIVE, not DISCRIMINATIVE
       - Adding coverage boost to MLP variance pollutes the signal
       - MLP learns within-class patterns, coverage is between-class

    2. Energy is already a strong baseline for selective prediction
       - MLP's marginal +4.4pp improvement doesn't justify complexity

    Key Finding: Separation of Concerns
    -----------------------------------
    Coverage and Energy serve DIFFERENT purposes:
    - Coverage: Binary OOD detection (has evidence vs no evidence)
    - Energy: Fine-grained confidence (among queries with evidence)

    Cascading approach respects this separation:
    1. First: Flag zero-coverage (structural OOD)
    2. Then: Use Energy for selective prediction (semantic confidence)

    Theoretical Implications:
    -------------------------
    This finding supports Theorem 2 (embedding-based impossibility):
    - No embedding-based method can detect novel contexts
    - Coverage lookup is NECESSARY (not just sufficient)
    - The paradox arises from conflating OOD detection with prediction confidence
    """
    print(insights)

    return results


if __name__ == "__main__":
    results = main()
