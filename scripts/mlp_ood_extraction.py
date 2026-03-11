#!/usr/bin/env python3
"""
MLP OOD Extraction Experiment

Key Question: Can a learned classifier extract OOD signal from embeddings that
existing methods miss?

Setup:
1. Load trained DistMult embeddings from FB15k-237
2. Create OOD labels: novel-context = 1, ID = 0
3. Train MLP: OOD(h,r,t) = MLP([phi(h); psi(r); phi(t)])
4. Train with BCE loss on 80% of test triples, evaluate on 20%
5. Compare to:
   - Coverage lookup (should be 1.0 by construction)
   - Random baseline (0.5)
   - Energy scoring (existing method)

Interpretation:
- If MLP achieves high AUROC (>0.8): coverage info IS extractable but existing methods don't try
- If MLP achieves low AUROC (~0.5): coverage info is NOT extractable from embeddings alone

This experiment tests Theorem 2's claim that relation-agnostic methods CANNOT extract
coverage information from embeddings.
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
import time
from datetime import datetime

from src.data.loaders import load_fb15k237


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# DistMult Model (for training embeddings)
# ============================================================

class DistMult(nn.Module):
    """DistMult model for training embeddings."""
    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        """DistMult score: <h, r, t> = sum(h * r * t)"""
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_energy_uncertainty(self, h, r, t):
        """Energy-based uncertainty: -score (lower score = higher uncertainty)"""
        return -self.forward(h, r, t)


# ============================================================
# MLP OOD Classifier
# ============================================================

class MLPOODClassifier(nn.Module):
    """
    MLP classifier that takes concatenated embeddings and predicts OOD probability.

    Input: [phi(h); psi(r); phi(t)] - concatenation of head, relation, tail embeddings
    Output: P(OOD | h, r, t)
    """
    def __init__(self, embedding_dim, hidden_dims=[256, 128]):
        super().__init__()
        input_dim = embedding_dim * 3  # h, r, t concatenated

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, h_emb, r_emb, t_emb):
        """
        Args:
            h_emb: Head embeddings [batch, dim]
            r_emb: Relation embeddings [batch, dim]
            t_emb: Tail embeddings [batch, dim]
        Returns:
            OOD logits [batch]
        """
        x = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        return self.network(x).squeeze(-1)


# ============================================================
# Training Functions
# ============================================================

def train_distmult(model, triples, device, epochs=50, lr=0.001):
    """Train DistMult embeddings."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    heads = torch.tensor(triples[:, 0])
    rels = torch.tensor(triples[:, 1])
    tails = torch.tensor(triples[:, 2])

    loader = DataLoader(TensorDataset(heads, rels, tails), batch_size=1024, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            # Positive scores
            pos_scores = model(h, r, t)

            # Negative sampling (corrupt tails)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            # BCE loss
            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )

            # L2 regularization
            reg = 1e-5 * (model.entity_emb.weight.norm(2) + model.relation_emb.weight.norm(2))
            loss = loss + reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    DistMult Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    return model


def train_mlp_classifier(mlp, distmult, train_h, train_r, train_t, train_labels,
                         device, epochs=50, lr=0.001):
    """Train MLP OOD classifier on frozen DistMult embeddings."""
    mlp = mlp.to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)

    # Get frozen embeddings
    distmult.eval()
    with torch.no_grad():
        h_emb = distmult.entity_emb(train_h.to(device)).detach()
        r_emb = distmult.relation_emb(train_r.to(device)).detach()
        t_emb = distmult.entity_emb(train_t.to(device)).detach()

    labels = train_labels.float().to(device)

    # Create balanced sampler if needed
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos

    print(f"    Training MLP: {int(n_pos)} OOD, {int(n_neg)} ID samples")

    # Simple training loop
    dataset = TensorDataset(h_emb, r_emb, t_emb, labels)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    for epoch in range(epochs):
        mlp.train()
        total_loss = 0
        for batch_h, batch_r, batch_t, batch_y in loader:
            logits = mlp(batch_h, batch_r, batch_t)

            # Weighted BCE to handle class imbalance
            pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)]).to(device)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    MLP Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    return mlp


# ============================================================
# Evaluation Functions
# ============================================================

def compute_coverage_labels(triples, train_triples, num_entities, num_relations):
    """
    Compute novel-context labels for test triples.

    Novel-context = 1 if either (h, r) or (t, r) is unseen in training.
    ID = 0 if both (h, r) and (t, r) are seen in training.
    """
    # Build coverage matrix from training
    coverage = np.zeros((num_entities, num_relations), dtype=bool)
    for h, r, t in train_triples:
        coverage[h, r] = True
        coverage[t, r] = True

    # Label each test triple
    labels = []
    for h, r, t in triples:
        h_covered = coverage[h, r]
        t_covered = coverage[t, r]
        is_novel_context = not (h_covered and t_covered)
        labels.append(int(is_novel_context))

    return np.array(labels), coverage


def evaluate_methods(distmult, mlp, test_h, test_r, test_t, test_labels, coverage, device):
    """
    Evaluate all OOD detection methods:
    1. Coverage lookup (oracle)
    2. Energy scoring (existing baseline)
    3. MLP classifier (our test)
    4. Random baseline
    """
    results = {}

    distmult.eval()
    mlp.eval()

    test_h_dev = test_h.to(device)
    test_r_dev = test_r.to(device)
    test_t_dev = test_t.to(device)

    with torch.no_grad():
        # 1. Coverage lookup (should be ~1.0 by construction)
        # Coverage score = 1 - min(cov(h,r), cov(t,r))
        # Higher score = more OOD
        coverage_scores = []
        for i in range(len(test_h)):
            h, r, t = test_h[i].item(), test_r[i].item(), test_t[i].item()
            h_cov = coverage[h, r]
            t_cov = coverage[t, r]
            # OOD score: 1 if either is uncovered, 0 if both covered
            ood_score = 1.0 if not (h_cov and t_cov) else 0.0
            coverage_scores.append(ood_score)
        coverage_scores = np.array(coverage_scores)

        # Coverage AUROC should be 1.0 since labels are defined by coverage
        results['Coverage (oracle)'] = roc_auc_score(test_labels, coverage_scores)

        # 2. Energy scoring: -score as OOD score (lower score = more uncertain)
        energy_scores = distmult.get_energy_uncertainty(test_h_dev, test_r_dev, test_t_dev).cpu().numpy()
        results['Energy'] = roc_auc_score(test_labels, energy_scores)

        # 3. MLP classifier
        h_emb = distmult.entity_emb(test_h_dev)
        r_emb = distmult.relation_emb(test_r_dev)
        t_emb = distmult.entity_emb(test_t_dev)
        mlp_logits = mlp(h_emb, r_emb, t_emb).cpu().numpy()
        mlp_probs = 1 / (1 + np.exp(-mlp_logits))  # sigmoid
        results['MLP'] = roc_auc_score(test_labels, mlp_probs)

        # 4. Random baseline
        random_scores = np.random.rand(len(test_labels))
        results['Random'] = roc_auc_score(test_labels, random_scores)

    return results


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(seed=42):
    """Run the MLP OOD extraction experiment."""

    print("=" * 70)
    print("MLP OOD EXTRACTION EXPERIMENT")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {seed}")

    device = setup_device()
    print(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load FB15k-237
    print("\n[1] Loading FB15k-237...")
    train_ds, _, test_ds = load_fb15k237()
    train_triples = train_ds.triples
    test_triples = test_ds.triples
    n_ent = train_ds.num_entities
    n_rel = train_ds.num_relations

    print(f"    Entities: {n_ent}, Relations: {n_rel}")
    print(f"    Train triples: {len(train_triples)}")
    print(f"    Test triples: {len(test_triples)}")

    # Compute OOD labels for test set
    print("\n[2] Computing OOD labels (novel-context)...")
    test_labels, coverage = compute_coverage_labels(
        test_triples, train_triples, n_ent, n_rel
    )
    n_ood = test_labels.sum()
    n_id = len(test_labels) - n_ood
    print(f"    Novel-context (OOD): {n_ood} ({100*n_ood/len(test_labels):.1f}%)")
    print(f"    In-distribution (ID): {n_id} ({100*n_id/len(test_labels):.1f}%)")

    # Split test set: 80% train MLP, 20% evaluate
    print("\n[3] Splitting test set (80% train, 20% eval)...")
    n_test = len(test_triples)
    indices = np.random.permutation(n_test)
    split_idx = int(0.8 * n_test)

    train_idx = indices[:split_idx]
    eval_idx = indices[split_idx:]

    mlp_train_triples = test_triples[train_idx]
    mlp_train_labels = test_labels[train_idx]

    eval_triples = test_triples[eval_idx]
    eval_labels = test_labels[eval_idx]

    print(f"    MLP training: {len(train_idx)} triples")
    print(f"    Evaluation: {len(eval_idx)} triples")
    print(f"    Eval OOD: {eval_labels.sum()} ({100*eval_labels.mean():.1f}%)")

    # Train DistMult on training KG
    print("\n[4] Training DistMult embeddings...")
    distmult = DistMult(n_ent, n_rel, dim=100)
    distmult = train_distmult(distmult, train_triples, device, epochs=50)

    # Prepare tensors
    mlp_train_h = torch.tensor(mlp_train_triples[:, 0])
    mlp_train_r = torch.tensor(mlp_train_triples[:, 1])
    mlp_train_t = torch.tensor(mlp_train_triples[:, 2])
    mlp_train_y = torch.tensor(mlp_train_labels)

    eval_h = torch.tensor(eval_triples[:, 0])
    eval_r = torch.tensor(eval_triples[:, 1])
    eval_t = torch.tensor(eval_triples[:, 2])

    # Train MLP classifier on frozen embeddings
    print("\n[5] Training MLP OOD classifier on frozen embeddings...")
    mlp = MLPOODClassifier(embedding_dim=100, hidden_dims=[256, 128, 64])
    mlp = train_mlp_classifier(
        mlp, distmult, mlp_train_h, mlp_train_r, mlp_train_t, mlp_train_y,
        device, epochs=100
    )

    # Evaluate all methods
    print("\n[6] Evaluating OOD detection methods...")
    results = evaluate_methods(
        distmult, mlp, eval_h, eval_r, eval_t, eval_labels, coverage, device
    )

    return results, {
        'n_entities': n_ent,
        'n_relations': n_rel,
        'n_train': len(train_triples),
        'n_test': len(test_triples),
        'n_eval': len(eval_idx),
        'ood_rate': eval_labels.mean(),
    }


def run_multiseed(seeds=[42, 123, 456]):
    """Run experiment with multiple seeds and aggregate results."""
    all_results = []

    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")
        results, meta = run_experiment(seed=seed)
        all_results.append(results)

    # Aggregate
    methods = list(all_results[0].keys())
    mean_results = {}
    std_results = {}

    for method in methods:
        values = [r[method] for r in all_results]
        mean_results[method] = np.mean(values)
        std_results[method] = np.std(values)

    return mean_results, std_results, meta


def main():
    """Main entry point."""

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "mlp_extraction_experiment.txt"

    # Run with multiple seeds
    seeds = [42, 123, 456]
    mean_results, std_results, meta = run_multiseed(seeds)

    # Format output
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("MLP OOD EXTRACTION EXPERIMENT RESULTS")
    output_lines.append("=" * 70)
    output_lines.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Seeds: {seeds}")
    output_lines.append(f"\nDataset: FB15k-237")
    output_lines.append(f"  Entities: {meta['n_entities']}")
    output_lines.append(f"  Relations: {meta['n_relations']}")
    output_lines.append(f"  Train triples: {meta['n_train']}")
    output_lines.append(f"  Eval triples: {meta['n_eval']}")
    output_lines.append(f"  OOD rate in eval: {100*meta['ood_rate']:.1f}%")

    output_lines.append("\n" + "-" * 70)
    output_lines.append("AUROC Results (Novel-Context Detection)")
    output_lines.append("-" * 70)
    output_lines.append(f"{'Method':<25} {'AUROC':>15} {'Std':>10}")
    output_lines.append("-" * 50)

    for method in ['Coverage (oracle)', 'MLP', 'Energy', 'Random']:
        auroc = mean_results[method]
        std = std_results[method]
        output_lines.append(f"{method:<25} {auroc:>15.4f} {std:>10.4f}")

    output_lines.append("\n" + "-" * 70)
    output_lines.append("INTERPRETATION")
    output_lines.append("-" * 70)

    mlp_auroc = mean_results['MLP']
    energy_auroc = mean_results['Energy']

    output_lines.append(f"\nMLP AUROC: {mlp_auroc:.4f}")
    output_lines.append(f"Energy AUROC: {energy_auroc:.4f}")
    output_lines.append(f"Random baseline: 0.5000")

    if mlp_auroc > 0.8:
        output_lines.append("\n*** FINDING: MLP achieves high AUROC (>0.8)")
        output_lines.append("    Coverage info IS extractable from embeddings,")
        output_lines.append("    but existing methods (like Energy) don't try to extract it.")
        output_lines.append("    This suggests the architecture choice matters, not the embeddings.")
    elif mlp_auroc > 0.6:
        output_lines.append("\n*** FINDING: MLP achieves moderate AUROC (0.6-0.8)")
        output_lines.append("    Some coverage signal exists in embeddings,")
        output_lines.append("    but extraction is imperfect.")
        output_lines.append("    Explicit coverage tracking remains necessary.")
    else:
        output_lines.append("\n*** FINDING: MLP achieves low AUROC (~0.5)")
        output_lines.append("    Coverage info is NOT extractable from embeddings alone.")
        output_lines.append("    This confirms Theorem 2: embedding-based methods")
        output_lines.append("    CANNOT distinguish novel-context from ID queries.")
        output_lines.append("    Only explicit coverage tracking can solve this.")

    output_lines.append("\n" + "-" * 70)
    output_lines.append("KEY INSIGHT")
    output_lines.append("-" * 70)

    if mlp_auroc < 0.6:
        output_lines.append("\nThe MLP had access to:")
        output_lines.append("  - Full entity embeddings phi(h), phi(t)")
        output_lines.append("  - Full relation embedding psi(r)")
        output_lines.append("  - Explicit supervision on OOD labels")
        output_lines.append("  - 80% of test data for training")
        output_lines.append("")
        output_lines.append("Despite all this, it CANNOT learn to detect novel contexts.")
        output_lines.append("This proves the information simply isn't there.")
        output_lines.append("")
        output_lines.append("Embeddings capture SEMANTIC similarity, not COVERAGE.")
        output_lines.append("Coverage must be tracked explicitly (hash table / Bloom filter).")

    # Print to console and save
    output_text = "\n".join(output_lines)
    print("\n" + output_text)

    with open(output_file, 'w') as f:
        f.write(output_text)

    print(f"\n\nResults saved to: {output_file}")

    return mean_results


if __name__ == "__main__":
    main()
