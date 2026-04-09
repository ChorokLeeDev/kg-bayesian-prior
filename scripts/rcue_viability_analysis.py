"""
RCUE Viability Analysis: Does MLP provide value beyond Coverage lookup?

Key questions:
1. Within-class discrimination: Among ID triples (coverage=1),
   can MLP distinguish easy vs hard predictions?
2. Selective prediction: Does RCUE beat Coverage+Energy combination?
3. Calibration: Does MLP provide meaningful uncertainty gradations?

If MLP contribution is marginal → Position paper (coverage blind spot)
If MLP contribution is significant → Full paper (RCUE method)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import load_fb15k237
from src.models.relation_conditioned.rcue import RCUE
from sklearn.metrics import roc_auc_score


def evaluate_within_class_discrimination(model, test_triples, train_triples):
    """
    Key experiment: Among ID triples (coverage=1 for both h and t),
    can MLP variance predict which are easy vs hard?

    Ground truth: Use ranking performance (MRR) as "hardness"
    """
    model.eval()
    device = next(model.parameters()).device

    # Split test triples by coverage
    h_cov = model.coverage[test_triples[:, 0], test_triples[:, 1]].cpu().numpy()
    t_cov = model.coverage[test_triples[:, 2], test_triples[:, 1]].cpu().numpy()

    # ID = both covered, OOD = at least one not covered
    id_mask = (h_cov == 1) & (t_cov == 1)
    ood_mask = ~id_mask

    print(f"Test triples: {len(test_triples)}")
    print(f"  ID (both covered): {id_mask.sum()} ({id_mask.mean()*100:.1f}%)")
    print(f"  OOD (novel context): {ood_mask.sum()} ({ood_mask.mean()*100:.1f}%)")

    # For ID triples, compute MLP variance and ranking
    id_triples = test_triples[id_mask]

    with torch.no_grad():
        h = torch.tensor(id_triples[:, 0], device=device)
        r = torch.tensor(id_triples[:, 1], device=device)
        t = torch.tensor(id_triples[:, 2], device=device)

        # MLP variance (without coverage boost since all covered)
        mlp_variance = model.get_uncertainty(h, r, t).cpu().numpy()

        # Compute ranks for each triple
        ranks = []
        batch_size = 500
        for i in range(0, len(id_triples), batch_size):
            batch_h = h[i:i+batch_size]
            batch_r = r[i:i+batch_size]
            batch_t = t[i:i+batch_size]

            # Score all tails
            scores = model.score_tails(batch_h, batch_r)  # [batch, num_entities]

            # Get rank of true tail
            true_scores = scores[torch.arange(len(batch_h)), batch_t]
            # Rank = number of entities with higher score + 1
            batch_ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
            ranks.extend(batch_ranks.cpu().numpy())

        ranks = np.array(ranks)

    # MRR for ID triples
    mrr_id = (1.0 / ranks).mean()
    print(f"\nID triples MRR: {mrr_id:.4f}")

    # Can MLP variance predict which triples are hard?
    # "Hard" = rank > median rank
    median_rank = np.median(ranks)
    is_hard = ranks > median_rank

    # AUROC: can MLP variance predict "hard"?
    auroc_mlp = roc_auc_score(is_hard, mlp_variance)
    print(f"\nMLP variance predicting hard triples:")
    print(f"  AUROC: {auroc_mlp:.4f}")
    print(f"  (0.5 = random, >0.6 = useful signal)")

    # Compare with Energy baseline
    with torch.no_grad():
        energy = -model(h, r, t).cpu().numpy()  # Negative score = uncertainty

    auroc_energy = roc_auc_score(is_hard, energy)
    print(f"\nEnergy predicting hard triples:")
    print(f"  AUROC: {auroc_energy:.4f}")

    print(f"\nMLP advantage: {(auroc_mlp - auroc_energy)*100:+.1f}pp")

    return {
        'id_count': id_mask.sum(),
        'ood_count': ood_mask.sum(),
        'mrr_id': mrr_id,
        'auroc_mlp_hard': auroc_mlp,
        'auroc_energy_hard': auroc_energy,
        'mlp_advantage': auroc_mlp - auroc_energy
    }


def evaluate_selective_prediction(model, test_triples, coverage_threshold=0.5):
    """
    Selective prediction: abstain on uncertain queries.

    Compare:
    1. RCUE uncertainty
    2. Coverage + Energy (simple combination)
    3. Coverage only (binary)
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        h = torch.tensor(test_triples[:, 0], device=device)
        r = torch.tensor(test_triples[:, 1], device=device)
        t = torch.tensor(test_triples[:, 2], device=device)

        # RCUE uncertainty
        rcue_unc = model.get_uncertainty(h, r, t).cpu().numpy()

        # Energy uncertainty
        energy_unc = -model(h, r, t).cpu().numpy()

        # Coverage (binary)
        h_cov = model.coverage[h, r].cpu().numpy()
        t_cov = model.coverage[t, r].cpu().numpy()
        coverage_unc = 2 - (h_cov + t_cov)  # 0=both covered, 2=neither covered

        # Combined: Coverage + Energy (normalized)
        energy_norm = (energy_unc - energy_unc.min()) / (energy_unc.max() - energy_unc.min() + 1e-8)
        combined_unc = coverage_unc + energy_norm

        # Compute ranks
        ranks = []
        batch_size = 500
        for i in range(0, len(test_triples), batch_size):
            batch_h = h[i:i+batch_size]
            batch_r = r[i:i+batch_size]
            batch_t = t[i:i+batch_size]

            scores = model.score_tails(batch_h, batch_r)
            true_scores = scores[torch.arange(len(batch_h)), batch_t]
            batch_ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1
            ranks.extend(batch_ranks.cpu().numpy())

        ranks = np.array(ranks)

    print("\n" + "="*60)
    print("SELECTIVE PREDICTION (50% abstain)")
    print("="*60)

    # For each method, select top 50% by uncertainty (abstain on high uncertainty)
    keep_ratio = 0.5
    n_keep = int(len(test_triples) * keep_ratio)

    methods = {
        'RCUE': rcue_unc,
        'Coverage+Energy': combined_unc,
        'Energy only': energy_norm,
        'Coverage only': coverage_unc
    }

    results = {}
    baseline_mrr = (1.0 / ranks).mean()
    print(f"\nBaseline MRR (all triples): {baseline_mrr:.4f}")
    print()

    for name, unc in methods.items():
        # Keep lowest uncertainty triples
        keep_idx = np.argsort(unc)[:n_keep]
        kept_ranks = ranks[keep_idx]
        mrr = (1.0 / kept_ranks).mean()
        improvement = (mrr - baseline_mrr) / baseline_mrr * 100

        print(f"{name:20s}: MRR = {mrr:.4f} ({improvement:+.1f}%)")
        results[name] = {'mrr': mrr, 'improvement': improvement}

    return results


def evaluate_ood_detection(model, test_triples):
    """Standard OOD detection AUROC."""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        h = torch.tensor(test_triples[:, 0], device=device)
        r = torch.tensor(test_triples[:, 1], device=device)
        t = torch.tensor(test_triples[:, 2], device=device)

        # Coverage labels
        h_cov = model.coverage[h, r].cpu().numpy()
        t_cov = model.coverage[t, r].cpu().numpy()
        is_ood = (h_cov == 0) | (t_cov == 0)

        # RCUE uncertainty
        rcue_unc = model.get_uncertainty(h, r, t).cpu().numpy()

        # Energy
        energy_unc = -model(h, r, t).cpu().numpy()

    auroc_rcue = roc_auc_score(is_ood, rcue_unc)
    auroc_energy = roc_auc_score(is_ood, energy_unc)

    print("\n" + "="*60)
    print("OOD DETECTION (Novel Context)")
    print("="*60)
    print(f"OOD fraction: {is_ood.mean()*100:.1f}%")
    print(f"RCUE AUROC:   {auroc_rcue:.4f}")
    print(f"Energy AUROC: {auroc_energy:.4f}")
    print(f"Improvement:  {(auroc_rcue - auroc_energy)*100:+.1f}pp")

    return {'auroc_rcue': auroc_rcue, 'auroc_energy': auroc_energy}


def main():
    print("="*60)
    print("RCUE VIABILITY ANALYSIS")
    print("="*60)
    print("\nKey question: Does MLP provide value beyond Coverage lookup?")
    print()

    # Load data
    print("Loading FB15k-237...")
    train_ds, valid_ds, test_ds = load_fb15k237()
    train = train_ds.triples
    test = test_ds.triples
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations
    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train: {len(train)}, Test: {len(test)}")

    # Initialize and train RCUE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = RCUE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=100,
        hidden_dim=64,
        use_coverage=True
    ).to(device)

    # Precompute coverage
    model.precompute_coverage(train)

    # Training
    print("\nTraining RCUE...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    margin = 1.0
    batch_size = 1024
    epochs = 30

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        # Shuffle training data
        perm = np.random.permutation(len(train))

        for i in range(0, len(train), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = train[batch_idx]

            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)

            # Negative sampling
            t_neg = torch.randint(0, num_entities, (len(batch),), device=device)

            # Scores
            pos_scores = model(h, r, t)
            neg_scores = model(h, r, t_neg)

            # Margin loss
            score_loss = torch.relu(margin - pos_scores + neg_scores).mean()

            # Uncertainty loss: higher uncertainty for negatives
            pos_unc = model.get_uncertainty(h, r, t)
            neg_unc = model.get_uncertainty(h, r, t_neg)
            unc_loss = torch.relu(pos_unc - neg_unc + 0.1).mean()

            loss = score_loss + 0.1 * unc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}: Loss = {total_loss/n_batches:.4f}")

    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    # 1. OOD Detection
    ood_results = evaluate_ood_detection(model, test)

    # 2. Within-class discrimination
    within_results = evaluate_within_class_discrimination(model, test, train)

    # 3. Selective prediction
    selective_results = evaluate_selective_prediction(model, test)

    # Summary
    print("\n" + "="*60)
    print("VIABILITY SUMMARY")
    print("="*60)

    mlp_useful = within_results['mlp_advantage'] > 0.05  # >5pp advantage
    rcue_beats_combo = selective_results['RCUE']['mrr'] > selective_results['Coverage+Energy']['mrr']

    print(f"\n1. MLP within-class advantage: {within_results['mlp_advantage']*100:+.1f}pp")
    print(f"   {'[PASS]' if mlp_useful else '[FAIL]'} MLP provides useful signal beyond coverage")

    print(f"\n2. RCUE vs Coverage+Energy selective prediction:")
    print(f"   RCUE MRR:           {selective_results['RCUE']['mrr']:.4f}")
    print(f"   Coverage+Energy MRR: {selective_results['Coverage+Energy']['mrr']:.4f}")
    print(f"   {'[PASS]' if rcue_beats_combo else '[FAIL]'} RCUE beats simple combination")

    print("\n" + "-"*60)
    if mlp_useful and rcue_beats_combo:
        print("RECOMMENDATION: Full paper (RCUE method)")
        print("MLP contributes meaningful signal beyond coverage lookup.")
    elif mlp_useful or rcue_beats_combo:
        print("RECOMMENDATION: Marginal - consider both options")
        print("MLP has some value but may not justify full method paper.")
    else:
        print("RECOMMENDATION: Position paper")
        print("MLP contribution is marginal; focus on coverage blind spot insight.")

    return {
        'ood': ood_results,
        'within_class': within_results,
        'selective': selective_results
    }


if __name__ == "__main__":
    results = main()
