#!/usr/bin/env python3
"""
Predictive Uncertainty Experiments

Validates the hypotheses from docs/predictive_uncertainty.md:
- H1: Adversarial OOD has higher prediction entropy than normal OOD
- H2: Adversarial OOD has lower prediction margin than normal OOD
- H3: Predictive uncertainty is orthogonal to GP and Coverage

Also runs ablation study comparing:
- CAGP (2-component: semantic + structural)
- Predictive CAGP (3-component: + predictive)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
import json
from collections import defaultdict

from src.data.loaders import load_fb15k237, load_wn18rr
from src.models.predictive_cagp import PredictiveCAGP


def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_model(model, train_triples, device, epochs=30, batch_size=1024, lr=0.001):
    """Train a model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    heads = torch.tensor(train_triples[:, 0])
    relations = torch.tensor(train_triples[:, 1])
    tails = torch.tensor(train_triples[:, 2])

    dataset = TensorDataset(heads, relations, tails)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_h, batch_r, batch_t in loader:
            batch_h = batch_h.to(device)
            batch_r = batch_r.to(device)
            batch_t = batch_t.to(device)

            pos_scores = model(batch_h, batch_r, batch_t, use_sampling=True)
            neg_t = torch.randint(0, model.num_entities, batch_t.shape, device=device)
            neg_scores = model(batch_h, batch_r, neg_t, use_sampling=True)

            loss = criterion(pos_scores, torch.ones_like(pos_scores))
            loss += criterion(neg_scores, torch.zeros_like(neg_scores))
            loss += 0.01 * model.kl_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model


def generate_ood_samples(test_triples, model, num_entities, device, corruption_type='random', k=10):
    """Generate OOD samples based on corruption type."""
    n_samples = min(len(test_triples), 1000)

    if corruption_type == 'random':
        ood_tails = np.random.randint(0, num_entities, n_samples)

    elif corruption_type == 'high_score':
        # Adversarial: choose tails with high model scores
        ood_tails = []
        model.eval()
        with torch.no_grad():
            for i in range(n_samples):
                h = torch.tensor([test_triples[i, 0]]).to(device)
                r = torch.tensor([test_triples[i, 1]]).to(device)
                scores = model.score_all_tails(h, r).squeeze()
                scores[test_triples[i, 2]] = float('-inf')  # Exclude true tail
                topk_idx = torch.topk(scores, k).indices
                ood_tails.append(topk_idx[np.random.randint(k)].item())
        ood_tails = np.array(ood_tails)

    elif corruption_type == 'embedding_similar':
        # Adversarial: choose tails with similar embeddings
        ood_tails = []
        with torch.no_grad():
            emb = model.entity_mean.cpu().numpy()
            for i in range(n_samples):
                t = test_triples[i, 2]
                dists = np.linalg.norm(emb - emb[t], axis=1)
                dists[t] = np.inf
                nn_idx = np.argsort(dists)[:k]
                ood_tails.append(nn_idx[np.random.randint(len(nn_idx))])
        ood_tails = np.array(ood_tails)

    elif corruption_type == 'type_constrained':
        # Adversarial: sample from entities seen with same relation
        ood_tails = []
        coverage = model.coverage.cpu().numpy()
        for i in range(n_samples):
            r = test_triples[i, 1]
            valid = np.where(coverage[:, r] > 0)[0]
            if len(valid) > 0:
                valid = valid[valid != test_triples[i, 2]]  # Exclude true
                if len(valid) > 0:
                    ood_tails.append(np.random.choice(valid))
                else:
                    ood_tails.append(np.random.randint(0, num_entities))
            else:
                ood_tails.append(np.random.randint(0, num_entities))
        ood_tails = np.array(ood_tails)

    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    return test_triples[:n_samples], ood_tails


def compute_metrics_for_samples(model, heads, relations, tails, device, top_k=None):
    """Compute all uncertainty metrics for given samples."""
    model.eval()
    with torch.no_grad():
        h = torch.tensor(heads).to(device)
        r = torch.tensor(relations).to(device)
        t = torch.tensor(tails).to(device)

        components = model.get_uncertainty_components(h, r, t, top_k=top_k)

        return {
            'semantic': components['semantic'].cpu().numpy(),
            'structural': components['structural'].cpu().numpy(),
            'predictive': components['predictive'].cpu().numpy(),
            'entropy': model.get_prediction_entropy(h, r, top_k=top_k).cpu().numpy(),
            'margin': model.get_prediction_margin(h, r).cpu().numpy(),
        }


def analyze_query_level_uncertainty(model, test_triples, device, top_k=100, num_samples=1000):
    """
    Analyze uncertainty at query level.

    Key insight: Entropy/margin are query-level properties.
    We should compare:
    - Queries where model predicts correctly (normal queries)
    - Queries where model predicts incorrectly (vulnerable queries)

    Vulnerable queries should have higher entropy/lower margin.
    """
    model.eval()
    n_samples = min(len(test_triples), num_samples)

    correct_queries = []
    incorrect_queries = []

    with torch.no_grad():
        for i in range(n_samples):
            h = torch.tensor([test_triples[i, 0]]).to(device)
            r = torch.tensor([test_triples[i, 1]]).to(device)
            t_true = test_triples[i, 2]

            # Get prediction
            scores = model.score_all_tails(h, r).squeeze()
            pred = scores.argmax().item()

            # Get entropy and margin
            entropy = model.get_prediction_entropy(h, r, top_k=top_k).item()
            margin = model.get_prediction_margin(h, r).item()

            if pred == t_true:
                correct_queries.append({'entropy': entropy, 'margin': margin})
            else:
                incorrect_queries.append({'entropy': entropy, 'margin': margin})

    if len(correct_queries) == 0 or len(incorrect_queries) == 0:
        return None

    correct_entropy = np.array([q['entropy'] for q in correct_queries])
    correct_margin = np.array([q['margin'] for q in correct_queries])
    incorrect_entropy = np.array([q['entropy'] for q in incorrect_queries])
    incorrect_margin = np.array([q['margin'] for q in incorrect_queries])

    # Statistical tests
    t_entropy, p_entropy = stats.ttest_ind(incorrect_entropy, correct_entropy, alternative='greater')
    t_margin, p_margin = stats.ttest_ind(incorrect_margin, correct_margin, alternative='less')

    return {
        'n_correct': len(correct_queries),
        'n_incorrect': len(incorrect_queries),
        'correct_entropy_mean': float(correct_entropy.mean()),
        'correct_entropy_std': float(correct_entropy.std()),
        'incorrect_entropy_mean': float(incorrect_entropy.mean()),
        'incorrect_entropy_std': float(incorrect_entropy.std()),
        'entropy_t_stat': float(t_entropy),
        'entropy_p_value': float(p_entropy),
        'entropy_effect_size': float((incorrect_entropy.mean() - correct_entropy.mean()) /
                                      np.sqrt((incorrect_entropy.std()**2 + correct_entropy.std()**2) / 2)),
        'correct_margin_mean': float(correct_margin.mean()),
        'correct_margin_std': float(correct_margin.std()),
        'incorrect_margin_mean': float(incorrect_margin.mean()),
        'incorrect_margin_std': float(incorrect_margin.std()),
        'margin_t_stat': float(t_margin),
        'margin_p_value': float(p_margin),
        'margin_effect_size': float((correct_margin.mean() - incorrect_margin.mean()) /
                                     np.sqrt((incorrect_margin.std()**2 + correct_margin.std()**2) / 2)),
    }


def hypothesis_1_test(id_metrics, ood_metrics, ood_type):
    """
    H1: Adversarial OOD has higher prediction entropy than normal (ID) samples.
    """
    id_entropy = id_metrics['entropy']
    ood_entropy = ood_metrics['entropy']

    # Statistical test
    t_stat, p_value = stats.ttest_ind(ood_entropy, id_entropy, alternative='greater')
    effect_size = (ood_entropy.mean() - id_entropy.mean()) / np.sqrt(
        (ood_entropy.std()**2 + id_entropy.std()**2) / 2
    )

    return {
        'id_entropy_mean': float(id_entropy.mean()),
        'id_entropy_std': float(id_entropy.std()),
        'ood_entropy_mean': float(ood_entropy.mean()),
        'ood_entropy_std': float(ood_entropy.std()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'effect_size_cohens_d': float(effect_size),
        'hypothesis_supported': p_value < 0.05 and effect_size > 0.2,
    }


def hypothesis_2_test(id_metrics, ood_metrics, ood_type):
    """
    H2: Adversarial OOD has lower prediction margin than normal (ID) samples.
    """
    id_margin = id_metrics['margin']
    ood_margin = ood_metrics['margin']

    # Statistical test (OOD should have LOWER margin)
    t_stat, p_value = stats.ttest_ind(ood_margin, id_margin, alternative='less')
    effect_size = (id_margin.mean() - ood_margin.mean()) / np.sqrt(
        (ood_margin.std()**2 + id_margin.std()**2) / 2
    )

    return {
        'id_margin_mean': float(id_margin.mean()),
        'id_margin_std': float(id_margin.std()),
        'ood_margin_mean': float(ood_margin.mean()),
        'ood_margin_std': float(ood_margin.std()),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'effect_size_cohens_d': float(effect_size),
        'hypothesis_supported': p_value < 0.05 and effect_size > 0.2,
    }


def hypothesis_3_test(id_metrics):
    """
    H3: Predictive uncertainty is orthogonal to GP and Coverage.

    Tests correlation between predictive component and semantic/structural.
    Low correlation = orthogonal = captures different information.
    """
    predictive = id_metrics['predictive']
    semantic = id_metrics['semantic']
    structural = id_metrics['structural']

    corr_pred_sem, p_sem = stats.pearsonr(predictive, semantic)
    corr_pred_str, p_str = stats.pearsonr(predictive, structural)
    corr_sem_str, p_semstr = stats.pearsonr(semantic, structural)

    return {
        'corr_predictive_semantic': float(corr_pred_sem),
        'corr_predictive_structural': float(corr_pred_str),
        'corr_semantic_structural': float(corr_sem_str),
        'p_value_pred_sem': float(p_sem),
        'p_value_pred_str': float(p_str),
        # Orthogonal if |correlation| < 0.3
        'predictive_orthogonal_to_semantic': abs(corr_pred_sem) < 0.3,
        'predictive_orthogonal_to_structural': abs(corr_pred_str) < 0.3,
        'hypothesis_supported': abs(corr_pred_sem) < 0.3 and abs(corr_pred_str) < 0.3,
    }


def evaluate_ood_detection(model, test_triples, ood_tails, device, uncertainty_type='combined', top_k=None):
    """Evaluate OOD detection AUROC."""
    model.eval()
    n_samples = len(ood_tails)
    test_subset = test_triples[:n_samples]

    with torch.no_grad():
        h = torch.tensor(test_subset[:, 0]).to(device)
        r = torch.tensor(test_subset[:, 1]).to(device)
        t_id = torch.tensor(test_subset[:, 2]).to(device)
        t_ood = torch.tensor(ood_tails).to(device)

        if uncertainty_type == 'combined':
            id_unc = model.get_uncertainty(h, r, t_id, top_k=top_k, use_triple_level_predictive=True).cpu().numpy()
            ood_unc = model.get_uncertainty(h, r, t_ood, top_k=top_k, use_triple_level_predictive=True).cpu().numpy()
        elif uncertainty_type == 'semantic':
            id_unc = model.get_semantic_uncertainty(h, t_id).cpu().numpy()
            ood_unc = model.get_semantic_uncertainty(h, t_ood).cpu().numpy()
        elif uncertainty_type == 'structural':
            id_unc = model.get_structural_uncertainty(h, r, t_id).cpu().numpy()
            ood_unc = model.get_structural_uncertainty(h, r, t_ood).cpu().numpy()
        elif uncertainty_type == 'predictive':
            # Use triple-level predictive (rank-based)
            id_unc = model.get_triple_predictive_uncertainty(h, r, t_id, method='combined').cpu().numpy()
            ood_unc = model.get_triple_predictive_uncertainty(h, r, t_ood, method='combined').cpu().numpy()
        elif uncertainty_type == 'predictive_rank':
            id_unc = model.get_tail_rank(h, r, t_id).cpu().numpy()
            ood_unc = model.get_tail_rank(h, r, t_ood).cpu().numpy()
        elif uncertainty_type == 'predictive_gap':
            id_unc = model.get_score_gap(h, r, t_id).cpu().numpy()
            ood_unc = model.get_score_gap(h, r, t_ood).cpu().numpy()
        elif uncertainty_type == 'semantic+structural':
            id_sem = model.get_semantic_uncertainty(h, t_id)
            id_str = model.get_structural_uncertainty(h, r, t_id)
            id_sem_norm = id_sem / (id_sem.mean() + 1e-8)
            id_str_norm = id_str / (id_str.mean() + 1e-8)
            id_unc = (0.5 * id_sem_norm + 0.5 * id_str_norm).cpu().numpy()

            ood_sem = model.get_semantic_uncertainty(h, t_ood)
            ood_str = model.get_structural_uncertainty(h, r, t_ood)
            ood_sem_norm = ood_sem / (ood_sem.mean() + 1e-8)
            ood_str_norm = ood_str / (ood_str.mean() + 1e-8)
            ood_unc = (0.5 * ood_sem_norm + 0.5 * ood_str_norm).cpu().numpy()
        else:
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")

    labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc, ood_unc])

    try:
        auroc = roc_auc_score(labels, scores)
    except:
        auroc = 0.5

    return auroc


def run_ablation_study(model, test_triples, num_entities, device, top_k=None):
    """Run ablation study on all OOD types."""
    corruption_types = ['random', 'high_score', 'embedding_similar', 'type_constrained']
    uncertainty_configs = [
        ('semantic', 'Semantic (GP) only'),
        ('structural', 'Structural (Coverage) only'),
        ('semantic+structural', 'CAGP (Semantic + Structural)'),
        ('predictive_rank', 'Predictive (Rank) only'),
        ('predictive_gap', 'Predictive (Gap) only'),
        ('predictive', 'Predictive (Combined) only'),
        ('combined', 'Full 3-Component'),
    ]

    results = {}

    for corr_type in corruption_types:
        print(f"\n  Corruption: {corr_type}")
        test_subset, ood_tails = generate_ood_samples(
            test_triples, model, num_entities, device, corr_type
        )

        results[corr_type] = {}

        for unc_type, unc_name in uncertainty_configs:
            auroc = evaluate_ood_detection(
                model, test_subset, ood_tails, device, unc_type, top_k
            )
            results[corr_type][unc_type] = auroc
            print(f"    {unc_name}: {auroc:.4f}")

    return results


def run_hypothesis_validation(model, test_triples, num_entities, device, top_k=None):
    """Run all hypothesis validation experiments."""
    results = {
        'H1_entropy': {},
        'H2_margin': {},
        'H3_orthogonality': None,
        'query_level_analysis': None,
    }

    # Get ID metrics
    print("  Computing ID metrics...")
    id_metrics = compute_metrics_for_samples(
        model,
        test_triples[:1000, 0],
        test_triples[:1000, 1],
        test_triples[:1000, 2],
        device,
        top_k
    )

    # H3: Orthogonality test (only needs ID samples)
    results['H3_orthogonality'] = hypothesis_3_test(id_metrics)
    print(f"\n  H3 (Orthogonality):")
    print(f"    Corr(pred, sem): {results['H3_orthogonality']['corr_predictive_semantic']:.3f}")
    print(f"    Corr(pred, str): {results['H3_orthogonality']['corr_predictive_structural']:.3f}")
    print(f"    Supported: {results['H3_orthogonality']['hypothesis_supported']}")

    # Query-level analysis (NEW - correct interpretation)
    print("\n  Query-level analysis (correct vs incorrect predictions)...")
    query_analysis = analyze_query_level_uncertainty(model, test_triples, device, top_k)
    results['query_level_analysis'] = query_analysis
    if query_analysis:
        print(f"    Correct predictions: n={query_analysis['n_correct']}")
        print(f"    Incorrect predictions: n={query_analysis['n_incorrect']}")
        print(f"    Entropy (correct): {query_analysis['correct_entropy_mean']:.3f} +/- {query_analysis['correct_entropy_std']:.3f}")
        print(f"    Entropy (incorrect): {query_analysis['incorrect_entropy_mean']:.3f} +/- {query_analysis['incorrect_entropy_std']:.3f}")
        print(f"    Entropy p-value: {query_analysis['entropy_p_value']:.4f}, effect size: {query_analysis['entropy_effect_size']:.3f}")
        print(f"    Margin (correct): {query_analysis['correct_margin_mean']:.3f} +/- {query_analysis['correct_margin_std']:.3f}")
        print(f"    Margin (incorrect): {query_analysis['incorrect_margin_mean']:.3f} +/- {query_analysis['incorrect_margin_std']:.3f}")
        print(f"    Margin p-value: {query_analysis['margin_p_value']:.4f}, effect size: {query_analysis['margin_effect_size']:.3f}")

    # H1 and H2 for each adversarial type (triple-level - for comparison)
    print("\n  Triple-level analysis (same query, different tails - expected to fail):")
    for corr_type in ['high_score', 'embedding_similar', 'type_constrained']:
        print(f"\n    Testing {corr_type}...")
        test_subset, ood_tails = generate_ood_samples(
            test_triples, model, num_entities, device, corr_type
        )

        # Get OOD metrics
        ood_metrics = compute_metrics_for_samples(
            model,
            test_subset[:, 0],
            test_subset[:, 1],
            ood_tails,
            device,
            top_k
        )

        # H1: Entropy test
        h1_result = hypothesis_1_test(id_metrics, ood_metrics, corr_type)
        results['H1_entropy'][corr_type] = h1_result
        print(f"      H1 (Entropy): ID={h1_result['id_entropy_mean']:.3f}, OOD={h1_result['ood_entropy_mean']:.3f}")

        # H2: Margin test
        h2_result = hypothesis_2_test(id_metrics, ood_metrics, corr_type)
        results['H2_margin'][corr_type] = h2_result
        print(f"      H2 (Margin): ID={h2_result['id_margin_mean']:.3f}, OOD={h2_result['ood_margin_mean']:.3f}")

    return results


def run_experiments(dataset_name='fb15k-237', epochs=30, dim=100, top_k=100):
    """Run all predictive uncertainty experiments."""
    print(f"\n{'='*70}")
    print(f"Predictive Uncertainty Experiments on {dataset_name}")
    print(f"{'='*70}\n")

    device = setup_device()

    # Load data
    print("Loading data...")
    if dataset_name == 'fb15k-237':
        train_ds, valid_ds, test_ds = load_fb15k237()
    elif dataset_name == 'wn18rr':
        train_ds, valid_ds, test_ds = load_wn18rr()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_triples = train_ds.triples
    test_triples = test_ds.triples
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    print(f"Entities: {num_entities}, Relations: {num_relations}")
    print(f"Train: {len(train_triples)}, Test: {len(test_triples)}")

    # Create and train model
    print("\nCreating PredictiveCAGP model...")
    model = PredictiveCAGP(
        num_entities, num_relations, dim,
        predictive_type='entropy',
        learn_temperature=True,
        learn_weights=True,
    )
    model.precompute_coverage(train_triples)

    print("Training model...")
    model = train_model(model, train_triples, device, epochs=epochs)

    results = {
        'dataset': dataset_name,
        'model_stats': model.get_stats(),
    }

    # Run hypothesis validation
    print("\n--- Hypothesis Validation ---")
    results['hypothesis_tests'] = run_hypothesis_validation(
        model, test_triples, num_entities, device, top_k
    )

    # Run ablation study
    print("\n--- Ablation Study ---")
    results['ablation'] = run_ablation_study(
        model, test_triples, num_entities, device, top_k
    )

    return results


def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Model stats
    print(f"\nModel learned weights:")
    stats = results['model_stats']
    print(f"  alpha (semantic):   {stats['alpha_semantic']:.3f}")
    print(f"  beta (structural):  {stats['beta_structural']:.3f}")
    print(f"  gamma (predictive): {stats['gamma_predictive']:.3f}")
    print(f"  temperature:        {stats['temperature']:.3f}")

    # Query-level analysis
    query_analysis = results['hypothesis_tests'].get('query_level_analysis')
    if query_analysis:
        print("\nQuery-Level Analysis (Correct vs Incorrect predictions):")
        print(f"  Margin effect: p={query_analysis['margin_p_value']:.4f}, d={query_analysis['margin_effect_size']:.3f}")
        if query_analysis['margin_p_value'] < 0.05:
            print("  -> Margin discriminates correct/incorrect predictions!")

    # Hypothesis results
    print("\nHypothesis Testing:")
    h3 = results['hypothesis_tests']['H3_orthogonality']
    print(f"  H3 (Orthogonality): {'SUPPORTED' if h3['hypothesis_supported'] else 'NOT SUPPORTED'}")
    print(f"      Corr(pred, sem): {h3['corr_predictive_semantic']:.3f}")
    print(f"      Corr(pred, str): {h3['corr_predictive_structural']:.3f}")

    # Ablation results
    print("\nAblation Study (AUROC):")
    print(f"{'Corruption':<18} {'Sem':<7} {'Str':<7} {'CAGP':<7} {'Rank':<7} {'Gap':<7} {'Pred':<7} {'Full':<7}")
    print("-" * 74)

    ablation = results['ablation']
    for corr_type in ['random', 'high_score', 'embedding_similar', 'type_constrained']:
        row = ablation[corr_type]
        print(f"{corr_type:<18} {row['semantic']:.3f}   {row['structural']:.3f}   "
              f"{row['semantic+structural']:.3f}   {row['predictive_rank']:.3f}   "
              f"{row['predictive_gap']:.3f}   {row['predictive']:.3f}   {row['combined']:.3f}")

    # Improvement analysis
    print("\nImprovement from adding Predictive component (CAGP -> Full):")
    for corr_type in ['random', 'high_score', 'embedding_similar', 'type_constrained']:
        cagp = ablation[corr_type]['semantic+structural']
        full = ablation[corr_type]['combined']
        improvement = (full - cagp) * 100
        marker = "**" if improvement > 5 else ""
        print(f"  {corr_type}: {cagp:.4f} -> {full:.4f} ({improvement:+.1f}%) {marker}")


def main():
    """Main entry point."""
    all_results = {}

    # Run on FB15k-237
    fb_results = run_experiments('fb15k-237', epochs=30, dim=100, top_k=100)
    all_results['fb15k-237'] = fb_results
    print_summary(fb_results)

    # Optionally run on WN18RR
    # wn_results = run_experiments('wn18rr', epochs=30, dim=100, top_k=100)
    # all_results['wn18rr'] = wn_results
    # print_summary(wn_results)

    # Save results
    output_path = project_root / 'outputs' / 'predictive_uncertainty_results.json'
    output_path.parent.mkdir(exist_ok=True)

    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_types(all_results), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
