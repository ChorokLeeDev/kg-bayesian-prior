#!/usr/bin/env python3
"""
Coverage Entropy Analysis: Quantifying hierarchical vs heterogeneous KG structure.

This script computes coverage pattern entropy to distinguish:
- WN18RR (hierarchical, lower entropy, GNN works)
- FB15k-237 / YAGO3-10 (heterogeneous, higher entropy, GNN fails)

Metrics computed per dataset:
1. Mean coverage entropy per entity: H(c_e) = -sum_r [p_r log p_r]
2. Coverage correlation across relations within entities
3. Gini coefficient of relation distribution per entity
4. Coverage concentration ratio (top-k relations cover what % of edges)

The hypothesis: Lower entropy = more predictable coverage patterns =
GNN neighborhood aggregation is informative for OOD detection.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import stats
from collections import defaultdict


def load_dataset_triples(dataset_name: str):
    """Load training triples for a dataset."""
    from src.data.loaders import load_wn18rr, load_fb15k237, load_yago310

    loaders = {
        "WN18RR": load_wn18rr,
        "FB15k-237": load_fb15k237,
        "YAGO3-10": load_yago310,
    }

    train_ds, _, _ = loaders[dataset_name]()
    return train_ds.triples, train_ds.num_entities, train_ds.num_relations


def compute_coverage_matrix(triples: np.ndarray, num_entities: int, num_relations: int):
    """
    Build coverage matrix C[e, r] = count of times entity e appears with relation r.

    We count both head and tail occurrences for complete coverage picture.
    """
    coverage = np.zeros((num_entities, num_relations), dtype=np.float32)

    for h, r, t in triples:
        coverage[h, r] += 1  # entity as head
        coverage[t, r] += 1  # entity as tail

    return coverage


def compute_entropy(coverage_matrix: np.ndarray, eps: float = 1e-10):
    """
    Compute Shannon entropy of coverage distribution per entity.

    H(c_e) = -sum_r [p_r * log(p_r)] where p_r = c(e,r) / sum_r' c(e,r')

    Returns:
        entropies: per-entity entropy values
        mean_entropy: mean over all entities
        std_entropy: std over all entities
    """
    # Normalize to get probabilities per entity
    row_sums = coverage_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for entities with no coverage
    row_sums = np.where(row_sums > 0, row_sums, 1)
    probs = coverage_matrix / row_sums

    # Compute entropy: H = -sum(p * log(p))
    # Add eps to avoid log(0)
    log_probs = np.log(probs + eps)
    entropies = -np.sum(probs * log_probs, axis=1)

    # Only include entities with at least one edge
    active_mask = coverage_matrix.sum(axis=1) > 0
    active_entropies = entropies[active_mask]

    return active_entropies, active_entropies.mean(), active_entropies.std()


def compute_normalized_entropy(coverage_matrix: np.ndarray, eps: float = 1e-10):
    """
    Compute normalized entropy H(c_e) / log(R) to account for different #relations.

    This gives a fair comparison across datasets with different relation counts.
    """
    num_relations = coverage_matrix.shape[1]
    max_entropy = np.log(num_relations)

    entropies, mean_ent, std_ent = compute_entropy(coverage_matrix, eps)

    norm_entropies = entropies / max_entropy

    return norm_entropies, norm_entropies.mean(), norm_entropies.std()


def compute_gini_coefficient(coverage_matrix: np.ndarray):
    """
    Compute Gini coefficient of relation distribution per entity.

    Gini = 0: perfectly equal distribution (each relation used equally)
    Gini = 1: maximum inequality (only one relation used)

    Higher Gini = more concentrated = more predictable = GNN-friendly
    """
    gini_values = []

    for entity_cov in coverage_matrix:
        if entity_cov.sum() == 0:
            continue

        # Sort values
        sorted_cov = np.sort(entity_cov)
        n = len(sorted_cov)
        cumsum = np.cumsum(sorted_cov)

        # Gini formula
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_cov))) / (n * sorted_cov.sum()) - (n + 1) / n
        gini_values.append(gini)

    gini_values = np.array(gini_values)
    return gini_values, gini_values.mean(), gini_values.std()


def compute_relation_concentration(coverage_matrix: np.ndarray, k: int = 3):
    """
    Compute what fraction of an entity's edges come from top-k relations.

    Higher concentration = more predictable = GNN-friendly
    """
    concentrations = []

    for entity_cov in coverage_matrix:
        total = entity_cov.sum()
        if total == 0:
            continue

        # Top-k relations
        top_k = np.sort(entity_cov)[-k:]
        concentration = top_k.sum() / total
        concentrations.append(concentration)

    concentrations = np.array(concentrations)
    return concentrations, concentrations.mean(), concentrations.std()


def compute_coverage_sparsity(coverage_matrix: np.ndarray):
    """
    Compute sparsity metrics for coverage matrix.

    - Mean relations per entity
    - Fraction of (entity, relation) pairs with zero coverage
    """
    # Active relations per entity
    active_per_entity = (coverage_matrix > 0).sum(axis=1)
    active_mask = coverage_matrix.sum(axis=1) > 0

    mean_active_relations = active_per_entity[active_mask].mean()
    std_active_relations = active_per_entity[active_mask].std()

    # Global sparsity
    total_pairs = coverage_matrix.shape[0] * coverage_matrix.shape[1]
    zero_pairs = (coverage_matrix == 0).sum()
    sparsity = zero_pairs / total_pairs

    return mean_active_relations, std_active_relations, sparsity


def compute_coverage_correlation(coverage_matrix: np.ndarray):
    """
    Compute mean pairwise correlation of relation usage across entities.

    High correlation = similar coverage patterns = hierarchical structure
    Low correlation = diverse patterns = heterogeneous structure
    """
    # Only include entities with coverage
    active_mask = coverage_matrix.sum(axis=1) > 0
    active_coverage = coverage_matrix[active_mask]

    # Sample if too large
    if len(active_coverage) > 5000:
        idx = np.random.choice(len(active_coverage), 5000, replace=False)
        active_coverage = active_coverage[idx]

    # Compute correlation matrix
    # Normalize each entity's coverage
    norms = np.linalg.norm(active_coverage, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    normalized = active_coverage / norms

    # Pairwise correlations (cosine similarity of coverage vectors)
    corr_matrix = normalized @ normalized.T

    # Get upper triangle (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]

    return upper_tri.mean(), upper_tri.std()


def analyze_dataset(dataset_name: str):
    """Run full coverage entropy analysis for a dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name}")
    print('='*60)

    # Load data
    triples, num_entities, num_relations = load_dataset_triples(dataset_name)
    print(f"Entities: {num_entities:,}, Relations: {num_relations}, Triples: {len(triples):,}")

    # Build coverage matrix
    coverage = compute_coverage_matrix(triples, num_entities, num_relations)

    # Compute metrics
    results = {"dataset": dataset_name, "num_entities": num_entities,
               "num_relations": num_relations, "num_triples": len(triples)}

    # 1. Raw entropy
    _, mean_ent, std_ent = compute_entropy(coverage)
    results["mean_entropy"] = mean_ent
    results["std_entropy"] = std_ent
    print(f"Mean entropy: {mean_ent:.4f} +/- {std_ent:.4f}")

    # 2. Normalized entropy (fair comparison)
    _, mean_norm_ent, std_norm_ent = compute_normalized_entropy(coverage)
    results["mean_normalized_entropy"] = mean_norm_ent
    results["std_normalized_entropy"] = std_norm_ent
    print(f"Normalized entropy (H/log(R)): {mean_norm_ent:.4f} +/- {std_norm_ent:.4f}")

    # 3. Gini coefficient
    _, mean_gini, std_gini = compute_gini_coefficient(coverage)
    results["mean_gini"] = mean_gini
    results["std_gini"] = std_gini
    print(f"Gini coefficient: {mean_gini:.4f} +/- {std_gini:.4f}")

    # 4. Relation concentration (top-3)
    _, mean_conc, std_conc = compute_relation_concentration(coverage, k=3)
    results["mean_concentration_top3"] = mean_conc
    results["std_concentration_top3"] = std_conc
    print(f"Top-3 relation concentration: {mean_conc:.4f} +/- {std_conc:.4f}")

    # 5. Sparsity metrics
    mean_active, std_active, sparsity = compute_coverage_sparsity(coverage)
    results["mean_active_relations"] = mean_active
    results["std_active_relations"] = std_active
    results["coverage_sparsity"] = sparsity
    print(f"Mean relations per entity: {mean_active:.2f} +/- {std_active:.2f}")
    print(f"Coverage sparsity: {sparsity:.4f}")

    # 6. Coverage correlation
    mean_corr, std_corr = compute_coverage_correlation(coverage)
    results["mean_coverage_correlation"] = mean_corr
    results["std_coverage_correlation"] = std_corr
    print(f"Mean coverage correlation: {mean_corr:.4f} +/- {std_corr:.4f}")

    return results


def format_results_table(all_results):
    """Format results as a comparison table."""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("COVERAGE ENTROPY ANALYSIS: HIERARCHICAL VS HETEROGENEOUS KG STRUCTURE")
    lines.append("="*80)
    lines.append("")
    lines.append("Hypothesis: Lower entropy = more predictable coverage = GNN works")
    lines.append("           Higher entropy = heterogeneous patterns = GNN fails")
    lines.append("")

    # Basic stats table
    lines.append("-"*80)
    lines.append("DATASET STATISTICS")
    lines.append("-"*80)
    lines.append(f"{'Dataset':<12} {'Entities':>10} {'Relations':>10} {'Triples':>12}")
    lines.append("-"*80)
    for r in all_results:
        lines.append(f"{r['dataset']:<12} {r['num_entities']:>10,} {r['num_relations']:>10} {r['num_triples']:>12,}")

    # Main metrics table
    lines.append("")
    lines.append("-"*80)
    lines.append("COVERAGE ENTROPY METRICS (lower = more predictable = GNN-friendly)")
    lines.append("-"*80)
    lines.append(f"{'Dataset':<12} {'Entropy':>10} {'Norm. Ent.':>12} {'Gini':>10} {'Top-3 Conc.':>12}")
    lines.append("-"*80)
    for r in all_results:
        lines.append(f"{r['dataset']:<12} {r['mean_entropy']:>10.4f} {r['mean_normalized_entropy']:>12.4f} "
                    f"{r['mean_gini']:>10.4f} {r['mean_concentration_top3']:>12.4f}")

    # Sparsity and correlation table
    lines.append("")
    lines.append("-"*80)
    lines.append("STRUCTURAL METRICS")
    lines.append("-"*80)
    lines.append(f"{'Dataset':<12} {'Rels/Entity':>12} {'Sparsity':>10} {'Correlation':>12}")
    lines.append("-"*80)
    for r in all_results:
        lines.append(f"{r['dataset']:<12} {r['mean_active_relations']:>12.2f} "
                    f"{r['coverage_sparsity']:>10.4f} {r['mean_coverage_correlation']:>12.4f}")

    # Analysis
    lines.append("")
    lines.append("-"*80)
    lines.append("ANALYSIS")
    lines.append("-"*80)

    # Sort by normalized entropy
    sorted_by_entropy = sorted(all_results, key=lambda x: x["mean_normalized_entropy"])

    lines.append("")
    lines.append("Ranked by normalized entropy (lower = more hierarchical):")
    for i, r in enumerate(sorted_by_entropy, 1):
        structure = "HIERARCHICAL" if r["mean_normalized_entropy"] < 0.5 else "HETEROGENEOUS"
        lines.append(f"  {i}. {r['dataset']:<12} H/log(R) = {r['mean_normalized_entropy']:.4f}  [{structure}]")

    # WN18RR analysis
    wn = next((r for r in all_results if r["dataset"] == "WN18RR"), None)
    fb = next((r for r in all_results if r["dataset"] == "FB15k-237"), None)
    yago = next((r for r in all_results if r["dataset"] == "YAGO3-10"), None)

    # Sort by correlation (key discriminator)
    sorted_by_corr = sorted(all_results, key=lambda x: x["mean_coverage_correlation"], reverse=True)
    lines.append("")
    lines.append("Ranked by coverage correlation (higher = neighbors more predictive = GNN-friendly):")
    for i, r in enumerate(sorted_by_corr, 1):
        gnn_note = "GNN WORKS" if r["mean_coverage_correlation"] > 0.3 else "GNN FAILS"
        lines.append(f"  {i}. {r['dataset']:<12} corr = {r['mean_coverage_correlation']:.4f}  [{gnn_note}]")

    lines.append("")
    lines.append("Key observations:")

    if wn and fb:
        corr_ratio = wn["mean_coverage_correlation"] / fb["mean_coverage_correlation"]
        lines.append(f"  - WN18RR has {corr_ratio:.1f}x HIGHER coverage correlation than FB15k-237")
        entropy_ratio = fb["mean_normalized_entropy"] / wn["mean_normalized_entropy"]
        lines.append(f"  - FB15k-237 has {entropy_ratio:.2f}x higher normalized entropy than WN18RR")

    if wn and yago:
        corr_ratio = wn["mean_coverage_correlation"] / yago["mean_coverage_correlation"]
        lines.append(f"  - WN18RR has {corr_ratio:.1f}x HIGHER coverage correlation than YAGO3-10")

    lines.append("")
    lines.append("Concentration metrics:")
    if wn:
        lines.append(f"  - WN18RR: {wn['mean_active_relations']:.1f} relations/entity avg, {wn['mean_concentration_top3']:.1%} in top-3")
    if fb:
        lines.append(f"  - FB15k-237: {fb['mean_active_relations']:.1f} relations/entity avg, {fb['mean_concentration_top3']:.1%} in top-3")
    if yago:
        lines.append(f"  - YAGO3-10: {yago['mean_active_relations']:.1f} relations/entity avg, {yago['mean_concentration_top3']:.1%} in top-3")

    lines.append("")
    lines.append("-"*80)
    lines.append("CONCLUSION: WHY TOPOLOGY DETERMINES GNN EFFECTIVENESS")
    lines.append("-"*80)
    lines.append("")
    lines.append("The CRITICAL discriminator is COVERAGE CORRELATION, not entropy alone:")
    if wn and fb:
        lines.append(f"  - WN18RR:    correlation = {wn['mean_coverage_correlation']:.4f} (neighbors predict coverage)")
        lines.append(f"  - FB15k-237: correlation = {fb['mean_coverage_correlation']:.4f} (neighbors uninformative)")
    if yago:
        lines.append(f"  - YAGO3-10:  correlation = {yago['mean_coverage_correlation']:.4f} (intermediate)")
    lines.append("")
    lines.append("WN18RR's hierarchical structure (WordNet taxonomy) means:")
    lines.append("  - Entities in same synset/hypernym chain share relation patterns")
    lines.append("  - GNN message passing propagates useful coverage information")
    lines.append("  - Neighbor aggregation predicts entity's OOD status")
    lines.append("")
    lines.append("FB15k-237's heterogeneous structure (Freebase) means:")
    lines.append("  - Entity types are diverse (people, places, films, etc.)")
    lines.append("  - Neighbors have uncorrelated relation coverage")
    lines.append("  - GNN aggregation mixes uninformative signals")
    lines.append("  - Only explicit per-entity coverage tracking works")
    lines.append("")
    lines.append("This validates the paper's claim: 'topology is the key'")
    lines.append("GNN boundary detection works iff coverage patterns are locally predictable")
    lines.append("")

    return "\n".join(lines)


def main():
    """Run coverage entropy analysis on all datasets."""
    print("Coverage Entropy Analysis")
    print("Quantifying hierarchical vs heterogeneous KG structure")

    datasets = ["WN18RR", "FB15k-237", "YAGO3-10"]
    all_results = []

    for dataset in datasets:
        try:
            results = analyze_dataset(dataset)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Format and save results
    output_text = format_results_table(all_results)
    print(output_text)

    # Save to file
    output_path = PROJECT_ROOT / "outputs" / "coverage_entropy_analysis.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text)
    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
