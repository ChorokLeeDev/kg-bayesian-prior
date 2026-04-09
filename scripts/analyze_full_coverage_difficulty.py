"""
Coverage Paradox Analysis: Why Full Coverage triples have LOWER prediction accuracy

Hypothesis: Full coverage = idiosyncratic/hard facts, Partial = compositional/easy facts

Analysis:
1. Training loss distribution: Full vs Partial coverage triples
2. Triple "uniqueness": Neighborhood diversity and pattern rarity
3. Relation distribution: Are full coverage triples in rare relations?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Set
from tqdm import tqdm

from src.data.loaders import load_fb15k237


def build_coverage_matrix(triples: np.ndarray, num_entities: int, num_relations: int) -> np.ndarray:
    """Build entity-relation coverage matrix from training triples."""
    coverage = np.zeros((num_entities, num_relations), dtype=bool)
    for h, r, t in triples:
        coverage[h, r] = True
        coverage[t, r] = True
    return coverage


def classify_test_triples(test_triples: np.ndarray, coverage: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Classify test triples by coverage status.

    Returns:
        Dictionary with:
        - 'full': Both head and tail have coverage for this relation
        - 'partial_head': Only head has coverage
        - 'partial_tail': Only tail has coverage
        - 'zero': Neither has coverage (novel context)
    """
    results = {'full': [], 'partial_head': [], 'partial_tail': [], 'zero': []}

    for idx, (h, r, t) in enumerate(test_triples):
        head_cov = coverage[h, r]
        tail_cov = coverage[t, r]

        if head_cov and tail_cov:
            results['full'].append(idx)
        elif head_cov and not tail_cov:
            results['partial_head'].append(idx)
        elif not head_cov and tail_cov:
            results['partial_tail'].append(idx)
        else:
            results['zero'].append(idx)

    return {k: np.array(v) for k, v in results.items()}


def compute_entity_neighborhood(triples: np.ndarray, num_entities: int) -> Dict[int, Set[Tuple[int, int, str]]]:
    """
    Compute the neighborhood of each entity.
    Neighborhood = set of (relation, neighbor_entity, direction) tuples.
    """
    neighborhoods = defaultdict(set)

    for h, r, t in triples:
        neighborhoods[h].add((r, t, 'out'))
        neighborhoods[t].add((r, h, 'in'))

    return neighborhoods


def compute_triple_uniqueness_scores(
    triples: np.ndarray,
    train_triples: np.ndarray,
    num_entities: int,
    num_relations: int
) -> np.ndarray:
    """
    Compute uniqueness score for each triple.

    Uniqueness based on:
    1. Relation frequency (rarer relations = higher uniqueness)
    2. Entity frequency (rarer entities = higher uniqueness)
    3. Neighborhood overlap (less overlap = higher uniqueness)
    """
    # Compute relation frequencies
    rel_counts = np.zeros(num_relations)
    for h, r, t in train_triples:
        rel_counts[r] += 1
    rel_freq = rel_counts / len(train_triples)

    # Compute entity frequencies
    ent_counts = np.zeros(num_entities)
    for h, r, t in train_triples:
        ent_counts[h] += 1
        ent_counts[t] += 1
    ent_freq = ent_counts / (2 * len(train_triples))

    # Compute neighborhoods for pattern analysis
    neighborhoods = compute_entity_neighborhood(train_triples, num_entities)

    # Build (h,r) and (r,t) pattern counts
    hr_pattern_counts = defaultdict(int)
    rt_pattern_counts = defaultdict(int)
    for h, r, t in train_triples:
        hr_pattern_counts[(h, r)] += 1
        rt_pattern_counts[(r, t)] += 1

    uniqueness_scores = []

    for h, r, t in tqdm(triples, desc="Computing uniqueness"):
        # Relation rarity score (higher = rarer)
        rel_rarity = 1.0 - rel_freq[r]

        # Entity rarity score
        ent_rarity = 1.0 - (ent_freq[h] + ent_freq[t]) / 2

        # Pattern rarity: how common is this (h,r,?) and (?,r,t) pattern?
        hr_count = hr_pattern_counts.get((h, r), 0)
        rt_count = rt_pattern_counts.get((r, t), 0)
        pattern_rarity = 1.0 - min(1.0, (hr_count + rt_count) / 10.0)  # Normalized

        # Neighborhood diversity: entities with diverse neighborhoods are in more "regular" positions
        h_diversity = len(neighborhoods.get(h, set()))
        t_diversity = len(neighborhoods.get(t, set()))
        avg_diversity = (h_diversity + t_diversity) / 2
        neighborhood_score = 1.0 / (1.0 + np.log1p(avg_diversity))  # Higher = less diverse = more unique

        # Combined uniqueness
        uniqueness = (rel_rarity + ent_rarity + pattern_rarity + neighborhood_score) / 4
        uniqueness_scores.append(uniqueness)

    return np.array(uniqueness_scores)


def compute_training_loss_proxy(
    test_triples: np.ndarray,
    train_triples: np.ndarray,
    num_entities: int,
    num_relations: int,
    embedding_dim: int = 100
) -> np.ndarray:
    """
    Train a simple DistMult model and compute per-triple loss.
    This serves as a proxy for "difficulty" - higher loss = harder to fit.
    """
    # Build training set for quick model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize embeddings
    entity_emb = nn.Embedding(num_entities, embedding_dim).to(device)
    relation_emb = nn.Embedding(num_relations, embedding_dim).to(device)
    nn.init.xavier_uniform_(entity_emb.weight)
    nn.init.xavier_uniform_(relation_emb.weight)

    # Convert to tensors
    train_tensor = torch.LongTensor(train_triples).to(device)

    # Simple training
    optimizer = torch.optim.Adam(
        list(entity_emb.parameters()) + list(relation_emb.parameters()),
        lr=0.001
    )

    batch_size = 1024
    n_epochs = 30  # Quick training

    print(f"Training DistMult model on {len(train_triples)} triples...")

    for epoch in range(n_epochs):
        total_loss = 0.0
        perm = torch.randperm(len(train_tensor))

        for i in range(0, len(train_tensor), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = train_tensor[batch_idx]

            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            # DistMult scoring
            h_emb = entity_emb(h)
            r_emb = relation_emb(r)
            t_emb = entity_emb(t)

            pos_scores = (h_emb * r_emb * t_emb).sum(dim=1)

            # Negative sampling
            neg_t = torch.randint(0, num_entities, (len(batch),), device=device)
            neg_t_emb = entity_emb(neg_t)
            neg_scores = (h_emb * r_emb * neg_t_emb).sum(dim=1)

            # Margin loss
            loss = torch.relu(1.0 - pos_scores + neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")

    # Compute per-triple loss on test set
    test_tensor = torch.LongTensor(test_triples).to(device)

    with torch.no_grad():
        h, r, t = test_tensor[:, 0], test_tensor[:, 1], test_tensor[:, 2]
        h_emb = entity_emb(h)
        r_emb = relation_emb(r)
        t_emb = entity_emb(t)

        # Score = how well model predicts this triple
        scores = (h_emb * r_emb * t_emb).sum(dim=1)

        # Convert to "difficulty": lower score = harder (use negative score as loss proxy)
        difficulty = -scores.cpu().numpy()

    return difficulty


def compute_compositional_score(
    triples: np.ndarray,
    train_triples: np.ndarray,
    num_entities: int,
    num_relations: int
) -> np.ndarray:
    """
    Compute compositionality score: how much can this triple be inferred from paths?

    Higher score = more compositional (can be derived from multi-hop paths).
    Lower score = more idiosyncratic (must be memorized).
    """
    # Build adjacency for path finding
    adj = defaultdict(lambda: defaultdict(set))
    for h, r, t in train_triples:
        adj[h][r].add(t)
        # Also store inverse
        adj[t][r + num_relations].add(h)  # Inverse relations

    compositional_scores = []

    for h, r, t in tqdm(triples, desc="Computing compositionality"):
        # Count 2-hop paths from h to t
        two_hop_count = 0
        for r1 in adj[h]:
            for mid in adj[h][r1]:
                for r2 in adj[mid]:
                    if t in adj[mid][r2]:
                        two_hop_count += 1

        # Normalize by relation frequency
        rel_count = sum(1 for _, r2, _ in train_triples if r2 == r)

        # Compositionality score: more paths = more compositional
        if rel_count > 0:
            score = min(1.0, two_hop_count / 5.0)  # Normalized
        else:
            score = 0.0

        compositional_scores.append(score)

    return np.array(compositional_scores)


def analyze_relation_distribution(
    test_triples: np.ndarray,
    classification: Dict[str, np.ndarray],
    train_triples: np.ndarray,
    num_relations: int
) -> Dict[str, Dict[int, float]]:
    """Analyze relation distribution for each coverage category."""
    # Compute relation frequencies in training
    train_rel_counts = np.zeros(num_relations)
    for h, r, t in train_triples:
        train_rel_counts[r] += 1
    train_rel_freq = train_rel_counts / len(train_triples)

    results = {}

    for category, indices in classification.items():
        if len(indices) == 0:
            results[category] = {'avg_rel_freq': 0.0, 'n_unique_rels': 0}
            continue

        cat_triples = test_triples[indices]
        cat_rels = cat_triples[:, 1]

        # Average relation frequency
        avg_freq = train_rel_freq[cat_rels].mean()
        n_unique = len(np.unique(cat_rels))

        results[category] = {
            'avg_rel_freq': float(avg_freq),
            'n_unique_rels': int(n_unique),
            'n_triples': len(indices)
        }

    return results


def main():
    print("=" * 70)
    print("COVERAGE PARADOX ANALYSIS: Why Full Coverage = Lower Accuracy")
    print("=" * 70)

    # Load FB15k-237
    print("\nLoading FB15k-237 dataset...")
    train_ds, valid_ds, test_ds = load_fb15k237()

    train_triples = train_ds.triples
    test_triples = test_ds.triples
    num_entities = train_ds.num_entities
    num_relations = train_ds.num_relations

    print(f"Train: {len(train_triples)} triples")
    print(f"Test: {len(test_triples)} triples")
    print(f"Entities: {num_entities}, Relations: {num_relations}")

    # Build coverage matrix
    print("\nBuilding coverage matrix...")
    coverage = build_coverage_matrix(train_triples, num_entities, num_relations)

    # Classify test triples
    print("Classifying test triples by coverage...")
    classification = classify_test_triples(test_triples, coverage)

    print("\nCoverage Distribution:")
    for cat, indices in classification.items():
        pct = 100 * len(indices) / len(test_triples)
        print(f"  {cat}: {len(indices)} ({pct:.1f}%)")

    # Combine partial categories
    partial_indices = np.concatenate([
        classification['partial_head'],
        classification['partial_tail']
    ])
    full_indices = classification['full']
    zero_indices = classification['zero']

    results = []
    results.append("=" * 70)
    results.append("COVERAGE PARADOX ANALYSIS RESULTS")
    results.append("=" * 70)
    results.append("")
    results.append("Dataset: FB15k-237")
    results.append(f"Train triples: {len(train_triples)}")
    results.append(f"Test triples: {len(test_triples)}")
    results.append(f"Entities: {num_entities}, Relations: {num_relations}")
    results.append("")
    results.append("Coverage Distribution:")
    for cat, indices in classification.items():
        pct = 100 * len(indices) / len(test_triples)
        results.append(f"  {cat}: {len(indices)} ({pct:.1f}%)")

    # Analysis 1: Relation Distribution
    print("\n" + "=" * 50)
    print("Analysis 1: Relation Distribution")
    print("=" * 50)

    rel_analysis = analyze_relation_distribution(
        test_triples, classification, train_triples, num_relations
    )

    results.append("")
    results.append("=" * 50)
    results.append("Analysis 1: Relation Distribution")
    results.append("=" * 50)

    for cat, stats in rel_analysis.items():
        print(f"\n{cat}:")
        print(f"  Avg relation frequency: {stats['avg_rel_freq']:.4f}")
        print(f"  Unique relations: {stats['n_unique_rels']}")
        results.append(f"\n{cat}:")
        results.append(f"  Avg relation frequency: {stats['avg_rel_freq']:.4f}")
        results.append(f"  Unique relations: {stats['n_unique_rels']}")

    # Analysis 2: Triple Uniqueness
    print("\n" + "=" * 50)
    print("Analysis 2: Triple Uniqueness Scores")
    print("=" * 50)

    uniqueness = compute_triple_uniqueness_scores(
        test_triples, train_triples, num_entities, num_relations
    )

    results.append("")
    results.append("=" * 50)
    results.append("Analysis 2: Triple Uniqueness Scores")
    results.append("(Higher = more idiosyncratic/rare pattern)")
    results.append("=" * 50)

    for cat_name, indices in [('full', full_indices), ('partial', partial_indices), ('zero', zero_indices)]:
        if len(indices) > 0:
            cat_uniqueness = uniqueness[indices]
            print(f"\n{cat_name} coverage:")
            print(f"  Mean uniqueness: {cat_uniqueness.mean():.4f}")
            print(f"  Std uniqueness: {cat_uniqueness.std():.4f}")
            print(f"  Median: {np.median(cat_uniqueness):.4f}")
            results.append(f"\n{cat_name} coverage:")
            results.append(f"  Mean uniqueness: {cat_uniqueness.mean():.4f}")
            results.append(f"  Std uniqueness: {cat_uniqueness.std():.4f}")
            results.append(f"  Median: {np.median(cat_uniqueness):.4f}")

    # Analysis 3: Training Loss (Difficulty)
    print("\n" + "=" * 50)
    print("Analysis 3: Model-Based Difficulty")
    print("=" * 50)

    difficulty = compute_training_loss_proxy(
        test_triples, train_triples, num_entities, num_relations
    )

    results.append("")
    results.append("=" * 50)
    results.append("Analysis 3: Model-Based Difficulty")
    results.append("(Higher = harder to predict)")
    results.append("=" * 50)

    for cat_name, indices in [('full', full_indices), ('partial', partial_indices), ('zero', zero_indices)]:
        if len(indices) > 0:
            cat_diff = difficulty[indices]
            print(f"\n{cat_name} coverage:")
            print(f"  Mean difficulty: {cat_diff.mean():.4f}")
            print(f"  Std difficulty: {cat_diff.std():.4f}")
            print(f"  Median: {np.median(cat_diff):.4f}")
            results.append(f"\n{cat_name} coverage:")
            results.append(f"  Mean difficulty: {cat_diff.mean():.4f}")
            results.append(f"  Std difficulty: {cat_diff.std():.4f}")
            results.append(f"  Median: {np.median(cat_diff):.4f}")

    # Analysis 4: Compositionality
    print("\n" + "=" * 50)
    print("Analysis 4: Compositionality Scores")
    print("=" * 50)

    compositionality = compute_compositional_score(
        test_triples, train_triples, num_entities, num_relations
    )

    results.append("")
    results.append("=" * 50)
    results.append("Analysis 4: Compositionality Scores")
    results.append("(Higher = more derivable from paths)")
    results.append("=" * 50)

    for cat_name, indices in [('full', full_indices), ('partial', partial_indices), ('zero', zero_indices)]:
        if len(indices) > 0:
            cat_comp = compositionality[indices]
            print(f"\n{cat_name} coverage:")
            print(f"  Mean compositionality: {cat_comp.mean():.4f}")
            print(f"  Std compositionality: {cat_comp.std():.4f}")
            print(f"  Fraction with paths: {(cat_comp > 0).mean():.4f}")
            results.append(f"\n{cat_name} coverage:")
            results.append(f"  Mean compositionality: {cat_comp.mean():.4f}")
            results.append(f"  Std compositionality: {cat_comp.std():.4f}")
            results.append(f"  Fraction with paths: {(cat_comp > 0).mean():.4f}")

    # Summary Statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Full vs Partial Coverage")
    print("=" * 70)

    results.append("")
    results.append("=" * 70)
    results.append("SUMMARY: Full vs Partial Coverage")
    results.append("=" * 70)

    if len(full_indices) > 0 and len(partial_indices) > 0:
        # Effect sizes
        full_unique = uniqueness[full_indices].mean()
        partial_unique = uniqueness[partial_indices].mean()

        full_diff = difficulty[full_indices].mean()
        partial_diff = difficulty[partial_indices].mean()

        full_comp = compositionality[full_indices].mean()
        partial_comp = compositionality[partial_indices].mean()

        summary = f"""
Comparison: Full Coverage vs Partial Zero-Coverage

| Metric           | Full    | Partial | Delta   | Interpretation |
|------------------|---------|---------|---------|----------------|
| Uniqueness       | {full_unique:.3f}   | {partial_unique:.3f}   | {full_unique - partial_unique:+.3f}   | {"Full more unique" if full_unique > partial_unique else "Partial more unique"} |
| Difficulty       | {full_diff:.3f}   | {partial_diff:.3f}   | {full_diff - partial_diff:+.3f}   | {"Full harder" if full_diff > partial_diff else "Partial harder"} |
| Compositionality | {full_comp:.3f}   | {partial_comp:.3f}   | {full_comp - partial_comp:+.3f}   | {"Full more compositional" if full_comp > partial_comp else "Partial more compositional"} |

Hypothesis: Full coverage = idiosyncratic/hard facts
"""
        print(summary)
        results.append(summary)

        # Statistical tests
        from scipy import stats

        t_unique, p_unique = stats.ttest_ind(uniqueness[full_indices], uniqueness[partial_indices])
        t_diff, p_diff = stats.ttest_ind(difficulty[full_indices], difficulty[partial_indices])
        t_comp, p_comp = stats.ttest_ind(compositionality[full_indices], compositionality[partial_indices])

        stat_results = f"""
Statistical Significance (t-test):
  Uniqueness:       t={t_unique:.2f}, p={p_unique:.2e} {'***' if p_unique < 0.001 else '**' if p_unique < 0.01 else '*' if p_unique < 0.05 else ''}
  Difficulty:       t={t_diff:.2f}, p={p_diff:.2e} {'***' if p_diff < 0.001 else '**' if p_diff < 0.01 else '*' if p_diff < 0.05 else ''}
  Compositionality: t={t_comp:.2f}, p={p_comp:.2e} {'***' if p_comp < 0.001 else '**' if p_comp < 0.01 else '*' if p_comp < 0.05 else ''}
"""
        print(stat_results)
        results.append(stat_results)

    # Conclusion
    conclusion = """
CONCLUSION:
-----------
If Full Coverage triples have HIGHER uniqueness and difficulty, this confirms
the hypothesis that these are "idiosyncratic facts" - patterns that must be
memorized rather than inferred compositionally.

This explains the Coverage Paradox:
- Partial coverage: Entity appears in compositional contexts -> easier to predict
- Full coverage: Entity appears in rare/unique patterns -> harder to predict
- Having "more evidence" (full coverage) can actually mean being in a harder distribution!
"""
    print(conclusion)
    results.append(conclusion)

    # Save results
    output_dir = Path("/Users/i767700/Github/kg-bayesian-prior/outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "full_coverage_difficulty_results.txt"

    with open(output_path, 'w') as f:
        f.write('\n'.join(results))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
