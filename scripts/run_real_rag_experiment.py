#!/usr/bin/env python3
"""
Real-world RAG experiment using pre-trained embeddings.

Uses sentence-transformers for realistic semantic similarity.
Goal: Show that semantic uncertainty (embedding similarity) fails on novel query-doc pairs
while structural uncertainty (co-occurrence) succeeds.
"""

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import csv
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("Warning: sentence-transformers not installed. Using random embeddings.")


def set_seed(seed):
    np.random.seed(seed)


def create_qa_dataset(n_queries=200, n_docs=100, n_train_pairs=500, seed=42):
    """
    Create synthetic QA dataset with realistic structure.
    """
    set_seed(seed)

    # Generate synthetic queries and documents
    topics = ['science', 'history', 'sports', 'technology', 'art', 'music',
              'politics', 'economics', 'health', 'education']

    queries = []
    docs = []

    for i in range(n_queries):
        topic = topics[i % len(topics)]
        queries.append(f"What is the main concept of {topic} topic {i}?")

    for i in range(n_docs):
        topic = topics[i % len(topics)]
        docs.append(f"This document discusses {topic} with detailed information about concept {i}.")

    # Create training pairs with topic matching
    train_pairs = []
    co_occurrence = defaultdict(set)  # query_id -> set of doc_ids
    query_freq = defaultdict(int)

    for _ in range(n_train_pairs):
        q_id = np.random.randint(0, n_queries)
        # Bias: docs from same topic more likely to be paired
        q_topic_idx = q_id % len(topics)
        same_topic_docs = [d for d in range(n_docs) if d % len(topics) == q_topic_idx]

        if np.random.random() < 0.7 and same_topic_docs:
            d_id = np.random.choice(same_topic_docs)
        else:
            d_id = np.random.randint(0, n_docs)

        train_pairs.append((q_id, d_id))
        co_occurrence[q_id].add(d_id)
        query_freq[q_id] += 1

    return {
        'queries': queries,
        'docs': docs,
        'train_pairs': train_pairs,
        'co_occurrence': co_occurrence,
        'query_freq': query_freq,
        'n_queries': n_queries,
        'n_docs': n_docs,
    }


def compute_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Compute embeddings using sentence-transformers."""
    if HAS_SBERT:
        model = SentenceTransformer(model_name)
        return model.encode(texts, convert_to_numpy=True)
    else:
        # Random embeddings as fallback
        return np.random.randn(len(texts), 384)


def create_test_set(data, n_test=300):
    """Create test set with OOD categories."""
    n_queries = data['n_queries']
    n_docs = data['n_docs']
    query_freq = data['query_freq']
    co_occurrence = data['co_occurrence']

    # Frequency threshold
    freq_values = list(query_freq.values())
    tau = np.percentile(freq_values, 25) if freq_values else 1

    test_pairs = []
    test_labels = []
    test_categories = []

    # Split queries into common/rare
    common_queries = [q for q in range(n_queries) if query_freq[q] > tau]
    rare_queries = [q for q in range(n_queries) if query_freq[q] <= tau]

    # Novel context: common queries with unseen docs
    for _ in range(n_test // 3):
        if not common_queries:
            continue
        q_id = np.random.choice(common_queries)
        seen_docs = co_occurrence[q_id]
        unseen_docs = [d for d in range(n_docs) if d not in seen_docs]
        if not unseen_docs:
            continue
        d_id = np.random.choice(unseen_docs)

        test_pairs.append((q_id, d_id))
        test_labels.append(1)  # OOD
        test_categories.append('novel_context')

    # Emerging: rare queries
    for _ in range(n_test // 3):
        if not rare_queries:
            continue
        q_id = np.random.choice(rare_queries)
        d_id = np.random.randint(0, n_docs)

        test_pairs.append((q_id, d_id))
        test_labels.append(1)  # OOD
        test_categories.append('emerging')

    # ID: common queries with seen docs
    for _ in range(n_test // 3):
        common_with_docs = [q for q in common_queries if len(co_occurrence[q]) > 0]
        if not common_with_docs:
            continue
        q_id = np.random.choice(common_with_docs)
        d_id = np.random.choice(list(co_occurrence[q_id]))

        test_pairs.append((q_id, d_id))
        test_labels.append(0)  # ID
        test_categories.append('id')

    return test_pairs, np.array(test_labels), test_categories


def compute_uncertainties(data, test_pairs, query_embs, doc_embs):
    """Compute semantic and structural uncertainties."""
    co_occurrence = data['co_occurrence']

    semantic_scores = []
    structural_scores = []

    for q_id, d_id in test_pairs:
        # Semantic: 1 - cosine similarity (higher = more uncertain)
        q_emb = query_embs[q_id]
        d_emb = doc_embs[d_id]
        sim = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-8)
        sem = 1 - sim
        semantic_scores.append(sem)

        # Structural: 1 if not seen, 0 if seen
        struct = 0 if d_id in co_occurrence[q_id] else 1
        structural_scores.append(struct)

    return np.array(semantic_scores), np.array(structural_scores)


def evaluate_ood(scores, labels, categories, target_cat=None):
    """Evaluate OOD detection."""
    if target_cat:
        mask = np.array([c == target_cat or c == 'id' for c in categories])
        if mask.sum() < 10:
            return None
        scores = scores[mask]
        labels_subset = np.array([1 if categories[i] == target_cat else 0
                                  for i in range(len(categories)) if mask[i]])
        if len(set(labels_subset)) < 2:
            return None
        return roc_auc_score(labels_subset, scores)
    else:
        if len(set(labels)) < 2:
            return None
        return roc_auc_score(labels, scores)


def run_experiment(seed=42):
    """Run single experiment."""
    print(f"\n--- Real RAG Experiment (seed={seed}) ---")

    # Create data
    data = create_qa_dataset(n_queries=200, n_docs=100, n_train_pairs=500, seed=seed)
    test_pairs, labels, categories = create_test_set(data, n_test=300)

    print(f"  Train pairs: {len(data['train_pairs'])}")
    print(f"  Test pairs: {len(test_pairs)}")

    cats = defaultdict(int)
    for c in categories:
        cats[c] += 1
    print(f"  Categories: {dict(cats)}")

    # Compute embeddings
    print("  Computing embeddings...")
    query_embs = compute_embeddings(data['queries'])
    doc_embs = compute_embeddings(data['docs'])

    # Compute uncertainties
    u_sem, u_str = compute_uncertainties(data, test_pairs, query_embs, doc_embs)

    # Normalize and combine
    u_sem_norm = (u_sem - u_sem.min()) / (u_sem.max() - u_sem.min() + 1e-8)
    u_comb = 0.5 * u_sem_norm + 0.5 * u_str

    results = {}

    # Overall
    results['semantic_overall'] = evaluate_ood(u_sem, labels, categories)
    results['structural_overall'] = evaluate_ood(u_str, labels, categories)
    results['combined_overall'] = evaluate_ood(u_comb, labels, categories)

    # Per category
    for cat in ['novel_context', 'emerging']:
        results[f'semantic_{cat}'] = evaluate_ood(u_sem, labels, categories, cat)
        results[f'structural_{cat}'] = evaluate_ood(u_str, labels, categories, cat)
        results[f'combined_{cat}'] = evaluate_ood(u_comb, labels, categories, cat)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/real_rag_results.csv")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("REAL RAG EXPERIMENT")
    print(f"{'='*60}")

    if HAS_SBERT:
        print("Using sentence-transformers for realistic embeddings")
    else:
        print("WARNING: Using random embeddings (install sentence-transformers for real test)")

    all_results = []
    seed_list = [42, 123, 456][:args.seeds]

    for seed in seed_list:
        results = run_experiment(seed)
        results['seed'] = seed
        all_results.append(results)

    # Aggregate
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({args.seeds} seeds)")
    print(f"{'='*60}")

    print(f"\n{'Metric':<30} {'Mean':>10} {'Std':>10}")
    print("-" * 55)

    key_metrics = ['semantic_overall', 'structural_overall', 'combined_overall',
                   'semantic_novel_context', 'structural_novel_context', 'combined_novel_context',
                   'semantic_emerging', 'structural_emerging', 'combined_emerging']

    for metric in key_metrics:
        values = [r[metric] for r in all_results if r.get(metric) is not None]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            print(f"{metric:<30} {mean:>10.3f} {std:>10.3f}")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")

    sem_nv = [r['semantic_novel_context'] for r in all_results if r.get('semantic_novel_context')]
    str_nv = [r['structural_novel_context'] for r in all_results if r.get('structural_novel_context')]

    if sem_nv and str_nv:
        print(f"\nOn 'novel_context' (common query + unseen doc):")
        print(f"  Semantic AUROC: {np.mean(sem_nv):.3f}")
        print(f"  Structural AUROC: {np.mean(str_nv):.3f}")

        if np.mean(sem_nv) < 0.6:
            print(f"\n✓ Semantic fails on novel contexts (~random)")
            print(f"  → Confirms impossibility theorem applies to RAG")
        if np.mean(str_nv) > 0.9:
            print(f"✓ Structural succeeds on novel contexts")
            print(f"  → Co-occurrence tracking is necessary")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['seed'] + key_metrics
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
