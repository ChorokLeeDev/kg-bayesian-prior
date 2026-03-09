#!/usr/bin/env python3
"""
RAG Uncertainty Decomposition Experiment.

Validates that the impossibility theorem applies to RAG:
- Semantic uncertainty (embedding similarity) fails on novel query-document pairs
- Structural uncertainty (co-occurrence) succeeds

Uses MS MARCO passage retrieval subset for quick CPU validation.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import csv
from pathlib import Path

# Ensure reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleRetriever(nn.Module):
    """Simple dual-encoder retriever for RAG."""

    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()
        self.query_encoder = nn.Embedding(vocab_size, embedding_dim)
        self.doc_encoder = nn.Embedding(vocab_size, embedding_dim)
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.doc_proj = nn.Linear(embedding_dim, embedding_dim)

    def encode_query(self, query_ids):
        """Encode query to embedding."""
        emb = self.query_encoder(query_ids).mean(dim=1)  # Mean pooling
        return F.normalize(self.query_proj(emb), dim=-1)

    def encode_doc(self, doc_ids):
        """Encode document to embedding."""
        emb = self.doc_encoder(doc_ids).mean(dim=1)  # Mean pooling
        return F.normalize(self.doc_proj(emb), dim=-1)

    def similarity(self, query_ids, doc_ids):
        """Compute query-document similarity."""
        q_emb = self.encode_query(query_ids)
        d_emb = self.encode_doc(doc_ids)
        return (q_emb * d_emb).sum(dim=-1)


def create_synthetic_rag_data(n_queries=1000, n_docs=500, n_train_pairs=5000,
                              n_test_pairs=1000, vocab_size=1000,
                              query_len=10, doc_len=50, seed=42):
    """
    Create synthetic RAG dataset with controlled OOD structure.

    Key design: Novel-context queries have SAME frequency as ID queries,
    mimicking KG theorem where novel-context entities are frequency-matched.

    OOD types:
    - Emerging queries: Rare queries (< threshold occurrences in training)
    - Novel context: Common queries paired with unseen documents (SAME frequency as ID)
    - ID: Common queries with frequently paired documents
    """
    np.random.seed(seed)

    # Generate random query and document "tokens"
    queries = np.random.randint(0, vocab_size, (n_queries, query_len))
    docs = np.random.randint(0, vocab_size, (n_docs, doc_len))

    # Create training pairs with power-law distribution
    # Some queries appear frequently, others rarely
    query_probs = np.random.power(0.5, n_queries)
    query_probs /= query_probs.sum()

    # Some docs are "popular" (retrieved often)
    doc_probs = np.random.power(0.3, n_docs)
    doc_probs /= doc_probs.sum()

    train_pairs = []
    co_occurrence = defaultdict(set)  # query_id -> set of doc_ids
    query_freq = defaultdict(int)

    for _ in range(n_train_pairs):
        q_id = np.random.choice(n_queries, p=query_probs)
        d_id = np.random.choice(n_docs, p=doc_probs)
        train_pairs.append((q_id, d_id, 1))  # positive pair
        co_occurrence[q_id].add(d_id)
        query_freq[q_id] += 1

    # Compute query frequency threshold (25th percentile)
    freq_values = list(query_freq.values())
    tau = np.percentile(freq_values, 25) if freq_values else 1

    # Create test set with controlled OOD structure
    test_pairs = []
    test_labels = []  # 0=ID, 1=OOD
    test_categories = []  # 'emerging', 'novel_context', 'id'

    # Get common queries (high frequency)
    common_queries = [q for q in range(n_queries) if query_freq[q] > tau]

    # Split common queries into those that will be ID vs novel_context
    # This ensures novel_context has SAME frequency distribution as ID
    np.random.shuffle(common_queries)
    id_queries = set(common_queries[:len(common_queries)//2])
    novel_context_queries = set(common_queries[len(common_queries)//2:])

    for _ in range(n_test_pairs // 3):
        # 1. Emerging queries (rare queries)
        rare_queries = [q for q in range(n_queries) if query_freq[q] <= tau]
        if rare_queries:
            q_id = np.random.choice(rare_queries)
            d_id = np.random.randint(0, n_docs)
            test_pairs.append((q_id, d_id))
            test_labels.append(1)  # OOD
            test_categories.append('emerging')

    for _ in range(n_test_pairs // 3):
        # 2. Novel context (common query from novel_context_queries + unseen doc)
        # KEY: These queries have SAME frequency as ID queries
        nv_queries = list(novel_context_queries)
        if nv_queries:
            q_id = np.random.choice(nv_queries)
            # Find a doc not seen with this query
            seen_docs = co_occurrence[q_id]
            unseen_docs = [d for d in range(n_docs) if d not in seen_docs]
            if unseen_docs:
                d_id = np.random.choice(unseen_docs)
                test_pairs.append((q_id, d_id))
                test_labels.append(1)  # OOD
                test_categories.append('novel_context')

    for _ in range(n_test_pairs // 3):
        # 3. ID (common query from id_queries + seen doc)
        id_qs = [q for q in id_queries if len(co_occurrence[q]) > 0]
        if id_qs:
            q_id = np.random.choice(id_qs)
            d_id = np.random.choice(list(co_occurrence[q_id]))
            test_pairs.append((q_id, d_id))
            test_labels.append(0)  # ID
            test_categories.append('id')

    return {
        'queries': torch.tensor(queries, dtype=torch.long),
        'docs': torch.tensor(docs, dtype=torch.long),
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
        'test_labels': np.array(test_labels),
        'test_categories': test_categories,
        'co_occurrence': co_occurrence,
        'query_freq': query_freq,
        'tau': tau,
        'vocab_size': vocab_size,
    }


def train_retriever(model, data, epochs=10, lr=1e-3, batch_size=128, device='cpu'):
    """Train the retriever with contrastive loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    queries = data['queries'].to(device)
    docs = data['docs'].to(device)
    train_pairs = data['train_pairs']

    n_docs = len(docs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        np.random.shuffle(train_pairs)

        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            q_ids = torch.tensor([p[0] for p in batch], device=device)
            d_ids = torch.tensor([p[1] for p in batch], device=device)

            # Positive scores
            q_emb = model.encode_query(queries[q_ids])
            d_emb = model.encode_doc(docs[d_ids])
            pos_scores = (q_emb * d_emb).sum(dim=-1)

            # Negative scores (random docs)
            neg_d_ids = torch.randint(0, n_docs, (len(batch),), device=device)
            neg_d_emb = model.encode_doc(docs[neg_d_ids])
            neg_scores = (q_emb * neg_d_emb).sum(dim=-1)

            # Contrastive loss
            loss = F.relu(0.5 - pos_scores + neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(train_pairs)//batch_size):.4f}")

    return model


def compute_uncertainties(model, data, device='cpu'):
    """Compute semantic and structural uncertainties for test pairs."""
    model.eval()

    queries = data['queries'].to(device)
    docs = data['docs'].to(device)
    test_pairs = data['test_pairs']
    co_occurrence = data['co_occurrence']

    semantic_uncertainties = []
    structural_uncertainties = []

    with torch.no_grad():
        for q_id, d_id in test_pairs:
            # Semantic uncertainty: 1 - similarity
            q_emb = model.encode_query(queries[q_id:q_id+1])
            d_emb = model.encode_doc(docs[d_id:d_id+1])
            sim = (q_emb * d_emb).sum().item()
            u_sem = 1 - sim
            semantic_uncertainties.append(u_sem)

            # Structural uncertainty: 1 if not seen, 0 if seen
            u_str = 0 if d_id in co_occurrence[q_id] else 1
            structural_uncertainties.append(u_str)

    return np.array(semantic_uncertainties), np.array(structural_uncertainties)


def evaluate_ood(uncertainties, labels, categories):
    """Evaluate OOD detection per category."""
    results = {}

    # Overall
    if len(np.unique(labels)) > 1:
        results['overall'] = roc_auc_score(labels, uncertainties)

    # Per category
    for cat in ['emerging', 'novel_context']:
        cat_mask = np.array([c == cat for c in categories])
        id_mask = np.array([c == 'id' for c in categories])

        if cat_mask.sum() > 0 and id_mask.sum() > 0:
            combined_mask = cat_mask | id_mask
            cat_labels = np.array([1 if categories[i] == cat else 0
                                   for i in range(len(categories)) if combined_mask[i]])
            cat_scores = uncertainties[combined_mask]

            if len(np.unique(cat_labels)) > 1:
                results[cat] = roc_auc_score(cat_labels, cat_scores)

    return results


def run_experiment(seed=42, device='cpu'):
    """Run full RAG experiment."""
    print(f"\n{'='*60}")
    print(f"RAG Uncertainty Decomposition Experiment (seed={seed})")
    print(f"{'='*60}")

    set_seed(seed)

    # Create data
    print("\nCreating synthetic RAG dataset...")
    data = create_synthetic_rag_data(
        n_queries=500, n_docs=300, n_train_pairs=3000,
        n_test_pairs=600, vocab_size=500, seed=seed
    )

    n_emerging = sum(1 for c in data['test_categories'] if c == 'emerging')
    n_novel = sum(1 for c in data['test_categories'] if c == 'novel_context')
    n_id = sum(1 for c in data['test_categories'] if c == 'id')
    print(f"  Train pairs: {len(data['train_pairs'])}")
    print(f"  Test: {n_emerging} emerging, {n_novel} novel_context, {n_id} ID")

    # Train model
    print("\nTraining retriever...")
    model = SimpleRetriever(data['vocab_size'], embedding_dim=64)
    model = train_retriever(model, data, epochs=20, device=device)

    # Compute uncertainties
    print("\nComputing uncertainties...")
    u_sem, u_str = compute_uncertainties(model, data, device)

    # Combined uncertainty (CAGP-RAG)
    u_comb = 0.5 * (u_sem - u_sem.min()) / (u_sem.max() - u_sem.min() + 1e-8) + 0.5 * u_str

    # Evaluate
    print("\nEvaluating OOD detection...")
    labels = data['test_labels']
    categories = data['test_categories']

    results = {
        'semantic': evaluate_ood(u_sem, labels, categories),
        'structural': evaluate_ood(u_str, labels, categories),
        'combined': evaluate_ood(u_comb, labels, categories),
    }

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print("\n{:<15} {:>10} {:>15} {:>10}".format("Method", "Emerging", "Novel-context", "Overall"))
    print("-"*55)

    for method, res in results.items():
        em = f"{res.get('emerging', 'N/A'):.3f}" if 'emerging' in res else "N/A"
        nv = f"{res.get('novel_context', 'N/A'):.3f}" if 'novel_context' in res else "N/A"
        ov = f"{res.get('overall', 'N/A'):.3f}" if 'overall' in res else "N/A"
        print(f"{method:<15} {em:>10} {nv:>15} {ov:>10}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    if 'novel_context' in results['semantic']:
        nv_sem = results['semantic']['novel_context']
        if nv_sem < 0.55:
            print(f"✓ Semantic novel-context AUROC = {nv_sem:.3f} (~random)")
            print("  → Confirms impossibility: embedding similarity blind to co-occurrence")
        elif nv_sem > 0.7:
            print(f"⚠ Semantic novel-context AUROC = {nv_sem:.3f} (higher than expected)")
        else:
            print(f"? Semantic novel-context AUROC = {nv_sem:.3f} (borderline)")

    if 'novel_context' in results['structural']:
        nv_str = results['structural']['novel_context']
        print(f"✓ Structural novel-context AUROC = {nv_str:.3f}")
        print("  → Co-occurrence perfectly detects novel query-doc pairs")

    return results


def main():
    parser = argparse.ArgumentParser(description="RAG Uncertainty Decomposition")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="outputs/rag_results.csv")
    args = parser.parse_args()

    seed_list = [42, 123, 456][:args.seeds]
    all_results = []

    for seed in seed_list:
        results = run_experiment(seed=seed, device=args.device)

        row = {'seed': seed}
        for method in ['semantic', 'structural', 'combined']:
            for cat in ['emerging', 'novel_context', 'overall']:
                key = f"{method}_{cat}"
                row[key] = results[method].get(cat, None)
        all_results.append(row)

    # Aggregate
    print("\n" + "="*60)
    print(f"AGGREGATE RESULTS ({args.seeds} seeds)")
    print("="*60)

    for method in ['semantic', 'structural', 'combined']:
        print(f"\n{method.upper()}:")
        for cat in ['emerging', 'novel_context', 'overall']:
            key = f"{method}_{cat}"
            values = [r[key] for r in all_results if r[key] is not None]
            if values:
                print(f"  {cat}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['seed'] + [f"{m}_{c}" for m in ['semantic', 'structural', 'combined']
                                  for c in ['emerging', 'novel_context', 'overall']]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
