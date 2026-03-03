#!/usr/bin/env python3
"""
R-GCN and CompGCN encoder experiments for CAGP.

Tests whether GNN-based embeddings still produce relation-agnostic variance,
confirming Theorem 1's applicability to message-passing architectures.

Implements:
  1. R-GCN encoder -> DistMult scoring -> OOD evaluation
  2. CompGCN encoder -> DistMult scoring -> OOD evaluation
  3. R-GCN + Coverage augmentation
  4. CompGCN + Coverage augmentation

Output: outputs/rgcn_compgcn_results.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import json
import gc
import time

from src.data.loaders import load_wn18rr, load_fb15k237

OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "rgcn_compgcn_results.json"
LOG_PATH = Path(__file__).parent.parent / "outputs" / "rgcn_compgcn.log"

EPOCHS = 30
SEEDS = [42, 123, 456]
DIM = 100
BATCH_SIZE = 512
LR = 0.001
NUM_BASES = 10  # R-GCN basis decomposition


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


# ---- R-GCN Encoder ----

class RGCNLayer(nn.Module):
    """Relational Graph Convolutional Layer with basis decomposition."""
    def __init__(self, in_dim, out_dim, num_relations, num_bases=NUM_BASES):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Basis decomposition: W_r = sum_b a_{r,b} * V_b
        self.bases = nn.Parameter(torch.randn(self.num_bases, in_dim, out_dim) * 0.01)
        self.coefficients = nn.Parameter(torch.randn(num_relations, self.num_bases) * 0.01)
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, entity_emb, edge_index, edge_type):
        """
        entity_emb: (num_entities, in_dim)
        edge_index: (2, num_edges) - source, target
        edge_type: (num_edges,) - relation ids
        """
        num_entities = entity_emb.size(0)
        out = torch.zeros(num_entities, self.out_dim, device=entity_emb.device)

        # Compute per-relation weights from bases
        # coefficients: (num_relations, num_bases)
        # bases: (num_bases, in_dim, out_dim)
        # W_r: (num_relations, in_dim, out_dim)
        W = torch.einsum('rb,bio->rio', self.coefficients, self.bases)

        src, dst = edge_index[0], edge_index[1]

        # Message passing per relation type
        for r in range(self.num_relations):
            mask = edge_type == r
            if mask.sum() == 0:
                continue
            src_r = src[mask]
            dst_r = dst[mask]
            msg = entity_emb[src_r] @ W[r]  # (num_edges_r, out_dim)

            # Aggregate with normalization
            out.index_add_(0, dst_r, msg)

        # Normalize by in-degree
        in_degree = torch.zeros(num_entities, device=entity_emb.device)
        in_degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=entity_emb.device))
        in_degree = in_degree.clamp(min=1).unsqueeze(1)
        out = out / in_degree

        # Self-loop + bias
        out = out + self.self_loop(entity_emb) + self.bias
        return F.relu(out)


class RGCNEncoder(nn.Module):
    """2-layer R-GCN encoder producing entity embeddings."""
    def __init__(self, num_entities, num_relations, dim=DIM):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        self.layer1 = RGCNLayer(dim, dim, num_relations * 2)  # *2 for inverse relations
        self.layer2 = RGCNLayer(dim, dim, num_relations * 2)

    def forward(self, edge_index, edge_type):
        x = self.entity_emb.weight
        x = self.layer1(x, edge_index, edge_type)
        x = self.layer2(x, edge_index, edge_type)
        return x


# ---- CompGCN Encoder ----

class CompGCNLayer(nn.Module):
    """Composition-based GCN layer."""
    def __init__(self, in_dim, out_dim, num_relations, composition='sub'):
        super().__init__()
        self.composition = composition
        self.W_in = nn.Linear(in_dim, out_dim, bias=False)
        self.W_out = nn.Linear(in_dim, out_dim, bias=False)
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_rel = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.bn = nn.BatchNorm1d(out_dim)

    def compose(self, entity_emb, relation_emb):
        if self.composition == 'sub':
            return entity_emb - relation_emb
        elif self.composition == 'mult':
            return entity_emb * relation_emb
        else:  # corr (circular correlation)
            return torch.fft.irfft(
                torch.fft.rfft(entity_emb) * torch.conj(torch.fft.rfft(relation_emb)),
                n=entity_emb.size(-1)
            )

    def forward(self, entity_emb, relation_emb, edge_index, edge_type, num_relations_orig):
        num_entities = entity_emb.size(0)
        out = torch.zeros(num_entities, self.W_in.out_features, device=entity_emb.device)

        src, dst = edge_index[0], edge_index[1]

        # Forward edges (original relations)
        fwd_mask = edge_type < num_relations_orig
        if fwd_mask.sum() > 0:
            src_fwd = src[fwd_mask]
            dst_fwd = dst[fwd_mask]
            rel_fwd = edge_type[fwd_mask]
            composed = self.compose(entity_emb[src_fwd], relation_emb[rel_fwd])
            msg = self.W_in(composed)
            out.index_add_(0, dst_fwd, msg)

        # Inverse edges
        inv_mask = edge_type >= num_relations_orig
        if inv_mask.sum() > 0:
            src_inv = src[inv_mask]
            dst_inv = dst[inv_mask]
            rel_inv = edge_type[inv_mask] - num_relations_orig
            composed = self.compose(entity_emb[src_inv], relation_emb[rel_inv])
            msg = self.W_out(composed)
            out.index_add_(0, dst_inv, msg)

        # Normalize
        in_degree = torch.zeros(num_entities, device=entity_emb.device)
        in_degree.index_add_(0, dst, torch.ones(len(dst), device=entity_emb.device))
        in_degree = in_degree.clamp(min=1).unsqueeze(1)
        out = out / in_degree

        # Self-loop
        out = out + self.W_self(entity_emb) + self.bias

        # Update relation embeddings
        new_rel_emb = self.W_rel(relation_emb)

        out = self.bn(out)
        return F.relu(out), new_rel_emb


class CompGCNEncoder(nn.Module):
    """2-layer CompGCN encoder."""
    def __init__(self, num_entities, num_relations, dim=DIM):
        super().__init__()
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.layer1 = CompGCNLayer(dim, dim, num_relations)
        self.layer2 = CompGCNLayer(dim, dim, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.entity_emb.weight
        r = self.relation_emb.weight
        x, r = self.layer1(x, r, edge_index, edge_type, self.num_relations)
        x, r = self.layer2(x, r, edge_index, edge_type, self.num_relations)
        return x, r


# ---- Full Models ----

class GNNModel(nn.Module):
    """GNN encoder + DistMult scoring + optional coverage."""
    def __init__(self, num_entities, num_relations, dim=DIM, encoder_type='rgcn', use_coverage=False):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.encoder_type = encoder_type
        self.use_coverage = use_coverage

        if encoder_type == 'rgcn':
            self.encoder = RGCNEncoder(num_entities, num_relations, dim)
            self.relation_emb = nn.Embedding(num_relations, dim)
        else:  # compgcn
            self.encoder = CompGCNEncoder(num_entities, num_relations, dim)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))
        self.register_buffer('edge_index', torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer('edge_type', torch.zeros(0, dtype=torch.long))
        self._entity_embeddings = None

    def build_graph(self, triples):
        """Build graph from training triples (with inverse relations)."""
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        # Forward + inverse edges
        edge_index = torch.stack([
            torch.cat([h, t]),
            torch.cat([t, h])
        ])
        edge_type = torch.cat([r, r + self.num_relations])
        self.edge_index = edge_index
        self.edge_type = edge_type

    def encode(self):
        """Run GNN encoder to get entity embeddings."""
        if self.encoder_type == 'rgcn':
            self._entity_embeddings = self.encoder(self.edge_index, self.edge_type)
        else:
            self._entity_embeddings, self._rel_embeddings = self.encoder(
                self.edge_index, self.edge_type
            )

    def forward(self, h, r, t):
        if self._entity_embeddings is None:
            self.encode()
        h_emb = self._entity_embeddings[h]
        t_emb = self._entity_embeddings[t]
        if self.encoder_type == 'compgcn' and hasattr(self, '_rel_embeddings'):
            r_emb = self._rel_embeddings[r]
        else:
            r_emb = self.relation_emb(r)
        return (h_emb * r_emb * t_emb).sum(-1)

    def get_uncertainty(self, h, r, t):
        """Energy-based uncertainty from GNN embeddings + optional coverage."""
        scores = self.forward(h, r, t)
        energy_unc = -scores  # negative score = high uncertainty

        if self.use_coverage:
            struct = 2.0 - self.coverage[h, r] - self.coverage[t, r]
            # Normalize energy to [0, 2] range
            e_min, e_max = energy_unc.min(), energy_unc.max()
            if e_max > e_min:
                energy_norm = 2.0 * (energy_unc - e_min) / (e_max - e_min)
            else:
                energy_norm = torch.zeros_like(energy_unc)
            return 0.5 * energy_norm + 0.5 * struct
        return energy_unc

    def precompute_coverage(self, triples):
        for i in range(len(triples)):
            h, r, t = triples[i, 0].item(), triples[i, 1].item(), triples[i, 2].item()
            if h < self.num_entities and r < self.num_relations:
                self.coverage[h, r] = 1
            if t < self.num_entities and r < self.num_relations:
                self.coverage[t, r] = 1


def train_gnn_model(model, triples, device, epochs=EPOCHS, batch_size=BATCH_SIZE):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    h_all, r_all, t_all = triples[:, 0], triples[:, 1], triples[:, 2]

    # Build graph first
    model.build_graph(triples)
    model.precompute_coverage(triples)

    dataset = TensorDataset(h_all, r_all, t_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        model._entity_embeddings = None  # Re-encode each epoch

        for h, r, t in loader:
            h, r, t = h.to(device), r.to(device), t.to(device)

            # Re-encode for gradient flow
            model.encode()

            pos_scores = model(h, r, t)
            neg_t = torch.randint(0, model.num_entities, t.shape, device=device)
            neg_scores = model(h, r, neg_t)

            loss = (F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) +
                    F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores)))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            model._entity_embeddings = None  # Invalidate cache

        if (epoch + 1) % 5 == 0:
            log(f"    Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")

    model.eval()
    model.encode()  # Final encoding
    return model


def evaluate_ood(model, train_triples, test_triples, device):
    model.eval()

    freq = torch.zeros(model.num_entities, dtype=torch.long)
    for col in [0, 2]:
        for e in train_triples[:, col]:
            freq[e.item()] += 1

    tau = int(np.percentile(freq[freq > 0].numpy(), 25))

    h_test, r_test, t_test = test_triples[:, 0], test_triples[:, 1], test_triples[:, 2]
    min_freq = torch.minimum(freq[h_test], freq[t_test])
    is_emerging = min_freq <= tau

    cov_h = model.coverage[h_test, r_test]
    cov_t = model.coverage[t_test, r_test]
    is_novel = (~is_emerging) & ((cov_h == 0) | (cov_t == 0))
    is_ood = is_emerging | is_novel

    if is_ood.sum() == 0 or (~is_ood).sum() == 0:
        return {'overall': float('nan'), 'emerging': float('nan'), 'novel': float('nan')}

    with torch.no_grad():
        uncertainties = model.get_uncertainty(h_test.to(device), r_test.to(device), t_test.to(device)).cpu().numpy()

    labels = is_ood.numpy().astype(int)
    results = {'overall': float(roc_auc_score(labels, uncertainties))}

    for cat_name, cat_mask in [('emerging', is_emerging), ('novel', is_novel)]:
        if cat_mask.sum() > 0:
            mask = cat_mask | (~is_ood)
            if cat_mask[mask].sum() > 0 and (~is_ood)[mask].sum() > 0:
                results[cat_name] = float(roc_auc_score(cat_mask[mask].numpy(), uncertainties[mask]))

    return results


def main():
    log(f"\n{'='*60}")
    log(f"R-GCN / CompGCN Experiment - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")

    device = torch.device('cpu')
    all_results = {}
    start_time = time.time()

    datasets = {}
    log("Loading WN18RR...")
    train_wn, _, test_wn = load_wn18rr()
    datasets['WN18RR'] = (
        torch.tensor(train_wn.triples, dtype=torch.long),
        torch.tensor(test_wn.triples, dtype=torch.long),
        train_wn.num_entities,
        train_wn.num_relations,
    )

    log("Loading FB15k-237...")
    try:
        train_fb, _, test_fb = load_fb15k237()
        datasets['FB15k-237'] = (
            torch.tensor(train_fb.triples, dtype=torch.long),
            torch.tensor(test_fb.triples, dtype=torch.long),
            train_fb.num_entities,
            train_fb.num_relations,
        )
    except Exception as e:
        log(f"Skipping FB15k-237: {e}")

    configs = [
        ('R-GCN', 'rgcn', False),
        ('R-GCN+Cov', 'rgcn', True),
        ('CompGCN', 'compgcn', False),
        ('CompGCN+Cov', 'compgcn', True),
    ]

    for ds_name, (train_t, test_t, n_ent, n_rel) in datasets.items():
        log(f"\n{'='*60}")
        log(f"Dataset: {ds_name} ({n_ent} entities, {n_rel} relations)")
        log(f"{'='*60}")

        ds_results = {}

        for seed in SEEDS:
            log(f"\n--- Seed {seed} ---")
            seed_results = {}

            for config_name, enc_type, use_cov in configs:
                log(f"  Training {config_name}...")
                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    model = GNNModel(n_ent, n_rel, DIM, enc_type, use_cov).to(device)
                    model = train_gnn_model(model, train_t, device, epochs=EPOCHS)

                    r = evaluate_ood(model, train_t, test_t, device)
                    seed_results[config_name] = r
                    log(f"    {config_name}: overall={r.get('overall', 'N/A'):.4f}, "
                        f"emerging={r.get('emerging', 'N/A')}, novel={r.get('novel', 'N/A')}")
                except Exception as e:
                    log(f"    ERROR in {config_name}: {e}")
                    seed_results[config_name] = {'overall': float('nan'), 'error': str(e)}

                del model; gc.collect()

            ds_results[f'seed_{seed}'] = seed_results

        # Aggregate
        summary = {}
        for config_name, _, _ in configs:
            for metric in ['overall', 'emerging', 'novel']:
                vals = []
                for seed in SEEDS:
                    v = ds_results.get(f'seed_{seed}', {}).get(config_name, {}).get(metric, float('nan'))
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        vals.append(v)
                if vals:
                    summary[f'{config_name}_{metric}'] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'values': vals,
                    }

        all_results[ds_name] = {
            'per_seed': ds_results,
            'summary': summary,
        }

    elapsed = time.time() - start_time
    all_results['metadata'] = {
        'epochs': EPOCHS,
        'seeds': SEEDS,
        'dim': DIM,
        'num_bases': NUM_BASES,
        'elapsed_seconds': elapsed,
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)

    log(f"\nDone! Elapsed: {elapsed:.0f}s")
    log(f"Results saved to {OUTPUT_PATH}")

    # Print summary
    log(f"\n{'='*80}")
    log(f"SUMMARY ({EPOCHS} epochs, {len(SEEDS)} seeds)")
    log(f"{'='*80}")
    for ds_name in datasets:
        log(f"\n{ds_name}:")
        s = all_results[ds_name]['summary']
        for key, val in sorted(s.items()):
            log(f"  {key}: {val['mean']:.4f} ± {val['std']:.4f}")


if __name__ == '__main__':
    main()
