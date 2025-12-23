"""
Score-Based Uncertainty Methods (UKGE, Energy-based)

These methods use the model's output scores directly for uncertainty estimation,
rather than learned variance or coverage signals.

Key insight from reviewer: "On random OOD, score-based methods (UKGE, Energy)
achieve near-perfect 0.99 AUROC, outperforming CAGP's 0.96."

We add these baselines and test them on adversarial settings where
they are expected to fail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class UKGEStyleUncertainty(nn.Module):
    """
    UKGE-style confidence scoring.

    UKGE (Chen et al., 2019) associates confidence scores with triples.
    For OOD detection, we use the model's predicted probability as confidence:
    - High confidence (probability close to 1) → low uncertainty
    - Low confidence (probability close to 0.5 or 0) → high uncertainty

    This works well when OOD samples produce obviously wrong predictions,
    but fails when adversarial corruptions are plausible.
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int,
                 scoring: str = 'distmult'):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.scoring = scoring

        # Entity and relation embeddings
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Confidence calibration network (optional, for learned confidence)
        self.confidence_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Coverage for comparison
        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, heads, relations, tails):
        """Score triples."""
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)

        if self.scoring == 'distmult':
            return (h * r * t).sum(dim=-1)
        elif self.scoring == 'transe':
            return -torch.norm(h + r - t, p=1, dim=-1)
        else:
            return (h * r * t).sum(dim=-1)

    def get_confidence(self, heads, relations, tails):
        """
        Get confidence score for triples.

        High score → high confidence → low uncertainty
        """
        scores = self.forward(heads, relations, tails)
        probs = torch.sigmoid(scores)

        # Confidence = how far from 0.5 (maximum uncertainty point)
        confidence = torch.abs(probs - 0.5) * 2  # Scale to [0, 1]

        return confidence

    def get_uncertainty(self, heads, relations, tails):
        """
        Uncertainty = 1 - confidence.

        This is the UKGE-style uncertainty that works well on random OOD
        but fails on adversarial corruptions.
        """
        confidence = self.get_confidence(heads, relations, tails)
        return 1 - confidence

    def get_uncertainty_with_coverage(self, heads, relations, tails, alpha=0.5):
        """
        Hybrid uncertainty combining score-based and coverage.

        This tests whether adding coverage helps score-based methods.
        """
        score_unc = self.get_uncertainty(heads, relations, tails)

        h_seen = self.coverage[heads, relations]
        t_seen = self.coverage[tails, relations]
        cov_unc = 2.0 - h_seen - t_seen

        # Normalize
        score_unc_norm = score_unc / (score_unc.mean() + 1e-8) * cov_unc.mean()

        return alpha * score_unc_norm + (1 - alpha) * cov_unc

    def precompute_coverage(self, triples, entity_to_idx, relation_to_idx):
        for h, r, t in triples:
            h_idx = entity_to_idx[h]
            r_idx = relation_to_idx[r]
            t_idx = entity_to_idx[t]
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0


class EnergyBasedUncertainty(nn.Module):
    """
    Energy-based OOD detection (Liu et al., 2020).

    Uses the negative log-sum-exp of logits as an energy score:
    E(x) = -T * log(Σ exp(f_i(x) / T))

    Lower energy → in-distribution
    Higher energy → out-of-distribution

    For KG embeddings, we adapt this to use triple scores.
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int,
                 temperature: float = 1.0, scoring: str = 'distmult'):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.temperature = temperature
        self.scoring = scoring

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.register_buffer('coverage', torch.zeros(num_entities, num_relations))

    def forward(self, heads, relations, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)

        if self.scoring == 'distmult':
            return (h * r * t).sum(dim=-1)
        elif self.scoring == 'transe':
            return -torch.norm(h + r - t, p=1, dim=-1)
        else:
            return (h * r * t).sum(dim=-1)

    def score_all_tails(self, heads, relations):
        """Score all possible tails for given (h, r, ?) queries."""
        h = self.entity_emb(heads)  # (B, dim)
        r = self.relation_emb(relations)  # (B, dim)
        all_t = self.entity_emb.weight  # (E, dim)

        if self.scoring == 'distmult':
            query = h * r  # (B, dim)
            scores = torch.mm(query, all_t.t())  # (B, E)
        elif self.scoring == 'transe':
            query = h + r  # (B, dim)
            scores = -torch.cdist(query, all_t, p=1)  # (B, E)
        else:
            query = h * r
            scores = torch.mm(query, all_t.t())

        return scores

    def get_energy(self, heads, relations, tails):
        """
        Compute energy for triples.

        Energy = -T * log(Σ_t' exp(score(h, r, t') / T))

        Lower energy = more in-distribution = lower uncertainty
        """
        # Get scores for all possible tails
        all_scores = self.score_all_tails(heads, relations)  # (B, E)

        # Energy is negative log-sum-exp
        energy = -self.temperature * torch.logsumexp(
            all_scores / self.temperature, dim=-1
        )

        return energy

    def get_uncertainty(self, heads, relations, tails):
        """
        Uncertainty = energy (higher energy → more uncertain).

        Note: We don't actually use the specific tail for energy computation,
        since energy is defined over the (h, r) query.
        """
        energy = self.get_energy(heads, relations, tails)

        # Normalize to positive values (shift by min energy)
        energy = energy - energy.min() + 1e-6

        return energy

    def get_score_based_uncertainty(self, heads, relations, tails):
        """
        Alternative: Use negative score as uncertainty.

        This is simpler and often works as well as energy.
        """
        scores = self.forward(heads, relations, tails)
        # Lower score → higher uncertainty
        return -scores

    def precompute_coverage(self, triples, entity_to_idx, relation_to_idx):
        for h, r, t in triples:
            h_idx = entity_to_idx[h]
            r_idx = relation_to_idx[r]
            t_idx = entity_to_idx[t]
            self.coverage[h_idx, r_idx] = 1.0
            self.coverage[t_idx, r_idx] = 1.0


class AdversarialOODGenerator:
    """
    Generate adversarial OOD samples that target specific uncertainty signals.

    This addresses the reviewer concern: "The paper hand-waves that [UKGE/Energy]
    may fail on adversarial corruptions but doesn't test them."
    """

    def __init__(self, model, num_entities: int, coverage: torch.Tensor,
                 entity_freq: Dict[int, int]):
        self.model = model
        self.num_entities = num_entities
        self.coverage = coverage
        self.entity_freq = entity_freq

    def generate_relation_plausible(self, triples, k=10):
        """
        Replace tail with entity that commonly appears with this relation.

        This fools coverage-based detection since the replacement has valid coverage.
        """
        ood_triples = []
        device = self.coverage.device

        for h, r, t in triples:
            # Find entities that appear with this relation
            relation_entities = (self.coverage[:, r] > 0).nonzero(as_tuple=True)[0]

            if len(relation_entities) > 0:
                # Sample from entities seen with this relation
                idx = torch.randint(0, len(relation_entities), (1,)).item()
                new_t = relation_entities[idx].item()
            else:
                new_t = torch.randint(0, self.num_entities, (1,)).item()

            ood_triples.append((h, r, new_t))

        return ood_triples

    def generate_embedding_similar(self, triples, k=10):
        """
        Replace tail with embedding-similar entity.

        This fools GP-variance since similar embeddings have similar uncertainty.
        """
        self.model.eval()
        ood_triples = []

        with torch.no_grad():
            # Precompute all entity embeddings
            if hasattr(self.model, 'entity_emb'):
                all_emb = self.model.entity_emb.weight
            elif hasattr(self.model, 'entity_mean'):
                all_emb = self.model.entity_mean
            else:
                # Fallback: random
                return [(h, r, torch.randint(0, self.num_entities, (1,)).item())
                        for h, r, t in triples]

            for h, r, t in triples:
                t_emb = all_emb[t:t+1]  # (1, dim)

                # Find k nearest neighbors
                dists = torch.cdist(t_emb, all_emb).squeeze(0)  # (E,)
                dists[t] = float('inf')  # Exclude self

                _, nn_indices = torch.topk(dists, k, largest=False)
                new_t = nn_indices[torch.randint(0, k, (1,)).item()].item()

                ood_triples.append((h, r, new_t))

        return ood_triples

    def generate_popularity_matched(self, triples):
        """
        Replace tail with entity of similar training frequency.

        This fools GP-variance since frequency correlates with variance.
        """
        ood_triples = []

        # Sort entities by frequency
        sorted_entities = sorted(self.entity_freq.items(), key=lambda x: x[1])
        freq_to_entities = {}
        for e, f in sorted_entities:
            if f not in freq_to_entities:
                freq_to_entities[f] = []
            freq_to_entities[f].append(e)

        for h, r, t in triples:
            t_freq = self.entity_freq.get(t, 0)

            # Find entities with similar frequency
            candidates = []
            for f in range(max(0, t_freq - 5), t_freq + 6):
                candidates.extend(freq_to_entities.get(f, []))

            if candidates:
                candidates = [c for c in candidates if c != t]
                if candidates:
                    new_t = candidates[torch.randint(0, len(candidates), (1,)).item()]
                else:
                    new_t = torch.randint(0, self.num_entities, (1,)).item()
            else:
                new_t = torch.randint(0, self.num_entities, (1,)).item()

            ood_triples.append((h, r, new_t))

        return ood_triples

    def generate_high_score(self, triples, k=10):
        """
        Replace tail with entity that gets high model score.

        This fools score-based methods (UKGE, Energy) since they rely on score magnitude.
        """
        self.model.eval()
        ood_triples = []

        with torch.no_grad():
            for h, r, t in triples:
                h_t = torch.tensor([h])
                r_t = torch.tensor([r])

                # Get scores for all tails
                if hasattr(self.model, 'score_all_tails'):
                    scores = self.model.score_all_tails(h_t, r_t).squeeze(0)
                else:
                    # Fallback: score individually
                    all_tails = torch.arange(self.num_entities)
                    h_exp = h_t.expand(self.num_entities)
                    r_exp = r_t.expand(self.num_entities)
                    scores = self.model(h_exp, r_exp, all_tails)

                # Exclude true tail and get top-k
                scores[t] = float('-inf')
                _, top_k = torch.topk(scores, k)
                new_t = top_k[torch.randint(0, k, (1,)).item()].item()

                ood_triples.append((h, r, new_t))

        return ood_triples


def run_adversarial_comparison(models: Dict[str, nn.Module],
                               test_triples,
                               train_triples,
                               entity_to_idx,
                               relation_to_idx,
                               device):
    """
    Run comprehensive adversarial OOD comparison.

    This directly addresses the reviewer's critique about missing comparisons.
    """
    from sklearn.metrics import roc_auc_score
    import numpy as np

    num_entities = len(entity_to_idx)

    # Compute entity frequencies
    entity_freq = {}
    for h, r, t in train_triples:
        h_idx = entity_to_idx[h]
        t_idx = entity_to_idx[t]
        entity_freq[h_idx] = entity_freq.get(h_idx, 0) + 1
        entity_freq[t_idx] = entity_freq.get(t_idx, 0) + 1

    # Convert test triples to indices
    test_idx = [(entity_to_idx[h], relation_to_idx[r], entity_to_idx[t])
                for h, r, t in test_triples]

    results = {name: {} for name in models}

    # Use first model to generate adversarial samples (they should transfer)
    first_model = list(models.values())[0]
    coverage = first_model.coverage if hasattr(first_model, 'coverage') else None

    if coverage is not None:
        generator = AdversarialOODGenerator(first_model, num_entities, coverage, entity_freq)

        ood_settings = {
            'random': [(h, r, np.random.randint(num_entities)) for h, r, t in test_idx],
            'relation_plausible': generator.generate_relation_plausible(test_idx),
            'embedding_similar': generator.generate_embedding_similar(test_idx),
            'popularity_matched': generator.generate_popularity_matched(test_idx),
        }
    else:
        ood_settings = {
            'random': [(h, r, np.random.randint(num_entities)) for h, r, t in test_idx],
        }

    for name, model in models.items():
        model.eval()
        model = model.to(device)

        for ood_name, ood_triples in ood_settings.items():
            with torch.no_grad():
                # ID uncertainty
                h_id = torch.tensor([h for h, _, _ in test_idx]).to(device)
                r_id = torch.tensor([r for _, r, _ in test_idx]).to(device)
                t_id = torch.tensor([t for _, _, t in test_idx]).to(device)
                id_unc = model.get_uncertainty(h_id, r_id, t_id).cpu().numpy()

                # OOD uncertainty
                h_ood = torch.tensor([h for h, _, _ in ood_triples]).to(device)
                r_ood = torch.tensor([r for _, r, _ in ood_triples]).to(device)
                t_ood = torch.tensor([t for _, _, t in ood_triples]).to(device)
                ood_unc = model.get_uncertainty(h_ood, r_ood, t_ood).cpu().numpy()

            # AUROC
            labels = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
            scores = np.concatenate([id_unc, ood_unc])

            try:
                auroc = roc_auc_score(labels, scores)
            except:
                auroc = 0.5

            results[name][ood_name] = auroc

    return results
