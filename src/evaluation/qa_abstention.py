"""
Downstream NLP Task: Question Answering with Abstention

This module evaluates uncertainty quantification on a downstream NLP task:
- Given a QA query over a knowledge graph, should the system answer or abstain?

Task Setup:
1. Simulate KG-QA queries as (h, r, ?) link prediction
2. Some queries are "answerable" (entity exists in KG with high confidence)
3. Some queries are "unanswerable" (entity missing, relation unseen, or low confidence)
4. The system should abstain on unanswerable queries

Metrics:
- Selective Prediction Accuracy: Accuracy when the system chooses to answer
- Abstention Quality: How well does uncertainty predict unanswerability?
- Risk-Coverage Curve: Trade-off between coverage and accuracy

This directly addresses reviewer concern:
"For EMNLP, reviewers will expect to see how this uncertainty helps actual
NLP systems make better decisions"
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from dataclasses import dataclass


@dataclass
class QAQuery:
    """Represents a QA query over a knowledge graph."""
    head: int
    relation: int
    true_answer: Optional[int]  # None if unanswerable
    is_answerable: bool
    query_type: str  # 'normal', 'unseen_entity', 'unseen_relation', 'rare_entity'


class QAAbstentionEvaluator:
    """
    Evaluates QA systems with abstention capability.

    The key insight: A well-calibrated uncertainty estimate allows the system
    to abstain on queries it cannot reliably answer, improving overall quality.
    """

    def __init__(self, model, entity_frequencies: Dict[int, int],
                 relation_frequencies: Dict[int, int],
                 coverage_matrix: Optional[torch.Tensor] = None):
        """
        Args:
            model: Trained KGE model with uncertainty estimation
            entity_frequencies: Count of each entity in training data
            relation_frequencies: Count of each relation in training data
            coverage_matrix: (num_entities, num_relations) coverage matrix
        """
        self.model = model
        self.entity_freq = entity_frequencies
        self.relation_freq = relation_frequencies
        self.coverage = coverage_matrix

    def generate_qa_benchmark(
        self,
        test_triples: List[Tuple[int, int, int]],
        num_entities: int,
        unanswerable_ratio: float = 0.3,
        seed: int = 42
    ) -> List[QAQuery]:
        """
        Generate a QA benchmark with both answerable and unanswerable queries.

        Unanswerable query types:
        1. Unseen entity-relation pair (entity exists but never with this relation)
        2. Rare entity (entity seen < 3 times in training)
        3. Random corruption (simulates completely wrong queries)
        """
        np.random.seed(seed)
        queries = []

        # Answerable queries from test set
        for h, r, t in test_triples:
            queries.append(QAQuery(
                head=h, relation=r, true_answer=t,
                is_answerable=True, query_type='normal'
            ))

        num_unanswerable = int(len(test_triples) * unanswerable_ratio)

        # Type 1: Unseen entity-relation pairs
        if self.coverage is not None:
            for h, r, t in test_triples[:num_unanswerable // 3]:
                # Find a relation this entity hasn't been seen with
                entity_coverage = self.coverage[h]
                unseen_relations = (entity_coverage == 0).nonzero(as_tuple=True)[0]
                if len(unseen_relations) > 0:
                    new_r = unseen_relations[np.random.randint(len(unseen_relations))].item()
                    queries.append(QAQuery(
                        head=h, relation=new_r, true_answer=None,
                        is_answerable=False, query_type='unseen_relation'
                    ))

        # Type 2: Rare entities
        rare_threshold = 3
        rare_entities = [e for e, freq in self.entity_freq.items() if freq < rare_threshold]
        if rare_entities:
            for _ in range(num_unanswerable // 3):
                h = np.random.choice(rare_entities)
                r = np.random.randint(0, len(self.relation_freq))
                queries.append(QAQuery(
                    head=h, relation=r, true_answer=None,
                    is_answerable=False, query_type='rare_entity'
                ))

        # Type 3: Random corruption (completely fabricated)
        for _ in range(num_unanswerable // 3):
            h = np.random.randint(0, num_entities)
            r = np.random.randint(0, len(self.relation_freq))
            queries.append(QAQuery(
                head=h, relation=r, true_answer=None,
                is_answerable=False, query_type='random_corruption'
            ))

        np.random.shuffle(queries)
        return queries

    def predict_with_abstention(
        self,
        queries: List[QAQuery],
        threshold: float,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Make predictions with abstention based on uncertainty threshold.

        Returns metrics on selective prediction quality.
        """
        self.model.eval()

        predictions = []
        uncertainties = []
        is_correct = []
        is_answerable = []

        with torch.no_grad():
            for query in queries:
                h = torch.tensor([query.head]).to(device)
                r = torch.tensor([query.relation]).to(device)

                # Get uncertainty by averaging over multiple candidate tails
                # This avoids bias from using a single placeholder entity
                num_samples = 10
                sample_tails = torch.randint(0, self.model.num_entities, (num_samples,)).to(device)
                h_exp = h.expand(num_samples)
                r_exp = r.expand(num_samples)
                sample_uncertainties = self.model.get_uncertainty(h_exp, r_exp, sample_tails)
                uncertainty = sample_uncertainties.mean().item()
                uncertainties.append(uncertainty)

                # Get prediction (top-1 tail)
                if hasattr(self.model, 'score_tails'):
                    scores = self.model.score_tails(h, r)
                    pred_tail = scores.argmax(dim=-1).item()
                else:
                    # Fallback: score all tails
                    num_entities = self.model.num_entities
                    all_tails = torch.arange(num_entities).to(device)
                    h_exp = h.expand(num_entities)
                    r_exp = r.expand(num_entities)
                    scores = self.model(h_exp, r_exp, all_tails)
                    pred_tail = scores.argmax().item()

                predictions.append(pred_tail)
                is_correct.append(pred_tail == query.true_answer if query.is_answerable else False)
                is_answerable.append(query.is_answerable)

        uncertainties = np.array(uncertainties)
        is_correct = np.array(is_correct)
        is_answerable = np.array(is_answerable)

        # Apply threshold: abstain if uncertainty > threshold
        should_answer = uncertainties <= threshold

        # Metrics
        results = {}

        # 1. Coverage: fraction of queries answered
        results['coverage'] = should_answer.mean()

        # 2. Selective accuracy: accuracy on answered queries
        if should_answer.sum() > 0:
            answered_correct = is_correct[should_answer].mean()
            results['selective_accuracy'] = answered_correct
        else:
            results['selective_accuracy'] = 0.0

        # 3. Abstention precision: of abstained queries, how many were truly unanswerable?
        should_abstain = ~should_answer
        if should_abstain.sum() > 0:
            abstain_precision = (~is_answerable[should_abstain]).mean()
            results['abstention_precision'] = abstain_precision
        else:
            results['abstention_precision'] = 0.0

        # 4. AUROC for detecting unanswerable queries
        results['unanswerable_auroc'] = roc_auc_score(~is_answerable, uncertainties)

        return results

    def compute_risk_coverage_curve(
        self,
        queries: List[QAQuery],
        device: torch.device,
        num_thresholds: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute risk-coverage curve.

        This shows the trade-off:
        - Higher coverage (answer more queries) → higher risk (more errors)
        - Lower coverage (abstain more) → lower risk

        A good uncertainty estimate creates a favorable curve.
        """
        self.model.eval()

        uncertainties = []
        is_correct = []

        with torch.no_grad():
            for query in queries:
                h = torch.tensor([query.head]).to(device)
                r = torch.tensor([query.relation]).to(device)

                # Get uncertainty by averaging over multiple candidate tails
                num_samples = 10
                sample_tails = torch.randint(0, self.model.num_entities, (num_samples,)).to(device)
                h_exp = h.expand(num_samples)
                r_exp = r.expand(num_samples)
                sample_uncertainties = self.model.get_uncertainty(h_exp, r_exp, sample_tails)
                uncertainty = sample_uncertainties.mean().item()
                uncertainties.append(uncertainty)

                # Check if prediction is correct
                if hasattr(self.model, 'score_tails'):
                    scores = self.model.score_tails(h, r)
                    pred_tail = scores.argmax(dim=-1).item()
                else:
                    num_entities = self.model.num_entities
                    all_tails = torch.arange(num_entities).to(device)
                    h_exp = h.expand(num_entities)
                    r_exp = r.expand(num_entities)
                    scores = self.model(h_exp, r_exp, all_tails)
                    pred_tail = scores.argmax().item()

                is_correct.append(pred_tail == query.true_answer if query.is_answerable else False)

        uncertainties = np.array(uncertainties)
        is_correct = np.array(is_correct)

        # Sort by uncertainty (ascending)
        sorted_indices = np.argsort(uncertainties)
        sorted_correct = is_correct[sorted_indices]

        # Compute cumulative accuracy at different coverage levels
        coverages = np.linspace(0.01, 1.0, num_thresholds)
        risks = []

        for cov in coverages:
            n_answered = int(cov * len(queries))
            if n_answered > 0:
                # Answer the n_answered queries with lowest uncertainty
                accuracy = sorted_correct[:n_answered].mean()
                risk = 1 - accuracy
            else:
                risk = 0
            risks.append(risk)

        return {
            'coverage': coverages,
            'risk': np.array(risks),
            'auc': np.trapz(risks, coverages)  # Area under risk-coverage curve (lower is better)
        }


def evaluate_qa_abstention(
    model,
    test_triples: List[Tuple[int, int, int]],
    train_triples: List[Tuple[int, int, int]],
    num_entities: int,
    num_relations: int,
    device: torch.device
) -> Dict[str, float]:
    """
    Main evaluation function for QA with abstention.

    This is the key downstream NLP evaluation that addresses reviewer concern.
    """
    # Compute entity and relation frequencies from training data
    entity_freq = {}
    relation_freq = {}
    for h, r, t in train_triples:
        entity_freq[h] = entity_freq.get(h, 0) + 1
        entity_freq[t] = entity_freq.get(t, 0) + 1
        relation_freq[r] = relation_freq.get(r, 0) + 1

    # Get coverage matrix if available
    coverage = None
    if hasattr(model, 'coverage'):
        coverage = model.coverage

    evaluator = QAAbstentionEvaluator(model, entity_freq, relation_freq, coverage)

    # Generate benchmark
    queries = evaluator.generate_qa_benchmark(
        test_triples, num_entities, unanswerable_ratio=0.3
    )

    print(f"Generated {len(queries)} QA queries")
    print(f"  Answerable: {sum(q.is_answerable for q in queries)}")
    print(f"  Unanswerable: {sum(not q.is_answerable for q in queries)}")

    # Evaluate at multiple thresholds
    results = {}

    # Find optimal threshold using validation split (20% of queries)
    np.random.seed(42)
    val_size = max(100, len(queries) // 5)
    val_indices = np.random.choice(len(queries), val_size, replace=False)
    val_queries = [queries[i] for i in val_indices]

    uncertainties = []
    model.eval()
    with torch.no_grad():
        for query in val_queries:
            h = torch.tensor([query.head]).to(device)
            r = torch.tensor([query.relation]).to(device)
            # Sample multiple tails to get robust uncertainty estimate
            num_samples = 10
            sample_tails = torch.randint(0, num_entities, (num_samples,)).to(device)
            h_exp = h.expand(num_samples)
            r_exp = r.expand(num_samples)
            unc = model.get_uncertainty(h_exp, r_exp, sample_tails).mean().item()
            uncertainties.append(unc)

    threshold = np.median(uncertainties)

    # Main evaluation
    abstention_results = evaluator.predict_with_abstention(queries, threshold, device)
    results.update(abstention_results)

    # Risk-coverage curve
    rc_curve = evaluator.compute_risk_coverage_curve(queries, device)
    results['risk_coverage_auc'] = rc_curve['auc']

    return results


# Convenience function for integration with existing experiments
def run_qa_evaluation(model, data: Dict, device: torch.device) -> Dict[str, float]:
    """
    Run QA abstention evaluation on a dataset.

    Args:
        model: Trained KGE model with get_uncertainty method
        data: Dataset dict with 'train', 'test', 'entity_to_idx', 'relation_to_idx'
        device: torch device

    Returns:
        Dict of evaluation metrics
    """
    # Convert triples to index format
    test_idx = [
        (data['entity_to_idx'][h], data['relation_to_idx'][r], data['entity_to_idx'][t])
        for h, r, t in data['test']
    ]
    train_idx = [
        (data['entity_to_idx'][h], data['relation_to_idx'][r], data['entity_to_idx'][t])
        for h, r, t in data['train']
    ]

    return evaluate_qa_abstention(
        model,
        test_idx,
        train_idx,
        data['num_entities'],
        data['num_relations'],
        device
    )
