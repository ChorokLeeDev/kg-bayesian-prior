"""
Full Experiment on Real Datasets (FB15k-237, WN18RR)

This script runs comprehensive experiments for the NeurIPS submission:
1. Train all models (DistMult, GGPN, GP-KGE)
2. Evaluate: Accuracy (MRR, Hits@k), Calibration (ECE, Brier), OOD (AUROC)
3. Generate publication-quality results

Optimizations for large datasets:
- Efficient batch processing
- Gradient accumulation
- Early stopping
- Subsampled evaluation for speed
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_fb15k237, load_wn18rr
from src.data.kg_dataset import KGDataset
from src.models import DistMult, GPKGE
from src.models.ggpn import GGPN
from src.models.uncertain_kge import MCDropoutKGE
from src.utils.training import set_seed, NegativeSampler
from src.evaluation.calibration import expected_calibration_error, brier_score
from src.evaluation.ood_detection import compute_auroc, create_ood_dataset


def train_model(
    model: nn.Module,
    train_data: KGDataset,
    valid_data: KGDataset,
    num_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 0.001,
    device: str = "cpu",
    patience: int = 10,
    eval_every: int = 10,
    num_negatives: int = 10,
) -> Dict[str, float]:
    """Train model with early stopping."""
    model = model.to(device)

    if hasattr(model, 'set_graph'):
        model.set_graph(train_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    neg_sampler = NegativeSampler(train_data.num_entities, num_negatives=num_negatives)
    neg_sampler.set_true_triples(train_data.triples)

    best_mrr = 0
    patience_counter = 0
    best_state = None

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        total_loss = 0
        num_batches = 0

        indices = np.random.permutation(len(train_data))

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            batch = train_data.triples[batch_idx]

            pos_triples = torch.tensor(batch, device=device)
            neg_triples = neg_sampler(pos_triples)

            optimizer.zero_grad()

            if hasattr(model, 'loss'):
                loss_dict = model.loss(pos_triples, neg_triples)
                loss = loss_dict['total'] if isinstance(loss_dict, dict) else loss_dict
            else:
                pos_scores = model(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
                neg_scores = model(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

                # Handle multiple negatives
                num_neg = len(neg_triples) // len(pos_triples)
                if num_neg > 1:
                    neg_scores = neg_scores.view(len(pos_triples), num_neg).mean(dim=1)

                loss = F.relu(1.0 - pos_scores + neg_scores).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best_mrr': f'{best_mrr:.4f}'})

        # Validation
        if (epoch + 1) % eval_every == 0:
            mrr = quick_mrr_eval(model, valid_data, device, num_samples=500)

            if mrr > best_mrr:
                best_mrr = mrr
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {"best_mrr": best_mrr}


def quick_mrr_eval(
    model: nn.Module,
    data: KGDataset,
    device: str,
    num_samples: int = 500,
) -> float:
    """Quick MRR evaluation on subset."""
    model.eval()

    sample_idx = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    sample = data.triples[sample_idx]

    with torch.no_grad():
        h = torch.tensor(sample[:, 0], device=device)
        r = torch.tensor(sample[:, 1], device=device)
        t = torch.tensor(sample[:, 2], device=device)

        if hasattr(model, 'score_tails'):
            scores = model.score_tails(h, r)
        else:
            # Fallback
            all_tails = torch.arange(data.num_entities, device=device)
            scores = []
            for i in range(len(h)):
                s = model(h[i].expand(data.num_entities), r[i].expand(data.num_entities), all_tails)
                scores.append(s)
            scores = torch.stack(scores)

        # Compute ranks
        target_scores = scores[torch.arange(len(t)), t]
        ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
        mrr = (1.0 / ranks.float()).mean().item()

    return mrr


def evaluate_link_prediction(
    model: nn.Module,
    test_data: KGDataset,
    device: str,
    num_samples: int = 2000,
) -> Dict[str, float]:
    """Evaluate link prediction metrics."""
    model.eval()

    sample_idx = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
    sample = test_data.triples[sample_idx]

    all_ranks = []

    with torch.no_grad():
        batch_size = 100
        for start in tqdm(range(0, len(sample), batch_size), desc="Eval LP", leave=False):
            end = min(start + batch_size, len(sample))
            batch = sample[start:end]

            h = torch.tensor(batch[:, 0], device=device)
            r = torch.tensor(batch[:, 1], device=device)
            t = torch.tensor(batch[:, 2], device=device)

            if hasattr(model, 'score_tails'):
                scores = model.score_tails(h, r)
            else:
                all_tails = torch.arange(test_data.num_entities, device=device)
                scores = []
                for i in range(len(h)):
                    s = model(h[i].expand(test_data.num_entities), r[i].expand(test_data.num_entities), all_tails)
                    scores.append(s)
                scores = torch.stack(scores)

            target_scores = scores[torch.arange(len(t)), t]
            ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
            all_ranks.extend(ranks.cpu().tolist())

    ranks = torch.tensor(all_ranks, dtype=torch.float)

    return {
        "mrr": (1.0 / ranks).mean().item(),
        "hits@1": (ranks <= 1).float().mean().item(),
        "hits@3": (ranks <= 3).float().mean().item(),
        "hits@10": (ranks <= 10).float().mean().item(),
        "mean_rank": ranks.mean().item(),
    }


def evaluate_calibration(
    model: nn.Module,
    test_data: KGDataset,
    device: str,
    num_samples: int = 2000,
) -> Dict[str, float]:
    """Evaluate calibration metrics."""
    model.eval()

    # Positive samples
    pos_idx = np.random.choice(len(test_data), min(num_samples // 2, len(test_data)), replace=False)
    pos_triples = test_data.triples[pos_idx]

    # Negative samples
    neg_triples = []
    for triple in pos_triples:
        h, r, t = triple
        neg_t = np.random.randint(test_data.num_entities)
        while neg_t == t:
            neg_t = np.random.randint(test_data.num_entities)
        neg_triples.append([h, r, neg_t])
    neg_triples = np.array(neg_triples)

    all_triples = np.vstack([pos_triples, neg_triples])
    labels = np.concatenate([np.ones(len(pos_triples)), np.zeros(len(neg_triples))])

    # Shuffle
    shuffle_idx = np.random.permutation(len(all_triples))
    all_triples = all_triples[shuffle_idx]
    labels = labels[shuffle_idx]

    with torch.no_grad():
        h = torch.tensor(all_triples[:, 0], device=device)
        r = torch.tensor(all_triples[:, 1], device=device)
        t = torch.tensor(all_triples[:, 2], device=device)

        if hasattr(model, 'base_model'):
            scores = model.base_model.score_triple(h, r, t)
        elif hasattr(model, 'score_triple'):
            scores = model.score_triple(h, r, t)
        else:
            scores = model(h, r, t)

        confidences = torch.sigmoid(scores).cpu().numpy()

    ece, ece_details = expected_calibration_error(confidences, labels)
    brier = brier_score(confidences, labels)

    predictions = (confidences > 0.5).astype(int)
    accuracy = (predictions == labels).mean()

    return {
        "ece": ece,
        "brier": brier,
        "accuracy": accuracy,
    }


def evaluate_ood(
    model: nn.Module,
    test_data: KGDataset,
    train_data: KGDataset,
    device: str,
    num_samples: int = 1000,
) -> Dict[str, float]:
    """Evaluate OOD detection."""
    model.eval()

    # ID samples
    id_idx = np.random.choice(len(test_data), min(num_samples, len(test_data)), replace=False)
    id_triples = test_data.triples[id_idx]

    # OOD samples
    ood_triples = create_ood_dataset(train_data, test_data, "random", num_samples)

    def get_uncertainty(triples):
        h = torch.tensor(triples[:, 0], device=device)
        r = torch.tensor(triples[:, 1], device=device)
        t = torch.tensor(triples[:, 2], device=device)

        with torch.no_grad():
            if hasattr(model, 'predict_with_uncertainty'):
                pred = model.predict_with_uncertainty(h, r, t)
                if isinstance(pred, dict):
                    return pred.get('total', pred.get('epistemic', torch.zeros(len(h)))).cpu().numpy()
                elif isinstance(pred, tuple):
                    return pred[1].cpu().numpy()
            elif hasattr(model, 'predict_with_mc_samples'):
                _, var = model.predict_with_mc_samples(h, r, t, num_samples=10)
                return var.cpu().numpy()
            else:
                # Use entropy as proxy
                if hasattr(model, 'score_triple'):
                    scores = model.score_triple(h, r, t)
                else:
                    scores = model(h, r, t)
                probs = torch.sigmoid(scores)
                entropy = -probs * torch.log(probs + 1e-10) - (1 - probs) * torch.log(1 - probs + 1e-10)
                return entropy.cpu().numpy()

    id_unc = get_uncertainty(id_triples)
    ood_unc = get_uncertainty(ood_triples)

    auroc = compute_auroc(id_unc, ood_unc)

    return {
        "auroc": auroc,
        "id_unc_mean": id_unc.mean(),
        "ood_unc_mean": ood_unc.mean(),
    }


def run_experiment(
    dataset_name: str = "fb15k237",
    embedding_dim: int = 100,
    num_epochs: int = 100,
    batch_size: int = 1024,
    device: str = "cpu",
):
    """Run full experiment on a dataset."""
    print("=" * 70)
    print(f"FULL EXPERIMENT: {dataset_name.upper()}")
    print("=" * 70)

    set_seed(42)

    # Load data
    print("\nLoading data...")
    if dataset_name == "fb15k237":
        train_data, valid_data, test_data = load_fb15k237()
    else:
        train_data, valid_data, test_data = load_wn18rr()

    print(f"  Entities: {train_data.num_entities:,}")
    print(f"  Relations: {train_data.num_relations}")
    print(f"  Train: {len(train_data):,}, Valid: {len(valid_data):,}, Test: {len(test_data):,}")

    # Models
    models = {
        "DistMult": DistMult(
            train_data.num_entities,
            train_data.num_relations,
            embedding_dim,
        ),
        "GGPN": GGPN(
            train_data.num_entities,
            train_data.num_relations * 2,
            embedding_dim,
            num_layers=2,
            num_rff=100,
        ),
        "GP-KGE (Ours)": GPKGE(
            train_data.num_entities,
            train_data.num_relations,
            embedding_dim,
            kernel_type="relation_aware",
            scoring_function="distmult",
            num_inducing=min(500, train_data.num_entities),
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_model(
            model, train_data, valid_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
        )
        print(f"Best validation MRR: {train_metrics['best_mrr']:.4f}")

        # Evaluate
        print("Evaluating...")

        lp = evaluate_link_prediction(model, test_data, device)
        print(f"  LP: MRR={lp['mrr']:.4f}, H@1={lp['hits@1']:.4f}, H@10={lp['hits@10']:.4f}")

        cal = evaluate_calibration(model, test_data, device)
        print(f"  Calibration: ECE={cal['ece']:.4f}, Brier={cal['brier']:.4f}")

        ood = evaluate_ood(model, test_data, train_data, device)
        print(f"  OOD: AUROC={ood['auroc']:.4f}")

        results[name] = {
            "link_prediction": lp,
            "calibration": cal,
            "ood": ood,
        }

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<20} {'MRR':>8} {'H@1':>8} {'H@10':>8} {'ECE↓':>8} {'Brier↓':>8} {'AUROC↑':>8}")
    print("-" * 70)

    for name, res in results.items():
        print(f"{name:<20} "
              f"{res['link_prediction']['mrr']:>8.4f} "
              f"{res['link_prediction']['hits@1']:>8.4f} "
              f"{res['link_prediction']['hits@10']:>8.4f} "
              f"{res['calibration']['ece']:>8.4f} "
              f"{res['calibration']['brier']:>8.4f} "
              f"{res['ood']['auroc']:>8.4f}")

    # Save results
    save_dir = Path("outputs") / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(save_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    print(f"\nResults saved to {save_dir}")

    return results


def run_ablations(
    train_data: KGDataset,
    valid_data: KGDataset,
    test_data: KGDataset,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """Run ablation studies."""
    print("\n" + "=" * 70)
    print("ABLATION STUDIES")
    print("=" * 70)

    results = {}

    # Ablation 1: Relation-aware vs Single kernel
    print("\n--- Ablation 1: Kernel Type ---")

    for kernel_type in ["relation_aware", "matern"]:
        model = GPKGE(
            train_data.num_entities,
            train_data.num_relations,
            embedding_dim=100,
            kernel_type=kernel_type,
            num_inducing=min(500, train_data.num_entities),
        )

        train_model(model, train_data, valid_data, num_epochs=50, device=device)
        lp = evaluate_link_prediction(model, test_data, device, num_samples=1000)
        cal = evaluate_calibration(model, test_data, device, num_samples=1000)

        results[f"kernel_{kernel_type}"] = {"mrr": lp["mrr"], "ece": cal["ece"]}
        print(f"  {kernel_type}: MRR={lp['mrr']:.4f}, ECE={cal['ece']:.4f}")

    # Ablation 2: Number of inducing points
    print("\n--- Ablation 2: Inducing Points ---")

    for num_inducing in [100, 300, 500]:
        if num_inducing > train_data.num_entities:
            continue

        model = GPKGE(
            train_data.num_entities,
            train_data.num_relations,
            embedding_dim=100,
            num_inducing=num_inducing,
        )

        train_model(model, train_data, valid_data, num_epochs=50, device=device)
        lp = evaluate_link_prediction(model, test_data, device, num_samples=1000)
        cal = evaluate_calibration(model, test_data, device, num_samples=1000)

        results[f"inducing_{num_inducing}"] = {"mrr": lp["mrr"], "ece": cal["ece"]}
        print(f"  M={num_inducing}: MRR={lp['mrr']:.4f}, ECE={cal['ece']:.4f}")

    # Ablation 3: Scoring function
    print("\n--- Ablation 3: Scoring Function ---")

    for scoring in ["distmult", "transe"]:
        model = GPKGE(
            train_data.num_entities,
            train_data.num_relations,
            embedding_dim=100,
            scoring_function=scoring,
            num_inducing=min(500, train_data.num_entities),
        )

        train_model(model, train_data, valid_data, num_epochs=50, device=device)
        lp = evaluate_link_prediction(model, test_data, device, num_samples=1000)
        cal = evaluate_calibration(model, test_data, device, num_samples=1000)

        results[f"scoring_{scoring}"] = {"mrr": lp["mrr"], "ece": cal["ece"]}
        print(f"  {scoring}: MRR={lp['mrr']:.4f}, ECE={cal['ece']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fb15k237", choices=["fb15k237", "wn18rr"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--ablation", action="store_true", help="Run ablation studies")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Run main experiment
    results = run_experiment(
        dataset_name=args.dataset,
        embedding_dim=args.dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Run ablations if requested
    if args.ablation:
        if args.dataset == "fb15k237":
            train_data, valid_data, test_data = load_fb15k237()
        else:
            train_data, valid_data, test_data = load_wn18rr()

        ablation_results = run_ablations(train_data, valid_data, test_data, args.device)
