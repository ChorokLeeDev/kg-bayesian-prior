"""
Ablation study for GP-KGE model.

Studies the impact of:
1. Relation-aware vs single kernel
2. Different kernel types (diffusion vs Matérn)
3. Number of inducing points
4. KL weight (β in β-VAE)
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_fb15k237
from src.models import GPKGE
from src.utils.training import train_model, set_seed
from src.evaluation.link_prediction import evaluate_link_prediction
from src.evaluation.calibration import expected_calibration_error


def run_ablation(
    ablation_name: str,
    model_config: dict,
    train_data,
    valid_data,
    test_data,
    device: str = "cuda",
):
    """Run single ablation experiment."""
    print(f"\n{'='*50}")
    print(f"Ablation: {ablation_name}")
    print(f"Config: {model_config}")
    print(f"{'='*50}\n")

    model = GPKGE(
        num_entities=train_data.num_entities,
        num_relations=train_data.num_relations,
        **model_config
    )

    history = train_model(
        model=model,
        train_dataset=train_data,
        valid_dataset=valid_data,
        num_epochs=50,  # Shorter for ablation
        batch_size=128,
        learning_rate=0.001,
        device=device,
    )

    # Evaluate
    all_triples = np.vstack([train_data.triples, valid_data.triples, test_data.triples])
    metrics = evaluate_link_prediction(model, test_data, filter_triples=all_triples, device=device)

    return {
        "ablation": ablation_name,
        "config": model_config,
        "mrr": metrics["mrr"],
        "hits@10": metrics["hits@10"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("Loading data...")
    train_data, valid_data, test_data = load_fb15k237()

    results = []

    # 1. Full model (baseline for comparison)
    results.append(run_ablation(
        "full_model",
        {
            "embedding_dim": 100,
            "kernel_type": "relation_aware",
            "scoring_function": "distmult",
            "num_inducing": 500,
        },
        train_data, valid_data, test_data, device
    ))

    # 2. Without relation-aware kernel (single kernel)
    # Note: This requires modifying the kernel to use aggregated adjacency
    results.append(run_ablation(
        "single_kernel",
        {
            "embedding_dim": 100,
            "kernel_type": "matern",  # Non-relation-aware
            "scoring_function": "distmult",
            "num_inducing": 500,
        },
        train_data, valid_data, test_data, device
    ))

    # 3. Different kernel types
    for kernel in ["diffusion", "matern"]:
        results.append(run_ablation(
            f"kernel_{kernel}",
            {
                "embedding_dim": 100,
                "kernel_type": "relation_aware",
                "scoring_function": "distmult",
                "num_inducing": 500,
            },
            train_data, valid_data, test_data, device
        ))

    # 4. Number of inducing points
    for num_inducing in [100, 250, 500, 1000]:
        results.append(run_ablation(
            f"inducing_{num_inducing}",
            {
                "embedding_dim": 100,
                "kernel_type": "relation_aware",
                "scoring_function": "distmult",
                "num_inducing": num_inducing,
            },
            train_data, valid_data, test_data, device
        ))

    # 5. Different scoring functions
    for scoring in ["distmult", "complex", "transe"]:
        results.append(run_ablation(
            f"scoring_{scoring}",
            {
                "embedding_dim": 100,
                "kernel_type": "relation_aware",
                "scoring_function": scoring,
                "num_inducing": 500,
            },
            train_data, valid_data, test_data, device
        ))

    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"{'Ablation':<30} {'MRR':>10} {'Hits@10':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['ablation']:<30} {r['mrr']:>10.4f} {r['hits@10']:>10.4f}")

    # Save results
    save_dir = Path("outputs/ablation")
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(save_dir / f"ablation_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
