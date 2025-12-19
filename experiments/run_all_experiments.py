"""
Run all experiments for the thesis.

This script runs:
1. Baseline models (TransE, DistMult, ComplEx)
2. Uncertainty baselines (MC Dropout, Ensemble, Gaussian)
3. Our GP-KGE model

On datasets:
- FB15k-237 (main benchmark)
- CN15k (uncertainty evaluation)
"""

import subprocess
import sys
from pathlib import Path
from itertools import product


def run_experiment(model: str, dataset: str, seed: int):
    """Run a single experiment."""
    cmd = [
        sys.executable,
        "experiments/train.py",
        "--model", model,
        "--dataset", dataset,
        "--seed", str(seed),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {model} on {dataset} (seed={seed})")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    # Experiment configurations
    models = [
        # Baselines
        "transe",
        "distmult",
        "complex",
        # Uncertainty methods
        "mc_dropout",
        "ensemble",
        "gaussian",
        # Our model
        "gp_kge",
    ]

    datasets = [
        "fb15k237",
        # "cn15k",  # Uncomment when data is available
    ]

    seeds = [42, 123, 456]  # Multiple seeds for statistical significance

    # Run experiments
    results = {}
    for model, dataset, seed in product(models, datasets, seeds):
        key = f"{model}_{dataset}_{seed}"
        success = run_experiment(model, dataset, seed)
        results[key] = success

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    for key, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {key}")

    num_success = sum(results.values())
    num_total = len(results)
    print(f"\nCompleted: {num_success}/{num_total}")


if __name__ == "__main__":
    main()
