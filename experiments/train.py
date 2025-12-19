"""
Main training script for KG-Bayesian-Prior experiments.

Usage:
    python experiments/train.py --config configs/default.yaml
    python experiments/train.py --model gp_kge --dataset fb15k237
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json

import torch
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_fb15k237, load_wn18rr, load_cn15k
from src.models import TransE, DistMult, ComplEx, GPKGE
from src.models.uncertain_kge import MCDropoutKGE, EnsembleKGE, GaussianEmbeddingKGE
from src.utils.training import train_model, set_seed
from src.evaluation.link_prediction import evaluate_link_prediction, evaluate_with_uncertainty
from src.evaluation.calibration import expected_calibration_error, brier_score
from src.evaluation.ood_detection import evaluate_ood_detection, create_ood_dataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_dataset(name: str, data_dir=None):
    """Load dataset by name."""
    loaders = {
        "fb15k237": load_fb15k237,
        "wn18rr": load_wn18rr,
        "cn15k": load_cn15k,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[name](data_dir)


def get_model(config: dict, num_entities: int, num_relations: int):
    """Create model from config."""
    model_name = config["model"]["name"]
    model_config = config["model"]

    common_args = {
        "num_entities": num_entities,
        "num_relations": num_relations,
        "embedding_dim": model_config["embedding_dim"],
    }

    if model_name == "transe":
        return TransE(**common_args)

    elif model_name == "distmult":
        return DistMult(**common_args, dropout=model_config.get("dropout_rate", 0.0))

    elif model_name == "complex":
        return ComplEx(**common_args, dropout=model_config.get("dropout_rate", 0.0))

    elif model_name == "gp_kge":
        return GPKGE(
            **common_args,
            kernel_type=model_config["kernel_type"],
            scoring_function=model_config["scoring_function"],
            num_inducing=model_config["num_inducing"],
            jitter=model_config["jitter"],
        )

    elif model_name == "mc_dropout":
        base = DistMult(**common_args, dropout=model_config["dropout_rate"])
        return MCDropoutKGE(
            base,
            dropout_rate=model_config["dropout_rate"],
            num_samples=model_config["num_mc_samples"],
        )

    elif model_name == "ensemble":
        return EnsembleKGE(
            DistMult,
            num_models=model_config["num_ensemble"],
            **common_args,
        )

    elif model_name == "gaussian":
        return GaussianEmbeddingKGE(**common_args)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train KGE models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset name")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.model:
        config["model"]["name"] = args.model
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.seed:
        config["experiment"]["seed"] = args.seed

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Device
    device = config["experiment"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"Configuration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Dataset: {config['dataset']['name']}")
    print(f"  Device: {device}")
    print()

    # Load data
    print("Loading dataset...")
    train_data, valid_data, test_data = get_dataset(
        config["dataset"]["name"],
        config["dataset"]["data_dir"]
    )

    print(f"  Entities: {train_data.num_entities}")
    print(f"  Relations: {train_data.num_relations}")
    print(f"  Train triples: {len(train_data)}")
    print(f"  Valid triples: {len(valid_data)}")
    print(f"  Test triples: {len(test_data)}")
    print()

    # Create model
    print("Creating model...")
    model = get_model(config, train_data.num_entities, train_data.num_relations)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Train
    print("Training...")
    history = train_model(
        model=model,
        train_dataset=train_data,
        valid_dataset=valid_data,
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        num_negatives=config["training"]["num_negatives"],
        patience=config["training"]["patience"],
        device=device,
        eval_every=config["training"]["eval_every"],
        kl_weight=config["training"]["kl_weight"],
    )
    print()

    # Final evaluation
    print("Final Evaluation on Test Set:")
    print("-" * 50)

    # Link prediction
    lp_metrics = evaluate_link_prediction(
        model, test_data,
        batch_size=config["evaluation"]["batch_size"],
        filter_triples=np.vstack([train_data.triples, valid_data.triples, test_data.triples]),
        device=device,
    )
    print(f"Link Prediction:")
    print(f"  MRR: {lp_metrics['mrr']:.4f}")
    print(f"  Hits@1: {lp_metrics['hits@1']:.4f}")
    print(f"  Hits@3: {lp_metrics['hits@3']:.4f}")
    print(f"  Hits@10: {lp_metrics['hits@10']:.4f}")
    print()

    # Uncertainty evaluation (if model supports it)
    if hasattr(model, 'predict_with_uncertainty') or hasattr(model, 'get_entity_uncertainty'):
        print("Uncertainty Evaluation:")

        unc_metrics = evaluate_with_uncertainty(
            model, test_data,
            batch_size=config["evaluation"]["batch_size"],
            device=device,
        )
        print(f"  Uncertainty-Rank Correlation: {unc_metrics['uncertainty_rank_correlation']:.4f}")
        print(f"  Mean Uncertainty (correct): {unc_metrics['mean_uncertainty_correct']:.4f}")
        print(f"  Mean Uncertainty (incorrect): {unc_metrics['mean_uncertainty_incorrect']:.4f}")
        print()

        # OOD Detection
        print("OOD Detection:")
        from src.data.kg_dataset import KGDataset

        ood_triples = create_ood_dataset(train_data, test_data, "random", num_samples=1000)
        ood_data = KGDataset(
            ood_triples,
            train_data.num_entities,
            train_data.num_relations,
        )

        ood_metrics = evaluate_ood_detection(
            model, test_data, ood_data,
            batch_size=config["evaluation"]["batch_size"],
            device=device,
        )
        print(f"  AUROC: {ood_metrics['auroc']:.4f}")
        print(f"  FPR@95TPR: {ood_metrics['fpr@95tpr']:.4f}")
        print()

    # Save results
    save_dir = Path(config["experiment"]["save_dir"]) / config["experiment"]["name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    torch.save(model.state_dict(), save_dir / f"model_{timestamp}.pt")

    # Save results
    results = {
        "config": config,
        "link_prediction": lp_metrics,
        "training_history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    if hasattr(model, 'predict_with_uncertainty'):
        results["uncertainty"] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                  for k, v in unc_metrics.items()}
        results["ood_detection"] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                    for k, v in ood_metrics.items()}

    with open(save_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
