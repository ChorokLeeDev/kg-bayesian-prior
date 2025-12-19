"""
Training utilities.
"""

from typing import Callable, Dict, List, Optional
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for metrics like MRR
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, value: float, model: nn.Module) -> bool:
        """
        Check if should stop.

        Args:
            value: Current validation metric
            model: Model to save if best

        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = value
            self.best_model_state = model.state_dict().copy()
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class NegativeSampler:
    """Negative sampling for KGE training."""

    def __init__(
        self,
        num_entities: int,
        num_negatives: int = 1,
        mode: str = "uniform",
        filter_positive: bool = True,
    ):
        """
        Args:
            num_entities: Total entities
            num_negatives: Negatives per positive
            mode: "uniform" or "self_adversarial"
            filter_positive: Filter out known true triples
        """
        self.num_entities = num_entities
        self.num_negatives = num_negatives
        self.mode = mode
        self.filter_positive = filter_positive
        self.true_triples = set()

    def set_true_triples(self, triples: np.ndarray):
        """Set known true triples for filtering."""
        self.true_triples = set(map(tuple, triples.tolist()))

    def __call__(
        self,
        positive_triples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate negative samples.

        Args:
            positive_triples: Batch of positive triples (B, 3)

        Returns:
            Negative triples (B * num_negatives, 3)
        """
        batch_size = positive_triples.size(0)
        negatives = []

        for i in range(batch_size):
            h, r, t = positive_triples[i].tolist()

            for _ in range(self.num_negatives):
                # Randomly corrupt head or tail
                if random.random() < 0.5:
                    # Corrupt head
                    neg_h = random.randint(0, self.num_entities - 1)
                    if self.filter_positive:
                        while (neg_h, r, t) in self.true_triples:
                            neg_h = random.randint(0, self.num_entities - 1)
                    negatives.append([neg_h, r, t])
                else:
                    # Corrupt tail
                    neg_t = random.randint(0, self.num_entities - 1)
                    if self.filter_positive:
                        while (h, r, neg_t) in self.true_triples:
                            neg_t = random.randint(0, self.num_entities - 1)
                    negatives.append([h, r, neg_t])

        return torch.tensor(negatives, dtype=torch.long, device=positive_triples.device)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    negative_sampler: NegativeSampler,
    device: str = "cuda",
    kl_weight: float = 0.001,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: KGE model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        negative_sampler: Negative sampler
        device: Device
        kl_weight: Weight for KL term (for Bayesian models)

    Returns:
        Dict with loss components
    """
    model.train()
    total_loss = 0
    total_likelihood = 0
    total_kl = 0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        # Get positive triples
        h = batch["head"].to(device)
        r = batch["relation"].to(device)
        t = batch["tail"].to(device)
        positive_triples = torch.stack([h, r, t], dim=1)

        # Generate negative samples
        negative_triples = negative_sampler(positive_triples)

        # Compute loss
        optimizer.zero_grad()

        if hasattr(model, 'loss') and callable(model.loss):
            loss_dict = model.loss(positive_triples, negative_triples, kl_weight=kl_weight)
            if isinstance(loss_dict, dict):
                loss = loss_dict["total"]
                total_likelihood += loss_dict.get("likelihood", 0)
                total_kl += loss_dict.get("kl", 0)
            else:
                loss = loss_dict
        else:
            # Fallback: margin ranking loss
            pos_scores = model(h, r, t)
            neg_scores = model(
                negative_triples[:, 0],
                negative_triples[:, 1],
                negative_triples[:, 2],
            )
            loss = torch.mean(torch.relu(1.0 - pos_scores + neg_scores.mean()))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "likelihood": total_likelihood / num_batches if total_likelihood else 0,
        "kl": total_kl / num_batches if total_kl else 0,
    }


def train_model(
    model: nn.Module,
    train_dataset,
    valid_dataset,
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    num_negatives: int = 10,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_every: int = 5,
    kl_weight: float = 0.001,
) -> Dict[str, List[float]]:
    """
    Full training loop.

    Args:
        model: KGE model
        train_dataset: Training KGDataset
        valid_dataset: Validation KGDataset
        num_epochs: Maximum epochs
        batch_size: Batch size
        learning_rate: Learning rate
        num_negatives: Negatives per positive
        patience: Early stopping patience
        device: Device
        eval_every: Evaluate every N epochs
        kl_weight: KL weight for Bayesian models

    Returns:
        Dict with training history
    """
    from ..evaluation.link_prediction import evaluate_link_prediction

    model = model.to(device)

    # Set graph structure if model supports it
    if hasattr(model, 'set_graph'):
        model.set_graph(train_dataset)

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Negative sampler
    negative_sampler = NegativeSampler(
        num_entities=train_dataset.num_entities,
        num_negatives=num_negatives,
    )
    negative_sampler.set_true_triples(train_dataset.triples)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode="max")

    # Training history
    history = {
        "train_loss": [],
        "valid_mrr": [],
    }

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, negative_sampler, device, kl_weight
        )
        history["train_loss"].append(train_metrics["loss"])

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_metrics['loss']:.4f}")

        # Evaluate
        if (epoch + 1) % eval_every == 0:
            valid_metrics = evaluate_link_prediction(
                model, valid_dataset, device=device
            )
            history["valid_mrr"].append(valid_metrics["mrr"])

            print(f"  Valid MRR: {valid_metrics['mrr']:.4f}, Hits@10: {valid_metrics['hits@10']:.4f}")

            # Early stopping
            if early_stopping(valid_metrics["mrr"], model):
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    early_stopping.load_best_model(model)

    return history
