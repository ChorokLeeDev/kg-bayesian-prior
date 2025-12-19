"""
Calibration Metrics for Uncertainty Quantification

A well-calibrated model's confidence should match its accuracy:
- When model says 80% confidence, it should be correct 80% of the time

Key metrics:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability Diagram
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute Expected Calibration Error.

    ECE = Σ_b (|B_b| / N) * |acc(b) - conf(b)|

    where B_b is the set of samples in bin b.

    Args:
        confidences: Model confidence scores (0-1)
        accuracies: Binary accuracy indicators (0 or 1)
        num_bins: Number of bins for calibration

    Returns:
        Tuple of (ECE, bin_details)
    """
    confidences = np.asarray(confidences)
    accuracies = np.asarray(accuracies)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = in_bin.sum()

        if bin_counts[i] > 0:
            bin_accuracies[i] = accuracies[in_bin].mean()
            bin_confidences[i] = confidences[in_bin].mean()

    # ECE
    ece = np.sum(bin_counts / len(confidences) * np.abs(bin_accuracies - bin_confidences))

    details = {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }

    return ece, details


def maximum_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Maximum Calibration Error - worst-case bin miscalibration.

    MCE = max_b |acc(b) - conf(b)|
    """
    _, details = expected_calibration_error(confidences, accuracies, num_bins)

    # Only consider bins with samples
    mask = details["bin_counts"] > 0
    if not mask.any():
        return 0.0

    gaps = np.abs(details["bin_accuracies"] - details["bin_confidences"])
    return gaps[mask].max()


def brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Brier Score.

    Brier = (1/N) * Σ (p_i - y_i)²

    Lower is better. Perfect predictions: 0, random: 0.25

    Args:
        probabilities: Predicted probabilities
        labels: True binary labels

    Returns:
        Brier score
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    return np.mean((probabilities - labels) ** 2)


def brier_score_decomposition(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Decompose Brier Score into components.

    Brier = Reliability - Resolution + Uncertainty

    - Reliability: How well calibrated (lower better)
    - Resolution: How much predictions vary (higher better)
    - Uncertainty: Inherent uncertainty in data

    Returns:
        Dict with Brier, reliability, resolution, uncertainty
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    n = len(labels)

    # Overall stats
    mean_label = labels.mean()
    uncertainty = mean_label * (1 - mean_label)

    # Bin statistics
    bin_boundaries = np.linspace(0, 1, num_bins + 1)

    reliability = 0
    resolution = 0

    for i in range(num_bins):
        in_bin = (probabilities > bin_boundaries[i]) & (probabilities <= bin_boundaries[i + 1])
        n_k = in_bin.sum()

        if n_k > 0:
            mean_prob_k = probabilities[in_bin].mean()
            mean_label_k = labels[in_bin].mean()

            reliability += n_k * (mean_label_k - mean_prob_k) ** 2
            resolution += n_k * (mean_label_k - mean_label) ** 2

    reliability /= n
    resolution /= n

    return {
        "brier": brier_score(probabilities, labels),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """
    Plot reliability diagram.

    A perfectly calibrated model lies on the diagonal.

    Args:
        confidences: Model confidences
        accuracies: Binary accuracies
        num_bins: Number of bins
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    ece, details = expected_calibration_error(confidences, accuracies, num_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.bar(
        details["bin_centers"],
        details["bin_accuracies"],
        width=1.0 / num_bins,
        edgecolor="black",
        alpha=0.7,
        label="Model"
    )
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{title}\nECE = {ece:.4f}")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Gap diagram
    gaps = details["bin_accuracies"] - details["bin_confidences"]
    colors = ["red" if g < 0 else "green" for g in gaps]
    ax2.bar(
        details["bin_centers"],
        gaps,
        width=1.0 / num_bins,
        color=colors,
        edgecolor="black",
        alpha=0.7,
    )
    ax2.axhline(y=0, color="black", linestyle="-")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Accuracy - Confidence (Gap)")
    ax2.set_title("Calibration Gap per Bin")
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def calibration_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve for plotting.

    Returns:
        Tuple of (mean_predicted_probs, fraction_of_positives)
    """
    _, details = expected_calibration_error(confidences, accuracies, num_bins)

    # Filter empty bins
    mask = details["bin_counts"] > 0

    return details["bin_confidences"][mask], details["bin_accuracies"][mask]


def temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    num_iterations: int = 100,
    lr: float = 0.01,
) -> float:
    """
    Learn temperature scaling parameter to improve calibration.

    Divides logits by temperature T before softmax.
    T > 1: softer predictions
    T < 1: sharper predictions

    Args:
        logits: Raw model outputs (before sigmoid/softmax)
        labels: True labels
        num_iterations: Optimization iterations
        lr: Learning rate

    Returns:
        Optimal temperature
    """
    import torch
    import torch.nn.functional as F

    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=num_iterations)

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return temperature.item()
