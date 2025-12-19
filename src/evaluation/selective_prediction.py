"""
Selective Prediction Metrics

Evaluate the quality of uncertainty by measuring:
- Can we improve accuracy by rejecting uncertain predictions?
- What's the risk-coverage tradeoff?

A good uncertainty estimate should:
- When we reject high-uncertainty predictions, accuracy on remaining should improve
- Coverage-risk curve should be monotonically decreasing
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def risk_coverage_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    num_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve.

    Coverage: fraction of samples we make predictions on
    Risk: error rate on covered samples

    Args:
        predictions: Model predictions
        labels: True labels
        uncertainties: Uncertainty scores (higher = more uncertain)
        num_thresholds: Number of threshold points

    Returns:
        Tuple of (coverage, risk) arrays
    """
    n = len(predictions)

    # Sort by uncertainty (ascending)
    sorted_indices = np.argsort(uncertainties)

    # Compute cumulative errors
    errors = (predictions != labels).astype(float)
    sorted_errors = errors[sorted_indices]

    coverages = []
    risks = []

    for k in range(1, n + 1, max(1, n // num_thresholds)):
        coverage = k / n
        risk = sorted_errors[:k].mean()

        coverages.append(coverage)
        risks.append(risk)

    return np.array(coverages), np.array(risks)


def area_under_risk_coverage(
    coverages: np.ndarray,
    risks: np.ndarray,
) -> float:
    """
    Compute area under risk-coverage curve (AURC).

    Lower is better. Represents average risk across all coverage levels.
    """
    # Trapezoidal integration
    return np.trapz(risks, coverages)


def excess_aurc(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
) -> float:
    """
    Compute Excess AURC (E-AURC).

    E-AURC = AURC - AURC_optimal

    where optimal AURC is achieved by perfect uncertainty ordering.
    This normalizes for the base error rate.
    """
    coverages, risks = risk_coverage_curve(predictions, labels, uncertainties)
    aurc = area_under_risk_coverage(coverages, risks)

    # Optimal: sort by actual errors
    errors = (predictions != labels).astype(float)
    optimal_coverages, optimal_risks = risk_coverage_curve(
        predictions, labels, errors  # Use actual error as "perfect uncertainty"
    )
    aurc_optimal = area_under_risk_coverage(optimal_coverages, optimal_risks)

    return aurc - aurc_optimal


def rejection_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    rejection_rates: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute accuracy as function of rejection rate.

    Args:
        predictions: Model predictions
        labels: True labels
        uncertainties: Uncertainty scores
        rejection_rates: Specific rejection rates to evaluate

    Returns:
        Tuple of (rejection_rates, accuracies)
    """
    if rejection_rates is None:
        rejection_rates = np.linspace(0, 0.9, 20)

    n = len(predictions)
    sorted_indices = np.argsort(uncertainties)

    accuracies = []
    for reject_rate in rejection_rates:
        num_keep = int(n * (1 - reject_rate))
        if num_keep == 0:
            accuracies.append(1.0)  # No predictions, assume perfect
            continue

        kept_indices = sorted_indices[:num_keep]
        kept_preds = predictions[kept_indices]
        kept_labels = labels[kept_indices]

        accuracy = (kept_preds == kept_labels).mean()
        accuracies.append(accuracy)

    return rejection_rates, np.array(accuracies)


def selective_prediction_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all selective prediction metrics.

    Returns:
        Dict with AURC, E-AURC, accuracy at various rejection rates
    """
    coverages, risks = risk_coverage_curve(predictions, labels, uncertainties)
    rejection_rates, accuracies = rejection_curve(predictions, labels, uncertainties)

    metrics = {
        "aurc": area_under_risk_coverage(coverages, risks),
        "e_aurc": excess_aurc(predictions, labels, uncertainties),
        "base_accuracy": (predictions == labels).mean(),
    }

    # Add accuracy at specific rejection rates
    for reject_rate in [0.1, 0.2, 0.3, 0.5]:
        idx = np.argmin(np.abs(rejection_rates - reject_rate))
        metrics[f"accuracy_at_{int(reject_rate*100)}%_rejection"] = accuracies[idx]

    # Compute how much accuracy improves per rejection %
    improvement_rate = (accuracies[-1] - accuracies[0]) / (rejection_rates[-1] - rejection_rates[0])
    metrics["improvement_rate"] = improvement_rate

    return metrics


def plot_selective_prediction(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainties: np.ndarray,
    title: str = "Selective Prediction Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot selective prediction analysis.

    Two subplots:
    1. Risk-coverage curve
    2. Rejection-accuracy curve
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Risk-coverage curve
    coverages, risks = risk_coverage_curve(predictions, labels, uncertainties)
    aurc = area_under_risk_coverage(coverages, risks)

    ax1.plot(coverages, risks, 'b-', linewidth=2, label=f'Model (AURC={aurc:.4f})')
    ax1.fill_between(coverages, 0, risks, alpha=0.3)

    # Optimal curve
    errors = (predictions != labels).astype(float)
    opt_cov, opt_risks = risk_coverage_curve(predictions, labels, errors)
    ax1.plot(opt_cov, opt_risks, 'g--', linewidth=2, label='Optimal')

    # Random baseline
    base_error = errors.mean()
    ax1.axhline(y=base_error, color='r', linestyle=':', label=f'Random (error={base_error:.3f})')

    ax1.set_xlabel('Coverage')
    ax1.set_ylabel('Risk (Error Rate)')
    ax1.set_title('Risk-Coverage Curve')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max(risks) * 1.1)

    # Rejection-accuracy curve
    rejection_rates, accuracies = rejection_curve(predictions, labels, uncertainties)

    ax2.plot(rejection_rates * 100, accuracies * 100, 'b-', linewidth=2, label='Model')
    ax2.axhline(y=(predictions == labels).mean() * 100, color='r', linestyle=':',
                label=f'No rejection: {(predictions == labels).mean()*100:.1f}%')

    ax2.set_xlabel('Rejection Rate (%)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Rejection Rate')
    ax2.legend()
    ax2.set_xlim(0, 90)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def compare_uncertainty_methods(
    predictions: np.ndarray,
    labels: np.ndarray,
    uncertainty_methods: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple uncertainty estimation methods.

    Args:
        predictions: Model predictions
        labels: True labels
        uncertainty_methods: Dict mapping method name -> uncertainties

    Returns:
        Dict mapping method name -> metrics
    """
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, uncertainties in uncertainty_methods.items():
        metrics = selective_prediction_metrics(predictions, labels, uncertainties)
        results[name] = metrics

        # Plot risk-coverage
        coverages, risks = risk_coverage_curve(predictions, labels, uncertainties)
        axes[0].plot(coverages, risks, label=f"{name} (AURC={metrics['aurc']:.4f})")

        # Plot rejection-accuracy
        rejection_rates, accuracies = rejection_curve(predictions, labels, uncertainties)
        axes[1].plot(rejection_rates * 100, accuracies * 100, label=name)

    axes[0].set_xlabel('Coverage')
    axes[0].set_ylabel('Risk')
    axes[0].set_title('Risk-Coverage Comparison')
    axes[0].legend()

    axes[1].set_xlabel('Rejection Rate (%)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Rejection-Accuracy Comparison')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return results
