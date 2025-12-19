"""
Evaluation metrics for link prediction and uncertainty quantification.
"""

from .link_prediction import compute_mrr, compute_hits_at_k
from .calibration import expected_calibration_error, brier_score, reliability_diagram
from .ood_detection import compute_auroc, compute_aupr
from .selective_prediction import risk_coverage_curve

__all__ = [
    "compute_mrr",
    "compute_hits_at_k",
    "expected_calibration_error",
    "brier_score",
    "reliability_diagram",
    "compute_auroc",
    "compute_aupr",
    "risk_coverage_curve",
]
