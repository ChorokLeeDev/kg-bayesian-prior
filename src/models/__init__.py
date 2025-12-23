"""
Model implementations for KG embedding and uncertainty quantification.
"""

from .base import BaseKGEModel
from .transe import TransE
from .distmult import DistMult
from .complex import ComplEx
from .uncertain_kge import UncertainKGE, MCDropoutKGE, EnsembleKGE, GaussianEmbeddingKGE
from .gp_kge import GPKGE
from .ggpn import GGPN
from .coverage_augmented_gpkge import CoverageAugmentedGPKGE
from .predictive_cagp import PredictiveCAGP
from .adaptive_uncertainty import AdaptiveUncertaintyKGE, EnsembleUncertaintyKGE

__all__ = [
    "BaseKGEModel",
    "TransE",
    "DistMult",
    "ComplEx",
    "UncertainKGE",
    "MCDropoutKGE",
    "EnsembleKGE",
    "GaussianEmbeddingKGE",
    "GPKGE",
    "GGPN",
    "CoverageAugmentedGPKGE",
    "PredictiveCAGP",
    "AdaptiveUncertaintyKGE",
    "EnsembleUncertaintyKGE",
]
