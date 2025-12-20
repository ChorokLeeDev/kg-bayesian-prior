"""
Data loading and processing for Knowledge Graph datasets.
"""

from .kg_dataset import KGDataset
from .loaders import load_fb15k237, load_wn18rr, load_cn15k, load_yago310

__all__ = [
    "KGDataset",
    "load_fb15k237",
    "load_wn18rr",
    "load_cn15k",
    "load_yago310",
]
