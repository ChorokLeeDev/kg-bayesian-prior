"""Ranking helpers shared by evaluation scripts and tests."""

from __future__ import annotations

import numpy as np


def compute_rank_from_scores(scores: np.ndarray, target_idx: int) -> int:
    """
    Repository rank convention:
    rank = (#entities with strictly higher score) + 1.
    """
    return int((scores > scores[target_idx]).sum() + 1)

