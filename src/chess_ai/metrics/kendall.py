"""Kendall tau correlation metric."""

import numpy as np


def kendall_tau(rank_a: list[int], rank_b: list[int]) -> float:
    """Calculate Kendall tau-b correlation between two rankings.

    Simple Kendall tau-b approximation for small k (k<=5).

    Args:
        rank_a: First ranking
        rank_b: Second ranking

    Returns:
        Kendall tau correlation coefficient
    """
    assert len(rank_a) == len(rank_b)
    n = len(rank_a)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = np.sign(rank_a[i] - rank_a[j])
            db = np.sign(rank_b[i] - rank_b[j])
            if da == db:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else 0.0
