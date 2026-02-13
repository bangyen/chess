"""Shared mathematical utilities for chess evaluation pipelines.

Houses conversion functions (e.g. centipawn-to-winrate) that are
needed by both the audit and the model-training paths, avoiding
duplication and ensuring consistent behaviour.
"""

from typing import Union

import numpy as np
import numpy.typing as npt


def cp_to_winrate(
    cp: Union[float, npt.NDArray[np.floating]], k: float = 111.0
) -> Union[float, npt.NDArray[np.floating]]:
    """Convert a centipawn score to a win probability via sigmoid.

    Stockfish uses this mapping internally so that extreme evaluations
    (e.g. +500 vs +700) are compressed while the mid-range (around 0)
    retains high sensitivity.  Using win-rate space as the training
    target produces better-behaved regression targets and reduces
    the influence of outlier evaluations on model fitting.

    Accepts both scalar floats and numpy arrays (element-wise).

    Args:
        cp: Centipawn evaluation (positive = side-to-move advantage).
        k: Sigmoid steepness.  Default 111 matches Stockfish's
           internal win-rate model.

    Returns:
        Win probability in [0, 1], same type as *cp*.
    """
    result: Union[float, npt.NDArray[np.floating]] = 1.0 / (1.0 + np.exp(-cp / k))
    return result
