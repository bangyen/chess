"""Tests for shared math utilities (cp_to_winrate)."""

import math

import numpy as np

from chess_ai.utils.math import cp_to_winrate


class TestCpToWinrate:
    """Verify the centipawn-to-winrate sigmoid conversion."""

    def test_zero_cp_is_fifty_percent(self):
        """An evaluation of 0 cp should map to exactly 0.5 win probability."""
        assert math.isclose(cp_to_winrate(0.0), 0.5, abs_tol=1e-9)

    def test_large_positive_approaches_one(self):
        """A very large positive eval should approach 1.0."""
        result = cp_to_winrate(10000.0)
        assert result > 0.999

    def test_large_negative_approaches_zero(self):
        """A very large negative eval should approach 0.0."""
        result = cp_to_winrate(-10000.0)
        assert result < 0.001

    def test_monotonically_increasing(self):
        """Higher cp values should always map to higher win probabilities."""
        values = [-500.0, -100.0, 0.0, 100.0, 500.0]
        winrates = [cp_to_winrate(v) for v in values]
        for i in range(len(winrates) - 1):
            assert winrates[i] < winrates[i + 1]

    def test_numpy_array_input(self):
        """The function should accept and return numpy arrays."""
        arr = np.array([-200.0, 0.0, 200.0])
        result = cp_to_winrate(arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert math.isclose(float(result[1]), 0.5, abs_tol=1e-9)
        assert result[0] < result[1] < result[2]

    def test_custom_k_parameter(self):
        """A smaller k should make the sigmoid steeper."""
        # With smaller k, the same cp value should be further from 0.5
        result_default = cp_to_winrate(100.0, k=111.0)
        result_steep = cp_to_winrate(100.0, k=50.0)
        # Both > 0.5, but steep should be closer to 1.0
        assert result_steep > result_default
