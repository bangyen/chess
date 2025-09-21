"""Tests for Kendall tau correlation metric."""

import numpy as np
import pytest

from chess_ai.metrics.kendall import kendall_tau


class TestKendallTau:
    """Test Kendall tau correlation calculation."""

    def test_perfect_agreement(self):
        """Test perfect agreement between rankings."""
        rank_a = [1, 2, 3, 4, 5]
        rank_b = [1, 2, 3, 4, 5]

        result = kendall_tau(rank_a, rank_b)
        assert result == 1.0

    def test_perfect_disagreement(self):
        """Test perfect disagreement between rankings."""
        rank_a = [1, 2, 3, 4, 5]
        rank_b = [5, 4, 3, 2, 1]

        result = kendall_tau(rank_a, rank_b)
        assert result == -1.0

    def test_no_agreement(self):
        """Test no agreement between rankings."""
        rank_a = [1, 2, 3, 4, 5]
        rank_b = [3, 1, 4, 2, 5]

        result = kendall_tau(rank_a, rank_b)
        # Should be between -1 and 1, closer to 0
        assert -1.0 <= result <= 1.0
        assert abs(result) < 0.5  # Not strong agreement or disagreement

    def test_single_element(self):
        """Test with single element rankings."""
        rank_a = [1]
        rank_b = [1]

        result = kendall_tau(rank_a, rank_b)
        assert result == 0.0  # No pairs to compare

    def test_two_elements_agreement(self):
        """Test with two elements in agreement."""
        rank_a = [1, 2]
        rank_b = [1, 2]

        result = kendall_tau(rank_a, rank_b)
        assert result == 1.0

    def test_two_elements_disagreement(self):
        """Test with two elements in disagreement."""
        rank_a = [1, 2]
        rank_b = [2, 1]

        result = kendall_tau(rank_a, rank_b)
        assert result == -1.0

    def test_three_elements_mixed(self):
        """Test with three elements showing mixed agreement."""
        rank_a = [1, 2, 3]
        rank_b = [1, 3, 2]

        result = kendall_tau(rank_a, rank_b)
        # 1,2 vs 1,3: concordant (both rank 1 higher)
        # 1,3 vs 1,2: discordant (1<3 in a, 1>2 in b)
        # 2,3 vs 3,2: discordant (2<3 in a, 3>2 in b)
        # 1 concordant, 2 discordant out of 3 total
        expected = (2 - 1) / 3  # (concordant - discordant) / total
        assert abs(result - expected) < 1e-10

    def test_identical_rankings(self):
        """Test with identical rankings."""
        rank_a = [3, 1, 4, 1, 5]
        rank_b = [3, 1, 4, 1, 5]

        result = kendall_tau(rank_a, rank_b)
        assert result == 1.0

    def test_rankings_with_ties(self):
        """Test with tied rankings."""
        rank_a = [1, 1, 3, 4, 5]  # Ties at position 1
        rank_b = [1, 2, 3, 4, 5]

        result = kendall_tau(rank_a, rank_b)
        # Should handle ties gracefully
        assert -1.0 <= result <= 1.0

    def test_different_lengths_raises_error(self):
        """Test that different length rankings raise an error."""
        rank_a = [1, 2, 3]
        rank_b = [1, 2, 3, 4]

        with pytest.raises(AssertionError):
            kendall_tau(rank_a, rank_b)

    def test_empty_rankings(self):
        """Test with empty rankings."""
        rank_a = []
        rank_b = []

        result = kendall_tau(rank_a, rank_b)
        assert result == 0.0

    def test_large_rankings(self):
        """Test with larger rankings."""
        # Create rankings with some correlation
        rank_a = list(range(10))
        rank_b = [x + np.random.randint(-2, 3) for x in rank_a]  # Add some noise

        result = kendall_tau(rank_a, rank_b)
        assert -1.0 <= result <= 1.0

    def test_chess_move_ranking_example(self):
        """Test with a realistic chess move ranking example."""
        # Engine ranking: best to worst moves
        engine_ranking = [1, 2, 3, 4, 5]  # Move 1 is best, move 5 is worst

        # Surrogate model ranking: similar but not identical
        surrogate_ranking = [1, 3, 2, 4, 5]  # Moves 2 and 3 are swapped

        result = kendall_tau(engine_ranking, surrogate_ranking)

        # Should show good but not perfect agreement
        assert 0.0 < result < 1.0
        assert result > 0.5  # Should be reasonably high

    def test_chess_move_ranking_poor_agreement(self):
        """Test with poor agreement between engine and surrogate."""
        # Engine ranking: best to worst moves
        engine_ranking = [1, 2, 3, 4, 5]

        # Surrogate model ranking: mostly wrong
        surrogate_ranking = [5, 4, 3, 2, 1]  # Completely reversed

        result = kendall_tau(engine_ranking, surrogate_ranking)

        # Should show poor agreement
        assert result < 0.0
        assert result < -0.5  # Should be reasonably negative

    def test_numpy_arrays(self):
        """Test that function works with numpy arrays."""
        rank_a = np.array([1, 2, 3, 4, 5])
        rank_b = np.array([1, 2, 3, 4, 5])

        result = kendall_tau(rank_a, rank_b)
        assert result == 1.0

    def test_float_rankings(self):
        """Test with float rankings."""
        rank_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        rank_b = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = kendall_tau(rank_a, rank_b)
        assert result == 1.0

    def test_negative_rankings(self):
        """Test with negative rankings."""
        rank_a = [-2, -1, 0, 1, 2]
        rank_b = [-2, -1, 0, 1, 2]

        result = kendall_tau(rank_a, rank_b)
        assert result == 1.0

    def test_rankings_with_zeros(self):
        """Test with rankings containing zeros."""
        rank_a = [0, 1, 2, 3, 4]
        rank_b = [0, 1, 2, 3, 4]

        result = kendall_tau(rank_a, rank_b)
        assert result == 1.0
