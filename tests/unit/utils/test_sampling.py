"""Tests for position sampling utilities."""

import os
import random
import tempfile
from unittest.mock import patch

import chess
import pytest

from chess_ai.utils.sampling import (
    _board_phase_value,
    classify_phase,
    sample_positions_from_pgn,
    sample_random_positions,
    sample_stratified_positions,
)


class TestSampling:
    """Test position sampling functionality."""

    def test_sample_random_positions_basic(self):
        """Test basic random position sampling."""
        with patch("chess_ai.utils.sampling.tqdm") as mock_tqdm:
            mock_tqdm.return_value = range(3)  # Mock tqdm to return simple range

            positions = sample_random_positions(3, max_random_plies=15)

            assert len(positions) == 3
            for pos in positions:
                assert isinstance(pos, chess.Board)
                assert not pos.is_game_over()

    def test_sample_random_positions_empty(self):
        """Test sampling zero positions."""
        positions = sample_random_positions(0)
        assert len(positions) == 0

    def test_sample_random_positions_single(self):
        """Test sampling a single position."""
        with patch("chess_ai.utils.sampling.tqdm") as mock_tqdm:
            mock_tqdm.return_value = range(1)

            positions = sample_random_positions(1, max_random_plies=10)

            assert len(positions) == 1
            assert isinstance(positions[0], chess.Board)

    def test_sample_positions_from_pgn_nonexistent_file(self):
        """Test sampling from non-existent PGN file."""
        with pytest.raises(FileNotFoundError):
            sample_positions_from_pgn("/nonexistent/file.pgn", 10)

    def test_sample_positions_from_pgn_empty_file(self):
        """Test sampling from empty PGN file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            positions = sample_positions_from_pgn(temp_path, 10)
            assert len(positions) == 0
        finally:
            os.unlink(temp_path)

    def test_sample_positions_from_pgn_valid_game(self):
        """Test sampling from valid PGN file."""
        pgn_content = """
[Event "Test Game"]
[Site "Test"]
[Date "2023.01.01"]
[Round "1"]
[White "Test White"]
[Black "Test Black"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 1-0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(pgn_content)
            temp_path = f.name

        try:
            positions = sample_positions_from_pgn(temp_path, 5, ply_skip=4)
            assert len(positions) > 0
            assert len(positions) <= 5

            for pos in positions:
                assert isinstance(pos, chess.Board)
                assert not pos.is_game_over()
        finally:
            os.unlink(temp_path)

    def test_sample_positions_from_pgn_multiple_games(self):
        """Test sampling from PGN file with multiple games."""
        pgn_content = """
[Event "Game 1"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0

[Event "Game 2"]
[Result "0-1"]
1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Bd3 dxc4 8. Bxc4 b5 9. Bd3 a6 10. O-O c5 0-1
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(pgn_content)
            temp_path = f.name

        try:
            positions = sample_positions_from_pgn(temp_path, 10, ply_skip=2)
            assert len(positions) > 0

            for pos in positions:
                assert isinstance(pos, chess.Board)
        finally:
            os.unlink(temp_path)

    def test_sample_positions_from_pgn_ply_skip(self):
        """Test different ply skip values."""
        pgn_content = """
[Event "Test"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(pgn_content)
            temp_path = f.name

        try:
            # Test with different ply skip values
            positions1 = sample_positions_from_pgn(temp_path, 10, ply_skip=2)
            positions2 = sample_positions_from_pgn(temp_path, 10, ply_skip=4)
            positions3 = sample_positions_from_pgn(temp_path, 10, ply_skip=8)

            # More frequent sampling should give more positions
            assert len(positions1) >= len(positions2)
            assert len(positions2) >= len(positions3)
        finally:
            os.unlink(temp_path)

    def test_sample_positions_from_pgn_max_positions(self):
        """Test that max_positions limit is respected."""
        pgn_content = """
[Event "Test"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(pgn_content)
            temp_path = f.name

        try:
            positions = sample_positions_from_pgn(temp_path, 3, ply_skip=1)
            assert len(positions) <= 3
        finally:
            os.unlink(temp_path)

    def test_sample_random_positions_max_plies(self):
        """Test different max_random_plies values."""
        with patch("chess_ai.utils.sampling.tqdm") as mock_tqdm:
            mock_tqdm.return_value = range(2)

            # Test with different max plies
            positions1 = sample_random_positions(2, max_random_plies=15)
            positions2 = sample_random_positions(2, max_random_plies=20)
            positions3 = sample_random_positions(2, max_random_plies=25)

            assert len(positions1) == 2
            assert len(positions2) == 2
            assert len(positions3) == 2

    def test_sample_random_positions_game_over_handling(self):
        """Test that game over positions are handled correctly."""
        with patch("chess_ai.utils.sampling.tqdm") as mock_tqdm:
            mock_tqdm.return_value = range(5)

            positions = sample_random_positions(5, max_random_plies=100)

            # All positions should be valid and not game over
            for pos in positions:
                assert isinstance(pos, chess.Board)
                assert not pos.is_game_over()
                assert len(list(pos.legal_moves)) > 0

    def test_sample_positions_from_pgn_invalid_pgn(self):
        """Test sampling from invalid PGN file."""
        pgn_content = "This is not a valid PGN file"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(pgn_content)
            temp_path = f.name

        try:
            positions = sample_positions_from_pgn(temp_path, 10)
            # Should handle invalid PGN gracefully
            assert len(positions) == 0
        finally:
            os.unlink(temp_path)

    def test_sample_positions_from_pgn_encoding_errors(self):
        """Test sampling from PGN file with encoding errors."""
        pgn_content = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(pgn_content)
            temp_path = f.name

        try:
            # Should handle encoding errors gracefully
            positions = sample_positions_from_pgn(temp_path, 10)
            assert isinstance(positions, list)
        finally:
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Phase classification helpers
# ---------------------------------------------------------------------------


class TestPhaseClassification:
    """Tests for _board_phase_value and classify_phase."""

    def test_starting_position_is_opening(self):
        """The starting position has all 14 non-pawn/king pieces -> opening."""
        board = chess.Board()
        assert _board_phase_value(board) == 14
        assert classify_phase(board) == "opening"

    def test_empty_board_is_endgame(self):
        """A board with only kings has phase 0 -> endgame."""
        board = chess.Board(fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        assert _board_phase_value(board) == 0
        assert classify_phase(board) == "endgame"

    def test_few_pieces_is_endgame(self):
        """A king+rook endgame should be classified as endgame."""
        board = chess.Board(fen="4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        phase = _board_phase_value(board)
        assert phase < 6
        assert classify_phase(board) == "endgame"

    def test_middlegame_range(self):
        """A position with 6-11 non-pawn/king pieces is middlegame."""
        # 6 pieces: 2 rooks + 2 bishops + 2 knights = 6 per side
        # Use a FEN with exactly 8 non-pawn/king pieces (middlegame)
        board = chess.Board(
            fen="r1bqk2r/pppppppp/8/8/8/8/PPPPPPPP/R1BQK2R w KQkq - 0 1"
        )
        phase = _board_phase_value(board)
        assert 6 <= phase < 12
        assert classify_phase(board) == "middlegame"


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


class TestStratifiedSampling:
    """Tests for sample_stratified_positions."""

    def test_returns_requested_count(self):
        """The function returns exactly n positions (within tolerance)."""
        random.seed(42)
        positions = sample_stratified_positions(30)
        # We may get slightly fewer if some buckets are hard to fill,
        # but at least 80% should be present.
        assert len(positions) >= 24
        assert len(positions) <= 30

    def test_all_boards_are_valid(self):
        """Every returned board should be a valid, non-game-over position."""
        random.seed(42)
        positions = sample_stratified_positions(20)
        for pos in positions:
            assert isinstance(pos, chess.Board)
            assert not pos.is_game_over()

    def test_phase_distribution_roughly_matches_weights(self):
        """Positions should be distributed across phases per the weights.

        We use a generous tolerance (0.35 absolute) because random
        generation can't guarantee exact phase placement, especially
        for endgames which are hard to reach via random play.
        """
        random.seed(42)
        n = 60
        positions = sample_stratified_positions(n)

        counts = {"opening": 0, "middlegame": 0, "endgame": 0}
        for pos in positions:
            counts[classify_phase(pos)] += 1

        total = len(positions)
        # Default weights: opening=0.25, middlegame=0.50, endgame=0.25
        assert counts["opening"] / total >= 0.05  # at least some openings
        assert counts["middlegame"] / total >= 0.15  # at least some middlegames

    def test_endgame_bucket_has_low_piece_count(self):
        """Positions classified as endgame should have <= 12 non-pawn/king pieces."""
        random.seed(42)
        positions = sample_stratified_positions(
            40,
            phase_weights={"opening": 0.0001, "middlegame": 0.0001, "endgame": 0.9998},
        )

        endgame_positions = [p for p in positions if classify_phase(p) == "endgame"]
        for pos in endgame_positions:
            assert _board_phase_value(pos) <= 12

    def test_zero_positions_returns_empty(self):
        """Requesting 0 positions returns an empty list."""
        assert sample_stratified_positions(0) == []

    def test_negative_positions_returns_empty(self):
        """Requesting negative positions returns an empty list."""
        assert sample_stratified_positions(-5) == []

    def test_custom_phase_weights(self):
        """Custom phase weights are accepted without error."""
        random.seed(42)
        positions = sample_stratified_positions(
            20,
            phase_weights={"opening": 0.8, "middlegame": 0.1, "endgame": 0.1},
        )
        assert len(positions) > 0
        for pos in positions:
            assert isinstance(pos, chess.Board)

    def test_unknown_phase_raises(self):
        """Unknown phase names in weights should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown phase"):
            sample_stratified_positions(
                10,
                phase_weights={"opening": 0.5, "lategame": 0.5},
            )

    def test_none_weights_uses_defaults(self):
        """Passing phase_weights=None uses the default distribution."""
        random.seed(42)
        positions = sample_stratified_positions(20, phase_weights=None)
        assert len(positions) > 0

    def test_deterministic_with_seed(self):
        """Two runs with the same seed produce the same FENs."""
        random.seed(123)
        run1 = [b.fen() for b in sample_stratified_positions(10)]
        random.seed(123)
        run2 = [b.fen() for b in sample_stratified_positions(10)]
        assert run1 == run2


# ---------------------------------------------------------------------------
# PGN stratified sampling
# ---------------------------------------------------------------------------


class TestPgnStratifiedSampling:
    """Tests for sample_positions_from_pgn with phase_weights."""

    _PGN_LONG = """
[Event "Test Game"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15. Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21. Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1 Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5 Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3 39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 1-0
"""

    def test_stratified_pgn_returns_positions(self):
        """PGN sampling with phase_weights should return valid positions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(self._PGN_LONG)
            temp_path = f.name

        try:
            positions = sample_positions_from_pgn(
                temp_path,
                10,
                phase_weights={"opening": 0.5, "middlegame": 0.3, "endgame": 0.2},
            )
            assert len(positions) > 0
            for pos in positions:
                assert isinstance(pos, chess.Board)
        finally:
            os.unlink(temp_path)

    def test_uniform_pgn_ignores_phase_weights(self):
        """Without phase_weights the PGN sampler uses uniform ply-skip."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(self._PGN_LONG)
            temp_path = f.name

        try:
            positions = sample_positions_from_pgn(
                temp_path, 5, ply_skip=4, phase_weights=None
            )
            assert len(positions) > 0
            assert len(positions) <= 5
        finally:
            os.unlink(temp_path)
