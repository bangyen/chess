"""Tests for position sampling utilities."""

import os
import tempfile
from unittest.mock import patch

import chess
import pytest

from chess_feature_audit.utils.sampling import (
    sample_positions_from_pgn,
    sample_random_positions,
)


class TestSampling:
    """Test position sampling functionality."""

    def test_sample_random_positions_basic(self):
        """Test basic random position sampling."""
        with patch("chess_feature_audit.utils.sampling.tqdm") as mock_tqdm:
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
        with patch("chess_feature_audit.utils.sampling.tqdm") as mock_tqdm:
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
        with patch("chess_feature_audit.utils.sampling.tqdm") as mock_tqdm:
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
        with patch("chess_feature_audit.utils.sampling.tqdm") as mock_tqdm:
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
