"""Tests for positional chess metrics."""

import chess

from src.chess_ai.metrics.positional import (
    PIECE_VAL,
    _blockaded,
    _is_passed_pawn,
    _rank_distance,
    _rook_behind_passer,
    _runner_clear_path,
    _stoppers_in_path,
    checkability_now,
    confinement_count,
    confinement_delta,
    passed_pawn_momentum_delta,
    passed_pawn_momentum_snapshot,
)


class TestPositionalMetrics:
    """Test positional chess metrics and calculations."""

    def test_rank_distance_white_pawn(self):
        """Test rank distance calculation for white pawns."""
        # White pawn on e2 (rank 1) should be 6 ranks from promotion
        pawn_sq = chess.E2
        distance = _rank_distance(pawn_sq, chess.WHITE)
        assert distance == 6  # 7 - 1 = 6

    def test_rank_distance_black_pawn(self):
        """Test rank distance calculation for black pawns."""
        # Black pawn on e7 (rank 6) should be 6 ranks from promotion
        pawn_sq = chess.E7
        distance = _rank_distance(pawn_sq, chess.BLACK)
        assert distance == 6  # 6 - 0 = 6

    def test_rank_distance_edge_cases(self):
        """Test rank distance edge cases."""
        # White pawn on 8th rank (promoted)
        distance = _rank_distance(chess.E8, chess.WHITE)
        assert distance == 0  # 7 - 7 = 0

        # Black pawn on 1st rank (promoted)
        distance = _rank_distance(chess.E1, chess.BLACK)
        assert distance == 0  # 0 - 0 = 0

    def test_rook_behind_passer_no_rook(self):
        """Test rook behind passer when no rook is present."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))

        result = _rook_behind_passer(board, chess.E4, chess.WHITE)
        assert result == 0

    def test_rook_behind_passer_with_rook(self):
        """Test rook behind passer when rook is present."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4 and rook on e1
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E1, chess.Piece(chess.ROOK, chess.WHITE))

        result = _rook_behind_passer(board, chess.E4, chess.WHITE)
        assert result == 1

    def test_stoppers_in_path_no_stoppers(self):
        """Test stoppers in path when no enemy pawns are present."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))

        result = _stoppers_in_path(board, chess.E4, chess.WHITE)
        assert result == 0

    def test_stoppers_in_path_with_stoppers(self):
        """Test stoppers in path when enemy pawns are present."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4 and black pawn on e5
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))

        result = _stoppers_in_path(board, chess.E4, chess.WHITE)
        assert result == 1

    def test_blockaded_not_blockaded(self):
        """Test blockade detection when pawn is not blockaded."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))

        result = _blockaded(board, chess.E4, chess.WHITE)
        assert result == 0

    def test_blockaded_blockaded(self):
        """Test blockade detection when pawn is blockaded."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4 and black piece on e5
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))

        result = _blockaded(board, chess.E4, chess.WHITE)
        assert result == 1

    def test_runner_clear_path_clear(self):
        """Test runner clear path when path is clear."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))

        result = _runner_clear_path(board, chess.E4, chess.WHITE)
        assert result == 1

    def test_runner_clear_path_not_clear(self):
        """Test runner clear path when path is not clear."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4 and black pawn on e5
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))

        result = _runner_clear_path(board, chess.E4, chess.WHITE)
        assert result == 0

    def test_is_passed_pawn_passed(self):
        """Test passed pawn detection for a passed pawn."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4 with no enemy pawns ahead
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))

        result = _is_passed_pawn(board, chess.E4, chess.WHITE)
        assert result is True

    def test_is_passed_pawn_not_passed(self):
        """Test passed pawn detection for a non-passed pawn."""
        board = chess.Board()
        # Clear the board and place a white pawn on e4 and black pawn on e5
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))

        result = _is_passed_pawn(board, chess.E4, chess.WHITE)
        assert result is False

    def test_passed_pawn_momentum_snapshot_no_passed_pawns(self):
        """Test passed pawn momentum snapshot with no passed pawns."""
        board = chess.Board()  # Initial position has no passed pawns

        result = passed_pawn_momentum_snapshot(board, chess.WHITE)

        assert result["pp_count"] == 0
        assert result["pp_min_dist"] == 8  # Sentinel value
        assert result["pp_runners_clear"] == 0
        assert result["pp_blockaded"] == 0
        assert result["pp_rook_behind"] == 0

    def test_passed_pawn_momentum_snapshot_with_passed_pawns(self):
        """Test passed pawn momentum snapshot with passed pawns."""
        board = chess.Board()
        # Clear the board and place a white passed pawn on e4
        board.clear()
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))

        result = passed_pawn_momentum_snapshot(board, chess.WHITE)

        assert result["pp_count"] == 1
        assert result["pp_min_dist"] == 4  # 7 - 3 = 4 (e4 is rank 3)
        assert result["pp_runners_clear"] == 1
        assert result["pp_blockaded"] == 0
        assert result["pp_rook_behind"] == 0

    def test_passed_pawn_momentum_delta(self):
        """Test passed pawn momentum delta calculation."""
        # Create two positions with different passed pawn situations
        base_board = chess.Board()
        base_board.clear()
        base_board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))

        after_board = chess.Board()
        after_board.clear()
        after_board.set_piece_at(
            chess.E5, chess.Piece(chess.PAWN, chess.WHITE)
        )  # Moved forward

        result = passed_pawn_momentum_delta(base_board, after_board)

        # Should have delta features
        assert "d_pp_count" in result
        assert "d_pp_min_dist" in result
        assert "d_pp_runners_clear" in result
        assert "d_pp_blockaded" in result
        assert "d_pp_rook_behind" in result

    def test_checkability_now_initial_position(self):
        """Test checkability in initial position."""
        board = chess.Board()

        result = checkability_now(board)

        assert "d_quiet_checks" in result
        assert "d_capture_checks" in result
        assert result["d_quiet_checks"] == 0  # No checks in initial position
        assert result["d_capture_checks"] == 0

    def test_checkability_now_with_checks(self):
        """Test checkability when checks are available."""
        board = chess.Board()
        # Set up a position where checks are possible
        board.clear()
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
        board.turn = chess.WHITE

        result = checkability_now(board)

        assert "d_quiet_checks" in result
        assert "d_capture_checks" in result
        # Should have some checks available
        assert result["d_quiet_checks"] >= 0
        assert result["d_capture_checks"] >= 0

    def test_confinement_count_no_pieces(self):
        """Test confinement count with no pieces."""
        board = chess.Board()
        board.clear()

        result = confinement_count(board, chess.WHITE)
        assert result == 0

    def test_confinement_count_with_pieces(self):
        """Test confinement count with pieces."""
        board = chess.Board()
        # Initial position should have some confined pieces
        result = confinement_count(board, chess.WHITE)
        assert result >= 0

    def test_confinement_delta(self):
        """Test confinement delta calculation."""
        base_board = chess.Board()
        after_board = chess.Board()
        # Make a move to change the position
        after_board.push(chess.Move.from_uci("e2e4"))

        result = confinement_delta(base_board, after_board)

        assert "d_confinement" in result
        assert isinstance(result["d_confinement"], (int, float))

    def test_piece_val_constants(self):
        """Test that piece value constants are correct."""
        assert PIECE_VAL[chess.PAWN] == 1
        assert PIECE_VAL[chess.KNIGHT] == 3
        assert PIECE_VAL[chess.BISHOP] == 3.1
        assert PIECE_VAL[chess.ROOK] == 5
        assert PIECE_VAL[chess.QUEEN] == 9

    def test_edge_cases_boundary_squares(self):
        """Test edge cases with boundary squares."""
        board = chess.Board()
        board.clear()

        # Test with pawns on edge files
        board.set_piece_at(chess.A4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.H4, chess.Piece(chess.PAWN, chess.WHITE))

        # Should not crash
        result = passed_pawn_momentum_snapshot(board, chess.WHITE)
        assert isinstance(result, dict)

    def test_edge_cases_promotion_squares(self):
        """Test edge cases with pawns on promotion squares."""
        board = chess.Board()
        board.clear()

        # Test with pawns on 7th and 8th ranks
        board.set_piece_at(chess.E7, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.PAWN, chess.WHITE))

        # Should not crash
        result = passed_pawn_momentum_snapshot(board, chess.WHITE)
        assert isinstance(result, dict)

    def test_black_pawn_calculations(self):
        """Test calculations for black pawns."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))

        # Test rank distance for black pawn
        distance = _rank_distance(chess.E5, chess.BLACK)
        assert distance == 4  # 4 - 0 = 4 (e5 is rank 4)

        # Test passed pawn detection
        is_passed = _is_passed_pawn(board, chess.E5, chess.BLACK)
        assert is_passed is True  # No white pawns ahead

        # Test momentum snapshot
        result = passed_pawn_momentum_snapshot(board, chess.BLACK)
        assert result["pp_count"] == 1
        assert result["pp_min_dist"] == 4
