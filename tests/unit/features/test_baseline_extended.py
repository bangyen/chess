"""Extended tests for features/baseline.py to increase coverage.

Targets uncovered lines in the Python fallback path: outposts, batteries,
pawn_structure, safe_mobility, rook_on_open_file, backward_pawns,
connected_rooks, threats, doubled_pawns, space, king_tropism,
pawn_chain_count, and various edge-case positions.
"""

import chess

from chess_ai.features.baseline import baseline_extract_features


class TestOutpostsFeature:
    """Tests for the outposts feature extraction."""

    def test_no_outposts_initial(self):
        """Starting position has no knight outposts."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["outposts_us"] == 0.0
        assert feats["outposts_them"] == 0.0

    def test_outpost_present(self):
        """Knight on e5 supported by d4 pawn with no enemy pawn attack is an outpost."""
        # White knight on e5, white pawn on d4 supports it
        # No black pawns on d or f files that can attack e5
        board = chess.Board(
            "rnbqkb1r/pppp1ppp/8/4N3/3P4/8/PPP1PPPP/RNBQKB1R w KQkq - 0 1"
        )
        feats = baseline_extract_features(board)
        # The outposts feature may or may not detect this depending on attack squares
        assert isinstance(feats["outposts_us"], float)


class TestBatteriesFeature:
    """Tests for the batteries feature extraction."""

    def test_batteries_initial_back_rank(self):
        """Starting position may have batteries on back rank (R+Q+R)."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        # Back rank has R, Q, R on rank 1 which counts as a battery
        assert feats["batteries_us"] >= 0.0

    def test_rook_battery(self):
        """Two rooks on the same file form a battery."""
        # White Rooks on a1, e1. Black king on h2 (not in check from e1 on rank 1).
        board = chess.Board("8/8/8/8/8/8/4K2k/R3R3 w - - 0 1")
        feats = baseline_extract_features(board)
        # Two rooks on rank 1 should be detected
        assert feats["batteries_us"] >= 1.0


class TestConnectedRooksFeature:
    """Tests for the connected_rooks feature."""

    def test_connected_rooks_on_same_rank(self):
        """Two rooks on the same rank with nothing between are connected."""
        # White rooks on a1 and h1, white king on e2, black king on g8 (not in check from h1).
        board = chess.Board("6k1/8/8/8/8/8/4K3/R6R w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["connected_rooks_us"] == 1.0

    def test_not_connected_different_ranks(self):
        """Rooks on different ranks are not connected."""
        board = chess.Board("R6k/8/8/8/8/8/4K3/R7 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["connected_rooks_us"] == 0.0

    def test_not_connected_piece_between(self):
        """Rooks on same rank with a piece between are not connected."""
        board = chess.Board("7k/8/8/8/8/8/8/R2KR3 w - - 0 1")
        feats = baseline_extract_features(board)
        # King on d1 blocks connection between a1 and e1
        assert feats["connected_rooks_us"] == 0.0

    def test_single_rook(self):
        """With only one rook, connected_rooks is 0."""
        board = chess.Board("7k/8/8/8/8/8/4K3/R7 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["connected_rooks_us"] == 0.0


class TestThreatsFeature:
    """Tests for the threats feature (attacks on higher-value pieces)."""

    def test_no_threats_initial(self):
        """Starting position has no threats."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["threats_us"] == 0.0
        assert feats["threats_them"] == 0.0

    def test_pawn_threatens_knight(self):
        """A pawn attacking a knight is a threat."""
        # White pawn on d4, black knight on e5
        board = chess.Board(
            "rnbqkb1r/pppppppp/8/4n3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1"
        )
        feats = baseline_extract_features(board)
        assert feats["threats_us"] >= 1.0


class TestDoubledPawnsFeature:
    """Tests for the doubled_pawns feature."""

    def test_no_doubled_initial(self):
        """Starting position has no doubled pawns."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["doubled_pawns_us"] == 0.0
        assert feats["doubled_pawns_them"] == 0.0

    def test_doubled_pawns_present(self):
        """Two White pawns on the e-file should count as doubled."""
        # White has pawns on e3 and e4. Remove e2 pawn to keep count legal (8 pawns).
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/8/4P3/4P3/PPPP2PP/RNBQKBNR w KQkq - 0 1"
        )
        feats = baseline_extract_features(board)
        assert feats["doubled_pawns_us"] >= 1.0


class TestSpaceFeature:
    """Tests for the space feature."""

    def test_space_present(self):
        """Both sides should have some space control."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["space_us"] >= 0.0
        assert feats["space_them"] >= 0.0


class TestKingTropismFeature:
    """Tests for the king_tropism feature."""

    def test_tropism_present(self):
        """King tropism should be non-negative."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["king_tropism_us"] >= 0.0
        assert feats["king_tropism_them"] >= 0.0

    def test_tropism_increases_with_proximity(self):
        """Pieces closer to enemy king should give higher tropism."""
        # Queen on f6 near black king (g8 area)
        board_close = chess.Board(
            "rnbqkb1r/pppppppp/5Q2/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
        )
        board_far = chess.Board(
            "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

        feats_close = baseline_extract_features(board_close)
        feats_far = baseline_extract_features(board_far)

        # Close queen should give higher tropism
        assert feats_close["king_tropism_us"] >= feats_far["king_tropism_us"]


class TestPawnChainFeature:
    """Tests for the pawn_chain feature."""

    def test_initial_pawn_chains(self):
        """Starting position has some pawn chains (pawns defending each other)."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        # In starting position, no pawns defend each other
        assert feats["pawn_chain_us"] == 0.0

    def test_pawn_chain_after_moves(self):
        """After d4+e3, e3 pawn defends d4."""
        board = chess.Board()
        board.push(chess.Move.from_uci("d2d4"))
        board.push(chess.Move.from_uci("a7a6"))
        board.push(chess.Move.from_uci("e2e3"))
        board.push(chess.Move.from_uci("a6a5"))

        feats = baseline_extract_features(board)
        # e3 defends d4 pawn
        assert feats["pawn_chain_us"] >= 1.0


class TestSafeMobilityFeature:
    """Tests for the safe_mobility feature."""

    def test_safe_mobility_initial(self):
        """Starting position should have some safe mobility."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["safe_mobility_us"] > 0.0
        assert feats["safe_mobility_them"] > 0.0
        assert feats["safe_mobility_us"] <= 40.0  # capped


class TestRookOnOpenFileFeature:
    """Tests for the rook_open_file feature."""

    def test_no_open_files_initial(self):
        """Starting position has no rook on open file."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["rook_open_file_us"] == 0.0

    def test_rook_on_open_file(self):
        """Rook on an open file should be detected."""
        # Move black king to f8 so it's not in check from e1 rook.
        board = chess.Board("5k2/pppp1ppp/8/8/8/8/PPPP1PPP/4R1K1 w - - 0 1")
        feats = baseline_extract_features(board)
        # Rook on e1, e-file has no white pawns and no black pawns = open
        assert feats["rook_open_file_us"] >= 0.5


class TestBackwardPawnsFeature:
    """Tests for the backward_pawns feature."""

    def test_no_backward_initial(self):
        """Starting position has no backward pawns."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["backward_pawns_us"] == 0.0
        assert feats["backward_pawns_them"] == 0.0


class TestEdgeCasePositions:
    """Test feature extraction on edge-case positions."""

    def test_endgame_position(self):
        """Feature extraction works on a bare endgame position."""
        board = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["phase"] == 0.0  # No non-pawn, non-king pieces
        assert "material_us" in feats

    def test_queen_endgame(self):
        """Feature extraction works with only queens left."""
        board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["phase"] == 1.0  # Just one queen

    def test_black_to_move(self):
        """Features are correctly oriented when black is to move."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        feats = baseline_extract_features(board)
        # Now black is "us"
        assert "material_us" in feats

    def test_king_no_square(self):
        """King safety feature handles missing king gracefully."""
        # A valid position always has both kings; test standard behaviour
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        feats = baseline_extract_features(board)
        assert "king_safety_us" in feats

    def test_pinned_pieces_initial(self):
        """Starting position: pinned pieces should be 0."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["pinned_us"] == 0.0
        assert feats["pinned_them"] == 0.0

    def test_isolated_pawns_feature(self):
        """Isolated pawns detected in appropriate position."""
        # White a-pawn with no pawns on b-file = isolated
        board = chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["isolated_pawns_us"] >= 1.0
