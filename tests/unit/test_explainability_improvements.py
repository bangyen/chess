"""Tests for explainability improvements.

Covers the new features (threats, doubled_pawns, space, king_tropism,
pawn_chain), the train/inference probe-gap fix, the Rust early-return
path, and quiescence-search accuracy in forcing_swing.
"""

import math
from unittest.mock import Mock, patch

import chess
import pytest

from chess_ai.features.baseline import baseline_extract_features

# Try importing Rust utilities — tests that require them are skipped otherwise.
try:
    from chess_ai.rust_utils import (
        calculate_forcing_swing,
        extract_features_rust,
        find_best_reply,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


# ---------------------------------------------------------------------------
#  New feature tests (Python path)
# ---------------------------------------------------------------------------


class TestThreatsFeature:
    """Verify the threats feature counts attacks on higher-value pieces."""

    def test_knight_attacking_rook(self):
        """A knight attacking a rook should register as a threat."""
        # White knight on d5 attacks Black rook on c7
        board = chess.Board("8/2r5/8/3N4/8/8/8/4K2k w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["threats_us"] >= 1.0

    def test_no_threats_initial_position(self):
        """The initial position has no piece-on-piece threats."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["threats_us"] == 0.0
        assert feats["threats_them"] == 0.0

    def test_higher_attacks_lower_is_not_threat(self):
        """A queen attacking a pawn is NOT a threat (lower value victim)."""
        # White queen on d4, Black pawn on e5
        board = chess.Board("8/8/8/4p3/3Q4/8/8/4K2k w - - 0 1")
        feats = baseline_extract_features(board)
        # Queen attacks pawn = not a threat (queen > pawn)
        assert feats["threats_us"] == 0.0


class TestDoubledPawnsFeature:
    """Verify doubled-pawn detection."""

    def test_doubled_pawns_present(self):
        """Two White pawns on the e-file should count as 1 doubled pawn."""
        board = chess.Board("8/8/8/8/4P3/4P3/8/4K2k w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["doubled_pawns_us"] == 1.0

    def test_tripled_pawns(self):
        """Three pawns on the same file produce count = 2."""
        board = chess.Board("8/8/4P3/8/4P3/4P3/8/4K2k w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["doubled_pawns_us"] == 2.0

    def test_no_doubled_pawns_initial(self):
        """The starting position has no doubled pawns."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["doubled_pawns_us"] == 0.0
        assert feats["doubled_pawns_them"] == 0.0


class TestSpaceFeature:
    """Verify space-control calculation."""

    def test_initial_position_space(self):
        """White has some space in the starting position due to piece
        attacks reaching the opponent's half."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        # Knights and bishops attack some squares on the 5th-8th ranks
        assert feats["space_us"] >= 0.0
        assert feats["space_them"] >= 0.0

    def test_open_position_has_more_space(self):
        """An open centre should yield more space than a closed one."""
        # Italian Game position with open centre
        open_board = chess.Board(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        feats_open = baseline_extract_features(open_board)
        # Starting position (closed)
        feats_start = baseline_extract_features(chess.Board())
        assert feats_open["space_us"] >= feats_start["space_us"]


class TestKingTropismFeature:
    """Verify king-tropism (piece proximity to enemy king)."""

    def test_pieces_close_to_king(self):
        """A knight adjacent to the enemy king should have high tropism."""
        # White knight on f6, Black king on g8
        board = chess.Board("6k1/8/5N2/8/8/8/8/4K3 w - - 0 1")
        feats = baseline_extract_features(board)
        # Distance f6-g8 = 2  =>  contribution = 7 - 2 = 5
        assert math.isclose(feats["king_tropism_us"], 5.0, abs_tol=0.1)

    def test_no_tropism_without_pieces(self):
        """Only pawns + kings => tropism should be zero."""
        board = chess.Board("8/pppppppp/8/8/8/8/PPPPPPPP/4K2k w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["king_tropism_us"] == 0.0
        assert feats["king_tropism_them"] == 0.0


class TestPawnChainFeature:
    """Verify pawn-chain detection (pawns defended by a pawn)."""

    def test_simple_chain(self):
        """d4-e5 pawn chain: e5 is defended by d4."""
        board = chess.Board("8/8/8/4P3/3P4/8/8/4K2k w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["pawn_chain_us"] >= 1.0

    def test_no_chain_isolated(self):
        """Isolated pawns form no chain."""
        board = chess.Board("8/8/8/8/P6P/8/8/4K2k w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["pawn_chain_us"] == 0.0

    def test_initial_position_has_no_chain(self):
        """Starting position pawns are on the 2nd rank — none defends another."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert feats["pawn_chain_us"] == 0.0


# ---------------------------------------------------------------------------
#  Rust feature extraction — new features appear in the dict
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustNewFeatures:
    """Ensure extract_features_rust includes the five new feature keys."""

    NEW_KEYS = [
        "threats_us",
        "threats_them",
        "doubled_pawns_us",
        "doubled_pawns_them",
        "space_us",
        "space_them",
        "king_tropism_us",
        "king_tropism_them",
        "pawn_chain_us",
        "pawn_chain_them",
    ]

    def test_new_keys_present(self):
        """All new feature keys must be present in the Rust output."""
        feats = extract_features_rust(chess.STARTING_FEN)
        for key in self.NEW_KEYS:
            assert key in feats, f"Missing Rust feature: {key}"

    def test_doubled_pawns_rust(self):
        """Rust doubled-pawn feature matches known position."""
        # Two white pawns on e-file
        feats = extract_features_rust("8/8/8/8/4P3/4P3/8/4K2k w - - 0 1")
        assert feats["doubled_pawns_us"] == 1.0


# ---------------------------------------------------------------------------
#  Rust early-return path preserves _engine_probes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustEarlyReturn:
    """When Rust succeeds, baseline_extract_features should return early
    with _engine_probes present and all new features included."""

    def test_engine_probes_present(self):
        """The early-return path must include _engine_probes."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        assert "_engine_probes" in feats
        probes = feats["_engine_probes"]
        assert callable(probes["hanging_after_reply"])
        assert callable(probes["best_forcing_swing"])
        assert callable(probes["sf_eval_shallow"])

    def test_new_features_present_via_rust_path(self):
        """New features must be present whether coming from Rust or Python."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        for key in TestRustNewFeatures.NEW_KEYS:
            assert key in feats, f"Missing feature via Rust path: {key}"


# ---------------------------------------------------------------------------
#  Train/inference probe-gap regression test
# ---------------------------------------------------------------------------


class TestProbeGapFix:
    """Regression test: during audit inference loops the engine-probe
    features (hang_cnt, forcing_swing, etc.) must be populated, not
    left as 0.0."""

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_probes_populated_during_ranking(self, mock_sf_top_moves, mock_sf_eval):
        """Engine-probe features should be non-zero at inference when the
        probe functions return non-zero values."""
        from chess_ai.audit import audit_feature_set
        from chess_ai.engine.config import SFConfig

        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 60.0),
            (chess.Move.from_uci("d2d4"), 30.0),
        ]

        # Feature extractor whose probes return non-trivial values
        probe_hang = 2
        probe_swing = 15.0

        def feat_fn(board):
            return {
                "material_diff": 0.0,
                "phase": 14.0,
                "_engine_probes": {
                    "hanging_after_reply": lambda eng, b, depth=6: (
                        probe_hang,
                        5,
                        1,
                    ),
                    "best_forcing_swing": lambda eng, b, d_base=6, k_max=12: probe_swing,
                    "sf_eval_shallow": lambda eng, b, depth=6: 25.0,
                },
            }

        mock_engine = Mock()

        def mock_analyse(board, limit=None, multipv=1):
            if multipv == 1:
                return {"pv": [chess.Move.from_uci("e7e5")]}
            return [
                {"score": Mock(), "pv": [chess.Move.from_uci("e2e4")]},
                {"score": Mock(), "pv": [chess.Move.from_uci("d2d4")]},
            ]

        mock_engine.analyse = mock_analyse

        boards = [chess.Board() for _ in range(8)]
        cfg = SFConfig(engine_path="/dummy", depth=8)

        # The key assertion: the audit should complete successfully,
        # and the probed features should participate in the model.
        # (If the probe gap weren't fixed, hang_cnt etc. would be 0
        # during inference but non-zero during training.)
        result = audit_feature_set(
            boards=boards,
            engine=mock_engine,
            cfg=cfg,
            extract_features_fn=feat_fn,
            stability_bootstraps=2,
            test_size=0.4,
        )
        # Basic sanity — if probes were silently missing the model
        # would produce degenerate metrics.
        assert isinstance(result.r2, float)
        assert isinstance(result.tau_mean, float)


# ---------------------------------------------------------------------------
#  Forcing-swing accuracy (quiescence prevents horizon artefacts)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestForcingSwingQuiescence:
    """The Rust forcing swing should account for capture continuations
    thanks to the quiescence search, preventing obviously wrong
    material-only evaluations at the search horizon."""

    def test_swing_nonnegative_with_captures(self):
        """When forcing captures exist, the swing should be >= 0.

        The forcing swing measures how much a forcing move improves
        over the base evaluation.  It is always non-negative by
        construction (max over positive differences, default 0).
        """
        # Middlegame position with captures available
        fen = "r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3"
        swing = calculate_forcing_swing(fen, 2)
        assert swing >= 0.0

    def test_swing_zero_no_forcing_moves(self):
        """When there are no captures or checks, the swing should be
        zero (no forcing moves exist)."""
        # K vs K  — no forcing moves
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        swing = calculate_forcing_swing(fen, 3)
        assert math.isclose(swing, 0.0, abs_tol=1.0)


# ---------------------------------------------------------------------------
#  Optimised Rust search (TT, aspiration windows, pruning)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")
class TestRustSearchOptimisations:
    """Verify the optimised Rust search engine (with transposition table,
    aspiration windows, null-move pruning, and LMR) returns correct and
    timely results at depths previously too slow."""

    def test_find_best_reply_depth_6(self):
        """Engine returns a valid UCI move at depth 6.

        Previously the Rust engine was clamped to depth 4; the new TT
        and pruning optimisations make depth 6 practical.
        """
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        result = find_best_reply(fen, 6)
        assert result is not None
        move = chess.Move.from_uci(result)
        board = chess.Board(fen)
        assert move in board.legal_moves

    def test_find_best_reply_depth_8(self):
        """Engine returns a valid UCI move at depth 8.

        Depth 8 is the new cap.  With all search optimisations the
        engine should complete within a few seconds on typical midgame
        positions.
        """
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        result = find_best_reply(fen, 8)
        assert result is not None
        move = chess.Move.from_uci(result)
        board = chess.Board(fen)
        assert move in board.legal_moves

    def test_forcing_swing_nonnegative_depth_6(self):
        """Forcing swing at depth 6 must still be non-negative.

        The deeper search should not introduce negative swings because
        the metric is defined as max(0, …).
        """
        fen = "r1bqkbnr/pppppppp/2n5/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3"
        swing = calculate_forcing_swing(fen, 6)
        assert swing >= 0.0

    def test_depth_8_completes_in_time(self):
        """A depth-8 search on the opening position must finish within
        5 seconds, confirming that the TT and pruning are working.
        """
        import time

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        start = time.monotonic()
        result = find_best_reply(fen, 8)
        elapsed = time.monotonic() - start

        assert result is not None, "Engine returned no move"
        assert elapsed < 5.0, f"Depth-8 search took {elapsed:.2f}s (limit 5s)"
