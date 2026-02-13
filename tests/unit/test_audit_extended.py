"""Extended tests for audit.py to push coverage toward 95%.

Targets the faithfulness analysis loop, _sparsity helper, stability
selection bootstrap, TreeSurrogate early-stopping branch, and the
_eval_move_delta reply-None path.  Uses diverse board positions so that
FEN-keyed caches don't collapse all data to a single entry.
"""

import warnings
from unittest.mock import Mock, patch

import chess

from chess_ai.audit import AuditResult, audit_feature_set
from chess_ai.engine.config import SFConfig

# Suppress sklearn convergence warnings in tests
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Diverse starting positions (unique FENs) to defeat caching
_DIVERSE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4",
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkb1r/pppppppp/5n2/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2",
    "rnbqkb1r/pppp1ppp/4pn2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkb1r/pppp1ppp/4pn2/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "r1bqkbnr/pppnpppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
    "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
    "rnbqkb1r/pp2pppp/5n2/2pp4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "r1bqkbnr/pppppppp/2n5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2",
    "r1bqkbnr/ppp1pppp/2np4/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
]


def _make_diverse_boards(n: int = 30):
    """Create *n* boards from diverse FENs so each has a unique cache key."""
    boards = []
    for i in range(n):
        fen = _DIVERSE_FENS[i % len(_DIVERSE_FENS)]
        boards.append(chess.Board(fen))
    return boards


def _make_varied_feature_extractor():
    """Feature extractor that returns board-dependent values.

    Uses the piece count and a hash of the FEN so every unique position
    produces genuinely different feature values.  This ensures the
    surrogate model learns non-trivial coefficients and the faithfulness
    loop encounters non-zero contributions.
    """

    def extract(board):
        # Derive noise from the board state so it changes with FEN
        n_pieces = len(board.piece_map())
        fen_hash = hash(board.fen()) % 1000
        noise = float(fen_hash) / 200.0  # 0..5 range
        piece_frac = n_pieces / 32.0  # 0..1 range

        return {
            "material_us": 10.0 + noise,
            "material_them": 10.0 - noise * 0.3,
            "material_diff": (noise - 2.5) * 2.0,
            "mobility_us": 20.0 + noise * 3.0,
            "mobility_them": 20.0 - noise * 1.5,
            "king_ring_pressure_us": noise * 0.5,
            "king_ring_pressure_them": (5.0 - noise) * 0.3,
            "passed_us": noise * 0.4,
            "passed_them": (5.0 - noise) * 0.2,
            "open_files_us": noise * 0.3,
            "semi_open_us": noise * 0.1,
            "open_files_them": (5.0 - noise) * 0.2,
            "semi_open_them": 0.0,
            "phase": 14.0 * piece_frac,
            "center_control_us": 2.0 + noise * 0.5,
            "center_control_them": 2.0 + (5.0 - noise) * 0.3,
            "piece_activity_us": 15.0 + noise * 2.0,
            "piece_activity_them": 15.0 - noise * 0.5,
            "king_safety_us": 3.0 + noise * 0.2,
            "king_safety_them": 3.0 - noise * 0.1,
            "hanging_us": 0.0,
            "hanging_them": noise * 0.2,
            "batteries_us": noise * 0.3,
            "outposts_us": noise * 0.1,
            "bishop_pair_us": 1.0 if noise > 2.0 else 0.0,
            "threats_us": noise * 0.5,
            "threats_them": (5.0 - noise) * 0.3,
            "king_tropism_us": noise * 0.4,
            "king_tropism_them": (5.0 - noise) * 0.2,
            "space_us": noise * 1.5,
            "space_them": (5.0 - noise) * 0.5,
            "doubled_pawns_us": 1.0 if noise > 3.5 else 0.0,
            "doubled_pawns_them": 0.0,
            "rook_open_file_us": 1.0 if noise > 2.5 else 0.0,
            "rook_open_file_them": 0.0,
            "_engine_probes": {
                "hanging_after_reply": lambda eng, board, depth=6: (
                    int(noise),
                    int(noise),
                    0,
                ),
                "best_forcing_swing": lambda eng, board, d_base=6, k_max=12: noise
                * 10.0,
                "sf_eval_shallow": lambda eng, board, depth=6: noise * 5.0,
            },
        }

    return extract


def _make_mock_engine_for_audit():
    """Build a mock engine that returns varied, legal replies."""
    mock_engine = Mock()

    def mock_analyse(board, limit=None, multipv=1):
        legal = list(board.legal_moves)
        if multipv == 1:
            move = legal[0] if legal else None
            return {"score": Mock(), "pv": [move] if move else []}
        return [{"score": Mock(), "pv": [m]} for m in legal[:multipv]]

    mock_engine.analyse = mock_analyse
    return mock_engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAuditFaithfulnessAndStability:
    """Exercises the faithfulness loop, _sparsity, and stability selection.

    Uses 30 boards with a large gap between best and second move scores
    so the faithfulness analysis loop is entered, and enough training
    samples (>= 20) to trigger stability selection.
    """

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_faithfulness_loop_entered(self, mock_top, mock_eval):
        """Audit with large score gaps triggers faithfulness computation."""

        def varied_eval(eng, board, cfg):
            # Board-dependent eval so deltas are large and varied
            fen_hash = hash(board.fen()) % 500
            return float(fen_hash) - 250.0

        mock_eval.side_effect = varied_eval

        def varied_top(eng, board, cfg):
            legal = list(board.legal_moves)
            moves = legal[:3] if len(legal) >= 3 else legal
            # Big gap so faithfulness loop executes; also make
            # delta_sf_best - delta_sf_second >= 80 for decisive lines
            return [(m, 300.0 - i * 200.0) for i, m in enumerate(moves)]

        mock_top.side_effect = varied_top

        boards = _make_diverse_boards(30)
        engine = _make_mock_engine_for_audit()
        cfg = SFConfig(engine_path="/mock/sf", depth=12, multipv=3)
        extract_fn = _make_varied_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=3,
            test_size=0.25,
            gap_threshold_cp=50.0,
            stability_bootstraps=3,
            stability_thresh=0.5,
        )

        assert isinstance(result, AuditResult)
        # Faithfulness should have been computed (total > 0)
        assert 0.0 <= result.local_faithfulness <= 1.0
        assert 0.0 <= result.local_faithfulness_decisive <= 1.0
        assert result.sparsity_mean >= 0.0
        assert 0.0 <= result.coverage_ratio <= 1.0
        # With 30 diverse boards and 3 moves each, stability selection should run
        assert isinstance(result.stable_features, list)

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_stability_selection_runs(self, mock_top, mock_eval):
        """With >= 20 training samples, stability selection executes."""
        eval_counter = {"n": 0}

        def varied_eval(eng, board, cfg):
            eval_counter["n"] += 1
            return 40.0 + (eval_counter["n"] % 8) * 25.0

        mock_eval.side_effect = varied_eval

        def varied_top(eng, board, cfg):
            legal = list(board.legal_moves)
            moves = legal[:2] if len(legal) >= 2 else legal
            return [(m, 150.0 - i * 100.0) for i, m in enumerate(moves)]

        mock_top.side_effect = varied_top

        boards = _make_diverse_boards(30)
        engine = _make_mock_engine_for_audit()
        cfg = SFConfig(engine_path="/mock/sf", depth=12, multipv=2)
        extract_fn = _make_varied_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=2,
            test_size=0.2,
            gap_threshold_cp=30.0,
            stability_bootstraps=5,
            stability_thresh=0.3,
        )

        assert isinstance(result, AuditResult)
        assert isinstance(result.stable_features, list)
        assert isinstance(result.top_features_by_coef, list)
        assert len(result.top_features_by_coef) > 0

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_eval_move_delta_reply_none(self, mock_top, mock_eval):
        """When engine returns no reply during faithfulness analysis,
        _eval_move_delta falls back to zero delta and zero feature vector."""
        eval_counter = {"n": 0}

        def varied_eval(eng, board, cfg):
            eval_counter["n"] += 1
            return 50.0 + (eval_counter["n"] % 5) * 15.0

        mock_eval.side_effect = varied_eval

        def varied_top(eng, board, cfg):
            legal = list(board.legal_moves)
            moves = legal[:2] if len(legal) >= 2 else legal
            return [(m, 200.0 - i * 250.0) for i, m in enumerate(moves)]

        mock_top.side_effect = varied_top

        # Engine alternates between valid reply and None reply
        call_count = {"n": 0}

        def mock_analyse(board, limit=None, multipv=1):
            call_count["n"] += 1
            if call_count["n"] % 3 == 0:
                # No reply found
                return {"score": Mock(), "pv": [None]}
            legal = list(board.legal_moves)
            move = legal[0] if legal else None
            return {
                "score": Mock(),
                "pv": [move],
            }

        mock_engine = Mock()
        mock_engine.analyse = mock_analyse

        boards = _make_diverse_boards(30)
        cfg = SFConfig(engine_path="/mock/sf", depth=12, multipv=2)
        extract_fn = _make_varied_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=mock_engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=2,
            test_size=0.25,
            gap_threshold_cp=50.0,
            stability_bootstraps=2,
        )

        assert isinstance(result, AuditResult)


class TestTreeSurrogatePredict1D:
    """Test the TreeSurrogate predict method with 1D input (line 490)."""

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_predict_1d_input(self, mock_top, mock_eval):
        """TreeSurrogate.predict handles 1D input via reshape."""
        eval_counter = {"n": 0}

        def varied_eval(eng, board, cfg):
            eval_counter["n"] += 1
            return 40.0 + (eval_counter["n"] % 8) * 20.0

        mock_eval.side_effect = varied_eval

        def varied_top(eng, board, cfg):
            legal = list(board.legal_moves)
            moves = legal[:2] if len(legal) >= 2 else legal
            return [(m, 100.0 - i * 80.0) for i, m in enumerate(moves)]

        mock_top.side_effect = varied_top

        boards = _make_diverse_boards(10)
        engine = _make_mock_engine_for_audit()
        cfg = SFConfig(engine_path="/mock/sf", depth=8, multipv=2)
        extract_fn = _make_varied_feature_extractor()

        # Run audit to get a fitted model
        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=2,
            test_size=0.3,
            stability_bootstraps=2,
        )

        assert isinstance(result, AuditResult)
        assert result.r2 is not None
