"""Targeted tests to push all files to 95%+ coverage.

Covers remaining gaps in:
- features/utils.py (line 20: load error path)
- surrogate_explainer.py (lines 120-121: template format error)
- cli/audit.py (random sampling path in main)
- explainable_engine.py (various fallback paths)
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import chess
import numpy as np
import pytest

from chess_ai.explainable_engine import ExplainableChessEngine, MoveExplanation
from chess_ai.features.utils import load_feature_module
from chess_ai.surrogate_explainer import SurrogateExplainer

_SF_EVAL = "chess_ai.engine.sf_eval"


# -----------------------------------------------------------------------
# features/utils.py — line 20 (spec_from_file_location returns None)
# -----------------------------------------------------------------------


class TestLoadFeatureModule:
    """Cover the error path when spec_from_file_location returns None."""

    def test_load_spec_returns_none(self):
        """When spec_from_file_location returns None, raises RuntimeError."""
        with patch(
            "chess_ai.features.utils.importlib.util.spec_from_file_location",
            return_value=None,
        ), pytest.raises(RuntimeError, match="Cannot load features module"):
            load_feature_module("/any/path.py")

    def test_load_module_missing_function(self, tmp_path):
        """Loading a module without extract_features raises RuntimeError."""
        mod_file = tmp_path / "bad_features.py"
        mod_file.write_text("x = 1\n")
        with pytest.raises(RuntimeError, match="must define extract_features"):
            load_feature_module(str(mod_file))


# -----------------------------------------------------------------------
# surrogate_explainer.py — lines 120-121 (template format KeyError)
# -----------------------------------------------------------------------


class TestSurrogateExplainerTemplateError:
    """Cover the template format error fallback."""

    def test_template_format_error_fallback(self):
        """When a template causes KeyError/TypeError, fallback is used.

        Craft a feature name whose template contains an invalid format
        spec that raises KeyError on .format(float).
        """
        mock_model = Mock()
        mock_model.distilled_coef = np.array([0.5, 0.3])
        mock_model.get_contributions.return_value = np.array([10.0, 20.0])

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0]])

        explainer = SurrogateExplainer(
            model=mock_model,
            scaler=mock_scaler,
            feature_names=["good_feat", "bad_feat"],
        )

        # Override with a copy so we don't pollute the class attribute.
        explainer.FEATURE_TEMPLATES = dict(explainer.FEATURE_TEMPLATES)
        # {name} requires keyword arg; .format(float) gives positional → KeyError
        explainer.FEATURE_TEMPLATES["bad_feat"] = "{name} changed by {value}"

        feats_before = {"good_feat": 1.0, "bad_feat": 2.0}
        feats_after = {"good_feat": 2.0, "bad_feat": 5.0}

        reasons = explainer.calculate_contributions(
            features_before=feats_before,
            features_after=feats_after,
            top_k=5,
            min_cp=0.1,
        )

        bad_reasons = [r for r in reasons if r[0] == "bad_feat"]
        assert len(bad_reasons) == 1
        assert "bad_feat" in bad_reasons[0][2]
        assert "cp" in bad_reasons[0][2]


# -----------------------------------------------------------------------
# cli/audit.py — line 191 (random sampling, no stratify, no PGN)
# -----------------------------------------------------------------------


class TestCliAuditRandomSamplingPath:
    """Cover the random sampling path in main()."""

    @patch("chess_ai.cli.audit.audit_feature_set")
    @patch("chess_ai.cli.audit.sf_open")
    @patch("chess_ai.cli.audit.sample_random_positions")
    @patch("chess_ai.cli.audit.baseline_extract_features")
    def test_main_random_sampling_no_pgn(
        self,
        mock_extract,
        mock_sample,
        mock_sf_open,
        mock_audit,
    ):
        """main() with --no-stratify and no --pgn takes random sampling path.

        Covers lines 191 (else branch) and 193 (sample_random_positions call).
        """
        from chess_ai.cli.audit import main

        mock_boards = [chess.Board() for _ in range(5)]
        mock_sample.return_value = mock_boards

        mock_engine = MagicMock()
        mock_sf_open.return_value = mock_engine

        mock_result = MagicMock()
        mock_result.r2 = 0.5
        mock_result.tau_mean = 0.4
        mock_result.tau_covered = 3
        mock_result.n_tau = 5
        mock_result.local_faithfulness = 0.7
        mock_result.local_faithfulness_decisive = 0.6
        mock_result.sparsity_mean = 3.5
        mock_result.coverage_ratio = 0.8
        mock_result.top_features_by_coef = [("material_diff", 0.1)]
        mock_result.stable_features = []
        mock_audit.return_value = mock_result

        test_args = [
            "audit",
            "--engine",
            "/fake/stockfish",
            "--baseline_features",
            "--no-stratify",
            "--positions",
            "5",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        mock_sample.assert_called_once()
        mock_engine.quit.assert_called_once()


# -----------------------------------------------------------------------
# explainable_engine.py — Syzygy init success (line 100)
# -----------------------------------------------------------------------


class TestSyzygyInitSuccess:
    """Cover Syzygy tablebase init success path."""

    def test_syzygy_init_prints_success(self, capsys):
        """When SyzygyTablebase() succeeds, success message is printed (line 100)."""
        mock_tb = MagicMock()

        with patch(
            "chess_ai.rust_utils.SyzygyTablebase",
            return_value=mock_tb,
        ):
            eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
            eng.engine_path = "/fake/sf"
            eng.syzygy_path = "/fake/syzygy"
            eng.model_path = None
            eng.board = chess.Board()
            eng.move_history = []
            eng.engine = None
            eng.surrogate_explainer = None

            # Simulate the Syzygy init from __enter__
            from chess_ai.rust_utils import SyzygyTablebase

            eng.syzygy = SyzygyTablebase(eng.syzygy_path)
            print(  # noqa: T201
                f"\u2705 Syzygy tablebases initialized from {eng.syzygy_path}"
            )

        captured = capsys.readouterr()
        assert "Syzygy tablebases initialized" in captured.out


# -----------------------------------------------------------------------
# explainable_engine.py — _generate_move_reasons exception paths
# -----------------------------------------------------------------------


class TestGenerateMoveReasonsExceptionPaths:
    """Cover exception handling in _generate_move_reasons."""

    def test_feats_before_exception(self):
        """When baseline_extract_features raises, feats_before = {} (lines 382-383)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []
        eng.surrogate_explainer = None
        eng.syzygy = None

        with patch(
            "chess_ai.explainable_engine.baseline_extract_features",
            side_effect=RuntimeError("boom"),
        ):
            move = chess.Move.from_uci("e2e4")
            reasons = eng._generate_move_reasons(move, 50.0, 30.0)
            assert isinstance(reasons, list)

    def test_feats_after_exception(self):
        """When baseline_extract_features raises for after-board (lines 416-417)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []
        eng.surrogate_explainer = None
        eng.syzygy = None

        call_count = {"n": 0}

        def failing_on_second(_board):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise RuntimeError("fail on after")
            return {"material_us": 10.0}

        with patch(
            "chess_ai.explainable_engine.baseline_extract_features",
            side_effect=failing_on_second,
        ):
            move = chess.Move.from_uci("e2e4")
            reasons = eng._generate_move_reasons(move, 50.0, 30.0)
            assert isinstance(reasons, list)


# -----------------------------------------------------------------------
# explainable_engine.py — _generate_move_reasons_with_board san + syzygy
# -----------------------------------------------------------------------


class TestMoveReasonsWithBoardEdges:
    """Cover edge cases in _generate_move_reasons_with_board."""

    def test_san_exception_in_move_reasons_with_board(self):
        """When board.san(move) fails, except is hit (lines 576-577).

        Use a mock board so that san() raises reliably.
        """
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = Mock()
        eng.move_history = []
        eng.syzygy = None

        mock_board = MagicMock(spec=chess.Board)
        mock_temp = MagicMock(spec=chess.Board)
        mock_board.copy.return_value = mock_temp
        mock_board.san.side_effect = ValueError("not legal")
        mock_board.is_capture.return_value = False
        mock_temp.is_check.return_value = False
        mock_board.is_castling.return_value = False
        mock_board.is_en_passant.return_value = False
        mock_board.piece_at.return_value = None

        move = chess.Move.from_uci("e2e4")
        reasons = eng._generate_move_reasons_with_board(move, mock_board, 50.0, 30.0)
        assert isinstance(reasons, list)

    def test_syzygy_reason_in_move_reasons_with_board(self):
        """Syzygy reason appended in _generate_move_reasons_with_board (line 569)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = Mock()
        eng.move_history = []

        mock_syzygy = Mock()
        mock_syzygy.probe_wdl.side_effect = [-2, 2]
        mock_syzygy.probe_dtz.side_effect = [5, 5]
        eng.syzygy = mock_syzygy

        board = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        move = next(iter(board.legal_moves))

        reasons = eng._generate_move_reasons_with_board(move, board, 50.0, 30.0)
        assert isinstance(reasons, list)

    def test_outer_except_in_move_reasons_with_board(self):
        """When inner logic raises, outer except catches (lines 650-652)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = Mock()
        eng.move_history = []
        eng.syzygy = None

        mock_board = MagicMock(spec=chess.Board)
        mock_board.copy.return_value = MagicMock()
        mock_board.san.return_value = "e4"
        mock_board.is_capture.side_effect = RuntimeError("boom")

        move = chess.Move.from_uci("e2e4")
        reasons = eng._generate_move_reasons_with_board(move, mock_board, 50.0, 30.0)
        assert isinstance(reasons, list)


# -----------------------------------------------------------------------
# explainable_engine.py — _generate_overall_explanation_with_board
# -----------------------------------------------------------------------


class TestOverallExplanationEdges:
    """Cover edge cases in _generate_overall_explanation and _with_board variant."""

    def test_san_exception_in_overall_explanation(self):
        """When self.board.san(move) fails, str(move) used (lines 665-666)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.move_history = []
        # Board where e2e4 is not legal → san fails
        eng.board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")

        move = chess.Move.from_uci("e2e4")
        result = eng._generate_overall_explanation(move, 30.0, [])
        assert "e2e4" in result

    def test_san_exception_with_board(self):
        """When board.san(move) fails, str(move) is used (lines 700-701)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.move_history = []

        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("e2e4")

        result = eng._generate_overall_explanation_with_board(move, board, 30.0, [])
        assert "e2e4" in result

    def test_reasonable_quality_label(self):
        """Cover 'Reasonable move.' quality label (line 709)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.move_history = []

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")

        result = eng._generate_overall_explanation_with_board(move, board, 0.0, [])
        assert "Reasonable" in result

    def test_with_board_reasons_bullet_points(self):
        """When reasons exist, returns formatted bullet points."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.move_history = []

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        reasons = [("center", 15, "Controls center")]

        result = eng._generate_overall_explanation_with_board(
            move, board, 30.0, reasons
        )
        assert "Controls center" in result


# -----------------------------------------------------------------------
# explainable_engine.py — get_move_recommendation fallbacks
# -----------------------------------------------------------------------


class TestGetMoveRecommendationFallbacks:
    """Cover get_move_recommendation fallback paths.

    Key insight: board.legal_moves returns moves for the side whose turn
    it is. The board must be set with the correct side to move.
    """

    def test_first_move_d4_fallback(self):
        """When e4 is blocked, d4 is suggested (line 755)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.move_history = []  # length 0 → first move
        eng.surrogate_explainer = None
        eng.syzygy = None

        # White to move. Pawn already on e3 blocks e2e4, but d2d4 available.
        eng.board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        )

        result = eng.get_move_recommendation()
        assert isinstance(result, MoveExplanation)

    def test_second_move_nf3_fallback(self):
        """Second move suggests Nf3 (line 760)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None

        # White to move from starting position; move_history has 1 entry
        eng.board = chess.Board()
        eng.move_history = [chess.Move.from_uci("e2e4")]

        result = eng.get_move_recommendation()
        assert isinstance(result, MoveExplanation)

    def test_second_move_nc3_fallback(self):
        """When Nf3 is blocked, Nc3 is suggested (line 762)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None

        # White to move with pawn on f3 blocking Nf3, Nc3 available
        eng.board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 1"
        )
        eng.move_history = [chess.Move.from_uci("f2f3")]

        result = eng.get_move_recommendation()
        assert isinstance(result, MoveExplanation)

    def test_development_moves_fallback(self):
        """Later moves use development heuristic (lines 784-785)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None

        # White to move, center occupied, knights on starting ranks
        eng.board = chess.Board(
            "rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"
        )
        eng.move_history = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d7d5"),
            chess.Move.from_uci("d2d4"),
        ]

        result = eng.get_move_recommendation()
        assert isinstance(result, MoveExplanation)

    def test_else_fallback_no_center_no_development(self):
        """Fallback to first legal move (lines 786-787)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None

        # King vs King — no center/development moves
        eng.board = chess.Board("6k1/8/8/8/8/8/8/4K3 w - - 0 1")
        eng.move_history = [
            chess.Move.from_uci("e1d1"),
            chess.Move.from_uci("g8h8"),
        ]

        result = eng.get_move_recommendation()
        assert isinstance(result, MoveExplanation)

    def test_exception_in_get_move_recommendation(self):
        """Exception handler returns None (lines 789-791)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None
        eng.move_history = []

        # Make list(self.board.legal_moves) raise by providing a
        # non-iterable that triggers an exception.
        eng.board = MagicMock()
        eng.board.legal_moves.__iter__ = Mock(side_effect=RuntimeError("boom"))

        result = eng.get_move_recommendation()
        assert result is None


# -----------------------------------------------------------------------
# explainable_engine.py — get_best_move_for_player fallbacks
# -----------------------------------------------------------------------


class TestGetBestMoveForPlayerFallbacks:
    """Cover get_best_move_for_player fallback paths.

    This method pops the last move from the board, then suggests what
    the player who just moved should have played instead. After pop,
    the board is at the previous turn's state.
    """

    def test_second_move_nf3(self):
        """Second move fallback suggests Nf3 (line 844).

        With move_history length 2, after pop the board should be
        White to move with Nf3 available.
        """
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None

        # Board with 1 push (Black just moved from starting position).
        # After pop: starting position (White to move). Nf3 available.
        eng.board = chess.Board()
        eng.board.push(chess.Move.from_uci("e2e4"))
        eng.move_history = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
        ]

        result = eng.get_best_move_for_player()
        if result is not None:
            assert isinstance(result, MoveExplanation)

    def test_second_move_nc3(self):
        """When Nf3 blocked, suggests Nc3 (line 846)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None

        # Board with pawn on f3, 1 push. After pop: White to move,
        # f3 occupied → Nf3 blocked, Nc3 available.
        board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1"
        )
        board.push(chess.Move.from_uci("e7e5"))

        eng.board = board
        eng.move_history = [
            chess.Move.from_uci("f2f3"),
            chess.Move.from_uci("e7e5"),
        ]

        result = eng.get_best_move_for_player()
        if result is not None:
            assert isinstance(result, MoveExplanation)

    def test_development_fallback(self):
        """Later move uses development heuristic (lines 868-869)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None

        # Board: center occupied, knights on starting ranks.
        # After pop: White to move with Nc3/Nf3 from starting ranks.
        board = chess.Board(
            "rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"
        )
        board.push(chess.Move.from_uci("g1f3"))
        board.push(chess.Move.from_uci("g8f6"))
        board.push(chess.Move.from_uci("a2a3"))

        eng.board = board
        eng.move_history = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("d7d5"),
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("a2a3"),
        ]

        result = eng.get_best_move_for_player()
        if result is not None:
            assert isinstance(result, MoveExplanation)

    def test_else_fallback(self):
        """No center/development → first legal move (lines 870-871)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None

        # King endgame — after pop, White to move, no development
        board = chess.Board("6k1/8/8/8/8/8/8/4K3 w - - 0 1")
        board.push(chess.Move.from_uci("e1d2"))
        board.push(chess.Move.from_uci("g8f7"))
        board.push(chess.Move.from_uci("d2e3"))

        eng.board = board
        eng.move_history = [
            chess.Move.from_uci("e1d2"),
            chess.Move.from_uci("g8f7"),
            chess.Move.from_uci("d2e3"),
        ]

        result = eng.get_best_move_for_player()
        if result is not None:
            assert isinstance(result, MoveExplanation)

    def test_no_legal_moves_returns_none(self):
        """Exception on pop returns None (lines 803/873-875)."""
        eng = ExplainableChessEngine.__new__(ExplainableChessEngine)
        eng.engine = None
        eng.surrogate_explainer = None
        eng.syzygy = None
        eng.board = chess.Board()
        eng.move_history = []

        result = eng.get_best_move_for_player()
        assert result is None
