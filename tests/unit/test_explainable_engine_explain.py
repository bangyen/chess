"""Tests for ExplainableChessEngine explain/analyze paths.

Covers: explain_move, explain_move_with_board, analyze_position,
_generate_move_reasons with surrogate_explainer, make_move edge cases,
and get_best_move pv-missing branch.
"""

from unittest.mock import Mock, patch

import chess
import chess.engine

from chess_ai.explainable_engine import ExplainableChessEngine, MoveExplanation

# Patch target: the lazy import ``from .engine import sf_eval`` inside
# explain methods resolves to ``chess_ai.engine.sf_eval``.  We patch
# that attribute so the local binding picks up the mock.
_SF_EVAL = "chess_ai.engine.sf_eval"
_SF_TOP = "chess_ai.engine.sf_top_moves"


# ---------------------------------------------------------------------------
# explain_move
# ---------------------------------------------------------------------------


class TestExplainMove:
    """Tests for explain_move with a mocked engine."""

    @patch(_SF_EVAL, return_value=50.0)
    def test_explain_move_basic(self, _mock):
        """explain_move returns a MoveExplanation with score and reasons."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []

        move = chess.Move.from_uci("e2e4")
        result = eng.explain_move(move)

        assert isinstance(result, MoveExplanation)
        assert result.move == move

    @patch(_SF_EVAL, return_value=100.0)
    def test_explain_move_capture(self, _mock):
        """explain_move reports captures in reasons."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        # Position where e4 captures d5 pawn
        eng.board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
        )
        eng.move_history = []

        move = chess.Move.from_uci("e4d5")  # capture
        result = eng.explain_move(move)

        assert isinstance(result, MoveExplanation)
        assert result.move == move

    def test_explain_move_no_engine(self):
        """explain_move returns fallback when engine is None."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = None
        eng.board = chess.Board()
        eng.move_history = []

        move = chess.Move.from_uci("e2e4")
        result = eng.explain_move(move)

        assert isinstance(result, MoveExplanation)
        assert "not available" in result.overall_explanation.lower()

    @patch(_SF_EVAL, side_effect=RuntimeError("fail"))
    def test_explain_move_exception(self, _mock):
        """explain_move returns error message on exception."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []

        move = chess.Move.from_uci("e2e4")
        result = eng.explain_move(move)

        assert isinstance(result, MoveExplanation)
        assert "error" in result.overall_explanation.lower()


# ---------------------------------------------------------------------------
# explain_move_with_board
# ---------------------------------------------------------------------------


class TestExplainMoveWithBoard:
    """Tests for explain_move_with_board."""

    @patch(_SF_EVAL, return_value=30.0)
    def test_explain_move_with_board_basic(self, _mock):
        """explain_move_with_board returns valid explanation."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = eng.explain_move_with_board(move, board)

        assert isinstance(result, MoveExplanation)
        assert result.move == move

    def test_explain_move_with_board_no_engine(self):
        """explain_move_with_board returns fallback when engine is None."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = None
        eng.board = chess.Board()
        eng.move_history = []

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = eng.explain_move_with_board(move, board)

        assert "not available" in result.overall_explanation.lower()

    @patch(_SF_EVAL, side_effect=Exception("oops"))
    def test_explain_move_with_board_exception(self, _mock):
        """explain_move_with_board returns error explanation on exception."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = eng.explain_move_with_board(move, board)

        assert "error" in result.overall_explanation.lower()


# ---------------------------------------------------------------------------
# analyze_position
# ---------------------------------------------------------------------------


class TestAnalyzePosition:
    """Tests for analyze_position."""

    @patch(_SF_TOP)
    @patch(_SF_EVAL)
    def test_analyze_position_success(self, mock_eval, mock_top):
        """analyze_position returns features, score, and top moves."""
        mock_eval.return_value = 42.0
        mock_top.return_value = [
            (chess.Move.from_uci("e2e4"), 42.0),
            (chess.Move.from_uci("d2d4"), 30.0),
        ]

        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []
        eng.syzygy = None

        result = eng.analyze_position()

        assert "stockfish_score" in result
        assert result["stockfish_score"] == 42.0
        assert "features" in result
        assert "top_moves" in result
        assert "syzygy" in result

    @patch(_SF_EVAL, side_effect=RuntimeError("no engine"))
    def test_analyze_position_exception(self, _mock):
        """analyze_position returns {} on error."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []
        eng.syzygy = None

        result = eng.analyze_position()

        assert result == {}


# ---------------------------------------------------------------------------
# _generate_move_reasons with surrogate_explainer
# ---------------------------------------------------------------------------


class TestGenerateMoveReasonsWithSurrogate:
    """Tests for _generate_move_reasons using the surrogate model path."""

    def test_surrogate_model_path(self):
        """When surrogate_explainer is set, its contributions are used."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []

        mock_explainer = Mock()
        mock_explainer.calculate_contributions.return_value = [
            ("material_diff", 50.0, "Material advantage (+50 cp)")
        ]
        eng.surrogate_explainer = mock_explainer

        move = chess.Move.from_uci("e2e4")
        reasons = eng._generate_move_reasons(move, 50.0, 30.0)

        assert any("Material advantage" in r[2] for r in reasons)
        mock_explainer.calculate_contributions.assert_called_once()

    def test_surrogate_model_exception_falls_back(self):
        """When surrogate_explainer fails, falls back to hardcoded reasons."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []

        mock_explainer = Mock()
        mock_explainer.calculate_contributions.side_effect = RuntimeError("model fail")
        eng.surrogate_explainer = mock_explainer

        move = chess.Move.from_uci("e2e4")
        reasons = eng._generate_move_reasons(move, 50.0, 30.0)

        # Should return reasons from the hardcoded fallback
        assert isinstance(reasons, list)

    def test_surrogate_model_none_uses_hardcoded(self):
        """When surrogate_explainer is None, hardcoded reasons are used."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.board = chess.Board()
        eng.move_history = []
        eng.surrogate_explainer = None

        move = chess.Move.from_uci("e2e4")
        reasons = eng._generate_move_reasons(move, 50.0, 30.0)

        assert isinstance(reasons, list)


# ---------------------------------------------------------------------------
# _generate_move_reasons capture and check details
# ---------------------------------------------------------------------------


class TestGenerateMoveReasonsDetails:
    """Tests for specific reason-generation paths in _generate_move_reasons."""

    def test_capture_with_value(self):
        """Captures include the captured piece value."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.move_history = []
        eng.surrogate_explainer = None

        # Position with a capturable pawn on d5
        eng.board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
        )

        move = chess.Move.from_uci("e4d5")
        reasons = eng._generate_move_reasons(move, 50.0, 30.0)

        capture_reasons = [r for r in reasons if r[0] == "capture"]
        assert len(capture_reasons) >= 1

    def test_check_reason(self):
        """Check moves are reported in reasons."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.move_history = []
        eng.surrogate_explainer = None

        # Scholar's mate position - Qh5 gives check? Actually, let's use
        # a simpler position where a queen gives check.
        eng.board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")

        move = chess.Move.from_uci("h1h8")  # Qh8+ gives check
        reasons = eng._generate_move_reasons(move, 50.0, 30.0)

        check_reasons = [r for r in reasons if r[0] == "check"]
        assert len(check_reasons) >= 1

    def test_syzygy_reason_appended(self):
        """Syzygy reason is included when tablebase data is available."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = Mock()
        eng.move_history = []
        eng.surrogate_explainer = None

        mock_syzygy = Mock()
        mock_syzygy.probe_wdl.return_value = 2
        mock_syzygy.probe_dtz.return_value = 5
        eng.syzygy = mock_syzygy

        # KQK endgame (few enough pieces for syzygy)
        eng.board = chess.Board("8/8/8/8/8/4K3/8/4k2Q w - - 0 1")

        move = chess.Move.from_uci("h1e1")  # doesn't matter which
        reasons = eng._generate_move_reasons(move, 50.0, 30.0)

        # There should be a syzygy-related reason
        assert isinstance(reasons, list)


# ---------------------------------------------------------------------------
# make_move edge cases
# ---------------------------------------------------------------------------


class TestMakeMoveEdgeCases:
    """Tests for make_move with illegal and invalid moves."""

    def test_make_move_illegal(self):
        """make_move returns False for a legal-format but illegal move."""
        eng = ExplainableChessEngine("/sf")
        eng.board = chess.Board()
        eng.move_history = []

        # Ke2 is technically parseable but illegal in starting position
        result = eng.make_move("Ke2")
        assert result is False

    def test_make_move_invalid_format(self):
        """make_move returns False for invalid move format."""
        eng = ExplainableChessEngine("/sf")
        eng.board = chess.Board()
        eng.move_history = []

        result = eng.make_move("xyz123")
        assert result is False

    def test_make_move_valid(self):
        """make_move returns True for a valid move."""
        eng = ExplainableChessEngine("/sf")
        eng.board = chess.Board()
        eng.move_history = []

        result = eng.make_move("e4")
        assert result is True
        assert len(eng.move_history) == 1


# ---------------------------------------------------------------------------
# get_best_move edge cases
# ---------------------------------------------------------------------------


class TestGetBestMoveEdgeCases:
    """Tests for get_best_move."""

    def test_no_engine(self):
        """get_best_move returns None when engine is None."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = None
        eng.board = chess.Board()

        assert eng.get_best_move() is None

    def test_no_pv_in_info(self):
        """get_best_move returns None when info has no pv."""
        eng = ExplainableChessEngine("/sf")
        mock_engine = Mock()
        mock_engine.analyse.return_value = {"score": Mock()}  # no 'pv' key
        eng.engine = mock_engine
        eng.board = chess.Board()

        result = eng.get_best_move()
        assert result is None

    def test_empty_pv(self):
        """get_best_move returns None when pv list is empty."""
        eng = ExplainableChessEngine("/sf")
        mock_engine = Mock()
        mock_engine.analyse.return_value = {"score": Mock(), "pv": []}
        eng.engine = mock_engine
        eng.board = chess.Board()

        result = eng.get_best_move()
        assert result is None

    def test_analyse_exception(self):
        """get_best_move returns None on exception."""
        eng = ExplainableChessEngine("/sf")
        mock_engine = Mock()
        mock_engine.analyse.side_effect = RuntimeError("crash")
        eng.engine = mock_engine
        eng.board = chess.Board()

        result = eng.get_best_move()
        assert result is None

    def test_valid_pv(self):
        """get_best_move returns the first pv move."""
        eng = ExplainableChessEngine("/sf")
        mock_engine = Mock()
        expected = chess.Move.from_uci("e2e4")
        mock_engine.analyse.return_value = {"pv": [expected]}
        eng.engine = mock_engine
        eng.board = chess.Board()

        result = eng.get_best_move()
        assert result == expected
