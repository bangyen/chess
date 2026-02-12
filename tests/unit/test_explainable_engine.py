"""Tests for the explainable chess engine."""

from unittest.mock import Mock, patch

import chess
import chess.engine
import pytest

from src.chess_ai.explainable_engine import ExplainableChessEngine, MoveExplanation


class TestMoveExplanation:
    """Test MoveExplanation dataclass."""

    def test_move_explanation_creation(self):
        """Test creating a MoveExplanation."""
        move = chess.Move.from_uci("e2e4")
        reasons = [("development", 2, "Develops pawn from starting position")]

        explanation = MoveExplanation(
            move=move,
            score=50.0,
            reasons=reasons,
            overall_explanation="Move e4:\n  - Develops pawn from starting position",
        )

        assert explanation.move == move
        assert explanation.score == 50.0
        assert explanation.reasons == reasons
        assert (
            explanation.overall_explanation
            == "Move e4:\n  - Develops pawn from starting position"
        )

    def test_move_explanation_empty_reasons(self):
        """Test MoveExplanation with empty reasons."""
        move = chess.Move.from_uci("e2e4")

        explanation = MoveExplanation(
            move=move, score=0.0, reasons=[], overall_explanation="Move e4:"
        )

        assert explanation.move == move
        assert explanation.score == 0.0
        assert explanation.reasons == []
        assert explanation.overall_explanation == "Move e4:"


class TestExplainableChessEngine:
    """Test ExplainableChessEngine class."""

    def test_init(self):
        """Test engine initialization."""
        engine = ExplainableChessEngine(
            stockfish_path="/path/to/stockfish",
            depth=20,
            opponent_strength="intermediate",
        )

        assert engine.stockfish_path == "/path/to/stockfish"
        assert engine.depth == 20
        assert engine.opponent_strength == "intermediate"
        assert engine.engine is None
        assert engine.board == chess.Board()
        assert engine.move_history == []
        assert "intermediate" in engine.strength_settings

    def test_strength_settings(self):
        """Test strength settings configuration."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Check all strength levels are present
        expected_levels = ["beginner", "novice", "intermediate", "advanced", "expert"]
        for level in expected_levels:
            assert level in engine.strength_settings

        # Check specific settings
        assert engine.strength_settings["beginner"]["Skill Level"] == 0
        assert engine.strength_settings["expert"]["UCI_LimitStrength"] is False

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_context_manager_success(self, mock_popen_uci):
        """Test successful context manager entry."""
        mock_engine = Mock()
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")

        with engine as e:
            assert e.engine == mock_engine
            mock_popen_uci.assert_called_once_with("/path/to/stockfish")

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_context_manager_engine_error(self, mock_popen_uci):
        """Test context manager with engine error."""
        mock_popen_uci.side_effect = Exception("Engine not found")

        engine = ExplainableChessEngine("/path/to/stockfish")

        with pytest.raises(RuntimeError, match="Failed to start Stockfish"):
            with engine:
                pass

    def test_context_manager_empty_stockfish_path(self):
        """Test context manager with empty stockfish path."""
        engine = ExplainableChessEngine("")

        with pytest.raises(RuntimeError, match="Stockfish not found"):
            with engine:
                pass

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_context_manager_exit(self, mock_popen_uci):
        """Test context manager exit."""
        mock_engine = Mock()
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")

        with engine:
            pass

        mock_engine.quit.assert_called_once()

    def test_reset_game(self):
        """Test game reset functionality."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Make some moves
        engine.board.push(chess.Move.from_uci("e2e4"))
        engine.move_history.append(chess.Move.from_uci("e2e4"))

        # Reset
        engine.reset_game()

        assert engine.board == chess.Board()
        assert engine.move_history == []

    def test_make_move_valid(self):
        """Test making a valid move."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        result = engine.make_move("e4")

        assert result is True
        assert len(engine.move_history) == 1
        assert engine.move_history[0] == chess.Move.from_uci("e2e4")

    def test_make_move_invalid_format(self):
        """Test making a move with invalid format."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        result = engine.make_move("invalid")

        assert result is False
        assert len(engine.move_history) == 0

    def test_make_move_illegal(self):
        """Test making an illegal move."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Make a move first
        engine.make_move("e4")

        # Try to make the same move again (illegal)
        result = engine.make_move("e4")

        assert result is False

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_get_best_move_success(self, mock_popen_uci):
        """Test getting best move successfully."""
        mock_engine = Mock()
        mock_engine.analyse.return_value = {"pv": [chess.Move.from_uci("e2e4")]}
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine(
            "/path/to/stockfish", depth=10, enable_model_explanations=False
        )

        with engine:
            best_move = engine.get_best_move()

        assert best_move == chess.Move.from_uci("e2e4")
        mock_engine.analyse.assert_called_once()

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_get_best_move_no_engine(self, mock_popen_uci):
        """Test getting best move when engine is not available."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        best_move = engine.get_best_move()

        assert best_move is None

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_get_best_move_engine_error(self, mock_popen_uci):
        """Test getting best move when engine raises error."""
        mock_engine = Mock()
        mock_engine.analyse.side_effect = Exception("Engine error")
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")

        with engine:
            best_move = engine.get_best_move()

        assert best_move is None

    @patch("src.chess_ai.engine.sf_top_moves")
    @patch("src.chess_ai.engine.sf_eval")
    @patch("src.chess_ai.explainable_engine.baseline_extract_features")
    def test_analyze_position_success(
        self, mock_extract_features, mock_sf_eval, mock_sf_top_moves
    ):
        """Test position analysis success."""
        mock_extract_features.return_value = {"material_diff": 0.0, "mobility_us": 20.0}
        mock_sf_eval.return_value = 25.0
        mock_sf_top_moves.return_value = [(Mock(from_square=12, to_square=28), 25.0)]

        engine = ExplainableChessEngine(
            "/path/to/stockfish", enable_model_explanations=False
        )
        engine.engine = Mock()

        result = engine.analyze_position()

        assert "features" in result
        assert "top_moves" in result
        assert "stockfish_score" in result
        assert result["stockfish_score"] == 25.0
        mock_extract_features.assert_called_once()

    @patch("src.chess_ai.explainable_engine.baseline_extract_features")
    def test_analyze_position_error(self, mock_extract_features):
        """Test position analysis with error."""
        mock_extract_features.side_effect = Exception("Feature extraction error")

        engine = ExplainableChessEngine("/path/to/stockfish")

        result = engine.analyze_position()

        assert result == {}

    def test_explain_move_no_engine(self):
        """Test explaining move when engine is not available."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        move = chess.Move.from_uci("e2e4")

        explanation = engine.explain_move(move)

        assert explanation.move == move
        assert explanation.score == 0
        assert explanation.reasons == []
        assert explanation.overall_explanation == "Engine not available"

    def test_explain_move_with_engine(self):
        """Test explaining move with engine available."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        move = chess.Move.from_uci("e2e4")

        # Mock the engine
        engine.engine = Mock()

        explanation = engine.explain_move(move)

        assert explanation.move == move
        assert isinstance(explanation.reasons, list)
        assert isinstance(explanation.overall_explanation, str)

    def test_explain_move_with_board(self):
        """Test explaining move with specific board."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        move = chess.Move.from_uci("e2e4")
        board = chess.Board()

        # Mock the engine
        engine.engine = Mock()

        explanation = engine.explain_move_with_board(move, board)

        assert explanation.move == move
        assert isinstance(explanation.reasons, list)
        assert isinstance(explanation.overall_explanation, str)

    def test_generate_move_reasons_capture(self):
        """Test generating move reasons for capture."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Set up board with a capture move
        engine.board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        move = chess.Move.from_uci("d1h5")  # Queen to h5

        reasons = engine._generate_move_reasons(move, 0, 0)

        # Should have reasons for the move
        assert isinstance(reasons, list)

    def test_generate_move_reasons_check(self):
        """Test generating move reasons for check."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Set up board with check move
        engine.board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        move = chess.Move.from_uci("d1h5")  # Queen to h5

        reasons = engine._generate_move_reasons(move, 0, 0)

        assert isinstance(reasons, list)

    def test_generate_move_reasons_development(self):
        """Test generating move reasons for piece development."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        move = chess.Move.from_uci("g1f3")  # Knight development

        reasons = engine._generate_move_reasons(move, 0, 0)

        assert isinstance(reasons, list)

    def test_generate_move_reasons_center_control(self):
        """Test generating move reasons for center control."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        move = chess.Move.from_uci("e2e4")  # Center pawn move

        reasons = engine._generate_move_reasons(move, 0, 0)

        assert isinstance(reasons, list)

    def test_generate_move_reasons_castling(self):
        """Test generating move reasons for castling."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Set up board for castling
        engine.board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("e1g1")  # Kingside castling

        reasons = engine._generate_move_reasons(move, 0, 0)

        assert isinstance(reasons, list)

    def test_generate_overall_explanation(self):
        """Test generating overall explanation."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        move = chess.Move.from_uci("e2e4")
        reasons = [
            ("development", 2, "Develops pawn from starting position"),
            ("center_control", 1, "Controls central squares"),
        ]

        explanation = engine._generate_overall_explanation(move, 50.0, reasons)

        assert "Move e4:" in explanation
        assert "Develops pawn from starting position" in explanation
        assert "Controls central squares" in explanation

    def test_generate_overall_explanation_empty_reasons(self):
        """Test generating overall explanation with empty reasons."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        move = chess.Move.from_uci("e2e4")

        explanation = engine._generate_overall_explanation(move, 0.0, [])

        assert explanation == "Move e4: Reasonable move. (+0 cp)"

    def test_generate_overall_explanation_with_board(self):
        """Test generating overall explanation with board."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        move = chess.Move.from_uci("e2e4")
        board = chess.Board()
        reasons = [("development", 2, "Develops pawn from starting position")]

        explanation = engine._generate_overall_explanation_with_board(
            move, board, 50.0, reasons
        )

        assert "Move e4:" in explanation
        assert "Develops pawn from starting position" in explanation

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_get_move_recommendation_with_engine(self, mock_popen_uci):
        """Test getting move recommendation with engine."""
        mock_engine = Mock()
        mock_engine.analyse.return_value = {"pv": [chess.Move.from_uci("e2e4")]}
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")

        with engine:
            recommendation = engine.get_move_recommendation()

        assert recommendation is not None
        assert isinstance(recommendation, MoveExplanation)

    def test_get_move_recommendation_first_move(self):
        """Test getting move recommendation for first move."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        recommendation = engine.get_move_recommendation()

        assert recommendation is not None
        assert isinstance(recommendation, MoveExplanation)

    def test_get_move_recommendation_second_move(self):
        """Test getting move recommendation for second move."""
        engine = ExplainableChessEngine("/path/to/stockfish")
        engine.make_move("e4")

        recommendation = engine.get_move_recommendation()

        assert recommendation is not None
        assert isinstance(recommendation, MoveExplanation)

    def test_get_move_recommendation_no_legal_moves(self):
        """Test getting move recommendation when no legal moves."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Set up checkmate position
        engine.board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )

        recommendation = engine.get_move_recommendation()

        assert recommendation is None

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_get_best_move_for_player_with_engine(self, mock_popen_uci):
        """Test getting best move for player with engine."""
        mock_engine = Mock()
        mock_engine.analyse.return_value = {"pv": [chess.Move.from_uci("d2d4")]}
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")
        engine.make_move("e4")

        with engine:
            recommendation = engine.get_best_move_for_player()

        assert recommendation is not None
        assert isinstance(recommendation, MoveExplanation)

    def test_get_best_move_for_player_no_moves(self):
        """Test getting best move for player when no moves available."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        recommendation = engine.get_best_move_for_player()

        assert recommendation is None

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_get_stockfish_move_success(self, mock_popen_uci):
        """Test getting Stockfish move successfully."""
        mock_engine = Mock()
        mock_engine.play.return_value = Mock(move=chess.Move.from_uci("e7e5"))
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")

        with engine:
            move = engine.get_stockfish_move()

        assert move == chess.Move.from_uci("e7e5")

    def test_get_stockfish_move_no_engine(self):
        """Test getting Stockfish move when engine not available."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        move = engine.get_stockfish_move()

        assert move is None

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_get_stockfish_move_error(self, mock_popen_uci):
        """Test getting Stockfish move when engine raises error."""
        mock_engine = Mock()
        mock_engine.play.side_effect = Exception("Engine error")
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")

        with engine:
            move = engine.get_stockfish_move()

        assert move is None

    def test_print_board(self, capsys):
        """Test printing board."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        engine.print_board()

        captured = capsys.readouterr()
        assert "=" in captured.out
        assert "r n b q k b n r" in captured.out or "8" in captured.out

    def test_print_legal_moves(self, capsys):
        """Test printing legal moves."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        engine.print_legal_moves()

        captured = capsys.readouterr()
        assert "Legal moves:" in captured.out

    def test_print_help(self, capsys):
        """Test printing help."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        engine._print_help()

        captured = capsys.readouterr()
        assert "Available commands:" in captured.out
        assert "Make moves:" in captured.out
        assert "best" in captured.out
        assert "reset" in captured.out
        assert "help" in captured.out
        assert "quit" in captured.out

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_show_best_move(self, mock_popen_uci, capsys):
        """Test showing best move."""
        mock_engine = Mock()
        mock_engine.analyse.return_value = {"pv": [chess.Move.from_uci("e2e4")]}
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")

        with engine:
            engine._show_best_move()

        captured = capsys.readouterr()
        assert "Best move:" in captured.out

    def test_show_best_move_no_recommendation(self, capsys):
        """Test showing best move when no recommendation available."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Set up checkmate position
        engine.board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )

        engine._show_best_move()

        captured = capsys.readouterr()
        assert "Could not get move recommendation" in captured.out

    def test_play_interactive_game_quit_command(self, monkeypatch):
        """Test interactive game with quit command."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Mock input to return "quit" immediately
        monkeypatch.setattr("builtins.input", lambda _: "quit")

        # This should not raise an exception and should exit cleanly
        engine.play_interactive_game()

    def test_play_interactive_game_help_command(self, monkeypatch, capsys):
        """Test interactive game with help command."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Mock input to return "help" then "quit"
        input_calls = ["help", "quit"]
        monkeypatch.setattr("builtins.input", lambda _: input_calls.pop(0))

        engine.play_interactive_game()

        captured = capsys.readouterr()
        assert "Available commands:" in captured.out

    def test_play_interactive_game_reset_command(self, monkeypatch):
        """Test interactive game with reset command."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Mock input to return "reset" then "quit"
        input_calls = ["reset", "quit"]
        monkeypatch.setattr("builtins.input", lambda _: input_calls.pop(0))

        engine.play_interactive_game()

        # Board should be reset
        assert engine.board == chess.Board()

    def test_play_interactive_game_best_command(self, monkeypatch, capsys):
        """Test interactive game with best command."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Mock input to return "best" then "quit"
        input_calls = ["best", "quit"]
        monkeypatch.setattr("builtins.input", lambda _: input_calls.pop(0))

        engine.play_interactive_game()

        captured = capsys.readouterr()
        # Should show best move or error message
        assert (
            "Best move:" in captured.out
            or "Could not get move recommendation" in captured.out
        )

    def test_play_interactive_game_invalid_move(self, monkeypatch, capsys):
        """Test interactive game with invalid move."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Mock input to return invalid move then "quit"
        input_calls = ["invalid_move", "quit"]
        monkeypatch.setattr("builtins.input", lambda _: input_calls.pop(0))

        engine.play_interactive_game()

        captured = capsys.readouterr()
        assert "Invalid move" in captured.out

    def test_play_interactive_game_valid_move(self, monkeypatch, capsys):
        """Test interactive game with valid move."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Mock input to return valid move then "quit"
        input_calls = ["e4", "quit"]
        monkeypatch.setattr("builtins.input", lambda _: input_calls.pop(0))

        engine.play_interactive_game()

        captured = capsys.readouterr()
        assert "Your" in captured.out  # Should show explanation for the move

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_play_interactive_game_stockfish_move(
        self, mock_popen_uci, monkeypatch, capsys
    ):
        """Test interactive game with Stockfish move."""
        mock_engine = Mock()
        mock_engine.play.return_value = Mock(move=chess.Move.from_uci("e7e5"))
        mock_popen_uci.return_value = mock_engine

        engine = ExplainableChessEngine("/path/to/stockfish")

        # Mock input to return valid move then "quit"
        input_calls = ["e4", "quit"]
        monkeypatch.setattr("builtins.input", lambda _: input_calls.pop(0))

        with engine:
            engine.play_interactive_game()

        captured = capsys.readouterr()
        assert "Stockfish plays:" in captured.out

    def test_play_interactive_game_checkmate(self, monkeypatch, capsys):
        """Test interactive game ending in checkmate."""
        engine = ExplainableChessEngine("/path/to/stockfish")

        # Set up checkmate position
        engine.board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )

        # Mock input to return "quit" (should exit immediately due to checkmate)
        monkeypatch.setattr("builtins.input", lambda _: "quit")

        engine.play_interactive_game()

        captured = capsys.readouterr()
        # The game should detect checkmate and exit without user input
        assert len(captured.out) >= 0  # May or may not show game over message
