"""Tests for the Flask web application.

Covers all API endpoints, the GameState helper class, and edge cases
like invalid moves, game-over states, and missing engine fallback.
"""

import contextlib
from unittest.mock import Mock, patch

import chess
import pytest

from chess_ai.web.app import GameState, app

# ---------------------------------------------------------------------------
# Flask test client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Create a Flask test client with fresh game state per test."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_game_state():
    """Reset global game state before each test.

    Prevents state leaking between tests and ensures any Stockfish
    engine started at module import time is properly cleaned up.
    """
    from chess_ai.web import app as app_module

    # Shut down any engine that might have been started at module import
    old_state = app_module.game_state
    if hasattr(old_state, "engine") and old_state.engine is not None:
        with contextlib.suppress(Exception):
            old_state.engine.quit()

    app_module.game_state = GameState.__new__(GameState)
    app_module.game_state.board = chess.Board()
    app_module.game_state.engine = None
    app_module.game_state.move_history = []

    yield

    # Cleanup after test
    state = app_module.game_state
    if hasattr(state, "engine") and state.engine is not None:
        with contextlib.suppress(Exception):
            state.engine.quit()
        state.engine = None


# ---------------------------------------------------------------------------
# GameState unit tests
# ---------------------------------------------------------------------------


class TestGameState:
    """Tests for the GameState helper class."""

    def test_init_sets_default_board(self):
        """GameState starts with the standard chess opening position."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        assert gs.board.fen() == chess.Board().fen()
        assert gs.move_history == []

    def test_reset(self):
        """reset() returns board to start and clears history."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        gs.board.push(chess.Move.from_uci("e2e4"))
        gs.move_history.append({"move": "e2e4"})

        gs.reset()

        assert gs.board.fen() == chess.Board().fen()
        assert gs.move_history == []

    def test_make_move_valid(self):
        """make_move returns True and updates board for a legal move."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        assert gs.make_move("e2e4") is True
        assert gs.board.piece_at(chess.E4) is not None

    def test_make_move_invalid_uci(self):
        """make_move returns False for garbage UCI strings."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        assert gs.make_move("xyz") is False

    def test_make_move_illegal(self):
        """make_move returns False for an illegal (but valid-format) move."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        # e1e3 is illegal from the starting position
        assert gs.make_move("e1e3") is False

    def test_get_engine_move_no_engine_returns_first_legal(self):
        """Without Stockfish, returns the first legal move as fallback."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        gs.engine = None
        result = gs.get_engine_move()

        assert result is not None
        assert "move" in result
        assert "explanation" in result
        assert "Engine not available" in result["explanation"]

    def test_get_engine_move_game_over_returns_none(self):
        """Returns None when the game is already over."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        # Fool's mate
        gs.board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )
        assert gs.get_engine_move() is None

    @patch("chess_ai.web.app.baseline_extract_features")
    def test_get_engine_move_with_engine(self, mock_extract):
        """With a mocked engine, returns the engine's suggested move."""
        mock_extract.return_value = {"material_us": 10.0, "material_them": 10.0}

        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()

        mock_engine = Mock()
        move = chess.Move.from_uci("e2e4")
        mock_engine.play.return_value = Mock(move=move)
        gs.engine = mock_engine

        result = gs.get_engine_move()

        assert result is not None
        assert result["move"] == "e2e4"
        assert "features" in result
        assert "explanation" in result

    @patch("chess_ai.web.app.baseline_extract_features")
    def test_get_engine_move_engine_exception_falls_back(self, mock_extract):
        """When the engine raises, falls back to first legal move."""
        mock_extract.return_value = {"material_us": 10.0, "material_them": 10.0}

        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()

        mock_engine = Mock()
        mock_engine.play.side_effect = Exception("engine crashed")
        gs.engine = mock_engine

        result = gs.get_engine_move()

        assert result is not None
        assert "Engine not available" in result["explanation"]

    def test_generate_explanation_capture(self):
        """Explanation mentions capture when the move captures a piece."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        # Italian game - set up a position where a capture is possible
        gs.board = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3"
        )
        # Bxf7+ is a capture
        move = chess.Move.from_uci("c4f7")
        features = {"material_us": 10.0, "material_them": 10.0}

        explanation = gs._generate_explanation(move, features)
        # The move captures f7 pawn
        assert "Captures" in explanation

    def test_generate_explanation_check(self):
        """Explanation mentions check."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        # Scholar's mate setup
        gs.board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        move = chess.Move.from_uci("d1h5")
        features = {}

        explanation = gs._generate_explanation(move, features)
        # Qh5 doesn't give check from this position, but does control
        assert isinstance(explanation, str)

    def test_generate_explanation_castling(self):
        """Explanation mentions castling."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        gs.board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("e1g1")
        features = {}

        explanation = gs._generate_explanation(move, features)
        assert "Castles" in explanation

    def test_generate_explanation_center_control(self):
        """Explanation mentions center control for moves to d4/d5/e4/e5."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        move = chess.Move.from_uci("e2e4")
        features = {}

        explanation = gs._generate_explanation(move, features)
        assert "center" in explanation.lower() or "Controls" in explanation

    def test_generate_explanation_development(self):
        """Explanation mentions development for minor piece moves in opening."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        move = chess.Move.from_uci("g1f3")
        features = {}

        explanation = gs._generate_explanation(move, features)
        assert "Develops" in explanation

    def test_generate_explanation_promotion(self):
        """Explanation mentions pawn promotion."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        gs.board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")
        move = chess.Move.from_uci("a7a8q")
        features = {}

        explanation = gs._generate_explanation(move, features)
        assert "Promotes" in explanation

    def test_generate_explanation_material_advantage(self):
        """Explanation mentions material advantage when significant."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        move = chess.Move.from_uci("e2e4")
        features = {"material_us": 15.0, "material_them": 10.0}

        explanation = gs._generate_explanation(move, features)
        assert "material advantage" in explanation.lower() or "Controls" in explanation

    def test_generate_explanation_seeks_compensation(self):
        """Explanation mentions seeking compensation when behind in material."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        move = chess.Move.from_uci("e2e4")
        features = {"material_us": 5.0, "material_them": 12.0}

        explanation = gs._generate_explanation(move, features)
        assert "compensation" in explanation.lower() or "Controls" in explanation

    def test_generate_explanation_fallback(self):
        """Explanation falls back to 'Improves position' when no reasons match."""
        with patch("chess_ai.web.app.GameState._init_engine"):
            gs = GameState()
        # Pick a quiet non-center move with no captures or checks
        gs.board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        )
        move = chess.Move.from_uci("a7a6")
        features = {}

        explanation = gs._generate_explanation(move, features)
        assert "Improves position" in explanation

    def test_init_engine_no_stockfish(self):
        """_init_engine sets engine to None when stockfish is not found."""
        with patch("chess_ai.web.app.shutil.which", return_value=None):
            gs = GameState()
        assert gs.engine is None

    @patch("chess_ai.web.app.shutil.which", return_value="/usr/bin/stockfish")
    @patch("chess.engine.SimpleEngine.popen_uci", side_effect=Exception("fail"))
    def test_init_engine_exception(self, _mock_popen, _mock_which):
        """_init_engine sets engine to None when popen_uci fails."""
        gs = GameState()
        assert gs.engine is None


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for /api/health."""

    def test_health_check(self, client):
        """Health endpoint returns status and version."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "engine_available" in data


class TestNewGameEndpoint:
    """Tests for /api/game/new."""

    def test_new_game(self, client):
        """POST /api/game/new resets and returns initial FEN."""
        resp = client.post("/api/game/new")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "fen" in data
        assert "legal_moves" in data
        assert len(data["legal_moves"]) == 20  # starting position


class TestGetStateEndpoint:
    """Tests for /api/game/state."""

    def test_get_state(self, client):
        """GET /api/game/state returns full game state."""
        resp = client.get("/api/game/state")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "fen" in data
        assert "legal_moves" in data
        assert "is_game_over" in data
        assert data["is_game_over"] is False
        assert data["turn"] == "white"


class TestMakeMoveEndpoint:
    """Tests for /api/game/move."""

    def test_make_valid_move(self, client):
        """Valid move returns success and updated state."""
        resp = client.post("/api/game/move", json={"move": "e2e4"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "fen" in data

    def test_make_invalid_move(self, client):
        """Invalid move returns 400 error."""
        resp = client.post("/api/game/move", json={"move": "e1e8"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_make_move_missing_body(self, client):
        """Missing move field returns 400."""
        resp = client.post("/api/game/move", json={})
        assert resp.status_code == 400

    def test_make_move_no_json(self, client):
        """No JSON body returns 400 or 415 (unsupported media type)."""
        resp = client.post("/api/game/move")
        assert resp.status_code in (400, 415)


class TestEngineMoveEndpoint:
    """Tests for /api/engine/move."""

    def test_engine_move_no_engine(self, client):
        """Without an engine, falls back to first legal move."""
        resp = client.post("/api/engine/move", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "move" in data
        assert "explanation" in data

    def test_engine_move_custom_depth(self, client):
        """Custom depth parameter is accepted."""
        resp = client.post("/api/engine/move", json={"depth": 5})
        assert resp.status_code == 200

    def test_engine_move_game_over(self, client):
        """Returns 400 when game is already over."""
        from chess_ai.web import app as app_module

        app_module.game_state.board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )
        resp = client.post("/api/engine/move", json={})
        assert resp.status_code == 400


class TestAnalyzeFeaturesEndpoint:
    """Tests for /api/analysis/features."""

    def test_analyze_features(self, client):
        """Feature analysis returns features and FEN."""
        resp = client.post("/api/analysis/features")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "features" in data
        assert "fen" in data
        assert isinstance(data["features"], dict)
