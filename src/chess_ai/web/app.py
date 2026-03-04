#!/usr/bin/env python3
"""
Flask web application for Chess AI.

Provides API endpoints for playing chess, analyzing positions,
and running feature explainability audits.
"""

import threading
from typing import Any, Dict, List, Optional

import chess
import chess.engine
from flask import Flask, jsonify, render_template, request

from chess_ai.explainable_engine import ExplainableChessEngine
from chess_ai.features.baseline import baseline_extract_features
from chess_ai.utils.engine_discovery import find_stockfish

app = Flask(__name__, template_folder="templates", static_folder="static")


class GameState:
    """Manages chess game state and engine analysis."""

    def __init__(self) -> None:
        self.board = chess.Board()
        try:
            self.stockfish_path = find_stockfish()
        except FileNotFoundError:
            self.stockfish_path = None
        self.engine: Optional[ExplainableChessEngine] = None
        self.move_history: List[Dict[str, Any]] = []
        self.model_ready = False
        self.training_error = False
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize ExplainableChessEngine in a background thread."""
        if not self.stockfish_path:
            self.training_error = True
            return

        def _train() -> None:
            try:
                # The engine should already be available and entered
                if self.engine:
                    self.engine.initialize_model()
                    self.model_ready = True
            except Exception:
                self.training_error = True

        try:
            if self.stockfish_path:
                self.engine = ExplainableChessEngine(
                    stockfish_path=self.stockfish_path,
                    depth=12,
                    enable_model_explanations=True,
                )
                self.engine.__enter__()

                # Start background training thread
                thread = threading.Thread(target=_train, daemon=True)
                thread.start()
        except Exception:
            self.training_error = True
            self.engine = None

    def reset(self) -> None:
        """Reset the game to initial position."""
        self.board.reset()
        self.move_history = []

    def make_move(self, uci: str) -> bool:
        """Execute a move in UCI notation."""
        try:
            move = chess.Move.from_uci(uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            return False
        except ValueError:
            return False

    def get_engine_move(self, depth: int = 15) -> Optional[Dict[str, Any]]:
        """Get engine move with explanation."""
        if self.board.is_game_over():
            return None

        feature_values = baseline_extract_features(self.board, include_probes=False)

        if self.engine:
            try:
                # Use ExplainableChessEngine to get recommended move and explanation
                explanation_obj = self.engine.get_move_recommendation(board=self.board)
                if explanation_obj:
                    move = explanation_obj.move
                    return {
                        "move": move.uci(),
                        "explanation": explanation_obj.overall_explanation,
                        "features": {k: float(v) for k, v in feature_values.items()},
                    }
            except Exception:  # noqa: S110
                pass

        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            move = legal_moves[0]
            explanation = "Engine not available. Suggesting first legal move."
            return {
                "move": move.uci(),
                "explanation": explanation,
                "features": {k: float(v) for k, v in feature_values.items()},
            }

        return None

    def _generate_explanation(
        self, move: chess.Move, features: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation for a move.

        Provides context-aware reasoning by analyzing the move's tactical
        and strategic implications based on position features.
        """
        reasons = []

        if self.board.is_capture(move):
            captured = self.board.piece_at(move.to_square)
            if captured:
                piece_names = {
                    chess.PAWN: "pawn",
                    chess.KNIGHT: "knight",
                    chess.BISHOP: "bishop",
                    chess.ROOK: "rook",
                    chess.QUEEN: "queen",
                }
                reasons.append(
                    f"Captures {piece_names.get(captured.piece_type, 'piece')}"
                )

        if self.board.gives_check(move):
            reasons.append("Delivers check")

        if self.board.is_castling(move):
            reasons.append("Castles for king safety")

        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        if move.to_square in center_squares:
            reasons.append("Controls center")

        piece = self.board.piece_at(move.from_square)
        if (
            piece
            and piece.piece_type in [chess.KNIGHT, chess.BISHOP]
            and self.board.fullmove_number <= 10
        ):
            reasons.append("Develops piece")

        # Check for promotions
        if move.promotion:
            reasons.append("Promotes pawn")

        # Analyze material advantage from features
        if "material_us" in features and "material_them" in features:
            material_diff = features["material_us"] - features["material_them"]
            if material_diff > 3:
                reasons.append("Maintains material advantage")
            elif material_diff < -3:
                reasons.append("Seeks compensation")

        if not reasons:
            reasons.append("Improves position")

        return " · ".join(reasons)


game_state = GameState()


@app.route("/")
def dashboard() -> Any:
    """Render the main dashboard interface."""
    return render_template("dashboard.html")


@app.route("/api/game/new", methods=["POST"])
def new_game() -> Any:
    """Start a new game."""
    game_state.reset()
    return jsonify(
        {
            "fen": game_state.board.fen(),
            "legal_moves": [m.uci() for m in game_state.board.legal_moves],
        }
    )


@app.route("/api/game/state", methods=["GET"])
def get_state() -> Any:
    """Get current game state."""
    return jsonify(
        {
            "fen": game_state.board.fen(),
            "legal_moves": [m.uci() for m in game_state.board.legal_moves],
            "is_game_over": game_state.board.is_game_over(),
            "result": (
                game_state.board.result() if game_state.board.is_game_over() else None
            ),
            "turn": "white" if game_state.board.turn == chess.WHITE else "black",
        }
    )


@app.route("/api/game/move", methods=["POST"])
def make_move() -> Any:
    """Make a move on the board."""
    data = request.get_json()
    if not data or "move" not in data:
        return jsonify({"error": "Move required"}), 400

    # Get SAN/UCI for explanation before pushing
    move_uci = data["move"]
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return jsonify({"error": "Invalid UCI format"}), 400

    if move not in game_state.board.legal_moves:
        return jsonify({"error": "Illegal move"}), 400

    # Generate explanation for the move BEFORE pushing it
    explanation = None
    if game_state.engine:
        try:
            explanation_obj = game_state.engine.explain_move(
                move, board=game_state.board
            )
            explanation = explanation_obj.overall_explanation
        except Exception:  # noqa: S110
            pass

    success = game_state.make_move(move_uci)
    if not success:
        return jsonify({"error": "Invalid move"}), 400

    return jsonify(
        {
            "success": True,
            "fen": game_state.board.fen(),
            "legal_moves": [m.uci() for m in game_state.board.legal_moves],
            "is_game_over": game_state.board.is_game_over(),
            "explanation": explanation,
        }
    )


@app.route("/api/engine/move", methods=["POST"])
def engine_move() -> Any:
    """Get engine move with explanation."""
    data = request.get_json() or {}
    depth = data.get("depth", 15)

    result = game_state.get_engine_move(depth=depth)
    if result is None:
        return jsonify({"error": "Game over or no moves available"}), 400

    return jsonify(result)


@app.route("/api/analysis/features", methods=["POST"])
def analyze_features() -> Any:
    """Analyze features for current position.

    Extracts comprehensive position features including material balance,
    piece mobility, king safety, and positional control metrics.
    """
    feature_values = baseline_extract_features(game_state.board, include_probes=False)

    return jsonify(
        {
            "features": {k: float(v) for k, v in feature_values.items()},
            "fen": game_state.board.fen(),
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check() -> Any:
    """Health check endpoint for monitoring."""
    return jsonify(
        {
            "status": "healthy",
            "engine_available": game_state.engine is not None,
            "version": "1.0.0",
        }
    )


@app.route("/api/engine/status", methods=["GET"])
def engine_status() -> Any:
    """Get surrogate model training status."""
    return jsonify(
        {
            "ready": game_state.model_ready,
            "error": game_state.training_error,
            "engine_available": game_state.engine is not None,
        }
    )


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Start the Flask development server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
