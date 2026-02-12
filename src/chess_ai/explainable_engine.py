"""
Explainable Chess Engine

An interactive chess engine that analyzes your moves and explains what you should have done instead.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn

from .features import baseline_extract_features


@dataclass
class MoveExplanation:
    """Explanation for why a move is good or bad."""

    move: chess.Move
    score: float
    reasons: List[Tuple[str, int, str]]  # (feature_name, contribution, explanation)
    overall_explanation: str


class ExplainableChessEngine:
    """Interactive chess engine that explains moves and provides recommendations."""

    def __init__(
        self,
        stockfish_path: str,
        depth: int = 16,
        opponent_strength: str = "beginner",
    ):
        """Initialize the explainable chess engine."""
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.opponent_strength = opponent_strength
        self.engine = None
        self.board = chess.Board()
        self.move_history: List[chess.Move] = []

        # Stockfish strength settings
        self.strength_settings = {
            "beginner": {"Skill Level": 0, "UCI_LimitStrength": True, "UCI_Elo": 800},
            "novice": {"Skill Level": 3, "UCI_LimitStrength": True, "UCI_Elo": 1000},
            "intermediate": {
                "Skill Level": 8,
                "UCI_LimitStrength": True,
                "UCI_Elo": 1400,
            },
            "advanced": {"Skill Level": 15, "UCI_LimitStrength": True, "UCI_Elo": 1800},
            "expert": {"Skill Level": 20, "UCI_LimitStrength": False},  # Full strength
        }

    def __enter__(self):
        """Context manager entry."""
        if not self.stockfish_path or not self.stockfish_path.strip():
            raise RuntimeError(
                "Stockfish not found! Please install Stockfish:\n"
                "  • Ubuntu/Debian: sudo apt install stockfish\n"
                "  • macOS: brew install stockfish\n"
                "  • Windows: Download from https://stockfishchess.org/\n"
                "  • Google Colab: !apt install stockfish\n"
                "  • Or add Stockfish to your PATH"
            )

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

            # Configure Stockfish strength
            if self.opponent_strength in self.strength_settings:
                settings = self.strength_settings[self.opponent_strength]
                for option, value in settings.items():
                    try:
                        self.engine.configure({option: value})
                    except chess.engine.EngineError:
                        pass  # Some options might not be available in all Stockfish versions

            return self
        except Exception as e:
            raise RuntimeError(
                f"Failed to start Stockfish at {self.stockfish_path}: {e}\n"
                "Please ensure Stockfish is properly installed and accessible."
            ) from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.engine:
            self.engine.quit()

    def reset_game(self):
        """Reset to starting position."""
        self.board = chess.Board()
        self.move_history = []
        print("🔄 Game reset to starting position")

    def make_move(self, move_str: str) -> bool:
        """Make a move and return True if successful."""
        try:
            move = self.board.parse_san(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move)
                return True
            else:
                print(f"❌ Illegal move: {move_str}")
                return False
        except ValueError:
            print(f"❌ Invalid move format: {move_str}")
            return False

    def get_best_move(self) -> Optional[chess.Move]:
        """Get the best move from Stockfish."""
        if not self.engine:
            return None

        try:
            # Use a simpler approach to get the best move
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=self.depth))
            if "pv" in info and info["pv"]:
                return info["pv"][0]
            return None
        except Exception as e:
            print(f"❌ Error getting best move: {e}")
            return None

    def analyze_position(self) -> Dict:
        """Analyze the current position using our explainable features."""
        try:
            # Extract features for current position
            features = baseline_extract_features(self.board)

            # Get top 3 candidate moves (simplified)
            top_moves = []
            legal_moves = list(self.board.legal_moves)
            for _i, move in enumerate(legal_moves[:3]):
                top_moves.append((move, 0))  # Simplified scoring

            return {
                "stockfish_score": 0,  # Simplified for now
                "features": features,
                "top_moves": top_moves,
            }
        except Exception as e:
            print(f"❌ Error analyzing position: {e}")
            return {}

    def explain_move(self, move: chess.Move) -> MoveExplanation:
        """Explain why a move is good or bad."""
        if not self.engine:
            return MoveExplanation(move, 0, [], "Engine not available")

        try:
            # Generate explanation based on move characteristics (without engine calls for now)
            reasons = self._generate_move_reasons(move, 0, 0)
            overall_explanation = self._generate_overall_explanation(move, 0, reasons)

            return MoveExplanation(
                move=move,
                score=0,
                reasons=reasons,
                overall_explanation=overall_explanation,
            )

        except Exception as e:
            return MoveExplanation(move, 0, [], f"Error analyzing move: {e}")

    def explain_move_with_board(
        self, move: chess.Move, board: chess.Board
    ) -> MoveExplanation:
        """Explain why a move is good or bad using a specific board state."""
        if not self.engine:
            return MoveExplanation(move, 0, [], "Engine not available")

        try:
            # Generate explanation based on move characteristics using the provided board
            reasons = self._generate_move_reasons_with_board(move, board, 0, 0)
            overall_explanation = self._generate_overall_explanation_with_board(
                move, board, 0, reasons
            )

            return MoveExplanation(
                move=move,
                score=0,
                reasons=reasons,
                overall_explanation=overall_explanation,
            )

        except Exception as e:
            return MoveExplanation(move, 0, [], f"Error analyzing move: {e}")

    def _generate_move_reasons(
        self, move: chess.Move, score: float, best_score: float
    ) -> List[Tuple[str, int, str]]:
        """Generate specific reasons why a move is good or bad using feature deltas."""
        reasons = []

        # 1. Base features (before move)
        try:
            feats_before = baseline_extract_features(self.board)
        except Exception:
            feats_before = {}

        # 2. Make move
        temp_board = self.board.copy()

        # Capture logic (keep hardcoded as it's salient)
        if temp_board.is_capture(move):
            captured_piece = temp_board.piece_at(move.to_square)
            if captured_piece:
                piece_value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}.get(
                    captured_piece.symbol().upper(), 0
                )
                reasons.append(
                    (
                        "capture",
                        piece_value,
                        f"Captures {captured_piece.symbol()} (worth {piece_value} points)",
                    )
                )

        temp_board.push(move)

        # Check logic (keep hardcoded)
        if temp_board.is_check():
            reasons.append(("check", 2, "Gives check to opponent's king"))

        # 3. After features (after move)
        # Note: Perspective flips! "us" becomes "them"
        try:
            feats_after = baseline_extract_features(temp_board)
        except Exception:
            feats_after = {}

        # 4. Calculate Deltas
        # We compare before['feature_us'] with after['feature_them'] (because side flipped)

        def get_delta(feature_name):
            val_before = feats_before.get(f"{feature_name}_us", 0.0)
            val_after = feats_after.get(f"{feature_name}_them", 0.0)
            return val_after - val_before

        def get_opp_delta(feature_name):
            # Did we reduce their feature?
            val_before = feats_before.get(f"{feature_name}_them", 0.0)
            val_after = feats_after.get(f"{feature_name}_us", 0.0)
            return val_after - val_before

        # Significant changes thresholds

        # Batteries
        delta = get_delta("batteries")
        if delta > 0.5:
            reasons.append(("batteries_us", 1, "Forms a battery arrangement"))

        # Outposts
        delta = get_delta("outposts")
        if delta > 0.5:
            reasons.append(("outposts_us", 2, "Establishes a knight outpost"))

        # King Ring Pressure (weighted)
        # Note: 'king_ring_pressure_them' (pressure ON them) vs after 'king_ring_pressure_us' (pressure ON them from our perspective? No.)
        # baseline.py: king_ring_pressure_us is pressure EXERTED BY US onto THEM.
        # So before: pressure_us (by White on Black)
        # After (Black to move): pressure_them (by White on Black)
        delta = get_delta("king_ring_pressure")
        if delta > 0.5:
            reasons.append(("king_pressure", 2, "Increases pressure on enemy king"))

        # Bishop Pair
        delta = get_delta("bishop_pair")
        if delta > 0.5:
            reasons.append(("bishop_pair", 1, "Secures the bishop pair"))

        # Passed Pawns
        delta = get_delta("passed")
        if delta > 0.5:
            reasons.append(("passed_pawns", 2, "Creates a passed pawn"))

        # Isolated Pawns (we want to force them to have isolated pawns)
        delta_opp = get_opp_delta("isolated_pawns")
        if delta_opp > 0.5:
            reasons.append(
                ("structure_damage", 1, "Creates an isolated pawn for opponent")
            )

        # Center Control
        delta = get_delta("center_control")
        if delta > 0.5:
            reasons.append(("center_control", 1, "Improves central control"))

        return reasons

    def _generate_move_reasons_with_board(
        self, move: chess.Move, board: chess.Board, score: float, best_score: float
    ) -> List[Tuple[str, int, str]]:
        """Generate specific reasons why a move is good or bad using a specific board state."""
        reasons = []

        try:
            # Analyze the move's characteristics
            # Get move string for analysis
            try:
                board.san(move)
            except Exception:
                pass

            # Check if it's a capture
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}.get(
                        captured_piece.symbol().upper(), 0
                    )
                    reasons.append(
                        (
                            "capture",
                            piece_value,
                            f"Captures {captured_piece.symbol()} (worth {piece_value} points)",
                        )
                    )

            # Check if it gives check
            if board.gives_check(move):
                reasons.append(("check", 2, "Gives check to opponent's king"))

            # Check if it's a tactical move
            if board.is_capture(move) or board.gives_check(move):
                reasons.append(("tactical", 1, "Tactical move (capture or check)"))

            # Check piece development
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                # Pawn moves
                if (
                    move.from_square < 16 or move.from_square > 47
                ):  # From starting ranks
                    reasons.append(
                        ("development", 1, "Develops pawn from starting position")
                    )
            elif piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Minor piece development
                if (
                    move.from_square < 16 or move.from_square > 47
                ):  # From starting ranks
                    reasons.append(
                        (
                            "development",
                            2,
                            "Develops minor piece from starting position",
                        )
                    )

            # Check center control
            center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
            if move.to_square in center_squares:
                reasons.append(("center_control", 1, "Controls central squares"))

            # Check king safety
            if piece and piece.piece_type == chess.KING:
                # King moves in opening/middlegame
                if len(self.move_history) < 20:
                    reasons.append(
                        (
                            "king_safety",
                            -1,
                            "Moves king in opening (reduces castling options)",
                        )
                    )

            # Check for castling
            if board.is_castling(move):
                reasons.append(("castling", 3, "Castles to improve king safety"))

            # Check for en passant
            if board.is_en_passant(move):
                reasons.append(("en_passant", 1, "En passant capture"))

        except Exception:
            # If there's an error, just return basic reasons
            pass

        return reasons

    def _generate_overall_explanation(
        self,
        move: chess.Move,
        move_quality: float,
        reasons: List[Tuple[str, int, str]],
    ) -> str:
        """Generate an overall explanation for the move."""
        try:
            move_str = self.board.san(move)
        except Exception:
            move_str = str(move)

        if not reasons:
            return f"Move {move_str}:"

        # Convert to bullet points with tab indentation
        bullet_points = []
        for reason in reasons[:3]:  # Limit to top 3 reasons
            bullet_points.append(f"  - {reason[2]}")

        return f"Move {move_str}:\n" + "\n".join(bullet_points)

    def _generate_overall_explanation_with_board(
        self,
        move: chess.Move,
        board: chess.Board,
        move_quality: float,
        reasons: List[Tuple[str, int, str]],
    ) -> str:
        """Generate an overall explanation for the move using a specific board state."""
        try:
            move_str = board.san(move)
        except Exception:
            move_str = str(move)

        if not reasons:
            return f"Move {move_str}:"

        # Convert to bullet points with tab indentation
        bullet_points = []
        for reason in reasons[:3]:  # Limit to top 3 reasons
            bullet_points.append(f"- {reason[2]}")

        return f"Move {move_str}:\n" + "\n".join(bullet_points)

    def get_move_recommendation(self) -> Optional[MoveExplanation]:
        """Get the best move recommendation with explanation."""
        try:
            # Get legal moves for the current position
            legal_moves = list(self.board.legal_moves)
            if not legal_moves:
                return None

            # Try to get Stockfish recommendation first
            if self.engine:
                try:
                    # Get Stockfish's best move
                    info = self.engine.analyse(self.board, chess.engine.Limit(depth=10))
                    if "pv" in info and info["pv"]:
                        best_move = info["pv"][0]
                        if best_move in legal_moves:
                            return self.explain_move(best_move)
                except Exception:
                    # If Stockfish fails, fall back to simple logic
                    pass

            # Fallback: suggest good opening moves
            if len(self.move_history) == 0:
                # First move - suggest e4 or d4
                for move in legal_moves:
                    if move.from_square == chess.E2 and move.to_square == chess.E4:
                        return self.explain_move(move)
                    elif move.from_square == chess.D2 and move.to_square == chess.D4:
                        return self.explain_move(move)
            elif len(self.move_history) == 1:
                # Second move - suggest Nf3 or Nc3
                for move in legal_moves:
                    if move.from_square == chess.G1 and move.to_square == chess.F3:
                        return self.explain_move(move)
                    elif move.from_square == chess.B1 and move.to_square == chess.C3:
                        return self.explain_move(move)

            # For other moves, pick a reasonable move (not just the first one)
            # Prioritize center control and development
            center_moves = []
            development_moves = []

            for move in legal_moves:
                # Check for center control
                if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                    center_moves.append(move)
                # Check for piece development
                piece = self.board.piece_at(move.from_square)
                if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if (
                        move.from_square < 16 or move.from_square > 47
                    ):  # From starting ranks
                        development_moves.append(move)

            # Prefer center moves, then development moves
            if center_moves:
                return self.explain_move(center_moves[0])
            elif development_moves:
                return self.explain_move(development_moves[0])
            else:
                return self.explain_move(legal_moves[0])

        except Exception as e:
            print(f"❌ Error getting move recommendation: {e}")
            return None

    def get_best_move_for_player(self) -> Optional[MoveExplanation]:
        """Get the best move that the player who just moved should have played."""
        try:
            # Create a temporary board to analyze what the player should have done
            temp_board = self.board.copy()
            temp_board.pop()  # Undo the last move

            # Get legal moves for the player who just moved
            legal_moves = list(temp_board.legal_moves)
            if not legal_moves:
                return None

            # Try to get Stockfish recommendation first
            if self.engine:
                try:
                    # Get Stockfish's best move for the previous position
                    info = self.engine.analyse(temp_board, chess.engine.Limit(depth=10))
                    if "pv" in info and info["pv"]:
                        best_move = info["pv"][0]
                        if best_move in legal_moves:
                            # Check if this is the same move that was just played
                            last_move = self.move_history[-1]
                            if best_move != last_move:
                                return self.explain_move_with_board(
                                    best_move, temp_board
                                )
                except Exception:
                    # If Stockfish fails, fall back to simple logic
                    pass

            # Fallback: suggest good opening moves
            if len(self.move_history) == 1:  # After first move
                # First move - suggest e4 or d4, but not the move that was just played
                last_move = self.move_history[-1]
                for move in legal_moves:
                    if (
                        move.from_square == chess.E2
                        and move.to_square == chess.E4
                        and move != last_move
                    ):
                        return self.explain_move_with_board(move, temp_board)
                    elif (
                        move.from_square == chess.D2
                        and move.to_square == chess.D4
                        and move != last_move
                    ):
                        return self.explain_move_with_board(move, temp_board)
            elif len(self.move_history) == 2:  # After second move
                # Second move - suggest Nf3 or Nc3
                for move in legal_moves:
                    if move.from_square == chess.G1 and move.to_square == chess.F3:
                        return self.explain_move_with_board(move, temp_board)
                    elif move.from_square == chess.B1 and move.to_square == chess.C3:
                        return self.explain_move_with_board(move, temp_board)

            # For other moves, pick a reasonable move (not just the first one)
            # Prioritize center control and development
            center_moves = []
            development_moves = []

            for move in legal_moves:
                # Check for center control
                if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                    center_moves.append(move)
                # Check for piece development
                piece = temp_board.piece_at(move.from_square)
                if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if (
                        move.from_square < 16 or move.from_square > 47
                    ):  # From starting ranks
                        development_moves.append(move)

            # Prefer center moves, then development moves
            if center_moves:
                return self.explain_move_with_board(center_moves[0], temp_board)
            elif development_moves:
                return self.explain_move_with_board(development_moves[0], temp_board)
            else:
                return self.explain_move_with_board(legal_moves[0], temp_board)

        except Exception as e:
            print(f"❌ Error getting best move for player: {e}")
            return None

    def get_stockfish_move(self) -> Optional[chess.Move]:
        """Get Stockfish's move for the current position."""
        if not self.engine:
            return None

        try:
            # Use a shorter time limit for opponent moves to keep the game moving
            opponent_depth = min(self.depth, 8)  # Cap at depth 8 for opponent
            result = self.engine.play(
                self.board, chess.engine.Limit(depth=opponent_depth)
            )
            return result.move
        except Exception:
            return None

    def print_board(self):
        """Print the current board position."""
        print("\n" + "=" * 50)
        print(self.board)
        print("=" * 50)

    def print_legal_moves(self):
        """Print all legal moves."""
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        print(
            f"Legal moves: {', '.join(legal_moves[:10])}{'...' if len(legal_moves) > 10 else ''}"
        )

    def play_interactive_game(self):
        """Play an interactive chess game against Stockfish with explanations."""

        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                # Only print board when it's the human player's turn
                self.print_board()
                user_input = input("\nWhite to move: ").strip()

                if user_input.lower() == "quit":
                    break
                elif user_input.lower() == "help":
                    self._print_help()
                    continue
                elif user_input.lower() == "reset":
                    self.reset_game()
                    continue
                elif user_input.lower() == "best":
                    self._show_best_move()
                    continue

                # Try to make the move
                if self.make_move(user_input):
                    # Analyze the human's move and provide explanation
                    last_move = self.move_history[-1]
                    temp_board = self.board.copy()
                    temp_board.pop()  # Undo the last move to analyze it
                    explanation = self.explain_move_with_board(last_move, temp_board)

                    print(f"\nYour {explanation.overall_explanation}")

                    # Show what the best move would have been ONLY if it differs from what was played
                    if not self.board.is_game_over():
                        best_recommendation = self.get_best_move_for_player()
                        if (
                            best_recommendation
                            and best_recommendation.move != last_move
                        ):
                            temp_board = self.board.copy()
                            temp_board.pop()  # Undo the last move
                            try:
                                move_str = temp_board.san(best_recommendation.move)
                            except Exception:
                                move_str = str(best_recommendation.move)

                            print(f"\nBest {best_recommendation.overall_explanation}")
                else:
                    print("❌ Invalid move. Try again.")
                    continue
            else:
                # Stockfish's turn
                stockfish_move = self.get_stockfish_move()

                if stockfish_move:
                    # Get SAN notation before making the move
                    try:
                        move_str = self.board.san(stockfish_move)
                    except Exception:
                        move_str = str(stockfish_move)

                    # Make Stockfish's move
                    self.board.push(stockfish_move)
                    self.move_history.append(stockfish_move)

                    print(f"\n🤖 Stockfish plays: {move_str}")
                else:
                    print("❌ Stockfish failed to make a move. Game over.")
                    break

            # Check for game over
            if self.board.is_game_over():
                result = self.board.result()
                print(f"\n🏁 Game Over! Result: {result}")

                if result == "1-0":
                    print("🎉 You win!")
                elif result == "0-1":
                    print("🎉 Stockfish wins!")
                else:
                    print("🤝 Draw!")
                break

    def _print_help(self):
        """Print help information."""
        print("\n📖 Available commands:")
        print("  • Make moves: e4, Nf3, O-O, etc.")
        print("  • 'best' - Show the best move recommendation")
        print("  • 'reset' - Reset the game")
        print("  • 'help' - Show this help")
        print("  • 'quit' - Exit the game")

    def _show_best_move(self):
        """Show the best move recommendation."""
        recommendation = self.get_move_recommendation()
        if recommendation:
            try:
                move_str = self.board.san(recommendation.move)
            except Exception:
                move_str = str(recommendation.move)
            print(f"\n💡 Best move: {move_str}")
            print(f"   {recommendation.overall_explanation}")

            if recommendation.reasons:
                print("   Why this move:")
                for reason in recommendation.reasons[:3]:
                    print(f"   • {reason[2]}")
        else:
            print("❌ Could not get move recommendation")


if __name__ == "__main__":
    from src.chess_ai.cli.explainable import main

    main()
