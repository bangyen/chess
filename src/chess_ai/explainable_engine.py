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
        stockfish_path: str = "/opt/homebrew/bin/stockfish",
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
        """Generate specific reasons why a move is good or bad."""
        reasons = []

        # Create a temporary board to analyze the move BEFORE it's played
        temp_board = self.board.copy()

        try:
            # Analyze the move's characteristics
            # Get move string for analysis
            try:
                temp_board.san(move)
            except Exception:
                pass

            # Check if it's a capture
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

            # Check if it gives check
            if temp_board.gives_check(move):
                reasons.append(("check", 2, "Gives check to opponent's king"))

            # Check if it's a tactical move
            if temp_board.is_capture(move) or temp_board.gives_check(move):
                reasons.append(("tactical", 1, "Tactical move (capture or check)"))

            # Check piece development
            piece = temp_board.piece_at(move.from_square)
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
            if temp_board.is_castling(move):
                reasons.append(("castling", 3, "Castles to improve king safety"))

            # Check for en passant
            if temp_board.is_en_passant(move):
                reasons.append(("en_passant", 1, "En passant capture"))

        except Exception:
            # If there's an error, just return basic reasons
            pass

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

        if move_quality > 50:
            quality = "excellent"
        elif move_quality > 20:
            quality = "good"
        elif move_quality > -20:
            quality = "reasonable"
        elif move_quality > -50:
            quality = "questionable"
        else:
            quality = "poor"

        explanation = f"Move {move_str} is {quality}."

        if reasons:
            positive_reasons = [r for r in reasons if r[1] > 0]
            negative_reasons = [r for r in reasons if r[1] < 0]

            if positive_reasons:
                explanation += f" Positive aspects: {', '.join([r[2] for r in positive_reasons[:2]])}."

            if negative_reasons:
                explanation += (
                    f" Concerns: {', '.join([r[2] for r in negative_reasons[:2]])}."
                )

        return explanation

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

        if move_quality > 50:
            quality = "excellent"
        elif move_quality > 20:
            quality = "good"
        elif move_quality > -20:
            quality = "reasonable"
        elif move_quality > -50:
            quality = "questionable"
        else:
            quality = "poor"

        explanation = f"Move {move_str} is {quality}."

        if reasons:
            positive_reasons = [r for r in reasons if r[1] > 0]
            negative_reasons = [r for r in reasons if r[1] < 0]

            if positive_reasons:
                explanation += f" Positive aspects: {', '.join([r[2] for r in positive_reasons[:2]])}."

            if negative_reasons:
                explanation += (
                    f" Concerns: {', '.join([r[2] for r in negative_reasons[:2]])}."
                )

        return explanation

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
        print(f"Turn: {'White' if self.board.turn else 'Black'}")
        print(f"FEN: {self.board.fen()}")
        print(f"Moves played: {len(self.move_history)}")
        print("=" * 50)

    def print_legal_moves(self):
        """Print all legal moves."""
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        print(
            f"Legal moves: {', '.join(legal_moves[:10])}{'...' if len(legal_moves) > 10 else ''}"
        )

    def play_interactive_game(self):
        """Play an interactive chess game against Stockfish with explanations."""
        print("🎯 Welcome to the Explainable Chess Engine!")
        print(f"Playing against Stockfish ({self.opponent_strength} level)")
        print(
            "You are White. Make moves in standard algebraic notation (e.g., 'e4', 'Nf3', 'O-O')"
        )
        print("Type 'help' for commands, 'quit' to exit")

        while not self.board.is_game_over():
            self.print_board()

            if self.board.turn == chess.WHITE:
                # Human player's turn
                self.print_legal_moves()
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
                    print(f"✅ Move {user_input} played")

                    # Analyze the human's move and provide explanation
                    last_move = self.move_history[-1]
                    temp_board = self.board.copy()
                    temp_board.pop()  # Undo the last move to analyze it
                    explanation = self.explain_move_with_board(last_move, temp_board)

                    print("\n📊 Your Move Analysis:")
                    print(f"   {explanation.overall_explanation}")

                    # Show what the best move would have been
                    if not self.board.is_game_over():
                        best_recommendation = self.get_best_move_for_player()
                        if best_recommendation:
                            temp_board = self.board.copy()
                            temp_board.pop()  # Undo the last move
                            try:
                                move_str = temp_board.san(best_recommendation.move)
                            except Exception:
                                move_str = str(best_recommendation.move)

                            print(f"\n💡 Best move would be: {move_str}")
                            print(f"   {best_recommendation.overall_explanation}")
                else:
                    print("❌ Invalid move. Try again.")
                    continue
            else:
                # Stockfish's turn
                print(f"\n🤖 Stockfish ({self.opponent_strength}) is thinking...")
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

                    print(f"🤖 Stockfish plays: {move_str}")

                    # Analyze Stockfish's move
                    temp_board = self.board.copy()
                    temp_board.pop()  # Undo the last move to analyze it
                    explanation = self.explain_move_with_board(
                        stockfish_move, temp_board
                    )

                    print("\n📊 Stockfish's Move Analysis:")
                    print(f"   {explanation.overall_explanation}")
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


def main():
    """Main function to run the explainable chess engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Explainable Chess Engine")
    parser.add_argument(
        "--engine",
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish engine",
    )
    parser.add_argument("--depth", type=int, default=16, help="Search depth")

    args = parser.parse_args()

    try:
        with ExplainableChessEngine(args.engine, args.depth) as engine:
            engine.play_interactive_game()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
