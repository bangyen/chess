"""
Explainable Chess Engine

An interactive chess engine that analyzes your moves and explains what you should have done instead.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import chess
import chess.engine
import chess.pgn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import SFConfig, sf_eval, sf_top_moves
from .engine.hardcoded_reasons import (
    generate_hardcoded_reasons,
    generate_move_reasons_with_board,
)
from .engine.move_explanation import MoveExplanation
from .engine.recommendations import get_heuristic_move
from .engine.syzygy import SyzygyManager
from .features import baseline_extract_features
from .model_trainer import train_surrogate_model
from .surrogate_explainer import SurrogateExplainer
from .utils.sampling import sample_stratified_positions

if TYPE_CHECKING:
    pass


class ExplainableChessEngine:
    """Interactive chess engine that explains moves and provides recommendations."""

    def __init__(
        self,
        stockfish_path: str,
        syzygy_path: str | None = None,
        depth: int = 16,
        opponent_strength: str = "beginner",
        enable_model_explanations: bool = True,
        model_training_positions: int = 200,
    ) -> None:
        """Initialize the explainable chess engine."""
        self.stockfish_path = stockfish_path
        self.syzygy_path = syzygy_path
        self.depth = depth
        self.opponent_strength = opponent_strength
        self.enable_model_explanations = enable_model_explanations
        self.model_training_positions = model_training_positions
        self.engine: chess.engine.SimpleEngine | None = None
        self.syzygy_manager = SyzygyManager(self.syzygy_path)
        self.board = chess.Board()
        self.move_history: list[chess.Move] = []
        self.surrogate_explainer: SurrogateExplainer | None = None
        self.console = Console()

        # Stockfish strength settings
        self.strength_settings: dict[str, dict[str, Any]] = {
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

    def __enter__(self) -> ExplainableChessEngine:
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

            # Configure Stockfish strength and Syzygy
            options: dict[str, Any] = {}
            if self.opponent_strength in self.strength_settings:
                options.update(self.strength_settings[self.opponent_strength])

            if self.syzygy_path:
                options["SyzygyPath"] = self.syzygy_path

            try:
                self.engine.configure(options)
            except chess.engine.EngineError as e:
                print(f"⚠️  Warning: Failed to configure Stockfish options: {e}")

            # Train surrogate model if enabled
            if self.enable_model_explanations:
                self.console.print(
                    "[bold cyan]Training surrogate model for explanations...[/bold cyan]"
                )
                self._initialize_model()

            return self
        except Exception as e:
            raise RuntimeError(
                f"Failed to start Stockfish at {self.stockfish_path}: {e}\n"
                "Please ensure Stockfish is properly installed and accessible."
            ) from e  # type: ignore[unreachable]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool | None:
        """Context manager exit."""
        if self.engine:
            self.engine.quit()
        return None

    def _initialize_model(self) -> None:
        """Train and cache surrogate model for explanations."""
        try:
            # Generate phase-stratified positions so the surrogate
            # model sees a representative mix of opening, middlegame,
            # and endgame states rather than only chaotic random play.
            training_boards = sample_stratified_positions(self.model_training_positions)

            cfg = SFConfig(
                engine_path=self.stockfish_path,
                depth=self.depth,
                multipv=3,
                threads=1,
            )

            model, scaler, feature_names = train_surrogate_model(
                boards=training_boards,
                engine=self.engine,
                cfg=cfg,
                extract_features_fn=baseline_extract_features,
            )

            self.surrogate_explainer = SurrogateExplainer(
                model=model,
                scaler=scaler,
                feature_names=feature_names,
            )
            print("✅ Surrogate model training complete!")
        except Exception as e:
            print(f"⚠️  Warning: Model training failed: {e}")
            print("Falling back to rule-based explanations.")
            self.surrogate_explainer = None

    def reset_game(self) -> None:
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

    def get_best_move(self) -> chess.Move | None:
        """Get the best move from Stockfish."""
        if not self.engine:
            return None

        try:
            # Use a simpler approach to get the best move
            info = self.engine.analyse(self.board, chess.engine.Limit(depth=self.depth))
            if info.get("pv"):
                return info["pv"][0]
            return None
        except Exception as e:
            print(f"❌ Error getting best move: {e}")
            return None

    def _get_syzygy_data(self, board: chess.Board) -> dict:
        """Get Syzygy tablebase data for the position."""
        return self.syzygy_manager.get_syzygy_data(board)

    def analyze_position(self) -> dict:
        """Analyze the current position using our explainable features and Syzygy."""
        try:
            engine = self.engine
            if engine is None:
                return {"error": "Engine not available"}
            # Extract features for current position
            features = baseline_extract_features(self.board)

            # Get real Stockfish evaluation
            cfg = SFConfig(
                engine_path=self.stockfish_path,
                depth=self.depth,
                multipv=3,
            )
            stockfish_score = sf_eval(engine, self.board, cfg)
            top_moves = sf_top_moves(engine, self.board, cfg)

            # Get Syzygy data
            syzygy_data = self._get_syzygy_data(self.board)

            return {
                "stockfish_score": stockfish_score,
                "features": features,
                "top_moves": top_moves,
                "syzygy": syzygy_data,
            }
        except Exception as e:
            msg = f"Error analyzing position: {e}"
            print(f"❌ {msg}")
            return {"error": msg}

    def explain_move(
        self, move: chess.Move, board: chess.Board | None = None
    ) -> MoveExplanation:
        """Explain why a move is good or bad. If board is None, uses self.board."""
        if not self.engine:
            return MoveExplanation(move, 0, [], "Engine not available")

        analyze_board = board if board is not None else self.board

        try:
            cfg = SFConfig(engine_path=self.stockfish_path, depth=self.depth, multipv=1)

            # Get score before move
            score_before = sf_eval(self.engine, analyze_board, cfg)

            # Make move on temp board
            temp_board = analyze_board.copy()
            temp_board.push(move)

            # Get score after move (if it's our turn, negate opponent's score)
            score_after = -sf_eval(self.engine, temp_board, cfg)
            move_score = score_after - score_before

            # Generate explanation
            if board is not None:
                reasons = self._generate_move_reasons_with_board(
                    move, board, score_after, score_before
                )
            else:
                reasons = self._generate_move_reasons(move, score_after, score_before)

            overall_explanation = self._generate_overall_explanation(
                move, move_score, reasons, board=analyze_board
            )

            return MoveExplanation(
                move=move,
                score=move_score,
                reasons=reasons,
                overall_explanation=overall_explanation,
            )

        except Exception as e:
            return MoveExplanation(move, 0, [], f"Error analyzing move: {e}")

    def _get_syzygy_reason(
        self, board_before: chess.Board, board_after: chess.Board
    ) -> tuple[str, float, str] | None:
        """Generate a reason based on Syzygy tablebase changes."""
        return self.syzygy_manager.get_syzygy_reason(board_before, board_after)

    def _generate_move_reasons(
        self, move: chess.Move, score: float, best_score: float
    ) -> list[tuple[str, float, str]]:
        """Generate specific reasons why a move is good or bad using surrogate model."""
        reasons = []

        # 1. Extract features before move
        try:
            feats_before = baseline_extract_features(self.board)
        except Exception:
            feats_before = {}

        # 2. Make move on temp board
        temp_board = self.board.copy()

        # Keep high-salience hardcoded reasons (captures, checks)
        if temp_board.is_capture(move):
            captured_piece = temp_board.piece_at(move.to_square)
            if captured_piece:
                values = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900}
                cp = float(values.get(captured_piece.symbol().upper(), 0))
                reasons.append(
                    (
                        "capture",
                        cp,
                        f"Captures {captured_piece.symbol()} (+{cp:.0f} cp)",
                    )
                )

        temp_board.push(move)

        # Check logic (keep hardcoded)
        if temp_board.is_check():
            reasons.append(("check", 30.0, "Gives check (+30 cp)"))

        # Syzygy Check
        syzygy_reason = self._get_syzygy_reason(self.board, temp_board)
        if syzygy_reason:
            reasons.append(syzygy_reason)

        # 3. Extract features after move
        try:
            feats_after = baseline_extract_features(temp_board)
        except Exception:
            feats_after = {}

        # 4. Use surrogate model if available
        if self.surrogate_explainer is not None:
            try:
                model_reasons = self.surrogate_explainer.calculate_contributions(
                    features_before=feats_before,
                    features_after=feats_after,
                    top_k=5,
                )
                reasons.extend(model_reasons)
            except Exception as e:
                print(f"Warning: Model-based explanation failed: {e}")
                # Fall back to hardcoded
                reasons.extend(generate_hardcoded_reasons(feats_before, feats_after))
        else:
            # Use hardcoded fallback
            reasons.extend(generate_hardcoded_reasons(feats_before, feats_after))

        return reasons

    def _generate_move_reasons_with_board(
        self, move: chess.Move, board: chess.Board, score: float, best_score: float
    ) -> list[tuple[str, float, str]]:
        """Generate specific reasons why a move is good or bad using a specific board state."""
        reasons: list[tuple[str, float, str]] = []

        # Syzygy Check
        temp_board = board.copy()
        temp_board.push(move)
        syzygy_reason = self._get_syzygy_reason(board, temp_board)
        if syzygy_reason:
            reasons.append((syzygy_reason[0], syzygy_reason[1], syzygy_reason[2]))

        reasons.extend(generate_move_reasons_with_board(move, board, self.move_history))

        return reasons

    def _generate_overall_explanation(
        self,
        move: chess.Move,
        move_quality: float,
        reasons: list[tuple[str, float, str]],
        board: chess.Board | None = None,
    ) -> str:
        """Generate an overall explanation for the move with centipawn quality."""
        analyze_board = board if board is not None else self.board
        try:
            move_str = analyze_board.san(move)
        except Exception:
            move_str = str(move)

        # Quality labels based on centipawn evaluation
        if move_quality > 50:
            quality = "Excellent move!"
        elif move_quality > 20:
            quality = "Good move."
        elif move_quality > -20:
            quality = "Reasonable move."
        elif move_quality > -50:
            quality = "Questionable move."
        else:
            quality = "Poor move!"

        if not reasons:
            return f"Move {move_str}: {quality} ({move_quality:+.0f} cp)"

        # Format top 5 reasons as bullet points
        bullet_points = [f"  - {reason[2]}" for reason in reasons[:5]]

        return f"Move {move_str}: {quality} ({move_quality:+.0f} cp)\n" + "\n".join(
            bullet_points
        )

    def get_move_recommendation(
        self, board: chess.Board | None = None
    ) -> MoveExplanation | None:
        """Get the best move recommendation with explanation."""
        analyze_board = board if board is not None else self.board
        try:
            legal_moves = list(analyze_board.legal_moves)
            if not legal_moves:
                return None

            # Try Stockfish first
            if self.engine:
                info = self.engine.analyse(analyze_board, chess.engine.Limit(depth=10))
                if info.get("pv"):
                    best_move = info["pv"][0]
                    if best_move in legal_moves:
                        return self.explain_move(best_move, board=analyze_board)

            # Heuristic fallback
            heuristic_move = self._get_heuristic_move(analyze_board, legal_moves)
            if heuristic_move:
                return self.explain_move(heuristic_move, board=analyze_board)

            # Ultimate fallback: return first legal move
            if legal_moves:
                return self.explain_move(legal_moves[0], board=analyze_board)

            return None
        except Exception as e:
            print(f"❌ Error getting move recommendation: {e}")
            try:
                legal_moves = list(analyze_board.legal_moves)
                heuristic_move = self._get_heuristic_move(analyze_board, legal_moves)
                if heuristic_move:
                    return self.explain_move(heuristic_move, board=analyze_board)
                if legal_moves:
                    return self.explain_move(legal_moves[0], board=analyze_board)
            except Exception:  # noqa: S110
                pass
            return None

    def _get_heuristic_move(
        self, board: chess.Board, legal_moves: list[chess.Move]
    ) -> chess.Move | None:
        """Pick a reasonable move based on simple heuristics."""
        return get_heuristic_move(board, legal_moves)

    def get_best_move_for_player(self) -> MoveExplanation | None:
        """Get the best move that the player who just moved should have played."""
        if not self.move_history:
            return None

        temp_board = self.board.copy()
        temp_board.pop()  # Undo the last move
        best_rec = self.get_move_recommendation(board=temp_board)

        if best_rec and best_rec.move != self.move_history[-1]:
            return best_rec
        return None

    def get_stockfish_move(self) -> chess.Move | None:
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

    def print_board(self) -> None:
        """Print the current board position using rich."""
        # Create a stylized board string
        board_str = str(self.board)
        # Add some color to pieces
        styled_board = Text()
        for char in board_str:
            if char.isupper():  # White pieces
                styled_board.append(char, style="bold white")
            elif char.islower():  # Black pieces
                styled_board.append(char, style="bold cyan")
            else:
                styled_board.append(char, style="dim white")

        self.console.print("\n")
        self.console.print(
            Panel(
                styled_board,
                title="Current Position",
                expand=False,
                border_style="blue",
            )
        )

    def print_legal_moves(self) -> None:
        """Print all legal moves."""
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]
        print(
            f"Legal moves: {', '.join(legal_moves[:10])}{'...' if len(legal_moves) > 10 else ''}"
        )

    def play_interactive_game(self) -> None:  # noqa: C901
        """Play an interactive chess game against Stockfish with explanations."""
        self.console.print(
            Panel(
                "[bold green]Welcome to the Explainable Chess Engine![/bold green]\n"
                "Play against Stockfish and learn from your moves."
            )
        )

        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:
                # Only print board when it's the human player's turn
                self.print_board()
                user_input = self.console.input(
                    "\n[bold white]White to move (or 'help', 'best', 'reset', 'quit'): [/bold white]"
                ).strip()

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
                # Parse the move first to get the SAN notation before pushing
                try:
                    move = self.board.parse_san(user_input)
                except ValueError:
                    try:
                        move = self.board.parse_uci(user_input)
                    except ValueError:
                        move = None

                if move and move in self.board.legal_moves:
                    move_san = self.board.san(move)
                    self.make_move(user_input)  # This pushes the move

                    # Analyze the human's move and provide explanation
                    last_move = self.move_history[-1]
                    temp_board = self.board.copy()
                    temp_board.pop()  # Undo the last move to analyze it
                    explanation = self.explain_move(last_move, board=temp_board)

                    self.console.print(
                        Panel(
                            f"[bold white]Your move {move_san}:[/bold white]\n{explanation.overall_explanation}",
                            title="Player Analysis",
                            border_style="white",
                        )
                    )

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

                            self.console.print(
                                Panel(
                                    f"[bold yellow]Better would have been {move_str}:[/bold yellow]\n{best_recommendation.overall_explanation}",
                                    title="Recommendation",
                                    border_style="yellow",
                                )
                            )
                else:
                    self.console.print(
                        "[bold red]❌ Invalid move. Try again.[/bold red]"
                    )
                    continue
            else:
                # Stockfish's turn
                with self.console.status(
                    "[bold cyan]🤖 Stockfish is thinking...[/bold cyan]"
                ):
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

                    self.console.print(
                        f"\n[bold cyan]🤖 Stockfish plays: {move_str}[/bold cyan]"
                    )
                else:
                    self.console.print(
                        "[bold red]❌ Stockfish failed to make a move. Game over.[/bold red]"
                    )
                    break

            # Check for game over
            if self.board.is_game_over():
                result = self.board.result()
                status_text = Text.assemble(
                    ("\n🏁 Game Over! Result: ", "bold"), (result, "bold green")
                )
                self.console.print(Panel(status_text, border_style="green"))

                if result == "1-0":
                    self.console.print("[bold green]🎉 You win![/bold green]")
                elif result == "0-1":
                    self.console.print("[bold cyan]🎉 Stockfish wins![/bold cyan]")
                else:
                    self.console.print("[bold white]🤝 Draw![/bold white]")
                break

    def _print_help(self) -> None:
        """Print help information using rich."""
        help_table = Table(title="📖 Available Commands", box=None, show_header=False)
        help_table.add_column("Command", style="bold magenta")
        help_table.add_column("Description")

        help_table.add_row("e4, Nf3, O-O", "Make a move using SAN notation")
        help_table.add_row("best", "Show the best move recommendation with explanation")
        help_table.add_row("reset", "Reset the game to starting position")
        help_table.add_row("help", "Show this help screen")
        help_table.add_row("quit", "Exit the game")

        self.console.print(help_table)

    def _show_best_move(self) -> None:
        """Show the best move recommendation using rich."""
        recommendation = self.get_move_recommendation()
        if recommendation:
            try:
                move_str = self.board.san(recommendation.move)
            except Exception:
                move_str = str(recommendation.move)

            reasons_text = Text()
            reasons_text.append(
                f"{recommendation.overall_explanation}\n\n", style="italic"
            )
            if recommendation.reasons:
                reasons_text.append("Why this move:\n", style="bold underline")
                for reason in recommendation.reasons[:3]:
                    reasons_text.append(f" • {reason[2]}\n", style="green")

            self.console.print(
                Panel(
                    reasons_text,
                    title=f"💡 Best Move: [bold cyan]{move_str}[/bold cyan]",
                    border_style="cyan",
                )
            )
        else:
            self.console.print(
                "[bold red]❌ Could not get move recommendation[/bold red]"
            )


if __name__ == "__main__":
    from src.chess_ai.cli.explainable import main

    main()
