import chess

from src.chess_ai.explainable_engine import ExplainableChessEngine


class TestExplanationContent:
    def test_battery_explanation(self):
        engine = ExplainableChessEngine(stockfish_path="mock")
        # White Rook on a1. White Queen on b1. Move Queen to a2 -> Battery on a-file.
        engine.board = chess.Board("8/8/8/8/8/8/1Q6/R3K3 w - - 0 1")
        move = chess.Move.from_uci("b2a2")

        reasons = engine._generate_move_reasons(move, 0.0, 0.0)

        # Check if battery reason is present
        # Note: Exact string match might be fragile, checking substring
        battery_reason = any("battery" in r[2].lower() for r in reasons)
        assert battery_reason, f"Expected battery explanation, got: {reasons}"

    def test_outpost_explanation(self):
        engine = ExplainableChessEngine(stockfish_path="mock")
        # White Knight on f3. White Pawn on e3 (supporting d4). Move Knight to d4.
        # d4 is rank 4 (0-indexed 3). e3 is rank 3 (0-indexed 2). e3 supports d4.
        engine.board = chess.Board("8/8/8/8/8/4P3/5N2/4K3 w - - 0 1")
        move = chess.Move.from_uci("f2d3")  # Wait, f2 to d3? No.
        # Let's be precise.
        # Pawn on e3. Knight on f3.
        # Move f3 -> d4.
        # Board: "8/8/8/8/8/4P3/5N2/4K3 w - - 0 1" has Knight on f2?
        # Let's reset.

        # Setup: White Pawn on e3. White Knight on f3. Black King on h8. White King on e1.
        engine.board = chess.Board("7k/8/8/8/8/4P1N1/8/4K3 w - - 0 1")
        # Move Knight f3 (actually g3 in FEN above? no N is Knight)
        # FEN: 4P1N1 -> Pawn e3, Knight g3.
        # Move g3 -> f5? (Rank 5). e4 support?
        # Let's do simple: Pawn c3, Knight b1 -> d4? No.
        # Pawn c3. Knight jumps to d4? From others.

        # Let's use: White Pawn c3. White Knight b1.
        # Move Nb1-d2-b3-d4 is too long.
        # Place Knight on f3. Pawn on e3. Move Nf3-d4. e3 supports d4.
        engine.board = chess.Board("7k/8/8/8/8/4PN2/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("f3d4")

        reasons = engine._generate_move_reasons(move, 0.0, 0.0)

        outpost_reason = any("outpost" in r[2].lower() for r in reasons)
        assert outpost_reason, f"Expected outpost explanation, got: {reasons}"

    def test_check_explanation(self):
        engine = ExplainableChessEngine(stockfish_path="mock")
        # Direct check. White Queen h1. White King e1. Black King e8.
        engine.board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")
        move = chess.Move.from_uci("h1e4")  # Check

        reasons = engine._generate_move_reasons(move, 0.0, 0.0)

        check_reason = any("check" in r[2].lower() for r in reasons)
        assert check_reason, f"Expected check explanation, got: {reasons}"
