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

    def test_safe_mobility_explanation(self):
        engine = ExplainableChessEngine(stockfish_path="mock")
        # Position: White Queen blocked by own pawns.
        # 8/8/8/3PPP2/3PQP2/3PPP2/8/8 w - - 0 1 (Queen boxed in on e4)
        # Move: Pawn d5 removes blockage? No, hard to make big delta.
        # Easier: White Queen on a1. Many enemy pawns attacking b2, c2 etc?
        # Let's create a move that massively increases safe squares.
        # Queen trapped in corner (a1) by friendly bishop b2?
        # Move Bishop away -> Queen has diagonal.
        engine.board = chess.Board("8/8/8/8/8/8/1B6/Q7 w - - 0 1")
        # Queen moves on rank: 7. Moves on file: 7. Diag: 0. Total 14.
        # Move Bishop b2 to c3.
        # Queen now has diagonal a1-h8 (7 moves).
        # Safe mobility increases by 7.
        move = chess.Move.from_uci("b2c3")

        reasons = engine._generate_move_reasons(move, 0.0, 0.0)

        reason = any("safe piece activity" in r[2].lower() for r in reasons)
        assert reason, f"Expected safe mobility explanation, got: {reasons}"

    def test_rook_open_file_explanation(self):
        engine = ExplainableChessEngine(stockfish_path="mock")
        # White Rook on b1. File b is closed by P on b2.
        # Move Rook to a1 (File a is open).
        # Delta: +1 open file rook.
        engine.board = chess.Board("8/8/8/8/8/8/1P6/R7 w - - 0 1")  # R on a1 is open.
        # Wait, setup: Rook on b1 (closed).
        engine.board = chess.Board("8/8/8/8/8/8/1P6/1R6 w - - 0 1")
        # Move Rb1-a1.
        move = chess.Move.from_uci("b1a1")

        reasons = engine._generate_move_reasons(move, 0.0, 0.0)

        reason = any("open or semi-open file" in r[2].lower() for r in reasons)
        assert reason, f"Expected rook open file explanation, got: {reasons}"

    def test_backward_pawn_creation_explanation(self):
        engine = ExplainableChessEngine(stockfish_path="mock")
        # White to move. We want to force BLACK to have a backward pawn.
        # Setup: Black pawn on d6. Black pawn on c7, e7. (Supported).
        # If White captures c7 or e7?
        # Or if White controls d5?

        # Let's try: Black pawn on d6. No support. Stop d5.
        # But we need a MOVE that changes this.
        # Start: Black pawn d6. Black pawn c5 (supports d6? No, ahead).
        # Black pawn e7 (supports d6).
        # White captures e7 (say with Rook).
        # Then d6 becomes backward?

        # Setup: Black King h8. Black Pawns d6, e7. White Rook e1.
        engine.board = chess.Board("7k/4p3/3p4/8/8/8/8/4R2K w - - 0 1")
        # Move Rook takes e7.
        # Before: d6 supported by e7. Backward count = 0.
        # After: d6 alone. (Stop d5 not controlled by enemy? Wait backward needs stop control).
        # Let's put White Pawn on c4 (controls d5).
        engine.board = chess.Board("7k/4p3/3p4/8/2P5/8/8/4R2K w - - 0 1")
        move = chess.Move.from_uci("e1e7")

        reasons = engine._generate_move_reasons(move, 0.0, 0.0)

        # "Creates a backward pawn weakness for opponent"
        # We need check backward_pawns_them count.
        # Before: e7 exists. d6 supported? e7 is rank 6. d6 is rank 5.
        # e7 supports d6 (d6 is rank 5; adjacent file e, rank 6? No rank BEHIND).
        # Black pawn d6 (rank 6 from Black persp).
        # e7 (rank 7 from Black persp).
        # e7 is behind d6. So e7 supports d6.
        # Capture e7 -> d6 unsupported.
        # d6 stop square (d5). Controlled by White Pawn c4?
        # White c4 attacks d5. Yes.
        # So d6 becomes backward.

        reason = any("backward pawn weakness" in r[2].lower() for r in reasons)
        assert reason, f"Expected backward pawn explanation, got: {reasons}"

    def test_pst_explanation(self):
        engine = ExplainableChessEngine(stockfish_path="mock")
        # Improve placement: Knight h1 -> g3.
        # h1 is bad (-50). g3 is better (+10ish?).
        engine.board = chess.Board("8/8/8/8/8/8/8/6N1 w - - 0 1")  # N on h1
        move = chess.Move.from_uci("g1h3")  # Move to h3 (edge but better than h1?)
        # Or g1f3. f3 is good.
        move = chess.Move.from_uci("g1f3")

        reasons = engine._generate_move_reasons(move, 0.0, 0.0)

        reason = any("piece placement" in r[2].lower() for r in reasons)
        assert reason, f"Expected piece placement explanation, got: {reasons}"

    def test_pin_explanation(self):
        engine = ExplainableChessEngine(stockfish_path="mock")
        # Pin creation: White Rook pins Black Pawn.
        # Black King e8, Black Pawn e7, White Rook e1.
        # Move Rook e1-e2? No, Rook on e1 already pins if open file.
        # Let's say Rook on d1. Move d1-e1 to pin e7.
        engine.board = chess.Board("4k3/4p3/8/8/8/8/8/3R4 w - - 0 1")  # R on d1.
        # Move Re1.
        move = chess.Move.from_uci("d1e1")

        reasons = engine._generate_move_reasons(move, 0.0, 0.0)

        reason = any("pins an opponent" in r[2].lower() for r in reasons)
        assert reason, f"Expected pin creation explanation, got: {reasons}"
