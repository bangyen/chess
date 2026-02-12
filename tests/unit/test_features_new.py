import chess

from src.chess_ai.features.baseline import baseline_extract_features


class TestNewFeatures:
    def test_outposts(self):
        # White knight on e5, supported by d4 pawn, no black pawns to attack it
        # Board:
        # r . . . k . . r
        # p p p . . p p p
        # . . . . . . . .
        # . . . . N . . .  <- Knight on e5
        # . . . P . . . .  <- Pawn on d4
        # . . . . . . . .
        # P P P . . P P P
        # R N B Q K B N R
        board = chess.Board("r3k2r/ppp2ppp/8/4N3/3P4/8/PPP2PPP/RNBQKB1R w KQkq - 0 1")
        feats = baseline_extract_features(board)
        # Verify White outpost
        assert feats["outposts_us"] >= 1.0

        # Create a position where it's NOT an outpost (attacked by pawn)
        # Black pawn on f6 attacks e5
        board = chess.Board("r3k2r/ppp3pp/5p2/4N3/3P4/8/PPP2PPP/RNBQKB1R w KQkq - 0 1")
        feats = baseline_extract_features(board)
        assert feats["outposts_us"] == 0.0

    def test_batteries_file(self):
        # White Rooks on a1, a2
        board = chess.Board("8/8/8/8/8/8/R7/R7 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["batteries_us"] >= 1.0

    def test_batteries_rank(self):
        # White Rooks on a1, b1
        board = chess.Board("8/8/8/8/8/8/8/RR6 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["batteries_us"] >= 1.0

    def test_isolated_pawns(self):
        # White pawn on e4, no neighbors on d-file or f-file
        board = chess.Board("8/8/8/8/4P3/8/8/8 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["isolated_pawns_us"] == 1.0

        # Add neighbor on d4 -> not isolated
        board = chess.Board("8/8/8/8/3PP3/8/8/8 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["isolated_pawns_us"] == 0.0

    def test_batteries_diagonal(self):
        # White Bishop on h8, Queen on a1 (main diagonal)
        board = chess.Board("7B/8/8/8/8/8/8/Q7 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["batteries_us"] >= 1.0
