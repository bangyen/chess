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

    def test_safe_mobility(self):
        # White Queen on d4. Black pawns on c5, e5.
        board = chess.Board("8/8/8/2p1p3/3Q4/8/8/8 w - - 0 1")
        feats = baseline_extract_features(board)

        # Total Legal Moves Calculation:
        # File d: 7 moves (d1-d3, d5-d8).
        # Rank 4: 7 moves (a4-c4, e4-h4).
        # Diag 1 (a1-h8): a1, b2, c3. e5 is capture. Stops there. (4 moves).
        # Diag 2 (a7-g1): g1, f2, e3. c5 is capture. Stops there. (4 moves).
        # Total Legal = 7 + 7 + 4 + 4 = 22.

        # Unsafe squares (attacked by c5, e5):
        # c5 (Black pawn on rank 5) attacks b4, d4.
        # e5 (Black pawn on rank 5) attacks d4, f4.
        # d4 is occupation, not move.
        # b4 and f4 are legal moves on Rank 4.
        # So 2 moves are unsafe.
        # Safe mobility = 22 - 2 = 20.

        assert feats["safe_mobility_us"] == 20.0

    def test_rook_open_file(self):
        # White Rook on a1 (open file). White Rook on b1 (semi-open, black pawn on b7).
        # White Rook on c1 (closed, white pawn on c2).
        board = chess.Board("8/1p6/8/8/8/8/2P5/RRR5 w - - 0 1")
        # a1: open (1.0). b1: semi-open (0.5). c1: closed (0.0).
        # Total: 1.5
        feats = baseline_extract_features(board)
        assert feats["rook_open_file_us"] == 1.5

    def test_backward_pawns(self):
        # White pawn on e4. Stop square e5.
        # To be backward, e5 must be controlled by enemy pawn.
        # e5 is rank 5. Enemy pawn must be on rank 6 (d6/f6) to attack rank 5.
        board = chess.Board("8/8/3p1p2/8/4P3/8/8/8 w - - 0 1")
        feats = baseline_extract_features(board)
        assert feats["backward_pawns_us"] == 1.0

        # Add support: White pawn on d3 (rank 3, file d). Supports e4.
        # Does it support e4? Yes, side-by-side or behind.
        # d3 is behind e4.
        board = chess.Board("8/8/3p1p2/8/4P3/3P4/8/8 w - - 0 1")
        feats = baseline_extract_features(board)
        # Should be 0 (supported).
        assert feats["backward_pawns_us"] == 0.0
