import chess

from chess_ai.features.baseline import baseline_extract_features


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
        board = chess.Board(None)
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.A2, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        # Put black king on b7 (safe from a-file rooks)
        board.set_piece_at(chess.B7, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats = baseline_extract_features(board)
        assert feats["batteries_us"] >= 1.0

    def test_batteries_rank(self):
        # White Rooks on a1, b1
        board = chess.Board(None)
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.B1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        # Put black king on h8 (safe from 1st rank rooks)
        board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats = baseline_extract_features(board)
        assert feats["batteries_us"] >= 1.0

    def test_isolated_pawns(self):
        # White pawn on e4
        board = chess.Board(None)
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats = baseline_extract_features(board)
        assert feats["isolated_pawns_us"] == 1.0

        # Add neighbor on d4 -> not isolated
        board.set_piece_at(chess.D4, chess.Piece(chess.PAWN, chess.WHITE))
        feats = baseline_extract_features(board)
        assert feats["isolated_pawns_us"] == 0.0

    def test_batteries_diagonal(self):
        # White Bishop on h8, Queen on a1
        board = chess.Board(None)
        board.set_piece_at(chess.H8, chess.Piece(chess.BISHOP, chess.WHITE))
        board.set_piece_at(chess.A1, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats = baseline_extract_features(board)
        assert feats["batteries_us"] >= 1.0

    def test_safe_mobility(self):
        # White Queen on d4. Black pawns on c5, e5.
        board = chess.Board(None)
        board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.C5, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.BLACK))
        # White king on h1. Mobility = 3 (g1, g2, h2).
        board.set_piece_at(chess.H1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats = baseline_extract_features(board)
        # Queen moves = 20.0. King moves = 3.0. Total = 23.0.
        assert feats["safe_mobility_us"] == 23.0

    def test_rook_open_file(self):
        # White Rooks.
        board = chess.Board(None)
        board.set_piece_at(chess.A1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.B1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.C1, chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(chess.B7, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.C2, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats = baseline_extract_features(board)
        assert feats["rook_open_file_us"] == 1.5

    def test_backward_pawns(self):
        # White pawn e4, black pawns d6, f6.
        board = chess.Board(None)
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.D6, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.F6, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats = baseline_extract_features(board)
        assert feats["backward_pawns_us"] == 1.0

        # Add support d3.
        board.set_piece_at(chess.D3, chess.Piece(chess.PAWN, chess.WHITE))
        feats = baseline_extract_features(board)
        assert feats["backward_pawns_us"] == 0.0

    def test_pst(self):
        # White Knight center vs corner.
        board = chess.Board(None)
        board.set_piece_at(chess.E4, chess.Piece(chess.KNIGHT, chess.WHITE))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats1 = baseline_extract_features(board)

        board = chess.Board(None)
        board.set_piece_at(chess.H1, chess.Piece(chess.KNIGHT, chess.WHITE))
        board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats2 = baseline_extract_features(board)
        assert feats1["pst_us"] > feats2["pst_us"]

    def test_pinned(self):
        board = chess.Board(None)
        board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        # Use knight instead of pawn to avoid rank 2 issues if it were rank 1
        board.set_piece_at(chess.E2, chess.Piece(chess.KNIGHT, chess.WHITE))
        board.set_piece_at(chess.E8, chess.Piece(chess.ROOK, chess.BLACK))
        board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        feats = baseline_extract_features(board)
        assert feats["pinned_us"] == 1.0

        # Unpin: Move king to d1.
        board.remove_piece_at(chess.E1)
        board.set_piece_at(chess.D1, chess.Piece(chess.KING, chess.WHITE))
        feats = baseline_extract_features(board)
        assert feats["pinned_us"] == 0.0
