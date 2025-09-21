"""Positional chess metrics and feature calculations."""

import chess

# Passed-pawn momentum helpers
PIECE_VAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.1,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _rank_distance(pawn_sq: int, color: bool) -> int:
    """Calculate distance from pawn to promotion square."""
    r = chess.square_rank(pawn_sq)
    return int(
        (7 - r) if color == chess.WHITE else r
    )  # plies to 8th/1st rank in ranks, not moves


def _rook_behind_passer(board: chess.Board, pawn_sq: int, color: bool) -> int:
    """Check if there's a rook behind a passed pawn."""
    file_idx = chess.square_file(pawn_sq)
    ranks = (
        range(chess.square_rank(pawn_sq) - 1, -1, -1)
        if color == chess.WHITE
        else range(chess.square_rank(pawn_sq) + 1, 8)
    )
    for r in ranks:
        sq = chess.square(file_idx, r)
        p = board.piece_at(sq)
        if p:
            return int(p.piece_type == chess.ROOK and p.color == color)
    return 0


def _stoppers_in_path(board: chess.Board, pawn_sq: int, color: bool) -> int:
    """Count enemy pawns on same/adjacent files ahead within 2 ranks."""
    f = chess.square_file(pawn_sq)
    r = chess.square_rank(pawn_sq)
    dirs = +1 if color == chess.WHITE else -1
    cnt = 0
    for df in (-1, 0, 1):
        nf = f + df
        if nf < 0 or nf > 7:
            continue
        for dr in (dirs, 2 * dirs):
            nr = r + dr
            if nr < 0 or nr > 7:
                continue
            sq = chess.square(nf, nr)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color != color:
                cnt += 1
    return cnt


def _blockaded(board: chess.Board, pawn_sq: int, color: bool) -> int:
    """Check if the square directly in front is occupied by enemy piece."""
    f = chess.square_file(pawn_sq)
    r = chess.square_rank(pawn_sq)
    step = +1 if color == chess.WHITE else -1
    nr = r + step
    if nr < 0 or nr > 7:
        return 0
    sq = chess.square(f, nr)
    p = board.piece_at(sq)
    return int(p is not None and p.color != color)


def _runner_clear_path(board: chess.Board, pawn_sq: int, color: bool) -> int:
    """Check if no enemy pawn stoppers nearby and not blockaded right now."""
    return int(
        _stoppers_in_path(board, pawn_sq, color) == 0
        and _blockaded(board, pawn_sq, color) == 0
    )


def _is_passed_pawn(board: chess.Board, sq: int, color: bool) -> bool:
    """Check if a pawn is passed (no enemy pawns in front on same or adjacent files)."""
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)

    # Check files: same file and adjacent files
    for check_file in [file - 1, file, file + 1]:
        if check_file < 0 or check_file > 7:
            continue
        # Check ranks in front of this pawn
        if color == chess.WHITE:
            ranks_ahead = range(rank + 1, 8)
        else:
            ranks_ahead = range(rank - 1, -1, -1)

        for check_rank in ranks_ahead:
            check_sq = chess.square(check_file, check_rank)
            piece = board.piece_at(check_sq)
            if piece and piece.piece_type == chess.PAWN and piece.color != color:
                return False
    return True


def passed_pawn_momentum_snapshot(board: chess.Board, color: bool):
    """Compute momentum features for one side in the current position."""
    features = {
        "pp_count": 0,
        "pp_min_dist": 8,  # lower is better
        "pp_runners_clear": 0,
        "pp_blockaded": 0,
        "pp_rook_behind": 0,
    }
    for sq in board.pieces(chess.PAWN, color):
        if not _is_passed_pawn(board, sq, color):
            continue
        features["pp_count"] += 1
        features["pp_min_dist"] = min(
            features["pp_min_dist"], _rank_distance(sq, color)
        )
        features["pp_runners_clear"] += _runner_clear_path(board, sq, color)
        features["pp_blockaded"] += _blockaded(board, sq, color)
        features["pp_rook_behind"] += _rook_behind_passer(board, sq, color)
    if features["pp_count"] == 0:
        features["pp_min_dist"] = 8  # sentinel
    return features


def passed_pawn_momentum_delta(base: chess.Board, after_reply: chess.Board):
    """Return Δ features (us – them, and sided deltas) suitable for Δ-training."""
    us = base.turn
    base_us = passed_pawn_momentum_snapshot(base, us)
    base_them = passed_pawn_momentum_snapshot(base, not us)
    after_us = passed_pawn_momentum_snapshot(after_reply, us)
    after_them = passed_pawn_momentum_snapshot(after_reply, not us)

    def d(key):
        return (after_us[key] - base_us[key]) - (
            after_them[key] - base_them[key]
        )  # favor our gain vs their gain

    return {
        "d_pp_count": d("pp_count"),
        "d_pp_min_dist": -(after_us["pp_min_dist"] - base_us["pp_min_dist"])
        + (after_them["pp_min_dist"] - base_them["pp_min_dist"]),
        "d_pp_runners_clear": d("pp_runners_clear"),
        "d_pp_blockaded": -(after_us["pp_blockaded"] - base_us["pp_blockaded"])
        + (after_them["pp_blockaded"] - base_them["pp_blockaded"]),
        "d_pp_rook_behind": d("pp_rook_behind"),
    }


def checkability_now(board: chess.Board):
    """Count available check moves."""
    quiet, capture = 0, 0
    for mv in board.legal_moves:
        if board.gives_check(mv):
            if board.is_capture(mv):
                capture += 1
            else:
                quiet += 1
    return {"d_quiet_checks": quiet, "d_capture_checks": capture}


def confinement_count(board: chess.Board, constrained_side: bool):
    """Count pieces with limited mobility."""
    c = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, constrained_side):
            safe = 0
            # Check all legal moves from this square
            for mv in board.legal_moves:
                if mv.from_square == sq:
                    to = mv.to_square
                    attackers = len(board.attackers(not constrained_side, to))
                    defenders = len(board.attackers(constrained_side, to))
                    if attackers <= defenders:
                        safe += 1
            if safe <= 2:
                c += 1
    return c


def confinement_delta(base: chess.Board, after_reply: chess.Board):
    """Calculate confinement delta between positions."""
    us = base.turn
    return {
        "d_confinement": (
            confinement_count(after_reply, not us) - confinement_count(base, not us)
        )
        - (confinement_count(after_reply, us) - confinement_count(base, us))
    }
