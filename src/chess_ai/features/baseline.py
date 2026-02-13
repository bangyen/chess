"""Baseline feature extraction functions."""

import sys
from typing import Dict

import chess
import chess.engine

try:
    pass
except Exception:
    print(
        "scikit-learn is required. Install with: pip install scikit-learn",
        file=sys.stderr,
    )
    raise


try:
    from chess_ai.rust_utils import calculate_forcing_swing, find_best_reply

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def baseline_extract_features(board: "chess.Board") -> Dict[str, float]:
    """Small, fast, interpretable baseline feature set.

    This is intentionally compact; use your own richer set for best results.

    Args:
        board: The chess board position to extract features from

    Returns:
        Dictionary of feature names to values
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.1,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def material(side):
        val = 0.0
        for p, v in piece_values.items():
            val += len(board.pieces(p, side)) * v
        return val

    def mobility(side):
        # quick mobility: count legal moves, capped to reduce variance
        if board.turn != side:
            board.turn = side
            moves = sum(1 for _ in board.legal_moves)
            board.turn = not side
        else:
            moves = sum(1 for _ in board.legal_moves)
        return min(moves, 40)

    def king_ring_pressure(attacking_side):
        """Phase-normalized king-ring pressure"""
        ksq = board.king(not attacking_side)
        if ksq is None:
            return 0.0

        # Get king ring squares
        ring = chess.SquareSet()
        for sq in chess.SQUARES:
            if chess.square_distance(sq, ksq) <= 1:
                ring.add(sq)

        weight = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3**0.7,
            chess.BISHOP: 3.1**0.7,
            chess.ROOK: 5**0.7,
            chess.QUEEN: 9**0.7,
        }
        s = 0.0

        for sq in ring:
            if board.is_attacked_by(attacking_side, sq):
                # Find the strongest attacker
                for pt, w in weight.items():
                    if any(
                        board.piece_type_at(x) == pt
                        and board.color_at(x) == attacking_side
                        for x in board.attackers(attacking_side, sq)
                    ):
                        s += w
                        break

        # Normalize by phase
        phase = sum(
            len(board.pieces(pt, True)) + len(board.pieces(pt, False))
            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        )
        return s / max(6, phase)

    def passed_pawns(side):
        count = 0
        # Simple passed pawn check: no enemy pawns in front on same or adjacent files
        for sq in board.pieces(chess.PAWN, side):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True

            # Check files: same file and adjacent files
            for check_file in [file - 1, file, file + 1]:
                if check_file < 0 or check_file > 7:
                    continue
                # Check ranks in front of this pawn (higher ranks for white, lower for black)
                if side:  # White pawns
                    for check_rank in range(rank + 1, 8):
                        check_sq = chess.square(check_file, check_rank)
                        piece = board.piece_type_at(check_sq)
                        if piece == chess.PAWN and board.color_at(check_sq) != side:
                            is_passed = False
                            break
                else:  # Black pawns
                    for check_rank in range(0, rank):
                        check_sq = chess.square(check_file, check_rank)
                        piece = board.piece_type_at(check_sq)
                        if piece == chess.PAWN and board.color_at(check_sq) != side:
                            is_passed = False
                            break
                if not is_passed:
                    break

            if is_passed:
                count += 1
        return float(count)

    feats = {}
    feats["material_us"] = material(board.turn)
    feats["material_them"] = material(not board.turn)
    feats["material_diff"] = feats["material_us"] - feats["material_them"]
    feats["mobility_us"] = mobility(board.turn)
    feats["mobility_them"] = mobility(not board.turn)
    feats["king_ring_pressure_us"] = king_ring_pressure(board.turn)
    feats["king_ring_pressure_them"] = king_ring_pressure(not board.turn)
    feats["passed_us"] = passed_pawns(board.turn)
    feats["passed_them"] = passed_pawns(not board.turn)

    # simple rook on open/semi-open file
    def file_state(side):
        open_files = 0
        semi_open = 0
        for file in range(8):
            pawns = [
                sq
                for sq in chess.SQUARES
                if chess.square_file(sq) == file
                and board.piece_type_at(sq) == chess.PAWN
            ]
            pawns_side = [sq for sq in pawns if board.color_at(sq) == side]
            pawns_opp = [sq for sq in pawns if board.color_at(sq) == (not side)]
            if not pawns_side and not pawns_opp:
                open_files += 1
            elif not pawns_opp:
                semi_open += 1
        return open_files, semi_open

    of, sof = file_state(board.turn)
    of2, sof2 = file_state(not board.turn)
    feats["open_files_us"] = of
    feats["semi_open_us"] = sof
    feats["open_files_them"] = of2
    feats["semi_open_them"] = sof2
    feats["phase"] = float(
        sum(
            len(board.pieces(pt, True)) + len(board.pieces(pt, False))
            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        )
    )

    # Add more positional features
    def center_control(side):
        # Count pieces in center squares (d4, d5, e4, e5)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        count = 0
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece and piece.color == side:
                count += 1
        return float(count)

    def piece_activity(side):
        # Count squares attacked by pieces of this side
        count = 0
        for sq in chess.SQUARES:
            if board.is_attacked_by(side, sq):
                count += 1
        return float(count)

    def king_safety(side):
        # Simple king safety: count pieces around own king
        king_sq = board.king(side)
        if king_sq is None:
            return 0.0
        safety = 0.0
        # Count friendly pieces in king's vicinity
        for sq in board.attacks(king_sq):
            piece = board.piece_at(sq)
            if piece and piece.color == side:
                safety += 1.0
        return safety

    feats["center_control_us"] = center_control(board.turn)
    feats["center_control_them"] = center_control(not board.turn)
    feats["piece_activity_us"] = piece_activity(board.turn)
    feats["piece_activity_them"] = piece_activity(not board.turn)
    feats["king_safety_us"] = king_safety(board.turn)
    feats["king_safety_them"] = king_safety(not board.turn)

    # Add tactical features
    def hanging_pieces(side):
        # Count pieces that are attacked but not defended
        count = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == side:
                if board.is_attacked_by(not side, sq) and not board.is_attacked_by(
                    side, sq
                ):
                    count += 1
        return float(count)

    def bishop_pair(side):
        # Check if side has both bishops
        bishops = board.pieces(chess.BISHOP, side)
        return 1.0 if len(bishops) >= 2 else 0.0

    def rook_on_7th(side):
        # Count rooks on the 7th rank (for white) or 2nd rank (for black)
        target_rank = 6 if side == chess.WHITE else 1
        count = 0
        for sq in board.pieces(chess.ROOK, side):
            if chess.square_rank(sq) == target_rank:
                count += 1
        return float(count)

    def king_pawn_shield(side):
        # Count friendly pawns in front of the king (3 files around)
        ks = board.king(side)
        if ks is None:
            return 0.0
        file = chess.square_file(ks)
        rank = chess.square_rank(ks)

        # Define shield area: 3 files, 1-2 ranks ahead
        shield_ranks = (
            [rank + 1, rank + 2] if side == chess.WHITE else [rank - 1, rank - 2]
        )
        shield_files = [file - 1, file, file + 1]

        count = 0
        for f in shield_files:
            if f < 0 or f > 7:
                continue
            for r in shield_ranks:
                if r < 0 or r > 7:
                    continue
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == side:
                    count += 1
        return float(count)

    feats["hanging_us"] = hanging_pieces(board.turn)
    feats["hanging_them"] = hanging_pieces(not board.turn)
    feats["bishop_pair_us"] = bishop_pair(board.turn)
    feats["bishop_pair_them"] = bishop_pair(not board.turn)
    feats["rook_on_7th_us"] = rook_on_7th(board.turn)
    feats["rook_on_7th_them"] = rook_on_7th(not board.turn)
    feats["king_pawn_shield_us"] = king_pawn_shield(board.turn)
    feats["king_pawn_shield_them"] = king_pawn_shield(not board.turn)

    # Add engine-based dynamic probes for better move ranking
    def sf_eval_shallow(engine, board, depth=6):
        """Shallow engine evaluation with depth limit"""
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
            if isinstance(info, list):
                info = info[0]
            score = info["score"].pov(board.turn)
            cp = score.score(mate_score=100000)
            # Clip mate scores to prevent instability
            return float(max(-1000, min(1000, cp)))
        except Exception:
            return 0.0

    def hanging_after_reply_real(engine, board, depth=6):
        """Real hanging-after-reply with engine analysis (or Rust optimization)"""
        try:
            reply = None
            if RUST_AVAILABLE:
                try:
                    # Rust alpha-beta is unoptimized (no TT/move ordering), so depth 6 is too slow.
                    # Clamp to 4 which is sufficient for hanging pieces and fast.
                    rust_depth = min(depth, 4)
                    uci = find_best_reply(board.fen(), rust_depth)
                    if uci:
                        reply = chess.Move.from_uci(uci)
                except Exception:
                    # Fallback on error
                    pass

            if reply is None:
                # Stockfish fallback
                info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
                if isinstance(info, list):
                    info = info[0]
                reply = info.get("pv", [None])[0]

            if reply is None:
                return 0, 0, 0

            board.push(reply)
            side = board.turn  # side to move after reply
            cnt, v_max, near_king = 0, 0, 0
            opp = not side
            opp_king_sq = board.king(opp)

            # Check for hanging pieces
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.color == opp:
                    defenders = board.attackers(opp, sq)
                    attackers = board.attackers(side, sq)
                    if not defenders and attackers:
                        cnt += 1
                        piece_value = {
                            chess.PAWN: 1,
                            chess.KNIGHT: 3,
                            chess.BISHOP: 3,
                            chess.ROOK: 5,
                            chess.QUEEN: 9,
                        }.get(piece.piece_type, 0)
                        v_max = max(v_max, piece_value)
                        if (
                            opp_king_sq is not None
                            and chess.square_distance(sq, opp_king_sq) <= 1
                        ):
                            near_king = 1

            board.pop()
            return cnt, v_max, near_king
        except Exception:
            return 0, 0, 0

    def best_forcing_swing_real(engine, board, d_base=6, k_max=12):
        """Real forcing swing with eval differences (or Rust optimization)"""
        if RUST_AVAILABLE:
            try:
                # Rust implementation handles the full swing calculation
                # Note: Rust returns float centipawns
                # Clamp depth to 4 for performance
                rust_depth = min(d_base, 4)
                return float(calculate_forcing_swing(board.fen(), rust_depth))
            except Exception:
                pass

        try:
            base = sf_eval_shallow(engine, board, d_base)
            forcing = [
                m
                for m in board.legal_moves
                if board.is_capture(m) or board.gives_check(m)
            ]
            swings = []
            for m in forcing[:k_max]:
                board.push(m)
                v = sf_eval_shallow(engine, board, d_base - 1)
                board.pop()
                swings.append(v - base)
            return max(swings) if swings else 0.0
        except Exception:
            return 0.0

    # Add advanced positional features
    def outposts(side):
        # Knight on rank 4-6 (relative), supported by pawn, no enemy pawn can attack
        count = 0
        knights = board.pieces(chess.KNIGHT, side)
        for sq in knights:
            rank = chess.square_rank(sq)

            # Check rank (4th, 5th, 6th relative to side)
            rel_rank = rank if side == chess.WHITE else 7 - rank
            if rel_rank < 3 or rel_rank > 5:
                continue

            # Check pawn support
            is_supported = False
            pawn_attacks = board.attackers(side, sq)
            for attacker in pawn_attacks:
                if board.piece_type_at(attacker) == chess.PAWN:
                    is_supported = True
                    break
            if not is_supported:
                continue

            # Check if enemy pawns can attack
            # This is complex, simplified: check if enemy pawns on adjacent files are ahead
            # or if the square is guarded by an enemy pawn
            if board.is_attacked_by(not side, sq):
                # If attacked by pawn, not a true outpost
                attacked_by_pawn = False
                enemy_attacks = board.attackers(not side, sq)
                for attacker in enemy_attacks:
                    if board.piece_type_at(attacker) == chess.PAWN:
                        attacked_by_pawn = True
                        break
                if attacked_by_pawn:
                    continue

            count += 1
        return float(count)

    def batteries(side):
        # R-R, R-Q, Q-Q on same file/rank; B-Q on same diagonal
        count = 0
        # Files and Ranks
        for i in range(8):
            # File
            pieces_on_file = []
            for r in range(8):
                sq = chess.square(i, r)
                p = board.piece_at(sq)
                if p and p.color == side and p.piece_type in [chess.ROOK, chess.QUEEN]:
                    pieces_on_file.append(p.piece_type)
            if len(pieces_on_file) >= 2:
                count += 1

            # Rank
            pieces_on_rank = []
            for f in range(8):
                sq = chess.square(f, i)
                p = board.piece_at(sq)
                if p and p.color == side and p.piece_type in [chess.ROOK, chess.QUEEN]:
                    pieces_on_rank.append(p.piece_type)
            if len(pieces_on_rank) >= 2:
                count += 1

        # Diagonals (B-Q)
        # Scan all diagonals
        # Positive diagonals (sum of rank+file is constant)
        for s in range(15):
            pieces_on_diag = []
            for f in range(8):
                r = s - f
                if 0 <= r < 8:
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if (
                        p
                        and p.color == side
                        and p.piece_type in [chess.BISHOP, chess.QUEEN]
                    ):
                        pieces_on_diag.append(p.piece_type)
            if len(pieces_on_diag) >= 2:
                count += 1

        # Negative diagonals (diff of rank-file is constant)
        for d in range(-7, 8):
            pieces_on_diag = []
            for f in range(8):
                r = f + d
                if 0 <= r < 8:
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if (
                        p
                        and p.color == side
                        and p.piece_type in [chess.BISHOP, chess.QUEEN]
                    ):
                        pieces_on_diag.append(p.piece_type)
            if len(pieces_on_diag) >= 2:
                count += 1

        return float(count)

    def pawn_structure(side):
        isolated = 0

        pawns = board.pieces(chess.PAWN, side)
        for sq in pawns:
            file = chess.square_file(sq)

            # Isolated
            has_neighbor = False
            for f in [file - 1, file + 1]:
                if 0 <= f <= 7:
                    for r in range(8):
                        check_sq = chess.square(f, r)
                        p = board.piece_at(check_sq)
                        if p and p.piece_type == chess.PAWN and p.color == side:
                            has_neighbor = True
                            break
                if has_neighbor:
                    break
            if not has_neighbor:
                isolated += 1

        return float(isolated)

    feats["outposts_us"] = outposts(board.turn)
    feats["outposts_them"] = outposts(not board.turn)

    feats["batteries_us"] = batteries(board.turn)
    feats["batteries_them"] = batteries(not board.turn)

    feats["isolated_pawns_us"] = pawn_structure(board.turn)
    feats["isolated_pawns_them"] = pawn_structure(not board.turn)

    # Phase 2 Features
    def safe_mobility(side):
        # Count legal moves that don't land on squares attacked by enemy pawns
        if board.turn != side:
            board.turn = side
            moves = list(board.legal_moves)
            board.turn = not side
        else:
            moves = list(board.legal_moves)

        # Get enemy pawn attacks
        enemy_pawn_attacks = chess.SquareSet()
        opp = not side
        for sq in board.pieces(chess.PAWN, opp):
            enemy_pawn_attacks |= board.attacks(sq)

        safe_count = 0
        for move in moves:
            if move.to_square not in enemy_pawn_attacks:
                safe_count += 1

        return min(float(safe_count), 40.0)

    def rook_on_open_file(side):
        count = 0
        for sq in board.pieces(chess.ROOK, side):
            file = chess.square_file(sq)
            # Check if file is open (no pawns)
            is_open = True
            is_semi_open = True

            for r in range(8):
                s = chess.square(file, r)
                p = board.piece_at(s)
                if p and p.piece_type == chess.PAWN:
                    is_open = False
                    if p.color == side:
                        is_semi_open = False

            if is_open:
                count += 1.0
            elif is_semi_open:
                count += 0.5
        return float(count)

    def backward_pawns(side):
        count = 0
        pawns = board.pieces(chess.PAWN, side)
        opp = not side

        # Get all enemy pawn attacks (control)
        enemy_pawn_attacks = chess.SquareSet()
        for sq in board.pieces(chess.PAWN, opp):
            enemy_pawn_attacks |= board.attacks(sq)

        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)

            # 1. Check if supported by friendly pawns
            is_supported = False
            # Support comes from pawns on adjacent files, 1 rank behind or same rank (guards advance?)
            # Strictly: pawns on adjacent files that are behind or same rank.
            # Simplified: adjacent files, rank <= current rank.
            for f in [file - 1, file + 1]:
                if 0 <= f <= 7:
                    # Let's use simple support check: existing friendly pawns on adjacent files BEHIND or EQUAL
                    for r in range(8):
                        if (side == chess.WHITE and r <= rank) or (
                            side == chess.BLACK and r >= rank
                        ):
                            s_sq = chess.square(f, r)
                            p = board.piece_at(s_sq)
                            if p and p.piece_type == chess.PAWN and p.color == side:
                                is_supported = True
                                break
                if is_supported:
                    break

            if is_supported:
                continue

            # 2. Check if advance is stopped by enemy pawns (control of stop square)
            stop_rank = rank + 1 if side == chess.WHITE else rank - 1
            if 0 <= stop_rank <= 7:
                stop_sq = chess.square(file, stop_rank)
                if stop_sq in enemy_pawn_attacks:
                    count += 1

        return float(count)

    feats["safe_mobility_us"] = safe_mobility(board.turn)
    feats["safe_mobility_them"] = safe_mobility(not board.turn)
    feats["rook_open_file_us"] = rook_on_open_file(board.turn)
    feats["rook_open_file_them"] = rook_on_open_file(not board.turn)
    feats["backward_pawns_us"] = backward_pawns(board.turn)
    feats["backward_pawns_them"] = backward_pawns(not board.turn)
    feats["_engine_probes"] = {
        "hanging_after_reply": hanging_after_reply_real,
        "best_forcing_swing": best_forcing_swing_real,
        "sf_eval_shallow": sf_eval_shallow,
    }

    # Phase 3: Piece-Square Tables (Simplified PeSTO/Stockfish-like values)
    # 0 = a1, 63 = h8.
    # We'll valid for White side. For Black, we flip the square (63 - sq).

    # Pawns (incentivize center control and advancement)
    PST_PAWN = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        5,
        5,
        10,
        25,
        25,
        10,
        5,
        5,
        0,
        0,
        0,
        20,
        20,
        0,
        0,
        0,
        5,
        -5,
        -10,
        0,
        0,
        -10,
        -5,
        5,
        5,
        10,
        10,
        -20,
        -20,
        10,
        10,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    # Knights (incentivize center)
    PST_KNIGHT = [
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        0,
        0,
        0,
        -20,
        -40,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -30,
        5,
        15,
        20,
        20,
        15,
        5,
        -30,
        -30,
        0,
        15,
        20,
        20,
        15,
        0,
        -30,
        -30,
        5,
        10,
        15,
        15,
        10,
        5,
        -30,
        -40,
        -20,
        0,
        5,
        5,
        0,
        -20,
        -40,
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
    ]

    # Bishops (incentivize diagonal control key squares)
    PST_BISHOP = [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        10,
        10,
        5,
        0,
        -10,
        -10,
        5,
        5,
        10,
        10,
        5,
        5,
        -10,
        -10,
        0,
        10,
        10,
        10,
        10,
        0,
        -10,
        -10,
        10,
        10,
        10,
        10,
        10,
        10,
        -10,
        -10,
        5,
        0,
        0,
        0,
        0,
        5,
        -10,
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
    ]

    # Rooks (incentivize 7th rank and center files slightly)
    PST_ROOK = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        0,
        0,
        0,
        5,
        5,
        0,
        0,
        0,
    ]

    # Queen (incentivize center but not too early)
    PST_QUEEN = [
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -5,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        0,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        -10,
        5,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        5,
        0,
        0,
        0,
        0,
        -10,
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
    ]

    # King (Middlegame: safety)
    PST_KING_MG = [
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -20,
        -30,
        -30,
        -40,
        -40,
        -30,
        -30,
        -20,
        -10,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -10,
        20,
        20,
        0,
        0,
        0,
        0,
        20,
        20,
        20,
        30,
        10,
        0,
        0,
        10,
        30,
        20,
    ]

    # King (Endgame: activity)
    PST_KING_EG = [
        -50,
        -40,
        -30,
        -20,
        -20,
        -30,
        -40,
        -50,
        -30,
        -20,
        -10,
        0,
        0,
        -10,
        -20,
        -30,
        -30,
        -10,
        20,
        30,
        30,
        20,
        -10,
        -30,
        -30,
        -10,
        30,
        40,
        40,
        30,
        -10,
        -30,
        -30,
        -10,
        30,
        40,
        40,
        30,
        -10,
        -30,
        -30,
        -10,
        20,
        30,
        30,
        20,
        -10,
        -30,
        -30,
        -30,
        0,
        0,
        0,
        0,
        -30,
        -30,
        -50,
        -30,
        -30,
        -30,
        -30,
        -30,
        -30,
        -50,
    ]

    def pst_value(side):
        score = 0.0
        # Phase (0=opening, 1=endgame)
        # We already have 'phase' feature but need it normalized 0..1?
        # feats['phase'] is sum of piece counts (~40 max?).
        # Let's say phase < 15 is endgame.

        # Current phase feature: sum of Q(9), R(5), B(3), N(3).
        # Max ~ 9+10+6+6 = 31 * 2 = 62?
        # feats["phase"] (L164) is count of pieces, not values?
        # Ah wait, L164: len(pieces)... just counts.
        # Max pieces = 16 (excluding K, P).
        # Let's use simple check: if no queens or few pieces => endgame.
        is_endgame = feats["phase"] < 10  # heuristic

        for pt, table in [
            (chess.PAWN, PST_PAWN),
            (chess.KNIGHT, PST_KNIGHT),
            (chess.BISHOP, PST_BISHOP),
            (chess.ROOK, PST_ROOK),
            (chess.QUEEN, PST_QUEEN),
        ]:
            if table is None:
                continue
            for sq in board.pieces(pt, side):
                # Access table (which is rank-flattened: 0-7 is rank 1)
                # chess.SQUARES: a1=0, b1=1...
                # Verify orientation: a1 is index 0.
                # Table above: Top-left is a8? No, usually a1 is bottom-left.
                # Standard Python-chess: square 0 is a1.
                # If valid for White: a1 should be row 7 (index 56)?
                # Or row 0 (index 0).
                # Usually printed tables are Rank 8 top.
                # Let's assume standard array order: index 0 is first element.
                # If I wrote:
                # 0, 0...
                # means the first row of array.
                # If I map index 0 -> a1.
                # Then first row of array corresponds to Rank 1.
                # My tables above:
                # PAWN: Row 1 (index 0-7) is 0s. (Rank 1). Correct.
                # Row 2 (index 8-15) is 50s. (Rank 2). This incentivizes start?
                # Usually Pawns on Rank 2 are 0?
                # Ah, advanced pawns should be higher.
                # If 50s are at index 8-15 (Rank 2), that means staying at home is good?
                # NO. Row 2 in array = Rank 2 on board (if mapped directly).
                # Wait, usually Rank 7 (promotion) is high.
                # In my table: Row 7 (index 48-55) is... 5, 10, 10...
                # Row 8 (index 56-63) is 0.
                # This seems flipped?
                # Let's assume the table is VISUAL (rank 8 a top).
                # Then index 0-7 is Rank 8.
                # If so, Pawn table row 0 (Rank 8) is 0s (promoted? usually handled by material).
                # Row 7 (Rank 2) would be the second to last row.
                # Let's standardize: Visual table (Rank 8 at top).
                # To map square `sq` (0-63, a1-h8) to visual table index:
                # Rank 8 is indices 0-7. Rank 1 is 56-63.
                # sq_rank 7 -> table row 0.
                # row = 7 - chess.square_rank(sq).
                # col = chess.square_file(sq).
                # index = row * 8 + col.

                # For Black: flip rank. sq_rank 7 (Black home) -> table row 7 (visual bottom)?
                # No, table is "Relative to side".
                # So for White: Rank 1 is bottom. For Black: Rank 8 is bottom.
                # If table is Visual (Top=Far, Bottom=Home):
                # White Home (Rank 1) -> Bottom (Row 7).
                # Black Home (Rank 8) -> Bottom (Row 7).
                # So:
                # visual_row = 7 - relative_rank.
                # relative_rank = rank if White else 7 - rank.
                # visual_row = 7 - (rank) [White]
                # visual_row = 7 - (7 - rank) = rank [Black]

                vis_r = (
                    7 - chess.square_rank(sq)
                    if side == chess.WHITE
                    else chess.square_rank(sq)
                )
                vis_c = chess.square_file(sq)

                score += table[vis_r * 8 + vis_c]

        # King
        table = PST_KING_EG if is_endgame else PST_KING_MG
        for sq in board.pieces(chess.KING, side):
            vis_r = (
                7 - chess.square_rank(sq)
                if side == chess.WHITE
                else chess.square_rank(sq)
            )
            vis_c = chess.square_file(sq)
            score += table[vis_r * 8 + vis_c]

        return float(score) / 100.0  # Standardize scale

    def pinned_pieces(side):
        # Count absolutely pinned pieces
        count = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == side:
                if board.is_pinned(side, sq):
                    count += 1
        return float(count)

    feats["pst_us"] = pst_value(board.turn)
    feats["pst_them"] = pst_value(not board.turn)
    feats["pinned_us"] = pinned_pieces(board.turn)
    feats["pinned_them"] = pinned_pieces(not board.turn)

    return feats
