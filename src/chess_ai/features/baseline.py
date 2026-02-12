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
        # Count attacks into the 8 squares around enemy king (ring1), weighted lightly
        ks = board.king(not attacking_side)
        if ks is None:
            return 0.0
        # Use king attacks squares (the 8 squares around the king)
        ring = board.attacks(ks)

        count = 0
        for sq in ring:
            if board.is_attacked_by(attacking_side, sq):
                count += 1
        return float(count)

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
        """Real hanging-after-reply with engine analysis"""
        try:
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
        """Real forcing swing with eval differences"""
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

    def king_ring_pressure_real(board, attacker):
        """Phase-normalized king-ring pressure"""
        ksq = board.king(not attacker)
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
            if board.is_attacked_by(attacker, sq):
                # Find the strongest attacker
                for pt, w in weight.items():
                    if any(
                        board.piece_type_at(x) == pt and board.color_at(x) == attacker
                        for x in board.attackers(attacker, sq)
                    ):
                        s += w
                        break

        # Normalize by phase
        phase = sum(
            len(board.pieces(pt, True)) + len(board.pieces(pt, False))
            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        )
        return s / max(6, phase)

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
        doubled = 0
        isolated = 0

        pawns = board.pieces(chess.PAWN, side)
        for sq in pawns:
            file = chess.square_file(sq)

            # Doubled
            pawns_on_file = 0
            for r in range(8):
                check_sq = chess.square(file, r)
                p = board.piece_at(check_sq)
                if p and p.piece_type == chess.PAWN and p.color == side:
                    pawns_on_file += 1
            if pawns_on_file > 1:
                doubled += 1  # Counts each pawn involved, maybe divide by 2? Sticking to count.

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

        return float(doubled), float(isolated)

    def space_advantage(side):
        # Count squares behind enemy pawn structure controlled by us
        # Refined: Focus on central files (c, d, e, f) and rank 5+ (for white)
        count = 0
        enemy_camp_ranks = range(4, 8) if side == chess.WHITE else range(0, 4)
        central_files = [2, 3, 4, 5]  # c, d, e, f

        for sq in chess.SQUARES:
            if (
                chess.square_rank(sq) in enemy_camp_ranks
                and chess.square_file(sq) in central_files
            ):
                # Controlled by our pawn?
                attackers = board.attackers(side, sq)
                for attacker in attackers:
                    if board.piece_type_at(attacker) == chess.PAWN:
                        count += 1
                        break
        return float(count)

    feats["outposts_us"] = outposts(board.turn)
    feats["outposts_them"] = outposts(not board.turn)

    feats["batteries_us"] = batteries(board.turn)
    feats["batteries_them"] = batteries(not board.turn)

    doubled_us, iso_us = pawn_structure(board.turn)
    doubled_them, iso_them = pawn_structure(not board.turn)
    feats["doubled_pawns_us"] = doubled_us
    feats["doubled_pawns_them"] = doubled_them
    feats["isolated_pawns_us"] = iso_us
    feats["isolated_pawns_them"] = iso_them

    feats["space_us"] = space_advantage(board.turn)
    feats["space_them"] = space_advantage(not board.turn)

    # Store functions for later use in move ranking
    feats["_engine_probes"] = {
        "hanging_after_reply": hanging_after_reply_real,
        "best_forcing_swing": best_forcing_swing_real,
        "sf_eval_shallow": sf_eval_shallow,
    }

    return feats
