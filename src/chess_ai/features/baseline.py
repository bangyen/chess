"""Baseline feature extraction functions."""

import os
from typing import Any, Dict, Optional, Tuple

import chess
import chess.engine

try:
    pass
except Exception:
    raise


try:
    from chess_ai.rust_utils import (
        SyzygyTablebase,
        calculate_forcing_swing,
        extract_features_rust,
        find_best_reply,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


_SYZYGY_TB: Optional["SyzygyTablebase"] = None


def baseline_extract_features(board: "chess.Board") -> Dict[str, float]:  # noqa: C901
    """Small, fast, interpretable baseline feature set.

    This is intentionally compact; use your own richer set for best results.

    Args:
        board: The chess board position to extract features from

    Returns:
        Dictionary of feature names to values
    """
    global _SYZYGY_TB

    feats: Dict[str, Any] = {}

    if RUST_AVAILABLE:
        try:
            feats = extract_features_rust(board.fen())

            # Add Syzygy tablebase features (if available) - Still in Python for path management
            syzygy_path = os.environ.get("SYZYGY_PATH")
            if syzygy_path:
                try:
                    if not _SYZYGY_TB:
                        _SYZYGY_TB = SyzygyTablebase(syzygy_path)

                    if len(board.piece_map()) <= 7:
                        wdl = _SYZYGY_TB.probe_wdl(board.fen())
                        dtz = _SYZYGY_TB.probe_dtz(board.fen())
                        if wdl is not None:
                            feats["syzygy_wdl"] = float(wdl) / 2.0
                        if dtz is not None:
                            feats["syzygy_dtz"] = float(dtz) / 100.0
                except Exception:  # noqa: S110
                    pass

            # Rust succeeded — define engine probes and return early,
            # skipping the slower Python feature recomputation.
            def _sf_eval_shallow(
                engine: Any, board: chess.Board, depth: int = 6
            ) -> float:
                """Shallow engine evaluation with depth limit."""
                try:
                    info = engine.analyse(
                        board, chess.engine.Limit(depth=depth), multipv=1
                    )
                    if isinstance(info, list):
                        info = info[0]
                    score = info["score"].pov(board.turn)
                    cp = score.score(mate_score=100000)
                    return float(max(-1000, min(1000, cp)))
                except Exception:
                    return 0.0

            def _hanging_after_reply(
                engine: Any, board: chess.Board, depth: int = 6
            ) -> Tuple[int, int, int]:
                """Hanging-after-reply via Rust or Stockfish fallback."""
                try:
                    reply = None
                    if RUST_AVAILABLE:
                        try:
                            # Rust search uses TT + pruning; depth 8 is safe.
                            rust_depth = min(depth, 8)
                            uci = find_best_reply(board.fen(), rust_depth)
                            if uci:
                                reply = chess.Move.from_uci(uci)
                        except Exception:  # noqa: S110
                            pass
                    if reply is None:
                        info = engine.analyse(
                            board,
                            chess.engine.Limit(depth=depth),
                            multipv=1,
                        )
                        if isinstance(info, list):
                            info = info[0]
                        reply = info.get("pv", [None])[0]
                    if reply is None:
                        return 0, 0, 0
                    board.push(reply)
                    side = board.turn
                    cnt, v_max, near_king = 0, 0, 0
                    opp = not side
                    opp_king_sq = board.king(opp)
                    for sq in chess.SQUARES:
                        piece = board.piece_at(sq)
                        if piece and piece.color == opp:
                            defenders = board.attackers(opp, sq)
                            attackers = board.attackers(side, sq)
                            if not defenders and attackers:
                                cnt += 1
                                pv = {
                                    chess.PAWN: 1,
                                    chess.KNIGHT: 3,
                                    chess.BISHOP: 3,
                                    chess.ROOK: 5,
                                    chess.QUEEN: 9,
                                }.get(piece.piece_type, 0)
                                v_max = max(v_max, pv)
                                if (
                                    opp_king_sq is not None
                                    and chess.square_distance(sq, opp_king_sq) <= 1
                                ):
                                    near_king = 1
                    board.pop()
                    return cnt, v_max, near_king
                except Exception:
                    return 0, 0, 0

            def _best_forcing_swing(
                engine: Any, board: chess.Board, d_base: int = 6, k_max: int = 12
            ) -> float:
                """Forcing swing via Rust or Stockfish fallback."""
                if RUST_AVAILABLE:
                    try:
                        # Rust search uses TT + pruning; depth 8 is safe.
                        rust_depth = min(d_base, 8)
                        return float(calculate_forcing_swing(board.fen(), rust_depth))
                    except Exception:  # noqa: S110
                        pass
                try:
                    base = _sf_eval_shallow(engine, board, d_base)
                    forcing = [
                        m
                        for m in board.legal_moves
                        if board.is_capture(m) or board.gives_check(m)
                    ]
                    swings = []
                    for m in forcing[:k_max]:
                        board.push(m)
                        v = _sf_eval_shallow(engine, board, d_base - 1)
                        board.pop()
                        swings.append(v - base)
                    return max(swings) if swings else 0.0
                except Exception:
                    return 0.0

            feats["_engine_probes"] = {
                "hanging_after_reply": _hanging_after_reply,
                "best_forcing_swing": _best_forcing_swing,
                "sf_eval_shallow": _sf_eval_shallow,
            }
            return feats

        except Exception:
            # Fallback to Python if Rust fails
            feats = {}

    # Python Implementation (Fallback)
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.1,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def material(side: bool) -> float:
        val = 0.0
        for p, v in piece_values.items():
            val += len(board.pieces(p, side)) * v
        return val

    def mobility(side: bool) -> int:
        # quick mobility: count legal moves, capped to reduce variance
        if board.turn != side:
            board.turn = side
            moves = sum(1 for _ in board.legal_moves)
            board.turn = not side
        else:
            moves = sum(1 for _ in board.legal_moves)
        return min(moves, 40)

    def king_ring_pressure(attacking_side: bool) -> float:
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

    def passed_pawns(side: bool) -> float:
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
                    for check_rank in range(rank):
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
    def file_state(side: bool) -> Tuple[int, int]:
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
    def center_control(side: bool) -> float:
        # Count pieces in center squares (d4, d5, e4, e5)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        count = 0
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece and piece.color == side:
                count += 1
        return float(count)

    def piece_activity(side: bool) -> float:
        # Count squares attacked by pieces of this side
        count = 0
        for sq in chess.SQUARES:
            if board.is_attacked_by(side, sq):
                count += 1
        return float(count)

    def king_safety(side: bool) -> float:
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
    def hanging_pieces(side: bool) -> float:
        # Count pieces that are attacked but not defended
        count = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if (
                piece
                and piece.color == side
                and board.is_attacked_by(not side, sq)
                and not board.is_attacked_by(side, sq)
            ):
                count += 1
        return float(count)

    def bishop_pair(side: bool) -> float:
        # Check if side has both bishops
        bishops = board.pieces(chess.BISHOP, side)
        return 1.0 if len(bishops) >= 2 else 0.0

    def rook_on_7th(side: bool) -> float:
        # Count rooks on the 7th rank (for white) or 2nd rank (for black)
        target_rank = 6 if side == chess.WHITE else 1
        count = 0
        for sq in board.pieces(chess.ROOK, side):
            if chess.square_rank(sq) == target_rank:
                count += 1
        return float(count)

    def king_pawn_shield(side: bool) -> float:
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

    # Add Syzygy tablebase features (if available)
    syzygy_path = os.environ.get("SYZYGY_PATH")
    if syzygy_path and RUST_AVAILABLE:
        try:
            # We initialize a temporary tablebase if not cached?
            # Better to cache it globally to avoid reloading files.
            if _SYZYGY_TB is None:
                _SYZYGY_TB = SyzygyTablebase(syzygy_path)

            if len(board.piece_map()) <= 7:
                wdl = _SYZYGY_TB.probe_wdl(board.fen())
                dtz = _SYZYGY_TB.probe_dtz(board.fen())
                if wdl is not None:
                    # Map WDL to -1..1 or similar. Syzygy WDL is usually -2..2
                    feats["syzygy_wdl"] = float(wdl) / 2.0
                if dtz is not None:
                    # DTZ can be large; normalize or just cap
                    feats["syzygy_dtz"] = float(dtz) / 100.0
        except Exception:  # noqa: S110
            pass

    # Add engine-based dynamic probes for better move ranking
    def sf_eval_shallow(engine: Any, board: chess.Board, depth: int = 6) -> float:
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

    def hanging_after_reply_real(
        engine: Any, board: chess.Board, depth: int = 6
    ) -> Tuple[int, int, int]:
        """Real hanging-after-reply with engine analysis (or Rust optimization)"""
        try:
            reply = None
            if RUST_AVAILABLE:
                try:
                    # Rust search uses TT + pruning; depth 8 is safe.
                    rust_depth = min(depth, 8)
                    uci = find_best_reply(board.fen(), rust_depth)
                    if uci:
                        reply = chess.Move.from_uci(uci)
                except Exception:  # noqa: S110
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

    def best_forcing_swing_real(
        engine: Any, board: chess.Board, d_base: int = 6, k_max: int = 12
    ) -> float:
        """Real forcing swing with eval differences (or Rust optimization)"""
        if RUST_AVAILABLE:
            try:
                # Rust search uses TT + pruning; depth 8 is safe.
                rust_depth = min(d_base, 8)
                return float(calculate_forcing_swing(board.fen(), rust_depth))
            except Exception:  # noqa: S110
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
    def outposts(side: bool) -> float:
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

    def batteries(side: bool) -> float:  # noqa: C901
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

    def pawn_structure(side: bool) -> float:
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
    def safe_mobility(side: bool) -> float:
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

    def rook_on_open_file(side: bool) -> float:
        count: float = 0.0
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

    def backward_pawns(side: bool) -> float:
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

    def connected_rooks(side: chess.Color) -> float:
        """Detect whether a side's rooks are connected (on the same rank with no pieces between).

        Connected rooks reinforce each other on open files and ranks,
        a key positional factor that the feature set was previously missing.
        """
        rooks = list(board.pieces(chess.ROOK, side))
        if len(rooks) < 2:
            return 0.0
        r0, r1 = rooks[0], rooks[1]
        if chess.square_rank(r0) != chess.square_rank(r1):
            return 0.0
        # Check no pieces between the two rooks on their shared rank
        lo, hi = min(r0, r1), max(r0, r1)
        for sq in range(lo + 1, hi):
            if board.piece_at(sq) is not None:
                return 0.0
        return 1.0

    feats["connected_rooks_us"] = connected_rooks(board.turn)
    feats["connected_rooks_them"] = connected_rooks(not board.turn)
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

    def pst_value(side: bool) -> float:
        """Phase-interpolated PST score.

        Uses a continuous phase factor (0.0 = endgame, 1.0 = opening)
        to smoothly blend middlegame and endgame king tables, giving
        more accurate positional scores in transitional positions.
        """
        score = 0.0
        # Continuous phase factor: 14 non-pawn/king pieces = 1.0 (opening).
        pf = min(feats["phase"] / 14.0, 1.0)

        for pt, table in [
            (chess.PAWN, PST_PAWN),
            (chess.KNIGHT, PST_KNIGHT),
            (chess.BISHOP, PST_BISHOP),
            (chess.ROOK, PST_ROOK),
            (chess.QUEEN, PST_QUEEN),
        ]:
            if table is None:
                continue  # type: ignore[unreachable]
            for sq in board.pieces(pt, side):
                vis_r = (
                    7 - chess.square_rank(sq)
                    if side == chess.WHITE
                    else chess.square_rank(sq)
                )
                vis_c = chess.square_file(sq)
                score += table[vis_r * 8 + vis_c]

        # King: smoothly interpolate between MG and EG tables.
        for sq in board.pieces(chess.KING, side):
            vis_r = (
                7 - chess.square_rank(sq)
                if side == chess.WHITE
                else chess.square_rank(sq)
            )
            vis_c = chess.square_file(sq)
            idx = vis_r * 8 + vis_c
            mg = PST_KING_MG[idx]
            eg = PST_KING_EG[idx]
            score += pf * mg + (1.0 - pf) * eg

        return float(score) / 100.0  # Standardize scale

    def pinned_pieces(side: bool) -> float:
        # Count absolutely pinned pieces
        count = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == side and board.is_pinned(side, sq):
                count += 1
        return float(count)

    feats["pst_us"] = pst_value(board.turn)
    feats["pst_them"] = pst_value(not board.turn)
    feats["pinned_us"] = pinned_pieces(board.turn)
    feats["pinned_them"] = pinned_pieces(not board.turn)

    # ── New explainability features ──────────────────────────────────

    def threats(side: bool) -> float:
        """Count attacks on higher-value enemy pieces.

        A threat is an attack by a lower-value piece on a higher-value
        enemy piece.  Captures the "why this move creates threats" signal
        that helps the surrogate predict Stockfish evaluation deltas.
        """
        count = 0
        them = not side
        pv = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == them and piece.piece_type != chess.KING:
                victim_val = pv.get(piece.piece_type, 0)
                for a_sq in board.attackers(side, sq):
                    attacker = board.piece_at(a_sq)
                    if attacker:
                        attacker_val = pv.get(attacker.piece_type, 0)
                        if attacker_val < victim_val:
                            count += 1
        return float(count)

    feats["threats_us"] = threats(board.turn)
    feats["threats_them"] = threats(not board.turn)

    def doubled_pawns(side: bool) -> float:
        """Count extra pawns on files with 2+ own pawns.

        Doubled pawns are a well-known structural weakness; this
        feature helps the surrogate distinguish positions where pawn
        structure quality drives the Stockfish evaluation.
        """
        my_pawns = board.pieces(chess.PAWN, side)
        count = 0
        for f in range(8):
            pawns_on_file = sum(1 for sq in my_pawns if chess.square_file(sq) == f)
            if pawns_on_file >= 2:
                count += pawns_on_file - 1
        return float(count)

    feats["doubled_pawns_us"] = doubled_pawns(board.turn)
    feats["doubled_pawns_them"] = doubled_pawns(not board.turn)

    def space(side: bool) -> float:
        """Squares attacked/controlled in the opponent's half.

        Space advantage captures positional squeezes that Stockfish
        values highly but pure material counts miss.
        """
        controlled: set = set()
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == side:
                for a_sq in board.attacks(sq):
                    controlled.add(a_sq)
        # Opponent's half: ranks 4-7 for White, ranks 0-3 for Black
        if side == chess.WHITE:
            opp_half = {s for s in chess.SQUARES if chess.square_rank(s) >= 4}
        else:
            opp_half = {s for s in chess.SQUARES if chess.square_rank(s) <= 3}
        return float(len(controlled & opp_half))

    feats["space_us"] = space(board.turn)
    feats["space_them"] = space(not board.turn)

    def king_tropism(side: bool) -> float:
        """Sum of (7 - distance) for non-pawn, non-king pieces to enemy
        king.

        Captures attack buildup: pieces clustered near the enemy king
        often signal dangerous initiative that Stockfish reflects in
        its evaluation.
        """
        them = not side
        enemy_king_sq = board.king(them)
        if enemy_king_sq is None:
            return 0.0
        tropism = 0.0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if (
                piece
                and piece.color == side
                and piece.piece_type not in (chess.KING, chess.PAWN)
            ):
                dist = chess.square_distance(sq, enemy_king_sq)
                tropism += 7.0 - dist
        return tropism

    feats["king_tropism_us"] = king_tropism(board.turn)
    feats["king_tropism_them"] = king_tropism(not board.turn)

    def pawn_chain_count(side: bool) -> float:
        """Pawns defended by another friendly pawn.

        Pawn chains provide structural integrity; this feature helps
        the surrogate model understand why Stockfish prefers positions
        with solid pawn structures.
        """
        my_pawns = board.pieces(chess.PAWN, side)
        count = 0
        for sq in my_pawns:
            # Squares that would be attacked by an *opponent* pawn on sq
            # are exactly the squares where a friendly pawn defends sq.
            rank = chess.square_rank(sq)
            f = chess.square_file(sq)
            # A pawn on `sq` is defended if a same-color pawn sits on an
            # adjacent file one rank behind.
            behind_rank = rank - 1 if side == chess.WHITE else rank + 1
            if 0 <= behind_rank <= 7:
                for adj_f in (f - 1, f + 1):
                    if 0 <= adj_f <= 7:
                        def_sq = chess.square(adj_f, behind_rank)
                        p = board.piece_at(def_sq)
                        if p and p.piece_type == chess.PAWN and p.color == side:
                            count += 1
                            break
        return float(count)

    feats["pawn_chain_us"] = pawn_chain_count(board.turn)
    feats["pawn_chain_them"] = pawn_chain_count(not board.turn)

    # ── SEE (Static Exchange Evaluation) features ─────────────────
    #
    # Simulates capture sequences on a target square to determine
    # net material gain/loss, giving far more accurate tactical
    # assessment than the binary "attacked and undefended" check.

    _see_piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    # Piece types ordered by ascending value for least-valuable-attacker.
    _PIECE_ORDER = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    def _least_valuable_attacker(sq: int, side: bool) -> Optional[int]:
        """Return the square of the least-valuable attacker of *sq*
        belonging to *side*, or ``None`` if there is no attacker.

        Iterates piece types in ascending value order so the first
        hit is always the cheapest attacker — the standard SEE
        convention.
        """
        for pt in _PIECE_ORDER:
            attackers = board.attackers(side, sq)
            for a_sq in attackers:
                if board.piece_type_at(a_sq) == pt:
                    return a_sq
        return None

    def _see(target: int, attacker_sq: int) -> int:
        """Static Exchange Evaluation for a capture on *target*
        initiated from *attacker_sq*.

        Simulates alternating least-valuable recaptures and returns
        the net material gain (positive = winning, negative = losing)
        from the initial attacker's perspective.
        """
        attacker_pt = board.piece_type_at(attacker_sq)
        victim_pt = board.piece_type_at(target)
        if attacker_pt is None or victim_pt is None:
            return 0

        gain = [0] * 33
        depth = 0
        gain[0] = _see_piece_values.get(victim_pt, 0)
        current_val = _see_piece_values.get(attacker_pt, 0)
        side = board.color_at(attacker_sq)
        if side is None:
            return 0

        # Temporarily remove pieces to reveal x-ray attacks.
        removed: list = [attacker_sq]
        board.remove_piece_at(attacker_sq)
        board.set_piece_at(target, chess.Piece(attacker_pt, side))

        while True:
            depth += 1
            side = not side
            gain[depth] = current_val - gain[depth - 1]

            if max(-gain[depth - 1], gain[depth]) < 0:
                break

            a_sq = _least_valuable_attacker(target, side)
            if a_sq is None:
                break

            pt = board.piece_type_at(a_sq)
            if pt is None:
                break
            current_val = _see_piece_values.get(pt, 0)
            removed.append((a_sq, board.piece_at(a_sq)))
            board.remove_piece_at(a_sq)
            board.set_piece_at(target, chess.Piece(pt, side))

        # NOTE: Board state is restored by _see_safe() via board.copy().

        # Propagate backwards
        while depth > 1:
            depth -= 1
            gain[depth - 1] = -(max(-gain[depth - 1], gain[depth]))

        return gain[0]

    def _see_safe(target: int, attacker_sq: int) -> int:
        """Copy-safe wrapper around SEE that preserves board state."""
        saved = board.copy()
        result = _see(target, attacker_sq)
        # Restore by copying back (board is local to the function).
        for sq in chess.SQUARES:
            p = saved.piece_at(sq)
            if p != board.piece_at(sq):
                if p is None:
                    board.remove_piece_at(sq)
                else:
                    board.set_piece_at(sq, p)
        return result

    def see_features(side: bool) -> Tuple[float, float]:
        """Compute SEE advantage and vulnerability for *side*.

        Returns ``(advantage, vulnerability)`` where advantage is the
        sum of positive SEE scores (in pawns) for captures our side
        can initiate, and vulnerability is the count of our pieces
        that the opponent can profitably capture.
        """
        them = not side
        advantage = 0.0
        vulnerability = 0.0

        # Advantage: positive-SEE captures we can make.
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == them and piece.piece_type != chess.KING:
                a_sq = _least_valuable_attacker(sq, side)
                if a_sq is not None:
                    val = _see_safe(sq, a_sq)
                    if val > 0:
                        advantage += val / 100.0

        # Vulnerability: our pieces the opponent can profitably capture.
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == side and piece.piece_type != chess.KING:
                a_sq = _least_valuable_attacker(sq, them)
                if a_sq is not None:
                    val = _see_safe(sq, a_sq)
                    if val > 0:
                        vulnerability += 1.0

        return advantage, vulnerability

    see_adv_us, see_vuln_us = see_features(board.turn)
    see_adv_them, see_vuln_them = see_features(not board.turn)
    feats["see_advantage_us"] = see_adv_us
    feats["see_advantage_them"] = see_adv_them
    feats["see_vulnerability_us"] = see_vuln_us
    feats["see_vulnerability_them"] = see_vuln_them

    # Add engine probes (callables)
    feats["_engine_probes"] = {
        "hanging_after_reply": hanging_after_reply_real,
        "best_forcing_swing": best_forcing_swing_real,
        "sf_eval_shallow": sf_eval_shallow,
    }

    return feats
