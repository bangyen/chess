"""Baseline feature extraction functions."""

import os
from typing import Any, Dict, Optional, Tuple

import chess
import chess.engine

from .engine_probes import (
    best_forcing_swing,
    hanging_after_reply,
    sf_eval_shallow,
)
from .pst_tables import pst_value
from .see_python import see_features

try:
    from chess_ai.rust_utils import (
        SyzygyTablebase,
        extract_features_rust,
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

            # Add Syzygy tablebase features (if available)
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

            feats["_engine_probes"] = {
                "hanging_after_reply": hanging_after_reply,
                "best_forcing_swing": best_forcing_swing,
                "sf_eval_shallow": sf_eval_shallow,
            }
            return feats

        except Exception:
            # Fallback to Python if Rust fails
            feats = {}

    # ── Python Implementation (Fallback) ─────────────────────────────
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
                for pt, w in weight.items():
                    if any(
                        board.piece_type_at(x) == pt
                        and board.color_at(x) == attacking_side
                        for x in board.attackers(attacking_side, sq)
                    ):
                        s += w
                        break

        phase = sum(
            len(board.pieces(pt, True)) + len(board.pieces(pt, False))
            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        )
        return s / max(6, phase)

    def passed_pawns(side: bool) -> float:
        count = 0
        for sq in board.pieces(chess.PAWN, side):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            is_passed = True

            for check_file in [file - 1, file, file + 1]:
                if check_file < 0 or check_file > 7:
                    continue
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

    def center_control(side: bool) -> float:
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        count = 0
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece and piece.color == side:
                count += 1
        return float(count)

    def piece_activity(side: bool) -> float:
        count = 0
        for sq in chess.SQUARES:
            if board.is_attacked_by(side, sq):
                count += 1
        return float(count)

    def king_safety(side: bool) -> float:
        king_sq = board.king(side)
        if king_sq is None:
            return 0.0
        safety = 0.0
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

    def hanging_pieces(side: bool) -> float:
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
        bishops = board.pieces(chess.BISHOP, side)
        return 1.0 if len(bishops) >= 2 else 0.0

    def rook_on_7th(side: bool) -> float:
        target_rank = 6 if side == chess.WHITE else 1
        count = 0
        for sq in board.pieces(chess.ROOK, side):
            if chess.square_rank(sq) == target_rank:
                count += 1
        return float(count)

    def king_pawn_shield(side: bool) -> float:
        ks = board.king(side)
        if ks is None:
            return 0.0
        file = chess.square_file(ks)
        rank = chess.square_rank(ks)

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
            if _SYZYGY_TB is None:
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

    # ── Advanced positional features ─────────────────────────────────

    def outposts(side: bool) -> float:
        count = 0
        knights = board.pieces(chess.KNIGHT, side)
        for sq in knights:
            rank = chess.square_rank(sq)
            rel_rank = rank if side == chess.WHITE else 7 - rank
            if rel_rank < 3 or rel_rank > 5:
                continue

            is_supported = False
            pawn_attacks = board.attackers(side, sq)
            for attacker in pawn_attacks:
                if board.piece_type_at(attacker) == chess.PAWN:
                    is_supported = True
                    break
            if not is_supported:
                continue

            if board.is_attacked_by(not side, sq):
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
        count = 0
        for i in range(8):
            pieces_on_file = []
            for r in range(8):
                sq = chess.square(i, r)
                p = board.piece_at(sq)
                if p and p.color == side and p.piece_type in [chess.ROOK, chess.QUEEN]:
                    pieces_on_file.append(p.piece_type)
            if len(pieces_on_file) >= 2:
                count += 1

            pieces_on_rank = []
            for f in range(8):
                sq = chess.square(f, i)
                p = board.piece_at(sq)
                if p and p.color == side and p.piece_type in [chess.ROOK, chess.QUEEN]:
                    pieces_on_rank.append(p.piece_type)
            if len(pieces_on_rank) >= 2:
                count += 1

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

    def safe_mobility(side: bool) -> float:
        if board.turn != side:
            board.turn = side
            moves = list(board.legal_moves)
            board.turn = not side
        else:
            moves = list(board.legal_moves)

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

        enemy_pawn_attacks = chess.SquareSet()
        for sq in board.pieces(chess.PAWN, opp):
            enemy_pawn_attacks |= board.attacks(sq)

        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)

            is_supported = False
            for f in [file - 1, file + 1]:
                if 0 <= f <= 7:
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
        """Detect whether a side's rooks are connected (on the same rank
        with no pieces between).

        Connected rooks reinforce each other on open files and ranks,
        a key positional factor that the feature set was previously missing.
        """
        rooks = list(board.pieces(chess.ROOK, side))
        if len(rooks) < 2:
            return 0.0
        r0, r1 = rooks[0], rooks[1]
        if chess.square_rank(r0) != chess.square_rank(r1):
            return 0.0
        lo, hi = min(r0, r1), max(r0, r1)
        for sq in range(lo + 1, hi):
            if board.piece_at(sq) is not None:
                return 0.0
        return 1.0

    feats["connected_rooks_us"] = connected_rooks(board.turn)
    feats["connected_rooks_them"] = connected_rooks(not board.turn)

    # ── PST, pins, and explainability features ───────────────────────

    feats["pst_us"] = pst_value(board, board.turn, feats["phase"])
    feats["pst_them"] = pst_value(board, not board.turn, feats["phase"])

    def pinned_pieces(side: bool) -> float:
        count = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == side and board.is_pinned(side, sq):
                count += 1
        return float(count)

    feats["pinned_us"] = pinned_pieces(board.turn)
    feats["pinned_them"] = pinned_pieces(not board.turn)

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
            rank = chess.square_rank(sq)
            f = chess.square_file(sq)
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

    # ── SEE features (using extracted module) ────────────────────────

    see_adv_us, see_vuln_us = see_features(board, board.turn)
    see_adv_them, see_vuln_them = see_features(board, not board.turn)
    feats["see_advantage_us"] = see_adv_us
    feats["see_advantage_them"] = see_adv_them
    feats["see_vulnerability_us"] = see_vuln_us
    feats["see_vulnerability_them"] = see_vuln_them

    # Engine probes (callables)
    feats["_engine_probes"] = {
        "hanging_after_reply": hanging_after_reply,
        "best_forcing_swing": best_forcing_swing,
        "sf_eval_shallow": sf_eval_shallow,
    }

    return feats
