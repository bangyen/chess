#!/usr/bin/env python3
import argparse
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.linear_model import Lasso
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
except Exception:
    print(
        "scikit-learn is required. Install with: pip install scikit-learn",
        file=sys.stderr,
    )
    raise

try:
    from tqdm import tqdm
except Exception:
    print(
        "tqdm is required for progress bars. Install with: pip install tqdm",
        file=sys.stderr,
    )
    raise

try:
    import chess
    import chess.engine
    import chess.pgn
except Exception:
    print(
        "python-chess is required. Install with: pip install python-chess",
        file=sys.stderr,
    )
    raise


# -----------------------------
# Utility: load feature module
# -----------------------------
def load_feature_module(path: str):
    spec = importlib.util.spec_from_file_location("user_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load features module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "extract_features"):
        raise RuntimeError(
            "Feature module must define extract_features(board) -> Dict[str, number|bool]"
        )
    return mod


# -----------------------------
# Baseline features (optional)
# -----------------------------
def baseline_extract_features(board: "chess.Board") -> Dict[str, float]:
    """Small, fast, interpretable baseline feature set.
    This is intentionally compact; use your own richer set for best results.
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

    feats["hanging_us"] = hanging_pieces(board.turn)
    feats["hanging_them"] = hanging_pieces(not board.turn)

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

    # Add engine-based features (these will be computed during move ranking)
    # For now, add simplified versions for the base features
    feats["king_ring_pressure_us"] = king_ring_pressure_real(board, board.turn)
    feats["king_ring_pressure_them"] = king_ring_pressure_real(board, not board.turn)

    # Store functions for later use in move ranking
    feats["_engine_probes"] = {
        "hanging_after_reply": hanging_after_reply_real,
        "best_forcing_swing": best_forcing_swing_real,
        "sf_eval_shallow": sf_eval_shallow,
    }

    return feats


# -----------------------------
# Stockfish helpers
# -----------------------------
@dataclass
class SFConfig:
    engine_path: str
    depth: int = 16
    movetime: int = 0
    multipv: int = 3
    threads: int = 1


def sf_open(cfg: SFConfig):
    engine = chess.engine.SimpleEngine.popen_uci(cfg.engine_path)
    engine.configure({"Threads": cfg.threads})
    return engine


def sf_eval(engine, board: "chess.Board", cfg: SFConfig) -> float:
    limit = (
        chess.engine.Limit(depth=cfg.depth)
        if cfg.movetime == 0
        else chess.engine.Limit(time=cfg.movetime / 1000.0)
    )
    info = engine.analyse(board, limit=limit, multipv=1)
    # engine.analyse returns a list when multipv > 1, but a dict when multipv=1
    if isinstance(info, list):
        info = info[0]
    score = info["score"].pov(board.turn)
    cp = score.score(mate_score=100000)
    # Clip mate scores to prevent instability
    return float(max(-1000, min(1000, cp)))


def sf_top_moves(
    engine, board: "chess.Board", cfg: SFConfig
) -> List[Tuple[chess.Move, float]]:
    limit = (
        chess.engine.Limit(depth=cfg.depth)
        if cfg.movetime == 0
        else chess.engine.Limit(time=cfg.movetime / 1000.0)
    )
    infos = engine.analyse(board, limit=limit, multipv=cfg.multipv)
    out = []
    for d in infos:
        pv = d.get("pv", [])
        if not pv:
            continue
        move = pv[0]
        score = d["score"].pov(board.turn).score(mate_score=100000)
        out.append((move, float(score)))
    return out


# -----------------------------
# Position sampling
# -----------------------------
def sample_positions_from_pgn(
    path: str, max_positions: int, ply_skip: int = 8
) -> List["chess.Board"]:
    boards = []
    print(f"Sampling positions from PGN file: {path}")
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            plies = 0
            for move in game.mainline_moves():
                board.push(move)
                plies += 1
                if plies % ply_skip == 0 and not board.is_game_over():
                    boards.append(board.copy(stack=False))
                    if len(boards) >= max_positions:
                        print(f"Sampled {len(boards)} positions from PGN")
                        return boards
    print(f"Sampled {len(boards)} positions from PGN")
    return boards[:max_positions]


def sample_random_positions(n: int, max_random_plies: int = 24) -> List["chess.Board"]:
    boards = []
    print(f"Generating {n} random positions...")
    for _ in tqdm(range(n), desc="Generating positions"):
        b = chess.Board()
        # play random but legal moves to get middlegame-ish positions
        plies = random.randint(10, max_random_plies)
        for __ in range(plies):
            if b.is_game_over():
                break
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(random.choice(moves))
        if not b.is_game_over():
            boards.append(b.copy(stack=False))
    print(f"Generated {len(boards)} valid positions")
    return boards


# -----------------------------
# Metrics
# -----------------------------
def kendall_tau(rank_a: List[int], rank_b: List[int]) -> float:
    # Simple Kendall tau-b approximation for small k (k<=5)
    assert len(rank_a) == len(rank_b)
    n = len(rank_a)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = np.sign(rank_a[i] - rank_a[j])
            db = np.sign(rank_b[i] - rank_b[j])
            if da == db:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom > 0 else 0.0


# Passed-pawn momentum helpers
PIECE_VAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.1,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _rank_distance(pawn_sq: int, color: bool) -> int:
    r = chess.square_rank(pawn_sq)
    return int(
        (7 - r) if color == chess.WHITE else r
    )  # plies to 8th/1st rank in ranks, not moves


def _rook_behind_passer(board: chess.Board, pawn_sq: int, color: bool) -> int:
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
    """Enemy pawns on same/adjacent files ahead within 2 ranks."""
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
    """Is the square directly in front occupied by enemy piece (classic blockade)"""
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
    """No enemy pawn stoppers nearby and not blockaded right now."""
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
    quiet, capture = 0, 0
    for mv in board.legal_moves:
        if board.gives_check(mv):
            if board.is_capture(mv):
                capture += 1
            else:
                quiet += 1
    return {"d_quiet_checks": quiet, "d_capture_checks": capture}


def confinement_count(board: chess.Board, constrained_side: bool):
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
    us = base.turn
    return {
        "d_confinement": (
            confinement_count(after_reply, not us) - confinement_count(base, not us)
        )
        - (confinement_count(after_reply, us) - confinement_count(base, us))
    }


@dataclass
class AuditResult:
    r2: float
    tau_mean: float
    tau_covered: int
    n_tau: int
    local_faithfulness: float
    local_faithfulness_decisive: float
    sparsity_mean: float
    coverage_ratio: float
    stable_features: List[str]
    top_features_by_coef: List[Tuple[str, float]]


def audit_feature_set(
    boards: List["chess.Board"],
    engine,
    cfg: SFConfig,
    extract_features_fn,
    multipv_for_ranking: int = 3,
    test_size: float = 0.25,
    l1_alpha: float = 0.01,
    gap_threshold_cp: float = 50.0,
    attribution_topk: int = 5,
    stability_bootstraps: int = 20,
    stability_thresh: float = 0.7,
) -> AuditResult:

    # 1) Collect dataset (X, y) for fidelity (move delta-level)
    print("Collecting move deltas for training...")
    feature_names = None
    X = []
    y = []

    for b in tqdm(boards, desc="Move delta collection"):
        # Get base evaluation
        base_eval = sf_eval(engine, b, cfg)

        # Store base board for delta computation
        base_board = b.copy()

        # Get top moves from Stockfish
        top_moves = sf_top_moves(engine, b, cfg)

        for move, _ in top_moves:
            # Push the move
            b.push(move)

            # Get Stockfish's best reply to this move
            reply_info = engine.analyse(
                b, chess.engine.Limit(depth=cfg.depth), multipv=1
            )
            if isinstance(reply_info, list):
                reply_info = reply_info[0]
            reply_move = reply_info.get("pv", [None])[0]

            if reply_move is not None:
                # Push the reply
                b.push(reply_move)

                # Get evaluation after move → best reply
                after_reply_eval = sf_eval(engine, b, cfg)

                # Calculate reply-consistent delta
                delta_eval = after_reply_eval - base_eval

                # Get features after move → best reply
                after_reply_feats = extract_features_fn(b)
                after_reply_probes = after_reply_feats.pop("_engine_probes", {})
                after_reply_feats = {
                    k: (1.0 if isinstance(v, bool) and v else float(v))
                    for k, v in after_reply_feats.items()
                }

                # Add engine-based features for training
                if after_reply_probes:
                    # Add hanging after reply features
                    hang_cnt, hang_max_val, hang_near_king = after_reply_probes[
                        "hanging_after_reply"
                    ](engine, b, depth=6)
                    after_reply_feats["hang_cnt"] = hang_cnt
                    after_reply_feats["hang_max_val"] = hang_max_val
                    after_reply_feats["hang_near_king"] = hang_near_king

                    # Add forcing swing
                    forcing_swing = after_reply_probes["best_forcing_swing"](
                        engine, b, d_base=6, k_max=12
                    )
                    after_reply_feats["forcing_swing"] = forcing_swing

                # Add passed-pawn momentum delta features
                pp_delta = passed_pawn_momentum_delta(base_board, b)
                after_reply_feats.update(pp_delta)

                # Add checkability delta features
                base_check = checkability_now(base_board)
                after_check = checkability_now(b)
                check_delta = {
                    "d_quiet_checks": after_check["d_quiet_checks"]
                    - base_check["d_quiet_checks"],
                    "d_capture_checks": after_check["d_capture_checks"]
                    - base_check["d_capture_checks"],
                }
                after_reply_feats.update(check_delta)

                # Add confinement delta features
                conf_delta = confinement_delta(base_board, b)
                after_reply_feats.update(conf_delta)

                # Calculate delta features
                if feature_names is None:
                    feature_names = list(after_reply_feats.keys())
                else:
                    # keep common keys only
                    feature_names = [k for k in feature_names if k in after_reply_feats]

                # For delta training, we use the after-reply features directly
                # (they represent the change from the base state)
                X.append(after_reply_feats)
                y.append(delta_eval)

                # Pop the reply
                b.pop()

            # Pop the move
            b.pop()

    # realign feature vectors to common feature set
    feature_names = feature_names or []
    X_mat = np.array(
        [[float(x.get(k, 0.0)) for k in feature_names] for x in X], dtype=float
    )
    y = np.array(y, dtype=float)

    # Train/test split on move deltas
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y, test_size=test_size, random_state=42
    )

    # For testing, we need to split boards separately
    # Since we have move deltas, we'll use a subset of boards for testing
    n_test_boards = max(1, int(len(boards) * test_size))
    B_test = boards[:n_test_boards]
    # B_train = boards[n_test_boards:]  # Unused for now

    # 2) Fit linear L1 surrogate (signs are easier to read; you can swap models if you wish)
    # Normalize features to prevent extreme coefficients
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use cross-validation with a balanced alpha range
    from sklearn.linear_model import LassoCV

    # Use a more reasonable alpha range
    alphas = np.logspace(-2, 2, 20)  # 0.01 to 100.0
    model = LassoCV(
        cv=min(5, X_train.shape[0] // 10),
        random_state=42,
        max_iter=10000,
        alphas=alphas,
    )
    model.fit(X_train_scaled, y_train)
    print(f"Selected alpha: {model.alpha_:.4f}")

    # If still overfitting, use a minimum alpha
    if model.alpha_ < 1.0:
        model.alpha_ = 1.0
        model = Lasso(
            alpha=model.alpha_, fit_intercept=True, random_state=42, max_iter=10000
        )
        model.fit(X_train_scaled, y_train)
        print(f"Using minimum alpha: {model.alpha_:.4f}")

    # Fidelity
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    # 3) Move-ranking agreement (Kendall tau over MultiPV)
    print("Computing move ranking agreement...")
    taus = []
    covered = 0
    n_tau = 0
    for b in tqdm(B_test, desc="Move ranking analysis"):
        # Get engine top-k
        cfg_local = SFConfig(
            cfg.engine_path,
            cfg.depth,
            cfg.movetime,
            multipv=min(multipv_for_ranking, cfg.multipv),
            threads=cfg.threads,
        )
        cand = sf_top_moves(engine, b, cfg_local)
        if len(cand) < 2:
            continue
        # Build surrogate scores by evaluating features AFTER each candidate move
        # base_feats = extract_features_fn(b)  # Unused for now
        # base_vec = np.array(
        #     [float(base_feats.get(k, 0.0)) for k in feature_names], dtype=float
        # )  # Unused for now
        sf_scores = []
        sur_scores = []
        # ranks = list(range(len(cand)))  # Unused for now

        # Get base evaluation
        base_eval = sf_eval(engine, b, cfg)

        # Store base board for delta computation
        base_board = b.copy()

        for mv, _ in cand:
            # Push the move
            b.push(mv)

            # Get Stockfish's best reply to this move
            reply_info = engine.analyse(
                b, chess.engine.Limit(depth=cfg.depth), multipv=1
            )
            if isinstance(reply_info, list):
                reply_info = reply_info[0]
            reply_move = reply_info.get("pv", [None])[0]

            if reply_move is not None:
                # Push the reply
                b.push(reply_move)

                # Get evaluation after move → best reply
                after_reply_eval = sf_eval(engine, b, cfg)

                # Calculate reply-consistent delta
                delta_eval = after_reply_eval - base_eval

                # Get features after move → best reply
                feats_after_reply = extract_features_fn(b)
                feats_after_reply.pop("_engine_probes", {})
                feats_after_reply = {
                    k: (1.0 if isinstance(v, bool) and v else float(v))
                    for k, v in feats_after_reply.items()
                }

                # Add passed-pawn momentum delta features
                pp_delta = passed_pawn_momentum_delta(base_board, b)
                feats_after_reply.update(pp_delta)

                # Add checkability delta features
                base_check = checkability_now(base_board)
                after_check = checkability_now(b)
                check_delta = {
                    "d_quiet_checks": after_check["d_quiet_checks"]
                    - base_check["d_quiet_checks"],
                    "d_capture_checks": after_check["d_capture_checks"]
                    - base_check["d_capture_checks"],
                }
                feats_after_reply.update(check_delta)

                # Add confinement delta features
                conf_delta = confinement_delta(base_board, b)
                feats_after_reply.update(conf_delta)

                vec_after_reply = np.array(
                    [float(feats_after_reply.get(k, 0.0)) for k in feature_names],
                    dtype=float,
                )

                # Use linear model on after-reply features (delta training)
                vec_after_reply_scaled = scaler.transform(
                    vec_after_reply.reshape(1, -1)
                )
                sur_delta = float(model.predict(vec_after_reply_scaled)[0])

                sf_scores.append(delta_eval)
                sur_scores.append(sur_delta)

                # Pop the reply
                b.pop()

            # Pop the move
            b.pop()

        # Convert to rankings (higher is better)
        sf_rank = np.argsort(np.argsort(-np.array(sf_scores))).tolist()
        sur_rank = np.argsort(np.argsort(-np.array(sur_scores))).tolist()

        tau = kendall_tau(sf_rank, sur_rank)
        taus.append(tau)
        n_tau += 1
        if len(cand) >= 3:
            covered += 1

    tau_mean = float(np.mean(taus)) if taus else 0.0

    # 4) Local counterfactual faithfulness & sparsity/coverage of explanations
    print("Computing local faithfulness and sparsity...")
    faithful_hits = 0
    faithful_total = 0
    sparsity_counts = []
    coverage_hits = 0
    coverage_total = 0

    coef = model.coef_
    abs_coef = np.abs(coef)
    # define a minimal weight to count a feature toward coverage
    weight_threshold = (
        np.percentile(abs_coef[abs_coef > 0], 50) if np.any(abs_coef > 0) else 0.0
    )

    for b in tqdm(B_test, desc="Faithfulness analysis"):
        cand = sf_top_moves(engine, b, cfg)
        if len(cand) < 2:
            continue
        # pick best and second-best
        cand_sorted = sorted(cand, key=lambda x: -x[1])
        (best_mv, best_cp), (second_mv, second_cp) = cand_sorted[0], cand_sorted[1]
        if abs(best_cp - second_cp) < gap_threshold_cp:
            continue  # ambiguous; skip

        # Get base evaluation
        base_eval = sf_eval(engine, b, cfg)

        # Store base board for delta computation
        base_board = b.copy()

        # Evaluate best move → best reply
        b.push(best_mv)
        reply_info = engine.analyse(b, chess.engine.Limit(depth=cfg.depth), multipv=1)
        if isinstance(reply_info, list):
            reply_info = reply_info[0]
        reply_move = reply_info.get("pv", [None])[0]

        if reply_move is not None:
            b.push(reply_move)
            after_reply_eval_best = sf_eval(engine, b, cfg)
            delta_sf_best = after_reply_eval_best - base_eval

            # Get features after best move → best reply
            f_best = extract_features_fn(b)
            f_best.pop("_engine_probes", {})
            f_best = {
                k: (1.0 if isinstance(v, bool) and v else float(v))
                for k, v in f_best.items()
            }

            # Add delta features
            pp_delta = passed_pawn_momentum_delta(base_board, b)
            f_best.update(pp_delta)

            base_check = checkability_now(base_board)
            after_check = checkability_now(b)
            check_delta = {
                "d_quiet_checks": after_check["d_quiet_checks"]
                - base_check["d_quiet_checks"],
                "d_capture_checks": after_check["d_capture_checks"]
                - base_check["d_capture_checks"],
            }
            f_best.update(check_delta)

            conf_delta = confinement_delta(base_board, b)
            f_best.update(conf_delta)

            vec_best = np.array(
                [float(f_best.get(k, 0.0)) for k in feature_names], dtype=float
            )
            b.pop()
        else:
            delta_sf_best = 0.0
            vec_best = np.zeros(len(feature_names))
        b.pop()

        # Evaluate second move → best reply
        b.push(second_mv)
        reply_info = engine.analyse(b, chess.engine.Limit(depth=cfg.depth), multipv=1)
        if isinstance(reply_info, list):
            reply_info = reply_info[0]
        reply_move = reply_info.get("pv", [None])[0]

        if reply_move is not None:
            b.push(reply_move)
            after_reply_eval_second = sf_eval(engine, b, cfg)
            delta_sf_second = after_reply_eval_second - base_eval

            # Get features after second move → best reply
            f_second = extract_features_fn(b)
            f_second.pop("_engine_probes", {})
            f_second = {
                k: (1.0 if isinstance(v, bool) and v else float(v))
                for k, v in f_second.items()
            }

            # Add delta features
            pp_delta = passed_pawn_momentum_delta(base_board, b)
            f_second.update(pp_delta)

            base_check = checkability_now(base_board)
            after_check = checkability_now(b)
            check_delta = {
                "d_quiet_checks": after_check["d_quiet_checks"]
                - base_check["d_quiet_checks"],
                "d_capture_checks": after_check["d_capture_checks"]
                - base_check["d_capture_checks"],
            }
            f_second.update(check_delta)

            conf_delta = confinement_delta(base_board, b)
            f_second.update(conf_delta)

            vec_second = np.array(
                [float(f_second.get(k, 0.0)) for k in feature_names], dtype=float
            )
            b.pop()
        else:
            delta_sf_second = 0.0
            vec_second = np.zeros(len(feature_names))
        b.pop()

        # Use linear model on after-reply features (delta training)
        vec_best_scaled = scaler.transform(vec_best.reshape(1, -1))
        vec_second_scaled = scaler.transform(vec_second.reshape(1, -1))
        sur_best = float(model.predict(vec_best_scaled)[0])
        sur_second = float(model.predict(vec_second_scaled)[0])

        # Calculate decisive gap
        decisive_gap = abs(delta_sf_best - delta_sf_second)
        # is_decisive = decisive_gap >= 80.0  # Unused for now

        # Signed attribution for faithfulness
        # Use the best move's features for attribution
        contrib_best = coef * vec_best  # signed contributions
        contrib_second = coef * vec_second  # signed contributions

        # sparsity: count top contributors above a small fraction of total
        def sparsity(contrib):
            tot = np.sum(np.abs(contrib))
            if tot <= 1e-9:
                return 0
            contrib_sorted = np.sort(np.abs(contrib))[::-1]
            cum = 0.0
            k = 0
            for v in contrib_sorted:
                cum += v
                k += 1
                if cum >= 0.8 * tot:
                    break
            return k

        sp = sparsity(contrib_best)
        if sp > 0:
            sparsity_counts.append(sp)

        # coverage: at least 2 features with meaningful weight
        strong_feats: int = int(
            np.sum((np.abs(coef) >= weight_threshold) & (np.abs(contrib_best) > 0))
        )
        coverage_total += 1
        if strong_feats >= 2:
            coverage_hits += 1

        # faithfulness: do the top features (by signed contribution) align with eval direction?
        # Compare direction of (coef*delta) sum with SF eval diff sign
        dir_sur = float(np.sum(contrib_best) - np.sum(contrib_second))
        dir_sf = float(delta_sf_best - delta_sf_second)
        if dir_sur * dir_sf > 0:
            faithful_hits += 1
        faithful_total += 1

    local_faithfulness = (faithful_hits / faithful_total) if faithful_total else 0.0
    sparsity_mean = float(np.mean(sparsity_counts)) if sparsity_counts else 0.0
    coverage_ratio = (coverage_hits / coverage_total) if coverage_total else 0.0

    # Calculate decisive faithfulness (gap ≥ 80 cp)
    faithful_decisive_hits = 0
    faithful_decisive_total = 0
    for b in B_test:
        cand = sf_top_moves(engine, b, cfg)
        if len(cand) < 2:
            continue
        cand_sorted = sorted(cand, key=lambda x: -x[1])
        (best_mv, best_cp), (second_mv, second_cp) = cand_sorted[0], cand_sorted[1]
        if abs(best_cp - second_cp) < gap_threshold_cp:
            continue

        # Get base evaluation
        base_eval = sf_eval(engine, b, cfg)

        # Evaluate best move → best reply
        b.push(best_mv)
        reply_info = engine.analyse(b, chess.engine.Limit(depth=cfg.depth), multipv=1)
        if isinstance(reply_info, list):
            reply_info = reply_info[0]
        reply_move = reply_info.get("pv", [None])[0]

        if reply_move is not None:
            b.push(reply_move)
            after_reply_eval_best = sf_eval(engine, b, cfg)
            delta_sf_best = after_reply_eval_best - base_eval
            b.pop()
        else:
            delta_sf_best = 0.0
        b.pop()

        # Evaluate second move → best reply
        b.push(second_mv)
        reply_info = engine.analyse(b, chess.engine.Limit(depth=cfg.depth), multipv=1)
        if isinstance(reply_info, list):
            reply_info = reply_info[0]
        reply_move = reply_info.get("pv", [None])[0]

        if reply_move is not None:
            b.push(reply_move)
            after_reply_eval_second = sf_eval(engine, b, cfg)
            delta_sf_second = after_reply_eval_second - base_eval
            b.pop()
        else:
            delta_sf_second = 0.0
        b.pop()

        # Check if decisive gap
        decisive_gap = abs(delta_sf_best - delta_sf_second)
        if decisive_gap >= 80.0:
            # Get features for both moves
            b.push(best_mv)
            if reply_move is not None:
                b.push(reply_move)
                f_best = extract_features_fn(b)
                f_best.pop("_engine_probes", {})
                vec_best = np.array(
                    [float(f_best.get(k, 0.0)) for k in feature_names], dtype=float
                )
                b.pop()
            else:
                vec_best = np.zeros(len(feature_names))
            b.pop()

            b.push(second_mv)
            if reply_move is not None:
                b.push(reply_move)
                f_second = extract_features_fn(b)
                f_second.pop("_engine_probes", {})
                vec_second = np.array(
                    [float(f_second.get(k, 0.0)) for k in feature_names], dtype=float
                )
                b.pop()
            else:
                vec_second = np.zeros(len(feature_names))
            b.pop()

            # Use linear model on after-reply features
            vec_best_scaled = scaler.transform(vec_best.reshape(1, -1))
            vec_second_scaled = scaler.transform(vec_second.reshape(1, -1))
            sur_best = float(model.predict(vec_best_scaled)[0])
            sur_second = float(model.predict(vec_second_scaled)[0])

            # Check faithfulness
            if (sur_best > sur_second) == (delta_sf_best > delta_sf_second):
                faithful_decisive_hits += 1
            faithful_decisive_total += 1

    local_faithfulness_decisive = (
        (faithful_decisive_hits / faithful_decisive_total)
        if faithful_decisive_total
        else 0.0
    )

    # 5) Stability selection (L1 bootstraps): how often is a feature chosen (non-zero)
    if X_train.shape[0] >= 20:
        print("Running stability selection...")
        picks = np.zeros(len(feature_names), dtype=int)
        for b in tqdm(range(stability_bootstraps), desc="Stability selection"):
            idx = np.random.choice(
                X_train.shape[0], size=X_train.shape[0], replace=True
            )
            Xb = X_train_scaled[idx]
            yb = y_train[idx]
            m = Lasso(
                alpha=model.alpha_,
                fit_intercept=True,
                random_state=42 + b,
                max_iter=10000,
            )
            m.fit(Xb, yb)
            picks += m.coef_ != 0.0
        pick_freq = picks / stability_bootstraps
        stable_idx = np.where(pick_freq >= stability_thresh)[0].tolist()
    else:
        pick_freq = np.zeros(len(feature_names))
        stable_idx = []

    stable_features = [feature_names[i] for i in stable_idx]
    top_features = sorted(
        [(feature_names[i], float(coef[i])) for i in range(len(feature_names))],
        key=lambda x: -abs(x[1]),
    )[:15]

    return AuditResult(
        r2=float(r2),
        tau_mean=float(tau_mean),
        tau_covered=covered,
        n_tau=n_tau,
        local_faithfulness=float(local_faithfulness),
        local_faithfulness_decisive=float(local_faithfulness_decisive),
        sparsity_mean=float(sparsity_mean),
        coverage_ratio=float(coverage_ratio),
        stable_features=stable_features,
        top_features_by_coef=top_features,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Audit explainability of a chess feature set against Stockfish."
    )
    ap.add_argument(
        "--engine",
        type=str,
        default=os.environ.get("STOCKFISH_PATH", ""),
        help="Path to Stockfish binary",
    )
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Fixed search depth (set 0 to use movetime)",
    )
    ap.add_argument(
        "--movetime", type=int, default=0, help="Milliseconds per evaluation if depth=0"
    )
    ap.add_argument(
        "--multipv", type=int, default=3, help="Max MultiPV used for ranking metric"
    )
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument(
        "--pgn", type=str, default="", help="Optional PGN file to sample positions"
    )
    ap.add_argument(
        "--ply-skip",
        type=int,
        default=8,
        help="Keep every Nth ply when sampling from PGN",
    )
    ap.add_argument(
        "--features_module",
        type=str,
        default="",
        help="Path to a Python module that defines extract_features(board)",
    )
    ap.add_argument(
        "--baseline_features",
        action="store_true",
        help="Use built-in small baseline features",
    )
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="L1 regularization strength for surrogate",
    )
    ap.add_argument(
        "--gap",
        type=float,
        default=50.0,
        help="CP gap to treat best vs second as decisive for faithfulness",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.engine:
        print(
            "Please provide --engine PATH or set STOCKFISH_PATH env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = SFConfig(
        engine_path=args.engine,
        depth=args.depth if args.depth > 0 else 0,
        movetime=args.movetime if args.depth == 0 else 0,
        multipv=max(2, args.multipv),
        threads=args.threads,
    )

    # Positions
    if args.pgn:
        boards = sample_positions_from_pgn(
            args.pgn, args.positions, ply_skip=args.ply_skip
        )
        if len(boards) < args.positions:
            # supplement with randoms
            boards += sample_random_positions(args.positions - len(boards))
    else:
        boards = sample_random_positions(args.positions)
    if not boards:
        print("No positions sampled.", file=sys.stderr)
        sys.exit(1)

    # Feature extractor
    if args.baseline_features:
        extract_fn = baseline_extract_features
    elif args.features_module:
        mod = load_feature_module(args.features_module)
        extract_fn = mod.extract_features  # type: ignore
    else:
        print(
            "Provide --features_module PATH or use --baseline_features to run.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Engine
    engine = sf_open(cfg)

    try:
        res = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=args.multipv,
            test_size=args.test_size,
            l1_alpha=args.alpha,
            gap_threshold_cp=args.gap,
        )
    finally:
        engine.quit()

    # Report
    print("\n=== Explainability Audit Report ===")
    print(
        f"Positions: {len(boards)}  |  Depth: {cfg.depth or 'movetime'}  |  MultiPV: {cfg.multipv}"
    )
    print(f"Fidelity (Delta-R^2):          {res.r2:0.3f}")
    print(
        f"Move ranking (Kendall tau):    {res.tau_mean:0.3f}  (positions covered: {res.tau_covered}/{res.n_tau})"
    )
    print(
        f"Local faithfulness (best vs 2): {res.local_faithfulness*100:0.1f}% (gap ≥ {args.gap} cp)"
    )
    print(
        f"Local faithfulness (decisive): {res.local_faithfulness_decisive*100:0.1f}% (gap ≥ 80.0 cp)"
    )
    print(
        f"Sparsity (reasons to cover 80% contribution for best move): {res.sparsity_mean:0.2f}"
    )
    print(f"Coverage (≥2 strong reasons):  {res.coverage_ratio*100:0.1f}%")
    print("\nTop features by |coef|:")
    for name, coef in res.top_features_by_coef:
        print(f"  {name:30s}  coef={coef:+.4f}")
    if res.stable_features:
        print(f"\nStable features (picked ≥{100 * 0.7:.0f}% of bootstraps):")
        for name in res.stable_features:
            print(f"  - {name}")
    else:
        print("\nStable features: (none reached threshold)")

    print("\nGuidance:")
    print(
        " - Aim for Delta-R^2 ≥ 0.60 on mixed middlegames at depth ~16 as a healthy baseline."
    )
    print(" - Tau ≥ 0.45 for top-3 move ranking is decent; higher is better.")
    print(
        " - Local faithfulness ≥ 80% on decisive positions shows explanations track preferences."
    )
    print(" - Sparsity around 3–5 suggests crisp, narratable reasons.")
    print(
        " - Coverage ≥ 70% with ≥2 strong reasons means you can explain most positions.\n"
    )


if __name__ == "__main__":
    main()
