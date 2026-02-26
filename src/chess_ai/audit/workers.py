"""Worker processes for parallelizing move delta collection, ranking, and faithfulness analysis."""

from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import chess
import numpy as np

from ..engine import SFConfig
from ..engine.interface import sf_eval, sf_open, sf_top_moves
from .audit_enrichment import enrich_features

# Interaction term definitions
INTERACTION_PAIRS = [
    ("material_diff", "phase"),
    ("passed_us", "phase"),
    ("mobility_us", "phase"),
    ("batteries_us", "open_files_us"),
    ("batteries_us", "semi_open_us"),
    ("outposts_us", "phase"),
    ("bishop_pair_us", "open_files_us"),
    ("threats_us", "phase"),
    ("king_tropism_us", "phase"),
    ("space_us", "phase"),
    ("doubled_pawns_us", "phase"),
    # Delta-based interactions
    ("d_material_diff", "phase"),
    ("d_threats_us", "phase"),
    ("d_mobility_us", "phase"),
    ("d_hanging_them", "phase"),
    ("d_space_us", "phase"),
    ("d_king_ring_pressure_them", "phase"),
    ("d_passed_us", "phase"),
    ("d_rook_open_file_us", "phase"),
    ("d_forcing_swing", "phase"),
]


def _audit_worker_process(
    board_fen: str,
    cfg: SFConfig,
    extract_features_fn: Callable[[chess.Board], Dict[str, float]],
    interaction_pairs: List[Tuple[str, str]],
) -> List[Tuple[Dict[str, float], float, float]]:
    """Worker process for parallelizing move delta collection."""
    engine = sf_open(cfg)
    try:
        board = chess.Board(board_fen)
        base_eval = sf_eval(engine, board, cfg)
        base_board = board.copy()
        base_feats_raw = extract_features_fn(base_board)

        top_moves = sf_top_moves(engine, board, cfg)
        results = []

        for move, _ in top_moves:
            board.push(move)
            reply_res = engine.analyse(
                board, chess.engine.Limit(depth=cfg.depth), multipv=1
            )
            res_info = reply_res[0] if isinstance(reply_res, list) else reply_res
            reply_move = cast(Dict[str, Any], res_info).get("pv", [None])[0]

            if reply_move is not None:
                board.push(reply_move)
                after_reply_eval = sf_eval(engine, board, cfg)
                delta_eval = after_reply_eval - base_eval

                feats = enrich_features(
                    board,
                    base_board,
                    base_feats_raw,
                    extract_features_fn,
                    engine,
                    interaction_pairs,
                    {},
                    {},
                )
                results.append((feats, delta_eval, base_eval))
                board.pop()

            board.pop()
        return results
    finally:
        engine.quit()


def _ranking_worker_process(
    board_fen: str,
    cfg: SFConfig,
    cfg_local: SFConfig,
    extract_features_fn: Callable[[chess.Board], Dict[str, float]],
    interaction_pairs: List[Tuple[str, str]],
    feature_names: List[str],
    scaler: Any,
    model: Any,
) -> Optional[Tuple[List[float], List[float]]]:
    """Worker process for parallelizing move ranking analysis."""
    engine = sf_open(cfg)
    try:
        board = chess.Board(board_fen)
        cand = sf_top_moves(engine, board, cfg_local)
        if len(cand) < 2:
            return None

        sf_scores = []
        sur_scores = []
        base_eval = sf_eval(engine, board, cfg)
        base_board = board.copy()
        base_feats_raw = extract_features_fn(base_board)

        for mv, _ in cand:
            board.push(mv)
            reply_res = engine.analyse(
                board, chess.engine.Limit(depth=cfg.depth), multipv=1
            )
            res_info = reply_res[0] if isinstance(reply_res, list) else reply_res
            reply_move = cast(Dict[str, Any], res_info).get("pv", [None])[0]

            if reply_move is not None:
                board.push(reply_move)
                after_reply_eval = sf_eval(engine, board, cfg)
                delta_eval = after_reply_eval - base_eval

                feats = enrich_features(
                    board,
                    base_board,
                    base_feats_raw,
                    extract_features_fn,
                    engine,
                    interaction_pairs,
                    {},
                    {},
                )

                x_vec = np.array(
                    [float(feats.get(k, 0.0)) for k in feature_names], dtype=float
                )
                x_scaled = scaler.transform(x_vec.reshape(1, -1))
                sur_val = float(model.predict(x_scaled)[0])

                sf_scores.append(delta_eval)
                sur_scores.append(sur_val)
                board.pop()

            board.pop()

        if len(sf_scores) >= 2:
            return sf_scores, sur_scores
        return None
    finally:
        engine.quit()


def _faithfulness_worker_process(
    board_fen: str,
    cfg: SFConfig,
    extract_features_fn: Callable[[chess.Board], Dict[str, float]],
    interaction_pairs: List[Tuple[str, str]],
    feature_names: List[str],
    scaler: Any,
    model: Any,
    gap_threshold_cp: float,
    weight_threshold: float,
    coef: np.ndarray,
    abs_coef: np.ndarray,
) -> Optional[Tuple[int, int, int, List[int], int, int]]:
    """Worker process for parallelizing faithfulness analysis."""
    engine = sf_open(cfg)
    try:
        board = chess.Board(board_fen)
        cand = sf_top_moves(engine, board, cfg)
        if len(cand) < 2:
            return None

        cand_sorted = sorted(cand, key=lambda x: -x[1])
        (best_mv, best_cp), (second_mv, second_cp) = cand_sorted[0], cand_sorted[1]
        if abs(best_cp - second_cp) < gap_threshold_cp:
            return None

        base_eval = sf_eval(engine, board, cfg)
        base_board = board.copy()
        base_feats_raw = extract_features_fn(base_board)

        def _eval_move(mv: chess.Move) -> Tuple[float, np.ndarray]:
            board.push(mv)
            reply_res = engine.analyse(
                board, chess.engine.Limit(depth=cfg.depth), multipv=1
            )
            res_info = reply_res[0] if isinstance(reply_res, list) else reply_res
            reply_move = cast(Dict[str, Any], res_info).get("pv", [None])[0]
            if reply_move is not None:
                board.push(reply_move)
                after_eval = sf_eval(engine, board, cfg)
                delta_sf = after_eval - base_eval
                feats = enrich_features(
                    board,
                    base_board,
                    base_feats_raw,
                    extract_features_fn,
                    engine,
                    interaction_pairs,
                    {},
                    {},
                )
                vec = np.array(
                    [float(feats.get(k, 0.0)) for k in feature_names], dtype=float
                )
                board.pop()
            else:
                delta_sf = 0.0
                vec = np.zeros(len(feature_names))
            board.pop()
            return delta_sf, vec

        delta_sf_best, vec_best = _eval_move(best_mv)
        delta_sf_second, vec_second = _eval_move(second_mv)

        vec_best_scaled = scaler.transform(vec_best.reshape(1, -1))
        vec_second_scaled = scaler.transform(vec_second.reshape(1, -1))
        sur_best = float(model.predict(vec_best_scaled)[0])
        sur_second = float(model.predict(vec_second_scaled)[0])
        contrib_best = coef * vec_best_scaled[0]

        tot = np.sum(np.abs(contrib_best))
        sparsity_counts = []
        if tot > 1e-9:
            c_sorted = np.sort(np.abs(contrib_best))[::-1]
            cum = 0.0
            sp = 0
            for v in c_sorted:
                cum += v
                sp += 1
                if cum >= 0.8 * tot:
                    break
            if sp > 0:
                sparsity_counts.append(sp)

        strong_feats = int(
            np.sum((abs_coef >= weight_threshold) & (np.abs(contrib_best) > 0))
        )
        cov_hits = 1 if strong_feats >= 2 else 0
        cov_total = 1

        f_hits = (
            1 if (sur_best - sur_second) * (delta_sf_best - delta_sf_second) > 0 else 0
        )

        f_decisive_hits = 0
        f_decisive_total = 0
        if abs(delta_sf_best - delta_sf_second) >= 80.0:
            if (sur_best > sur_second) == (delta_sf_best > delta_sf_second):
                f_decisive_hits = 1
            f_decisive_total = 1

        return (
            f_hits,
            f_decisive_hits,
            f_decisive_total,
            sparsity_counts,
            cov_hits,
            cov_total,
        )
    finally:
        engine.quit()
