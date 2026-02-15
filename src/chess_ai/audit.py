"""Main audit functionality."""

import concurrent.futures
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import chess
import numpy as np
import numpy.typing as npt

from .audit_enrichment import enrich_features
from .engine import SFConfig, sf_eval, sf_top_moves
from .metrics.kendall import kendall_tau
from .tree_surrogate import TreeSurrogate
from .utils.math import cp_to_winrate

logger = logging.getLogger(__name__)

# Suppress sklearn convergence warnings for small datasets
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    warnings.filterwarnings("ignore", message=".*convergence.*", module="sklearn")
    warnings.filterwarnings("ignore", module="sklearn")
except Exception:
    raise

try:
    from tqdm import tqdm
except Exception:
    raise


# _cp_to_winrate has been factored into utils.math.cp_to_winrate
_cp_to_winrate = cp_to_winrate


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


@dataclass
class AuditResult:
    """Results from feature set audit."""

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


def audit_feature_set(  # noqa: C901
    boards: List["chess.Board"],
    engine: Any,
    cfg: SFConfig,
    extract_features_fn: Callable[..., Dict[str, Any]],
    multipv_for_ranking: int = 3,
    test_size: float = 0.25,
    l1_alpha: float = 0.01,
    gap_threshold_cp: float = 50.0,
    attribution_topk: int = 5,
    stability_bootstraps: int = 20,
    stability_thresh: float = 0.7,
) -> AuditResult:
    """Audit a feature set for explainability.

    Args:
        boards: List of chess positions to analyze
        engine: Stockfish engine instance
        cfg: Stockfish configuration
        extract_features_fn: Function to extract features from a board
        multipv_for_ranking: Number of top moves to consider for ranking
        test_size: Fraction of data to use for testing
        l1_alpha: L1 regularization strength
        gap_threshold_cp: CP gap threshold for decisive positions
        attribution_topk: Number of top features to show in attribution
        stability_bootstraps: Number of bootstrap samples for stability
        stability_thresh: Threshold for feature stability

    Returns:
        AuditResult with all metrics
    """
    # -- FEN-keyed caches --
    _eval_cache: Dict[str, float] = {}
    _top_moves_cache: Dict[str, List[Tuple[chess.Move, float]]] = {}
    _analyse_cache: Dict[str, chess.Move] = {}

    def _cached_sf_eval(eng: Any, board: chess.Board, c: SFConfig) -> float:
        key = board.fen()
        if key not in _eval_cache:
            _eval_cache[key] = sf_eval(eng, board, c)
        return _eval_cache[key]

    def _cached_sf_top_moves(
        eng: Any, board: chess.Board, c: SFConfig
    ) -> List[Tuple[chess.Move, float]]:
        key = f"{board.fen()}|mpv{c.multipv}"
        if key not in _top_moves_cache:
            _top_moves_cache[key] = sf_top_moves(eng, board, c)
        return _top_moves_cache[key]

    def _cached_best_reply(
        eng: Any, board: chess.Board, depth: int
    ) -> Optional[chess.Move]:
        key = board.fen()
        if key in _analyse_cache:
            return _analyse_cache[key]
        reply_info = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
        info = reply_info[0] if isinstance(reply_info, list) else reply_info
        reply_move = cast(Dict[str, Any], info).get("pv", [None])[0]
        _analyse_cache[key] = reply_move
        return cast(Optional[chess.Move], reply_move)

    def _enrich(
        board: chess.Board, base_board: chess.Board, base_feats_raw: Dict[str, float]
    ) -> Dict[str, float]:
        return enrich_features(
            board,
            base_board,
            base_feats_raw,
            extract_features_fn,
            engine,
            INTERACTION_PAIRS,
            {},
            {},
        )

    logger.info("Collecting move deltas for training using %d workers...", cfg.threads)

    # We parallelize the collection of move deltas.
    # Each board is processed by a worker which opens its own engine.
    X_raw = []
    y_raw: List[float] = []
    y_base_evals_raw: List[float] = []

    if cfg.threads > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cfg.threads
        ) as audit_executor:
            audit_futures = [
                audit_executor.submit(
                    _audit_worker_process,
                    board_fen=b.fen(),
                    cfg=cfg,
                    extract_features_fn=extract_features_fn,
                    interaction_pairs=INTERACTION_PAIRS,
                )
                for b in boards
            ]

            for future in tqdm(
                concurrent.futures.as_completed(audit_futures),
                total=len(audit_futures),
                desc="Move delta collection",
            ):
                worker_results = future.result()
                for feats, delta_eval, base_eval in worker_results:
                    X_raw.append(feats)
                    y_raw.append(delta_eval)
                    y_base_evals_raw.append(base_eval)
    else:
        # Sequential fallback for threads=1 or non-pickleable extractors
        for b in tqdm(boards, desc="Move delta collection (sequential)"):
            base_eval = sf_eval(engine, b, cfg)
            base_board = b.copy()
            base_feats_raw = extract_features_fn(base_board)
            top_moves = sf_top_moves(engine, b, cfg)
            for move, _ in top_moves:
                b.push(move)
                reply_res = engine.analyse(
                    b, chess.engine.Limit(depth=cfg.depth), multipv=1
                )
                res_info = reply_res[0] if isinstance(reply_res, list) else reply_res
                reply_move = cast(Dict[str, Any], res_info).get("pv", [None])[0]

                if reply_move is not None:
                    b.push(reply_move)
                    after_reply_eval = sf_eval(engine, b, cfg)
                    delta_eval = after_reply_eval - base_eval
                    feats = enrich_features(
                        b,
                        base_board,
                        base_feats_raw,
                        extract_features_fn,
                        engine,
                        INTERACTION_PAIRS,
                        {},
                        {},
                    )
                    X_raw.append(feats)
                    y_raw.append(delta_eval)
                    y_base_evals_raw.append(base_eval)
                    b.pop()
                b.pop()

    # Collect all feature names
    _seen_features: Dict[str, None] = {}
    for feats in X_raw:
        for k in feats:
            if k not in _seen_features:
                _seen_features[k] = None

    X = X_raw
    y = y_raw
    y_base_evals = y_base_evals_raw

    feature_names: List[str] = list(_seen_features.keys())

    # Update feature_names to include interaction term names
    for p1, p2 in INTERACTION_PAIRS:
        if p1 in feature_names and p2 in feature_names:
            combined_name = f"{p1}_x_{p2}"
            if combined_name not in feature_names:
                feature_names.append(combined_name)

    # Remove redundant absolute features that have delta counterparts.
    _keep_absolute = {"phase", "material_us", "material_them"}
    _delta_originals = {k[2:] for k in feature_names if k.startswith("d_")}
    feature_names = [
        k
        for k in feature_names
        if k.startswith("d_")
        or "_x_" in k
        or k in _keep_absolute
        or k not in _delta_originals
    ]

    X_mat = np.array(
        [[float(x.get(k, 0.0)) for k in feature_names] for x in X], dtype=float
    )
    y_arr_cp: npt.NDArray[np.floating] = np.array(y, dtype=float)

    # Convert centipawn deltas to win-rate deltas.
    y_base_arr = np.array(y_base_evals, dtype=float)
    y_arr: npt.NDArray[np.floating] = np.asarray(
        _cp_to_winrate(y_base_arr + y_arr_cp) - _cp_to_winrate(y_base_arr),
        dtype=float,
    )

    # Train/test split on move deltas
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y_arr, test_size=test_size, random_state=42
    )
    _, _, _y_train_cp, _y_test_cp = train_test_split(
        X_mat, y_arr_cp, test_size=test_size, random_state=42
    )

    n_test_boards = max(1, int(len(boards) * test_size))
    B_test = boards[:n_test_boards]

    # 2) Fit nonlinear surrogate (GBT) with distilled Lasso explanations
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = TreeSurrogate(n_samples=X_train_scaled.shape[0])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        model.fit(X_train_scaled, y_train, y_raw=y_train)

    logger.info("GBT iterations: %d", model.gbt.n_iter_)
    logger.info(
        "Distilled ElasticNet alpha: %.4f, l1_ratio: %.2f",
        model.alpha_,
        model.l1_ratio_,
    )

    # Fidelity
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    # 3) Move-ranking agreement (Kendall tau over MultiPV)
    logger.info("Computing move ranking agreement using %d workers...", cfg.threads)
    taus = []
    covered = 0
    n_tau = 0
    cfg_local = SFConfig(
        cfg.engine_path,
        cfg.depth,
        cfg.movetime,
        multipv=min(multipv_for_ranking, cfg.multipv),
        threads=cfg.threads,
    )

    if cfg.threads > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cfg.threads
        ) as ranking_executor:
            ranking_futures = [
                ranking_executor.submit(
                    _ranking_worker_process,
                    board_fen=b.fen(),
                    cfg=cfg,
                    cfg_local=cfg_local,
                    extract_features_fn=extract_features_fn,
                    interaction_pairs=INTERACTION_PAIRS,
                    feature_names=feature_names,
                    scaler=scaler,
                    model=model,
                )
                for b in B_test
            ]

            for future in tqdm(
                concurrent.futures.as_completed(ranking_futures),
                total=len(ranking_futures),
                desc="Move ranking analysis",
            ):
                res = future.result()
                if res is not None:
                    sf_scores, sur_scores = res
                    tau = kendall_tau(sf_scores, sur_scores)
                    taus.append(tau)
                    if tau > 0.5:
                        covered += 1
                    n_tau += 1
    else:
        # Sequential fallback
        for b in tqdm(B_test, desc="Move ranking analysis (sequential)"):
            cand = _cached_sf_top_moves(engine, b, cfg_local)
            if len(cand) < 2:
                continue
            sf_scores, sur_scores = [], []
            base_eval = _cached_sf_eval(engine, b, cfg)
            base_board = b.copy()
            base_feats_raw = extract_features_fn(base_board)
            for mv, _ in cand:
                b.push(mv)
                reply_move = _cached_best_reply(engine, b, cfg.depth)
                if reply_move is not None:
                    b.push(reply_move)
                    after_reply_eval = _cached_sf_eval(engine, b, cfg)
                    delta_eval = after_reply_eval - base_eval
                    feats = _enrich(b, base_board, base_feats_raw)
                    x_vec = np.array(
                        [float(feats.get(k, 0.0)) for k in feature_names], dtype=float
                    )
                    x_scaled = scaler.transform(x_vec.reshape(1, -1))
                    sur_val = float(model.predict(x_scaled)[0])
                    sf_scores.append(delta_eval)
                    sur_scores.append(sur_val)
                    b.pop()
                b.pop()
            if len(sf_scores) >= 2:
                tau = kendall_tau(sf_scores, sur_scores)
                taus.append(tau)
                if tau > 0.5:
                    covered += 1
                n_tau += 1

    tau_mean = float(np.mean(taus)) if taus else 0.0

    # 4) Local faithfulness, sparsity, coverage, AND decisive faithfulness
    logger.info("Computing local faithfulness and sparsity...")
    faithful_hits = 0
    faithful_total = 0
    faithful_decisive_hits = 0
    faithful_decisive_total = 0
    sparsity_counts = []
    coverage_hits = 0
    coverage_total = 0

    coef = model.distilled_coef
    abs_coef = np.abs(coef)
    weight_threshold = (
        np.percentile(abs_coef[abs_coef > 0], 50) if np.any(abs_coef > 0) else 0.0
    )

    if cfg.threads > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cfg.threads
        ) as faith_executor:
            faith_futures = [
                faith_executor.submit(
                    _faithfulness_worker_process,
                    board_fen=b.fen(),
                    cfg=cfg,
                    extract_features_fn=extract_features_fn,
                    interaction_pairs=INTERACTION_PAIRS,
                    feature_names=feature_names,
                    scaler=scaler,
                    model=model,
                    gap_threshold_cp=gap_threshold_cp,
                    weight_threshold=weight_threshold,
                    coef=coef,
                    abs_coef=abs_coef,
                )
                for b in B_test
            ]

            for future in tqdm(
                concurrent.futures.as_completed(faith_futures),
                total=len(faith_futures),
                desc="Faithfulness analysis",
            ):
                res = future.result()
                if res is not None:
                    (
                        f_hits,
                        f_decisive_hits,
                        f_decisive_total,
                        sp_counts,
                        cov_hits,
                        cov_total,
                    ) = res

                    faithful_hits += f_hits
                    faithful_total += 1
                    faithful_decisive_hits += f_decisive_hits
                    faithful_decisive_total += f_decisive_total
                    sparsity_counts.extend(sp_counts)
                    coverage_hits += cov_hits
                    coverage_total += cov_total
    else:
        # Sequential fallback
        for b in tqdm(B_test, desc="Faithfulness analysis (sequential)"):
            cand = _cached_sf_top_moves(engine, b, cfg)
            if len(cand) < 2:
                continue
            cand_sorted = sorted(cand, key=lambda x: -x[1])
            (best_mv, best_cp), (second_mv, second_cp) = cand_sorted[0], cand_sorted[1]
            if abs(best_cp - second_cp) < gap_threshold_cp:
                continue

            base_eval = _cached_sf_eval(engine, b, cfg)
            base_board = b.copy()
            base_feats_raw = extract_features_fn(base_board)

            # Evaluate best move
            b.push(best_mv)
            reply_move_best = _cached_best_reply(engine, b, cfg.depth)
            if reply_move_best is not None:
                b.push(reply_move_best)
                after_eval_best = _cached_sf_eval(engine, b, cfg)
                delta_sf_best = after_eval_best - base_eval
                feats_best = _enrich(b, base_board, base_feats_raw)
                vec_best = np.array(
                    [float(feats_best.get(k, 0.0)) for k in feature_names], dtype=float
                )
                b.pop()
            else:
                delta_sf_best = 0.0
                vec_best = np.zeros(len(feature_names))
            b.pop()

            # Evaluate second best move
            b.push(second_mv)
            reply_move_second = _cached_best_reply(engine, b, cfg.depth)
            if reply_move_second is not None:
                b.push(reply_move_second)
                after_eval_second = _cached_sf_eval(engine, b, cfg)
                delta_sf_second = after_eval_second - base_eval
                feats_second = _enrich(b, base_board, base_feats_raw)
                vec_second = np.array(
                    [float(feats_second.get(k, 0.0)) for k in feature_names],
                    dtype=float,
                )
                b.pop()
            else:
                delta_sf_second = 0.0
                vec_second = np.zeros(len(feature_names))
            b.pop()

            vec_best_scaled = scaler.transform(vec_best.reshape(1, -1))
            vec_second_scaled = scaler.transform(vec_second.reshape(1, -1))
            sur_best = float(model.predict(vec_best_scaled)[0])
            sur_second = float(model.predict(vec_second_scaled)[0])
            contrib_best = coef * vec_best_scaled[0]

            tot = np.sum(np.abs(contrib_best))
            sp = 0
            if tot > 1e-9:
                c_sorted = np.sort(np.abs(contrib_best))[::-1]
                cum = 0.0
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
            coverage_total += 1
            if strong_feats >= 2:
                coverage_hits += 1

            if (sur_best - sur_second) * (delta_sf_best - delta_sf_second) > 0:
                faithful_hits += 1
            faithful_total += 1

            if abs(delta_sf_best - delta_sf_second) >= 80.0:
                if (sur_best > sur_second) == (delta_sf_best > delta_sf_second):
                    faithful_decisive_hits += 1
                faithful_decisive_total += 1

    local_faithfulness = (faithful_hits / faithful_total) if faithful_total else 0.0
    sparsity_mean = float(np.mean(sparsity_counts)) if sparsity_counts else 0.0
    coverage_ratio = (coverage_hits / coverage_total) if coverage_total else 0.0
    local_faithfulness_decisive = (
        (faithful_decisive_hits / faithful_decisive_total)
        if faithful_decisive_total
        else 0.0
    )

    # 5) Stability selection (ElasticNet bootstraps)
    if X_train.shape[0] >= 20:
        logger.info("Running stability selection...")
        top_k = model.top_k_idx
        X_train_topk = X_train_scaled[:, top_k]
        y_gbt_train = model.predict(X_train_scaled)
        picks_topk = np.zeros(len(top_k), dtype=int)
        for b in tqdm(range(stability_bootstraps), desc="Stability selection"):
            idx = np.random.choice(
                X_train.shape[0], size=X_train.shape[0], replace=True
            )
            Xb = X_train_topk[idx]
            yb = y_gbt_train[idx]
            m = ElasticNet(
                alpha=model.alpha_,
                l1_ratio=model.l1_ratio_,
                fit_intercept=True,
                random_state=42 + b,
                max_iter=10000,
            )
            m.fit(Xb, yb)
            picks_topk += m.coef_ != 0.0
        pick_freq_topk = picks_topk / stability_bootstraps
        stable_topk = np.where(pick_freq_topk >= stability_thresh)[0]
        stable_idx = [int(top_k[i]) for i in stable_topk]
    else:
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


def _audit_worker_process(
    board_fen: str,
    cfg: SFConfig,
    extract_features_fn: Callable[[chess.Board], Dict[str, float]],
    interaction_pairs: List[Tuple[str, str]],
) -> List[Tuple[Dict[str, float], float, float]]:
    """Worker process for parallelizing move delta collection.

    Opens its own Stockfish engine instance and processes one board.
    """
    from .audit_enrichment import enrich_features
    from .engine.interface import sf_eval, sf_open, sf_top_moves

    engine = sf_open(cfg)
    try:
        board = chess.Board(board_fen)
        base_eval = sf_eval(engine, board, cfg)
        base_board = board.copy()

        # We need the base features for delta calculation if we are NOT using the Rust batch.
        # But if we ARE using Rust batch, extract_features_delta_rust handles it.
        # For simplicity and correctness with the existing enrich logic:
        base_feats_raw = extract_features_fn(base_board)

        top_moves = sf_top_moves(engine, board, cfg)
        results = []

        for move, _ in top_moves:
            board.push(move)
            # Use same depth as in the sequential version
            reply_res = engine.analyse(
                board, chess.engine.Limit(depth=cfg.depth), multipv=1
            )
            res_info = reply_res[0] if isinstance(reply_res, list) else reply_res
            reply_move = cast(Dict[str, Any], res_info).get("pv", [None])[0]

            if reply_move is not None:
                board.push(reply_move)
                after_reply_eval = sf_eval(engine, board, cfg)
                delta_eval = after_reply_eval - base_eval

                # enrich_features will handle the Rust batching if available
                feats = enrich_features(
                    board,
                    base_board,
                    base_feats_raw,
                    extract_features_fn,
                    engine,
                    interaction_pairs,
                    {},  # local hanging_cache
                    {},  # local swing_cache
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
    model: Any,  # Changed from model_coef/intercept
) -> Optional[Tuple[List[float], List[float]]]:
    """Worker process for parallelizing move ranking analysis."""
    import chess
    import numpy as np

    from .audit_enrichment import enrich_features
    from .engine.interface import sf_eval, sf_open, sf_top_moves

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

                # Predict with the surrogate (GBT)
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
    import chess
    import numpy as np

    from .audit_enrichment import enrich_features
    from .engine.interface import sf_eval, sf_open, sf_top_moves

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

        # Sparsity check
        tot = np.sum(np.abs(contrib_best))
        sp = 0
        if tot > 1e-9:
            c_sorted = np.sort(np.abs(contrib_best))[::-1]
            cum = 0.0
            for v in c_sorted:
                cum += v
                sp += 1
                if cum >= 0.8 * tot:
                    break

        sp_counts = [sp] if sp > 0 else []

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
            f_decisive_total = 1
            if (sur_best > sur_second) == (delta_sf_best > delta_sf_second):
                f_decisive_hits = 1

        return f_hits, f_decisive_hits, f_decisive_total, sp_counts, cov_hits, cov_total
    finally:
        engine.quit()
