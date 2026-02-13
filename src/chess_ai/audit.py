"""Main audit functionality."""

import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import chess
import numpy as np
import numpy.typing as npt

from .engine import SFConfig, sf_eval, sf_top_moves
from .metrics.kendall import kendall_tau
from .metrics.positional import (
    checkability_now,
    confinement_delta,
    passed_pawn_momentum_delta,
)

# Suppress sklearn convergence warnings for small datasets
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Suppress sklearn convergence warnings for small datasets
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    warnings.filterwarnings("ignore", message=".*convergence.*", module="sklearn")
    warnings.filterwarnings("ignore", module="sklearn")
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
    # -- FEN-keyed caches to avoid redundant Stockfish / Rust calls --
    _eval_cache: Dict[str, float] = {}
    _top_moves_cache: Dict[str, List[Tuple[chess.Move, float]]] = {}
    _analyse_cache: Dict[str, chess.Move] = {}
    _hanging_cache: Dict[str, Tuple[int, float, int]] = {}
    _swing_cache: Dict[str, float] = {}

    def cached_sf_eval(eng, board: chess.Board, c: SFConfig) -> float:
        """Return sf_eval result, caching by FEN to skip repeat calls."""
        key = board.fen()
        if key not in _eval_cache:
            _eval_cache[key] = sf_eval(eng, board, c)
        return _eval_cache[key]

    def cached_sf_top_moves(
        eng, board: chess.Board, c: SFConfig
    ) -> List[Tuple[chess.Move, float]]:
        """Return sf_top_moves result, caching by FEN + multipv."""
        key = f"{board.fen()}|mpv{c.multipv}"
        if key not in _top_moves_cache:
            _top_moves_cache[key] = sf_top_moves(eng, board, c)
        return _top_moves_cache[key]

    def cached_best_reply(eng, board: chess.Board, depth: int):
        """Find Stockfish's best reply, caching by FEN.

        Returns the best reply Move or None.
        """
        key = board.fen()
        if key in _analyse_cache:
            return _analyse_cache[key]
        reply_info = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
        if isinstance(reply_info, list):
            reply_info = reply_info[0]
        reply_move = reply_info.get("pv", [None])[0]
        _analyse_cache[key] = reply_move
        return reply_move

    # -- Helper to enrich features (avoids 4x copy-paste) --
    def _enrich_features(
        board: chess.Board,
        base_board: chess.Board,
        base_feats_raw: Dict[str, float],
        extract_fn: Callable,
        eng,
        apply_inter: Callable,
    ) -> Dict[str, float]:
        """Extract features and add probes, deltas, and interactions.

        Centralises the feature-enrichment pipeline that was previously
        duplicated in every loop of the audit.
        """
        feats = extract_fn(board)
        probes = feats.pop("_engine_probes", {})
        feats = {
            k: (1.0 if isinstance(v, bool) and v else float(v))
            for k, v in feats.items()
        }

        # Engine-based probe features (cached by FEN to skip
        # redundant Rust searches on repeated positions).
        if probes:
            fen_key = board.fen()
            if fen_key in _hanging_cache:
                hang_cnt, hang_max_val, hang_near_king = _hanging_cache[fen_key]
            else:
                hang_cnt, hang_max_val, hang_near_king = probes[
                    "hanging_after_reply"
                ](eng, board, depth=6)
                _hanging_cache[fen_key] = (hang_cnt, hang_max_val, hang_near_king)
            feats["hang_cnt"] = hang_cnt
            feats["hang_max_val"] = hang_max_val
            feats["hang_near_king"] = hang_near_king

            if fen_key in _swing_cache:
                forcing_swing = _swing_cache[fen_key]
            else:
                forcing_swing = probes["best_forcing_swing"](
                    eng, board, d_base=6, k_max=12
                )
                _swing_cache[fen_key] = forcing_swing
            feats["forcing_swing"] = forcing_swing

        # Passed-pawn momentum delta
        pp_delta = passed_pawn_momentum_delta(base_board, board)
        feats.update(pp_delta)

        # Checkability delta
        base_check = checkability_now(base_board)
        after_check = checkability_now(board)
        check_delta = {
            "d_quiet_checks": after_check["d_quiet_checks"]
            - base_check["d_quiet_checks"],
            "d_capture_checks": after_check["d_capture_checks"]
            - base_check["d_capture_checks"],
        }
        feats.update(check_delta)

        # Confinement delta
        conf_delta = confinement_delta(base_board, board)
        feats.update(conf_delta)

        # Compute delta features: d_<key> = after - base
        for k in base_feats_raw:
            if k in feats:
                feats[f"d_{k}"] = feats[k] - base_feats_raw[k]

        # Interaction terms
        apply_inter(feats)
        return feats

    def _extract_base_feats(
        extract_fn: Callable, board: chess.Board
    ) -> Dict[str, float]:
        """Extract and clean base features for delta computation."""
        raw = extract_fn(board)
        raw.pop("_engine_probes", None)
        return {
            k: (1.0 if isinstance(v, bool) and v else float(v)) for k, v in raw.items()
        }

    # Interaction term definitions (needed by _enrich_features)
    interaction_pairs = [
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
    ]

    def apply_interactions(feats):
        """Add pairwise interaction features in-place."""
        for p1, p2 in interaction_pairs:
            if p1 in feats and p2 in feats:
                feats[f"{p1}_x_{p2}"] = float(feats[p1]) * float(feats[p2])

    # 1) Collect dataset (X, y) for fidelity (move delta-level)
    print("Collecting move deltas for training...")
    feature_names = None
    X = []
    y: List[float] = []

    for b in tqdm(boards, desc="Move delta collection"):
        base_eval = cached_sf_eval(engine, b, cfg)
        base_board = b.copy()
        base_feats_raw = _extract_base_feats(extract_features_fn, base_board)

        top_moves = cached_sf_top_moves(engine, b, cfg)

        for move, _ in top_moves:
            b.push(move)
            reply_move = cached_best_reply(engine, b, cfg.depth)

            if reply_move is not None:
                b.push(reply_move)
                after_reply_eval = cached_sf_eval(engine, b, cfg)
                delta_eval = after_reply_eval - base_eval

                after_reply_feats = _enrich_features(
                    b,
                    base_board,
                    base_feats_raw,
                    extract_features_fn,
                    engine,
                    apply_interactions,
                )

                if feature_names is None:
                    feature_names = list(after_reply_feats.keys())
                else:
                    feature_names = [k for k in feature_names if k in after_reply_feats]

                X.append(after_reply_feats)
                y.append(delta_eval)
                b.pop()

            b.pop()

    # realign feature vectors to common feature set
    # (interactions are already applied by _enrich_features)
    feature_names = feature_names or []

    # Update feature_names to include interaction term names
    for p1, p2 in interaction_pairs:
        if p1 in feature_names and p2 in feature_names:
            combined_name = f"{p1}_x_{p2}"
            if combined_name not in feature_names:
                feature_names.append(combined_name)

    # Remove redundant absolute features that have delta counterparts.
    # Keep a small set of useful absolute features (phase for context,
    # material_us/them for scale) and all features without delta versions
    # (engine probes, interaction terms, etc.).
    _keep_absolute = {"phase", "material_us", "material_them"}
    _delta_originals = {k[2:] for k in feature_names if k.startswith("d_")}
    feature_names = [
        k
        for k in feature_names
        if k.startswith("d_")  # keep all delta features
        or "_x_" in k  # keep all interaction terms
        or k in _keep_absolute  # keep selected absolute features
        or k not in _delta_originals  # keep features without a delta counterpart
    ]

    X_mat = np.array(
        [[float(x.get(k, 0.0)) for k in feature_names] for x in X], dtype=float
    )
    y_arr: npt.NDArray[np.floating] = np.array(y, dtype=float)

    # Train/test split on move deltas
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y_arr, test_size=test_size, random_state=42
    )

    # For testing, we need to split boards separately
    # Since we have move deltas, we'll use a subset of boards for testing
    n_test_boards = max(1, int(len(boards) * test_size))
    B_test = boards[:n_test_boards]

    # 2) Fit nonlinear surrogate (GBT) with distilled Lasso explanations
    # Normalize features (GBT is invariant but the distilled LassoCV
    # for stability selection still benefits from scaling).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class TreeSurrogate:
        """Gradient-boosted tree surrogate with distilled linear explanations.

        Wraps HistGradientBoostingRegressor for prediction and distils
        its predictions into a sparse Lasso whose coefficients drive the
        sparsity, coverage, and top-feature metrics.
        """

        def __init__(self, n_samples: int = 100, distill_top_k: int = 10):
            # Adapt GBT complexity to dataset size: use early stopping
            # only when there are enough samples for a reliable validation
            # split; otherwise use a fixed, conservative iteration count.
            use_early = n_samples >= 400
            gbt_kwargs = {
                "max_depth": 4,
                "learning_rate": 0.05,
                "max_iter": 500 if use_early else 300,
                "min_samples_leaf": max(5, n_samples // 50),
                "random_state": 42,
            }
            if use_early:
                gbt_kwargs.update(
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                )
            else:
                gbt_kwargs["early_stopping"] = False
            self.gbt = HistGradientBoostingRegressor(**gbt_kwargs)
            self._lasso_alpha: float = 1.0
            self._distill_top_k = distill_top_k
            self.feature_importances: np.ndarray = np.zeros(0)
            self.distilled_coef: np.ndarray = np.zeros(0)
            self.top_k_idx: np.ndarray = np.zeros(0, dtype=int)

        def fit(self, X, y):
            """Fit the GBT and distil into a sparse Lasso.

            The distilled Lasso is trained on the GBT's *predictions*
            (not raw Stockfish targets), giving a sparse linear
            approximation of the learned decision boundary.  Its
            coefficients drive the sparsity, coverage, and top-feature
            metrics.  Feature pre-selection uses the GBT's built-in
            split-gain importances to keep only the top-K features.
            """
            self.gbt.fit(X, y)
            # feature_importances_ may be unavailable when all targets
            # are constant (no splits), so fall back to uniform weights.
            self.feature_importances = getattr(
                self.gbt,
                "feature_importances_",
                np.ones(X.shape[1]) / X.shape[1],
            )

            # Distill GBT into a sparse linear model for crisp explanations.
            # Restrict the Lasso to the top-K important features so that
            # the resulting coefficients are concentrated on the features the
            # GBT actually relies on, yielding sparsity in the 3-5 range.
            y_distill = self.gbt.predict(X)
            n_features = X.shape[1]
            k = min(self._distill_top_k, n_features)
            self.top_k_idx = np.argsort(self.feature_importances)[-k:]
            X_distill = X[:, self.top_k_idx]

            n_samples = X.shape[0]
            cv_folds = max(2, min(5, n_samples // 10)) if n_samples >= 10 else 2
            alphas = (
                np.logspace(-4, 2, 30).tolist()
                if n_samples >= 10
                else [1.0, 10.0, 100.0]
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="sklearn")
                distill_lasso = LassoCV(
                    cv=cv_folds,
                    alphas=alphas,
                    random_state=42,
                    max_iter=10000,
                )
                distill_lasso.fit(X_distill, y_distill)

            # Expand back to full-length coefficient vector (zeros for
            # features excluded by the importance pre-selection).
            full_coef = np.zeros(n_features)
            full_coef[self.top_k_idx] = distill_lasso.coef_
            self.distilled_coef = full_coef
            self._lasso_alpha = float(distill_lasso.alpha_)

        def predict(self, X):
            """Predict eval deltas."""
            if len(X.shape) == 1:
                return self.gbt.predict(X.reshape(1, -1))
            return self.gbt.predict(X)

        @property
        def alpha_(self):
            """Lasso alpha from the distilled fit, used by stability selection."""
            return self._lasso_alpha

    model = TreeSurrogate(n_samples=X_train_scaled.shape[0])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        model.fit(X_train_scaled, y_train)

    print(f"GBT iterations: {model.gbt.n_iter_}")
    print(f"Distilled Lasso alpha: {model.alpha_:.4f}")

    # Fidelity
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    # 3) Move-ranking agreement (Kendall tau over MultiPV)
    print("Computing move ranking agreement...")
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
    for b in tqdm(B_test, desc="Move ranking analysis"):
        cand = cached_sf_top_moves(engine, b, cfg_local)
        if len(cand) < 2:
            continue

        sf_scores = []
        sur_scores = []
        base_eval = cached_sf_eval(engine, b, cfg)
        base_board = b.copy()
        base_feats_raw = _extract_base_feats(extract_features_fn, base_board)

        for mv, _ in cand:
            b.push(mv)
            reply_move = cached_best_reply(engine, b, cfg.depth)

            if reply_move is not None:
                b.push(reply_move)
                after_reply_eval = cached_sf_eval(engine, b, cfg)
                delta_eval = after_reply_eval - base_eval

                feats_after_reply = _enrich_features(
                    b,
                    base_board,
                    base_feats_raw,
                    extract_features_fn,
                    engine,
                    apply_interactions,
                )
                vec_after_reply = np.array(
                    [float(feats_after_reply.get(k, 0.0)) for k in feature_names],
                    dtype=float,
                )
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

    # 4) Local faithfulness, sparsity, coverage, AND decisive faithfulness
    #    (merged into a single pass to avoid redundant Stockfish calls)
    print("Computing local faithfulness and sparsity...")
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

    def _sparsity(contrib):
        """Count features needed to cover 80% of total |contribution|."""
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

    def _eval_move_delta(b, mv, base_eval, base_board, base_feats_raw):
        """Evaluate a single move: push move+reply, get delta and features.

        Returns (delta_sf, vec) where vec is the feature vector or zeros.
        """
        b.push(mv)
        reply_move = cached_best_reply(engine, b, cfg.depth)

        if reply_move is not None:
            b.push(reply_move)
            after_eval = cached_sf_eval(engine, b, cfg)
            delta_sf = after_eval - base_eval

            feats = _enrich_features(
                b,
                base_board,
                base_feats_raw,
                extract_features_fn,
                engine,
                apply_interactions,
            )
            vec = np.array(
                [float(feats.get(k, 0.0)) for k in feature_names], dtype=float
            )
            b.pop()
        else:
            delta_sf = 0.0
            vec = np.zeros(len(feature_names))
        b.pop()
        return delta_sf, vec

    for b in tqdm(B_test, desc="Faithfulness analysis"):
        cand = cached_sf_top_moves(engine, b, cfg)
        if len(cand) < 2:
            continue
        cand_sorted = sorted(cand, key=lambda x: -x[1])
        (best_mv, best_cp), (second_mv, second_cp) = cand_sorted[0], cand_sorted[1]
        if abs(best_cp - second_cp) < gap_threshold_cp:
            continue

        base_eval = cached_sf_eval(engine, b, cfg)
        base_board = b.copy()
        base_feats_raw = _extract_base_feats(extract_features_fn, base_board)

        delta_sf_best, vec_best = _eval_move_delta(
            b, best_mv, base_eval, base_board, base_feats_raw
        )
        delta_sf_second, vec_second = _eval_move_delta(
            b, second_mv, base_eval, base_board, base_feats_raw
        )

        # Predict with GBT; use distilled linear coefs for attribution
        vec_best_scaled = scaler.transform(vec_best.reshape(1, -1))
        vec_second_scaled = scaler.transform(vec_second.reshape(1, -1))
        sur_best = float(model.predict(vec_best_scaled)[0])
        sur_second = float(model.predict(vec_second_scaled)[0])

        # Distilled linear attribution: coef * scaled_feature
        contrib_best = coef * vec_best_scaled[0]

        sp = _sparsity(contrib_best)
        if sp > 0:
            sparsity_counts.append(sp)

        # coverage: at least 2 features with meaningful |coef| and
        # non-zero local linear contribution
        strong_feats: int = int(
            np.sum((abs_coef >= weight_threshold) & (np.abs(contrib_best) > 0))
        )
        coverage_total += 1
        if strong_feats >= 2:
            coverage_hits += 1

        # faithfulness: does the surrogate rank best > second in the same
        # direction as Stockfish?
        dir_sur = sur_best - sur_second
        dir_sf = float(delta_sf_best - delta_sf_second)
        if dir_sur * dir_sf > 0:
            faithful_hits += 1
        faithful_total += 1

        # decisive faithfulness (gap ≥ 80 cp) — computed in the same pass
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

    # 5) Stability selection (L1 bootstraps): how often is a feature chosen (non-zero)
    #    Bootstrap Lassos run on the same top-K feature subset used for
    #    distillation, so the alpha is calibrated correctly and only
    #    features the GBT actually uses can be selected.
    if X_train.shape[0] >= 20:
        print("Running stability selection...")
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
            m = Lasso(
                alpha=model.alpha_,
                fit_intercept=True,
                random_state=42 + b,
                max_iter=10000,
            )
            m.fit(Xb, yb)
            picks_topk += m.coef_ != 0.0
        pick_freq_topk = picks_topk / stability_bootstraps
        stable_topk = np.where(pick_freq_topk >= stability_thresh)[0]
        # Map back to full feature indices
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
