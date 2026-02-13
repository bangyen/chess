"""Main audit functionality."""

import sys
import warnings
from dataclasses import dataclass
from typing import List, Tuple

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
    # 1) Collect dataset (X, y) for fidelity (move delta-level)
    print("Collecting move deltas for training...")
    feature_names = None
    X = []
    y: List[float] = []

    for b in tqdm(boards, desc="Move delta collection"):
        # Get base evaluation
        base_eval = sf_eval(engine, b, cfg)

        # Store base board for delta computation
        base_board = b.copy()

        # Extract base features once per position for delta computation
        base_feats_raw = extract_features_fn(base_board)
        base_feats_raw.pop("_engine_probes", None)
        base_feats_raw = {
            k: (1.0 if isinstance(v, bool) and v else float(v))
            for k, v in base_feats_raw.items()
        }

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

                # Compute delta features: d_<key> = after - base
                for k in base_feats_raw:
                    if k in after_reply_feats:
                        after_reply_feats[f"d_{k}"] = (
                            after_reply_feats[k] - base_feats_raw[k]
                        )

                # Calculate delta features
                if feature_names is None:
                    feature_names = list(after_reply_feats.keys())
                else:
                    # keep common keys only
                    feature_names = [k for k in feature_names if k in after_reply_feats]

                # Features include both a handful of absolute values (phase,
                # material) and d_<key> deltas that capture the positional
                # change caused by the move+reply sequence.
                X.append(after_reply_feats)
                y.append(delta_eval)

                # Pop the reply
                b.pop()

            # Pop the move
            b.pop()

    # realign feature vectors to common feature set
    feature_names = feature_names or []

    # Add interaction terms before matrix conversion
    interaction_pairs = [
        ("material_diff", "phase"),
        ("passed_us", "phase"),
        ("mobility_us", "phase"),
        ("batteries_us", "open_files_us"),
        ("batteries_us", "semi_open_us"),
        ("outposts_us", "phase"),
        ("bishop_pair_us", "open_files_us"),
        # New: interactions for explainability features
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
    ]

    def apply_interactions(feats):
        for p1, p2 in interaction_pairs:
            if p1 in feats and p2 in feats:
                feats[f"{p1}_x_{p2}"] = float(feats[p1]) * float(feats[p2])

    # First, apply interactions to all training samples
    for x in X:
        apply_interactions(x)

    # Update feature_names to include interactions
    for p1, p2 in interaction_pairs:
        # Check if both parents are in the common feature set
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
                "max_depth": 3,
                "learning_rate": 0.1,
                "max_iter": 300 if use_early else 200,
                "min_samples_leaf": max(5, n_samples // 50),
                "random_state": 42,
            }
            if use_early:
                gbt_kwargs.update(
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                )
            else:
                gbt_kwargs["early_stopping"] = False
            self.gbt = HistGradientBoostingRegressor(**gbt_kwargs)
            self._lasso_alpha: float = 1.0
            self._distill_top_k = distill_top_k
            self.feature_importances: np.ndarray = np.zeros(0)
            self.distilled_coef: np.ndarray = np.zeros(0)

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
            top_k_idx = np.argsort(self.feature_importances)[-k:]
            X_distill = X[:, top_k_idx]

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
            full_coef[top_k_idx] = distill_lasso.coef_
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

        sf_scores = []
        sur_scores = []

        # Get base evaluation
        base_eval = sf_eval(engine, b, cfg)

        # Store base board for delta computation
        base_board = b.copy()

        # Extract base features once per position for delta computation
        base_feats_raw = extract_features_fn(base_board)
        base_feats_raw.pop("_engine_probes", None)
        base_feats_raw = {
            k: (1.0 if isinstance(v, bool) and v else float(v))
            for k, v in base_feats_raw.items()
        }

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
                after_reply_probes = feats_after_reply.pop("_engine_probes", {})
                feats_after_reply = {
                    k: (1.0 if isinstance(v, bool) and v else float(v))
                    for k, v in feats_after_reply.items()
                }

                # Add engine-based features (matching training)
                if after_reply_probes:
                    hang_cnt, hang_max_val, hang_near_king = after_reply_probes[
                        "hanging_after_reply"
                    ](engine, b, depth=6)
                    feats_after_reply["hang_cnt"] = hang_cnt
                    feats_after_reply["hang_max_val"] = hang_max_val
                    feats_after_reply["hang_near_king"] = hang_near_king

                    forcing_swing = after_reply_probes["best_forcing_swing"](
                        engine, b, d_base=6, k_max=12
                    )
                    feats_after_reply["forcing_swing"] = forcing_swing

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

                # Compute delta features: d_<key> = after - base
                for k in base_feats_raw:
                    if k in feats_after_reply:
                        feats_after_reply[f"d_{k}"] = (
                            feats_after_reply[k] - base_feats_raw[k]
                        )

                # Add interactions
                apply_interactions(feats_after_reply)

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

    coef = model.distilled_coef
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

        # Extract base features once per position for delta computation
        base_feats_raw = extract_features_fn(base_board)
        base_feats_raw.pop("_engine_probes", None)
        base_feats_raw = {
            k: (1.0 if isinstance(v, bool) and v else float(v))
            for k, v in base_feats_raw.items()
        }

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
            best_probes = f_best.pop("_engine_probes", {})
            f_best = {
                k: (1.0 if isinstance(v, bool) and v else float(v))
                for k, v in f_best.items()
            }

            # Add engine-based features (matching training)
            if best_probes:
                hang_cnt, hang_max_val, hang_near_king = best_probes[
                    "hanging_after_reply"
                ](engine, b, depth=6)
                f_best["hang_cnt"] = hang_cnt
                f_best["hang_max_val"] = hang_max_val
                f_best["hang_near_king"] = hang_near_king

                forcing_swing = best_probes["best_forcing_swing"](
                    engine, b, d_base=6, k_max=12
                )
                f_best["forcing_swing"] = forcing_swing

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

            # Compute delta features: d_<key> = after - base
            for k in base_feats_raw:
                if k in f_best:
                    f_best[f"d_{k}"] = f_best[k] - base_feats_raw[k]

            # Add interactions
            apply_interactions(f_best)

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
            second_probes = f_second.pop("_engine_probes", {})
            f_second = {
                k: (1.0 if isinstance(v, bool) and v else float(v))
                for k, v in f_second.items()
            }

            # Add engine-based features (matching training)
            if second_probes:
                hang_cnt, hang_max_val, hang_near_king = second_probes[
                    "hanging_after_reply"
                ](engine, b, depth=6)
                f_second["hang_cnt"] = hang_cnt
                f_second["hang_max_val"] = hang_max_val
                f_second["hang_near_king"] = hang_near_king

                forcing_swing = second_probes["best_forcing_swing"](
                    engine, b, d_base=6, k_max=12
                )
                f_second["forcing_swing"] = forcing_swing

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

            # Compute delta features: d_<key> = after - base
            for k in base_feats_raw:
                if k in f_second:
                    f_second[f"d_{k}"] = f_second[k] - base_feats_raw[k]

            # Add interactions
            apply_interactions(f_second)

            vec_second = np.array(
                [float(f_second.get(k, 0.0)) for k in feature_names], dtype=float
            )
            b.pop()
        else:
            delta_sf_second = 0.0
            vec_second = np.zeros(len(feature_names))
        b.pop()

        # Predict with GBT; use distilled linear coefs for attribution
        vec_best_scaled = scaler.transform(vec_best.reshape(1, -1))
        vec_second_scaled = scaler.transform(vec_second.reshape(1, -1))
        sur_best = float(model.predict(vec_best_scaled)[0])
        sur_second = float(model.predict(vec_second_scaled)[0])

        # Distilled linear attribution: coef * scaled_feature
        contrib_best = coef * vec_best_scaled[0]

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

        # coverage: at least 2 features with meaningful |coef| and
        # non-zero local linear contribution
        strong_feats: int = int(
            np.sum((abs_coef >= weight_threshold) & (np.abs(contrib_best) > 0))
        )
        coverage_total += 1
        if strong_feats >= 2:
            coverage_hits += 1

        # faithfulness: does the surrogate rank best > second in the same
        # direction as Stockfish?  Use prediction difference directly.
        dir_sur = sur_best - sur_second
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
    for b in tqdm(B_test, desc="Decisive faithfulness"):
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
                f_best.update(pp_delta)
                apply_interactions(f_best)
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
                f_second.update(pp_delta)
                apply_interactions(f_second)
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
    #    Bootstrap Lassos are trained on the GBT's predictions (not raw
    #    Stockfish targets) so they measure which features the sparse
    #    approximation consistently selects.
    if X_train.shape[0] >= 20:
        print("Running stability selection...")
        y_gbt_train = model.predict(X_train_scaled)
        picks = np.zeros(len(feature_names), dtype=int)
        for b in tqdm(range(stability_bootstraps), desc="Stability selection"):
            idx = np.random.choice(
                X_train.shape[0], size=X_train.shape[0], replace=True
            )
            Xb = X_train_scaled[idx]
            yb = y_gbt_train[idx]
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
