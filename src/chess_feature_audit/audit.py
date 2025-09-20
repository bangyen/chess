"""Main audit functionality."""

import sys
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import chess
import numpy as np

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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use cross-validation with a balanced alpha range
    # Adjust parameters based on dataset size to prevent convergence issues
    n_samples, n_features = X_train.shape

    if n_samples < 10:
        # For very small datasets, use simpler approach
        alphas = [1.0, 10.0, 100.0]  # Fewer alphas, higher values
        cv_folds = 2
        max_iter = 1000  # Fewer iterations
    else:
        # For larger datasets, use full approach
        alphas = np.logspace(-2, 2, 20)  # 0.01 to 100.0
        cv_folds = max(2, min(5, n_samples // 10))
        max_iter = 10000

    model = LassoCV(
        cv=cv_folds,
        random_state=42,
        max_iter=max_iter,
        alphas=alphas,
    )

    # Suppress convergence warnings for small datasets
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        model.fit(X_train_scaled, y_train)
    print(f"Selected alpha: {model.alpha_:.4f}")

    # If still overfitting, use a minimum alpha
    if model.alpha_ < 1.0:
        model.alpha_ = 1.0
        model = Lasso(
            alpha=model.alpha_, fit_intercept=True, random_state=42, max_iter=max_iter
        )

        # Suppress convergence warnings for small datasets
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
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
