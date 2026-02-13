"""Model training utilities for explainable chess engine."""

import warnings
from typing import Callable, Dict, List, Tuple

import chess
import numpy as np

from .engine import SFConfig, sf_eval, sf_top_moves
from .metrics.positional import (
    checkability_now,
    confinement_delta,
    passed_pawn_momentum_delta,
)
from .utils.math import cp_to_winrate

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    warnings.filterwarnings("ignore", message=".*convergence.*", module="sklearn")
    warnings.filterwarnings("ignore", module="sklearn")
except Exception as e:
    raise ImportError(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e


class PhaseEnsemble:
    """Phase-aware ElasticNet regression ensemble for chess evaluation.

    Trains separate models for opening, middlegame, and endgame phases
    using ElasticNet (L1+L2 regularisation) instead of pure Lasso.
    ElasticNet handles correlated chess features (e.g. mobility_us and
    safe_mobility_us) more stably, keeping groups of related features
    rather than arbitrarily zeroing one.  Falls back to a global model
    if insufficient data for a phase.
    """

    #: L1 ratio values explored during cross-validation.
    #: Includes 1.0 (pure Lasso) so the CV can recover the old
    #: behaviour when appropriate.
    L1_RATIOS = [0.3, 0.5, 0.7, 0.9, 1.0]

    def __init__(self, feature_names, alphas, cv_folds, max_iter):
        """Initialise the ensemble with hyper-parameter search grids.

        Args:
            feature_names: Ordered list of feature names matching
                the columns of the training matrix.
            alphas: Regularisation strengths to try during CV.
            cv_folds: Number of cross-validation folds.
            max_iter: Maximum coordinate-descent iterations.
        """
        self.feature_names = feature_names
        self.phase_idx = (
            feature_names.index("phase") if "phase" in feature_names else -1
        )
        self.alphas = alphas
        self.cv_folds = cv_folds
        self.max_iter = max_iter
        self.models = {}  # "opening", "middlegame", "endgame"
        self.global_model = None
        self.scaler = None

    def get_phase(self, x):
        """Get game phase for a feature vector."""
        if self.phase_idx == -1:
            return "middlegame"
        p = x[self.phase_idx]
        if p > 24:
            return "opening"
        if p > 12:
            return "middlegame"
        return "endgame"

    def _make_model(self):
        """Create a fresh ElasticNetCV with the ensemble's search grid.

        Factored out so that both the global and per-phase models are
        constructed consistently.
        """
        return ElasticNetCV(
            cv=self.cv_folds,
            random_state=42,
            max_iter=self.max_iter,
            alphas=self.alphas,
            l1_ratio=self.L1_RATIOS,
        )

    def fit(self, X, y):
        """Train phase-specific models.

        A global model is always trained as a fallback.  Per-phase
        models are only trained when the phase has enough samples
        for reliable cross-validation.
        """
        # Train global model backup
        self.global_model = self._make_model()
        self.global_model.fit(X, y)

        # Train phase-specific models
        phases = [self.get_phase(x) for x in X]
        for p_name in ["opening", "middlegame", "endgame"]:
            idx = [i for i, p in enumerate(phases) if p == p_name]
            if len(idx) > self.cv_folds * 2:
                m = self._make_model()
                m.fit(X[idx], y[idx])
                self.models[p_name] = m
            else:
                self.models[p_name] = self.global_model

    def predict(self, X):
        """Predict evaluation deltas."""
        if len(X.shape) == 1:
            p_name = self.get_phase(X)
            return self.models.get(p_name, self.global_model).predict(X.reshape(1, -1))

        preds = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            p_name = self.get_phase(x)
            preds[i] = self.models.get(p_name, self.global_model).predict(
                x.reshape(1, -1)
            )[0]
        return preds

    def get_contributions(self, features_normalized: np.ndarray) -> np.ndarray:
        """Get per-feature contributions in centipawns.

        Args:
            features_normalized: Normalized feature vector

        Returns:
            Array where contribution[i] = coef[i] × feature_normalized[i]
        """
        if len(features_normalized.shape) == 1:
            phase_name = self.get_phase(features_normalized)
            model = self.models.get(phase_name, self.global_model)
            result: np.ndarray = model.coef_ * features_normalized  # type: ignore
            return result
        # Multiple vectors
        contributions: np.ndarray = np.zeros_like(features_normalized)
        for i, feat_vec in enumerate(features_normalized):
            phase_name = self.get_phase(feat_vec)
            model = self.models.get(phase_name, self.global_model)
            contributions[i] = model.coef_ * feat_vec  # type: ignore
        return contributions

    @property
    def coef_(self):
        """Get model coefficients (from middlegame or global)."""
        return self.models.get("middlegame", self.global_model).coef_

    @property
    def alpha_(self):
        """Get selected regularization parameter."""
        return self.global_model.alpha_


def train_surrogate_model(
    boards: List[chess.Board],
    engine,
    cfg: SFConfig,
    extract_features_fn: Callable,
    test_size: float = 0.25,
    l1_alpha: float = 0.01,
) -> Tuple[PhaseEnsemble, StandardScaler, List[str]]:
    """Train a surrogate model for chess evaluation.

    Args:
        boards: List of chess positions
        engine: Stockfish engine instance
        cfg: Stockfish configuration
        extract_features_fn: Function to extract features from board
        test_size: Fraction for test set
        l1_alpha: L1 regularization parameter

    Returns:
        Tuple of (PhaseEnsemble model, StandardScaler, feature_names)
    """
    print("Collecting move deltas for training...")
    # Use a canonical (union) feature set so that features appearing
    # in only some positions are kept rather than silently dropped.
    # Missing values are filled with 0.0 during matrix construction.
    _seen_features: Dict[str, None] = {}
    X = []
    y_cp: List[float] = []
    y_base_evals: List[float] = []

    for b in boards:
        # Get base evaluation
        base_eval = sf_eval(engine, b, cfg)
        base_board = b.copy()

        # Get top moves from Stockfish
        top_moves = sf_top_moves(engine, b, cfg)

        for move, _ in top_moves:
            b.push(move)

            # Get Stockfish's best reply
            reply_info = engine.analyse(
                b, chess.engine.Limit(depth=cfg.depth), multipv=1
            )
            if isinstance(reply_info, list):
                reply_info = reply_info[0]
            reply_move = reply_info.get("pv", [None])[0]

            if reply_move is not None:
                b.push(reply_move)

                # Get evaluation after move → best reply
                after_reply_eval = sf_eval(engine, b, cfg)
                delta_eval = after_reply_eval - base_eval

                # Get features after move → best reply
                after_reply_feats = extract_features_fn(b)
                after_reply_feats.pop("_engine_probes", {})
                after_reply_feats = {
                    k: (1.0 if isinstance(v, bool) and v else float(v))
                    for k, v in after_reply_feats.items()
                }

                # Add delta features if available
                try:
                    pp_delta = passed_pawn_momentum_delta(base_board, b)
                    after_reply_feats.update(pp_delta)
                except Exception:
                    pass

                try:
                    base_check = checkability_now(base_board)
                    after_check = checkability_now(b)
                    check_delta = {
                        "d_quiet_checks": after_check["d_quiet_checks"]
                        - base_check["d_quiet_checks"],
                        "d_capture_checks": after_check["d_capture_checks"]
                        - base_check["d_capture_checks"],
                    }
                    after_reply_feats.update(check_delta)
                except Exception:
                    pass

                try:
                    conf_delta = confinement_delta(base_board, b)
                    after_reply_feats.update(conf_delta)
                except Exception:
                    pass

                for k in after_reply_feats:
                    if k not in _seen_features:
                        _seen_features[k] = None

                X.append(after_reply_feats)
                y_cp.append(delta_eval)
                y_base_evals.append(base_eval)

                b.pop()
            b.pop()

    # Prepare data
    feature_names: List[str] = list(_seen_features.keys())
    X_mat = np.array(
        [[float(x.get(k, 0.0)) for k in feature_names] for x in X], dtype=float
    )

    # Convert centipawn deltas to win-rate deltas for consistency
    # with the audit pipeline.  This compresses extreme evaluations
    # while preserving sensitivity around 0, producing better-behaved
    # regression targets.
    y_cp_arr = np.array(y_cp, dtype=float)
    y_base_arr = np.array(y_base_evals, dtype=float)
    y_arr = np.asarray(
        cp_to_winrate(y_base_arr + y_cp_arr) - cp_to_winrate(y_base_arr),
        dtype=float,
    )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mat)

    # Train model with appropriate hyperparameters
    n_samples = X_scaled.shape[0]
    if n_samples < 10:
        alphas = [1.0, 10.0, 100.0]
        cv_folds = 2
        max_iter = 1000
    else:
        alphas = np.logspace(-2, 2, 20).tolist()
        cv_folds = max(2, min(5, n_samples // 10))
        max_iter = 10000

    model = PhaseEnsemble(feature_names, alphas, cv_folds, max_iter)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        model.fit(X_scaled, y_arr)

    print(f"Model trained with {n_samples} samples, alpha={model.alpha_:.4f}")

    return model, scaler, feature_names
