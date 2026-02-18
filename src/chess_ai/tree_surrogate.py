"""Gradient-boosted tree surrogate with distilled linear explanations.

Extracted from ``audit.py`` to reduce file size and allow the surrogate
to be tested and reused independently.
"""

import warnings
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV


class TreeSurrogate:
    """Gradient-boosted tree surrogate with distilled linear explanations.

    Wraps HistGradientBoostingRegressor for prediction and distils
    its predictions into a sparse ElasticNet whose coefficients drive
    the sparsity, coverage, and top-feature metrics.

    Improvements over plain Lasso distillation:
    - **ElasticNet (L1+L2)** handles correlated features more stably
      than pure L1, keeping groups of related features rather than
      arbitrarily picking one.
    - **Mixed distillation target** blends the GBT's predictions with
      the raw Stockfish targets so the linear model stays grounded
      in the real evaluation while still benefiting from the GBT's
      learned decision boundary.
    """

    #: Fraction of the raw Stockfish target blended into the
    #: distillation target.  0.0 = pure GBT predictions (original
    #: behaviour), 1.0 = ignore GBT entirely.
    DISTILL_MIX: float = 0.3

    def __init__(self, n_samples: int = 100, distill_top_k: int = 20):
        """Build a GBT surrogate sized for *n_samples* training rows.

        Uses early stopping when the dataset is large enough for a
        reliable validation split; otherwise a conservative fixed
        iteration count avoids overfitting on small datasets.
        """
        use_early = n_samples >= 400
        gbt_kwargs: dict = {
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
        self._distill_alpha: float = 1.0
        self._distill_l1_ratio: float = 0.5
        self._distill_top_k = distill_top_k
        self.feature_importances: np.ndarray = np.zeros(0)
        self.distilled_coef: np.ndarray = np.zeros(0)
        self.top_k_idx: np.ndarray = np.zeros(0, dtype=int)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_raw: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the GBT and distil into a sparse ElasticNet.

        The distilled ElasticNet is trained on a *mixed* target that
        blends the GBT's predictions with the raw Stockfish targets
        (controlled by ``DISTILL_MIX``).  This keeps the sparse
        linear approximation grounded in the real evaluation while
        still benefiting from the GBT's smoothed decision boundary.

        Feature pre-selection uses the GBT's built-in split-gain
        importances to keep only the top-K features.

        Args:
            X: Feature matrix (n_samples, n_features), scaled.
            y: Training targets (win-rate deltas).
            y_raw: Optional raw targets (same space as *y* when no
                win-rate scaling, or the original cp deltas).  When
                provided the mixed distillation target blends GBT
                predictions with *y_raw*; otherwise *y* is used.
        """
        self.gbt.fit(X, y)
        # feature_importances_ may be unavailable when all targets
        # are constant (no splits), so fall back to uniform weights.
        if hasattr(self.gbt, "feature_importances_"):
            self.feature_importances = self.gbt.feature_importances_
        elif X.shape[1] > self._distill_top_k:
            # Fallback for HistGradientBoostingRegressor which lacks
            # the feature_importances_ attribute.
            r = permutation_importance(
                self.gbt, X, y, n_repeats=5, random_state=42, n_jobs=1
            )
            self.feature_importances = r.importances_mean
        else:
            # If features are few, skip selection.
            self.feature_importances = np.ones(X.shape[1])

        # Distill GBT into a sparse linear model for crisp
        # explanations.  The mixed target keeps the Lasso grounded.
        y_gbt = self.gbt.predict(X)
        y_ground = y_raw if y_raw is not None else y
        mix = self.DISTILL_MIX
        y_distill = (1.0 - mix) * y_gbt + mix * y_ground

        n_features = X.shape[1]
        k = min(self._distill_top_k, n_features)
        self.top_k_idx = np.argsort(self.feature_importances)[-k:]
        X_distill = X[:, self.top_k_idx]

        n_samples = X.shape[0]
        cv_folds = max(2, min(5, n_samples // 3)) if n_samples >= 6 else 2
        alphas = np.logspace(-4, 2, 30).tolist()
        l1_ratios = [0.3, 0.5, 0.7, 0.9, 1.0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="sklearn")
            distill_model = ElasticNetCV(
                cv=cv_folds,
                alphas=alphas,
                l1_ratio=l1_ratios,
                random_state=42,
                max_iter=10000,
            )
            distill_model.fit(X_distill, y_distill)

        # Expand back to full-length coefficient vector (zeros for
        # features excluded by the importance pre-selection).
        full_coef = np.zeros(n_features)
        full_coef[self.top_k_idx] = distill_model.coef_
        self.distilled_coef = full_coef
        self._distill_alpha = float(distill_model.alpha_)
        self._distill_l1_ratio = float(distill_model.l1_ratio_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict eval deltas (in win-rate space)."""
        if len(X.shape) == 1:
            return np.asarray(self.gbt.predict(X.reshape(1, -1)), dtype=float)
        return np.asarray(self.gbt.predict(X), dtype=float)

    @property
    def alpha_(self) -> float:
        """ElasticNet alpha from the distilled fit, used by stability selection."""
        return self._distill_alpha

    @property
    def l1_ratio_(self) -> float:
        """ElasticNet l1_ratio from the distilled fit."""
        return self._distill_l1_ratio
