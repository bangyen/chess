"""Extended tests for model_trainer.py to increase coverage.

Targets uncovered lines: coef_ property routing, alpha_ property,
train_surrogate_model with small samples, and predict on 1-D input.
"""

import warnings
from unittest.mock import Mock, patch

import chess
import numpy as np
import pytest

from chess_ai.engine.config import SFConfig
from chess_ai.model_trainer import PhaseEnsemble, train_surrogate_model

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
except ImportError:
    pass


class TestPhaseEnsembleProperties:
    """Cover coef_ and alpha_ property accessors."""

    @pytest.fixture
    def fitted_ensemble(self):
        """Return a PhaseEnsemble fitted on synthetic data."""
        np.random.seed(42)
        n, d = 40, 5
        names = ["a", "b", "c", "d", "phase"]
        X = np.random.randn(n, d)
        y = X[:, 0] * 2.0 + np.random.randn(n) * 0.1

        alphas = np.logspace(-2, 2, 10).tolist()
        ens = PhaseEnsemble(names, alphas, cv_folds=2, max_iter=5000)
        ens.fit(X, y)
        return ens

    def test_coef_returns_array(self, fitted_ensemble):
        """coef_ property returns an ndarray of model coefficients."""
        coef = fitted_ensemble.coef_
        assert isinstance(coef, np.ndarray)
        assert coef.shape == (5,)

    def test_alpha_returns_float(self, fitted_ensemble):
        """alpha_ property returns the selected regularisation parameter."""
        alpha = fitted_ensemble.alpha_
        assert isinstance(alpha, float)
        assert alpha > 0

    def test_predict_single_vector(self, fitted_ensemble):
        """predict() handles a single 1-D feature vector."""
        x = np.random.randn(5)
        pred = fitted_ensemble.predict(x)
        assert pred.shape == (1,)

    def test_predict_batch(self, fitted_ensemble):
        """predict() handles a 2-D batch of feature vectors."""
        X = np.random.randn(10, 5)
        pred = fitted_ensemble.predict(X)
        assert pred.shape == (10,)


class TestTrainSurrogateModelSmallSamples:
    """Test train_surrogate_model with very few samples (< 10).

    Covers the branch where simpler hyperparameters are chosen.
    """

    @patch("chess_ai.model_trainer.sf_eval", return_value=50.0)
    @patch(
        "chess_ai.model_trainer.sf_top_moves",
        return_value=[
            (chess.Move.from_uci("e2e4"), 50.0),
        ],
    )
    def test_small_sample_training(self, _mock_top, _mock_eval):
        """Training with < 10 samples uses simplified hyperparameters."""
        mock_engine = Mock()
        mock_engine.analyse.return_value = {"pv": [chess.Move.from_uci("e7e5")]}

        cfg = SFConfig(engine_path="/mock/sf", depth=12)
        boards = [chess.Board(), chess.Board()]

        def extract(board):
            return {"material_diff": 0.0, "mobility_us": 20.0, "phase": 10.0}

        model, _scaler, names = train_surrogate_model(boards, mock_engine, cfg, extract)

        assert isinstance(model, PhaseEnsemble)
        assert len(names) > 0
