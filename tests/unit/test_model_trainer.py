"""Tests for surrogate model training utilities."""

import warnings
from typing import Dict
from unittest.mock import Mock, patch

import chess
import numpy as np
import pytest

from chess_ai.engine.config import SFConfig
from chess_ai.model_trainer import PhaseEnsemble, train_surrogate_model

# Suppress sklearn convergence warnings in tests
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_extractor(vary: bool = False):
    """Return a mock feature extractor.

    When *vary* is True, features change between calls so that
    deltas are non-trivially zero.
    """
    call_count = {"n": 0}

    def extract(board):
        call_count["n"] += 1
        is_after = call_count["n"] % 3 != 1
        return {
            "material_us": 12.0 if (vary and is_after) else 10.0,
            "material_them": 10.0,
            "material_diff": 2.0 if (vary and is_after) else 0.0,
            "mobility_us": 25.0 if (vary and is_after) else 20.0,
            "mobility_them": 20.0,
            "king_ring_pressure_us": 0.0,
            "king_ring_pressure_them": 0.0,
            "passed_us": 0.0,
            "passed_them": 0.0,
            "open_files_us": 0.0,
            "semi_open_us": 0.0,
            "open_files_them": 0.0,
            "semi_open_them": 0.0,
            "phase": 10.0,
            "center_control_us": 2.0,
            "center_control_them": 2.0,
            "piece_activity_us": 15.0,
            "piece_activity_them": 15.0,
            "king_safety_us": 3.0,
            "king_safety_them": 3.0,
            "hanging_us": 0.0,
            "hanging_them": 0.0,
        }

    return extract


def _make_mock_engine():
    """Return a mock Stockfish engine."""
    mock_engine = Mock()

    def mock_analyse(board, limit=None, multipv=1):
        if multipv == 1:
            return {"pv": [chess.Move.from_uci("e7e5")]}
        return [
            {"score": Mock(), "pv": [chess.Move.from_uci("e2e4")]},
            {"score": Mock(), "pv": [chess.Move.from_uci("d2d4")]},
        ]

    mock_engine.analyse = mock_analyse
    return mock_engine


# ---------------------------------------------------------------------------
# PhaseEnsemble unit tests
# ---------------------------------------------------------------------------


class TestPhaseEnsemble:
    """Tests for the PhaseEnsemble model wrapper."""

    def test_uses_elasticnet_cv(self):
        """PhaseEnsemble now wraps ElasticNetCV, not LassoCV."""
        alphas = np.logspace(-2, 2, 10).tolist()
        ens = PhaseEnsemble(["a", "b", "phase"], alphas, cv_folds=2, max_iter=1000)
        model = ens._make_model()
        # The internal model should be ElasticNetCV
        from sklearn.linear_model import ElasticNetCV

        assert isinstance(model, ElasticNetCV)

    def test_l1_ratios_include_lasso(self):
        """The L1_RATIOS grid includes 1.0 so pure Lasso can be recovered."""
        assert 1.0 in PhaseEnsemble.L1_RATIOS

    def test_fit_predict_basic(self):
        """Fit on synthetic data and predict without errors."""
        np.random.seed(42)
        n, d = 40, 5
        X = np.random.randn(n, d)
        y = X[:, 0] * 2.0 + np.random.randn(n) * 0.1

        names = [f"f{i}" for i in range(d)]
        alphas = np.logspace(-2, 2, 10).tolist()
        ens = PhaseEnsemble(names, alphas, cv_folds=2, max_iter=5000)
        ens.fit(X, y)

        preds = ens.predict(X)
        assert preds.shape == (n,)
        # Should have some correlation with truth
        assert np.corrcoef(y, preds)[0, 1] > 0.5

    def test_get_contributions_shape(self):
        """get_contributions returns array with same shape as input."""
        np.random.seed(42)
        n, d = 30, 4
        X = np.random.randn(n, d)
        y = X[:, 0] + np.random.randn(n) * 0.1

        names = [f"f{i}" for i in range(d)]
        alphas = np.logspace(-2, 2, 10).tolist()
        ens = PhaseEnsemble(names, alphas, cv_folds=2, max_iter=5000)
        ens.fit(X, y)

        # Single vector
        contrib_single = ens.get_contributions(X[0])
        assert contrib_single.shape == (d,)

        # Batch
        contrib_batch = ens.get_contributions(X[:5])
        assert contrib_batch.shape == (5, d)

    def test_phase_routing(self):
        """get_phase correctly classifies based on phase feature value."""
        names = ["a", "phase", "b"]
        alphas = [1.0]
        ens = PhaseEnsemble(names, alphas, cv_folds=2, max_iter=1000)

        assert ens.get_phase(np.array([0.0, 30.0, 0.0])) == "opening"
        assert ens.get_phase(np.array([0.0, 18.0, 0.0])) == "middlegame"
        assert ens.get_phase(np.array([0.0, 5.0, 0.0])) == "endgame"

    def test_phase_missing_defaults_to_middlegame(self):
        """When 'phase' feature is absent, default to middlegame."""
        names = ["a", "b"]
        alphas = [1.0]
        ens = PhaseEnsemble(names, alphas, cv_folds=2, max_iter=1000)
        assert ens.get_phase(np.array([1.0, 2.0])) == "middlegame"


# ---------------------------------------------------------------------------
# Canonical feature set tests
# ---------------------------------------------------------------------------


class TestCanonicalFeatureSet:
    """Test that the union-based feature set works correctly."""

    @pytest.fixture()
    def _sparse_extractor(self):
        """Feature extractor that returns different keys on different calls.

        Simulates positions where some features are only available
        in certain board states (e.g. syzygy features in endgames).
        """
        call_count = {"n": 0}

        def extract(board):
            call_count["n"] += 1
            feats: Dict[str, float] = {
                "material_diff": 0.0,
                "mobility_us": 20.0,
                "phase": 20.0,
            }
            # Only every other call includes this feature
            if call_count["n"] % 2 == 0:
                feats["rare_feature"] = 1.0
            return feats

        return extract

    def test_union_preserves_rare_features(self, _sparse_extractor):
        """Features present in only some samples are kept, not dropped."""
        # We test at the data-structure level: collect features the way
        # train_surrogate_model does and verify the union.
        seen: Dict[str, None] = {}
        for _i in range(6):
            feats = _sparse_extractor(chess.Board())
            for k in feats:
                if k not in seen:
                    seen[k] = None

        feature_names = list(seen.keys())
        assert "rare_feature" in feature_names
        assert "material_diff" in feature_names


# ---------------------------------------------------------------------------
# train_surrogate_model integration test
# ---------------------------------------------------------------------------


class TestTrainSurrogateModel:
    """Integration tests for the full training pipeline."""

    @pytest.fixture()
    def _mock_setup(self):
        """Common mock objects for training tests."""
        engine = _make_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=12)
        extract_fn = _make_feature_extractor(vary=True)
        boards = [chess.Board() for _ in range(8)]
        return engine, cfg, extract_fn, boards

    @patch("chess_ai.model_trainer.sf_eval", return_value=50.0)
    @patch(
        "chess_ai.model_trainer.sf_top_moves",
        return_value=[
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
        ],
    )
    def test_returns_correct_types(self, _mock_top, _mock_eval, _mock_setup):
        """train_surrogate_model returns (PhaseEnsemble, scaler, names)."""
        engine, cfg, extract_fn, boards = _mock_setup

        model, scaler, names = train_surrogate_model(boards, engine, cfg, extract_fn)

        assert isinstance(model, PhaseEnsemble)
        assert isinstance(names, list)
        assert len(names) > 0
        # Model should be able to predict
        vec = np.zeros(len(names))
        pred = model.predict(vec)
        assert pred.shape == (1,)

    @patch("chess_ai.model_trainer.sf_eval", return_value=50.0)
    @patch(
        "chess_ai.model_trainer.sf_top_moves",
        return_value=[
            (chess.Move.from_uci("e2e4"), 50.0),
        ],
    )
    def test_feature_names_are_union(self, _mock_top, _mock_eval, _mock_setup):
        """Returned feature names use the union, not intersection."""
        engine, cfg, _, boards = _mock_setup

        call_count = {"n": 0}

        def sparse_extract(board):
            call_count["n"] += 1
            feats = {
                "material_diff": 0.0,
                "mobility_us": 20.0,
                "phase": 20.0,
            }
            if call_count["n"] % 3 == 0:
                feats["endgame_only"] = 5.0
            return feats

        _, _, names = train_surrogate_model(boards, engine, cfg, sparse_extract)

        # Under intersection semantics "endgame_only" would be dropped;
        # under union semantics it is preserved.
        assert "endgame_only" in names
