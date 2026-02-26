"""Consolidated tests for surrogate model training."""

from unittest.mock import Mock, patch

import chess
import numpy as np

from chess_ai.engine import SFConfig
from chess_ai.model_trainer import PhaseEnsemble, train_surrogate_model


class TestPhaseEnsemble:
    """Tests for PhaseEnsemble model wrapper."""

    def test_init(self):
        ens = PhaseEnsemble(["feat1", "phase"], [0.1], cv_folds=2, max_iter=100)
        assert ens.feature_names == ["feat1", "phase"]

    def test_phase_routing(self):
        ens = PhaseEnsemble(["f1", "phase", "f2"], [1.0], cv_folds=2, max_iter=100)
        assert ens.get_phase(np.array([0, 30, 0])) == "opening"
        assert ens.get_phase(np.array([0, 15, 0])) == "middlegame"
        assert ens.get_phase(np.array([0, 5, 0])) == "endgame"

    def test_fit_predict(self):
        X = np.random.randn(20, 2)
        y = X[:, 0] * 2.0
        ens = PhaseEnsemble(["f1", "f2"], [0.1], cv_folds=2, max_iter=100)
        ens.fit(X, y)
        preds = ens.predict(X[0])
        assert preds.shape == (1,)


class TestTrainingPipeline:
    """Integration tests for the training pipeline."""

    @patch("chess_ai.model_trainer.sf_eval", return_value=50.0)
    @patch(
        "chess_ai.model_trainer.sf_top_moves",
        return_value=[(chess.Move.from_uci("e2e4"), 50.0)],
    )
    def test_train_surrogate_model_basic(self, mock_top, mock_eval):
        engine = Mock()
        engine.analyse.return_value = {"pv": [chess.Move.from_uci("e7e5")]}
        cfg = SFConfig(engine_path="sf", depth=1)
        boards = [chess.Board() for _ in range(10)]

        def mock_extract(b):
            return {"f1": 1.0, "phase": 24.0}

        model, _scaler, names = train_surrogate_model(boards, engine, cfg, mock_extract)
        assert isinstance(model, PhaseEnsemble)
        assert "f1" in names
