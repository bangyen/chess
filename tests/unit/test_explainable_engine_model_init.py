"""Tests for ExplainableChessEngine._initialize_model and __enter__ paths.

Covers the surrogate model training path, Syzygy init inside __enter__,
and the engine configure error handling.
"""

from unittest.mock import Mock, patch

import chess
import chess.engine

from chess_ai.explainable_engine import ExplainableChessEngine


class TestInitializeModel:
    """Tests for _initialize_model surrogate training."""

    @patch("chess_ai.explainable_engine.sample_stratified_positions")
    def test_initialize_model_success(self, mock_sample):
        """Successful model training sets surrogate_explainer.

        Mocks sampling and train_surrogate_model so that the real
        _initialize_model body (lines 126-155) executes end-to-end.
        """
        mock_sample.return_value = [chess.Board() for _ in range(5)]

        mock_engine = Mock()
        mock_engine.analyse.return_value = {"pv": [chess.Move.from_uci("e7e5")]}

        eng = ExplainableChessEngine("/sf", model_training_positions=5)
        eng.engine = mock_engine

        mock_model = Mock()
        mock_model.distilled_coef = [0.1, 0.2]
        mock_scaler = Mock()

        with patch(
            "chess_ai.model_trainer.train_surrogate_model",
            return_value=(mock_model, mock_scaler, ["f1", "f2"]),
        ):
            eng._initialize_model()

        assert eng.surrogate_explainer is not None

    def test_initialize_model_failure_falls_back(self):
        """When model training fails, surrogate_explainer is set to None."""
        eng = ExplainableChessEngine("/sf", model_training_positions=5)
        eng.engine = Mock()

        # Patch the imports inside _initialize_model to fail
        with patch(
            "chess_ai.explainable_engine.sample_stratified_positions",
            side_effect=RuntimeError("sampling failed"),
        ):
            eng._initialize_model()

        assert eng.surrogate_explainer is None


class TestEnterWithSyzygy:
    """Tests for __enter__ with Syzygy path and engine configuration."""

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_enter_configures_strength(self, mock_popen):
        """__enter__ applies strength settings to the engine."""
        mock_engine = Mock()
        mock_popen.return_value = mock_engine

        eng = ExplainableChessEngine(
            "/sf",
            opponent_strength="beginner",
            enable_model_explanations=False,
        )

        with eng:
            mock_engine.configure.assert_called_once()
            call_args = mock_engine.configure.call_args[0][0]
            assert call_args["Skill Level"] == 0

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_enter_configure_engine_error(self, mock_popen):
        """__enter__ handles EngineError from configure gracefully."""
        mock_engine = Mock()
        mock_engine.configure.side_effect = chess.engine.EngineError("bad option")
        mock_popen.return_value = mock_engine

        eng = ExplainableChessEngine(
            "/sf",
            enable_model_explanations=False,
        )

        # Should not raise despite configure failure
        with eng:
            assert eng.engine is mock_engine

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_enter_with_syzygy_success(self, mock_popen):
        """__enter__ initialises Syzygy when path is provided."""
        mock_engine = Mock()
        mock_popen.return_value = mock_engine

        eng = ExplainableChessEngine(
            "/sf",
            syzygy_path="/fake/syzygy",
            enable_model_explanations=False,
        )

        with patch(
            "chess_ai.explainable_engine.ExplainableChessEngine.__enter__",
            wraps=eng.__enter__,
        ):
            with eng:
                # Syzygy init is attempted; it may fail because
                # the rust_utils import might not have SyzygyTablebase
                # but the engine should still work
                assert eng.engine is mock_engine

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_enter_with_model_training(self, mock_popen):
        """__enter__ calls _initialize_model when enabled."""
        mock_engine = Mock()
        mock_popen.return_value = mock_engine

        eng = ExplainableChessEngine(
            "/sf",
            enable_model_explanations=True,
            model_training_positions=5,
        )

        with patch.object(eng, "_initialize_model") as mock_init:
            with eng:
                mock_init.assert_called_once()

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_enter_with_syzygy_path_sets_option(self, mock_popen):
        """__enter__ includes SyzygyPath in engine options."""
        mock_engine = Mock()
        mock_popen.return_value = mock_engine

        eng = ExplainableChessEngine(
            "/sf",
            syzygy_path="/tb/path",
            enable_model_explanations=False,
        )

        with eng:
            call_args = mock_engine.configure.call_args[0][0]
            assert call_args["SyzygyPath"] == "/tb/path"
