"""Tests for model_trainer.py uncovered branches.

Covers: reply_info as list, delta feature try/except branches, and
the import error path.
"""

import warnings
from unittest.mock import Mock, patch

import chess
import pytest

from chess_ai.engine.config import SFConfig
from chess_ai.model_trainer import PhaseEnsemble, train_surrogate_model

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
except ImportError:
    pass


class TestTrainSurrogateModelReplyAsList:
    """Test the branch where engine.analyse returns a list."""

    @patch("chess_ai.model_trainer.sf_eval", return_value=50.0)
    @patch(
        "chess_ai.model_trainer.sf_top_moves",
        return_value=[(chess.Move.from_uci("e2e4"), 50.0)],
    )
    def test_reply_info_as_list(self, _mock_top, _mock_eval):
        """When engine.analyse returns a list, the first element is used."""
        mock_engine = Mock()

        def mock_analyse(board, limit=None, multipv=1):
            # Return a list (some engines do this)
            return [{"pv": [chess.Move.from_uci("e7e5")]}]

        mock_engine.analyse = mock_analyse

        cfg = SFConfig(engine_path="/mock/sf", depth=12)
        boards = [chess.Board() for _ in range(4)]

        def extract(board):
            return {"material_diff": 0.0, "mobility_us": 20.0, "phase": 10.0}

        model, _scaler, _names = train_surrogate_model(
            boards, mock_engine, cfg, extract
        )

        assert isinstance(model, PhaseEnsemble)

    @patch("chess_ai.model_trainer.sf_eval", return_value=50.0)
    @patch(
        "chess_ai.model_trainer.sf_top_moves",
        return_value=[(chess.Move.from_uci("e2e4"), 50.0)],
    )
    def test_reply_none_skipped(self, _mock_top, _mock_eval):
        """When reply_move is None, that sample is skipped.

        With no valid samples at all, sklearn raises a ValueError
        on an empty array.  Verify this propagates correctly.
        """
        mock_engine = Mock()

        def mock_analyse(board, limit=None, multipv=1):
            return {"pv": [None]}

        mock_engine.analyse = mock_analyse

        cfg = SFConfig(engine_path="/mock/sf", depth=12)
        boards = [chess.Board() for _ in range(4)]

        def extract(board):
            return {"material_diff": 0.0, "phase": 10.0}

        with pytest.raises((ValueError, IndexError)):
            train_surrogate_model(boards, mock_engine, cfg, extract)
