"""Tests for audit worker processes."""

from unittest.mock import Mock, patch

import chess

from chess_ai.audit.workers import (
    INTERACTION_PAIRS,
    _audit_worker_process,
)
from chess_ai.engine import SFConfig


def test_audit_worker_process_basic():
    """Test the audit worker process."""
    cfg = SFConfig(engine_path="stockfish", depth=1, threads=1)
    board = chess.Board()

    with patch("chess_ai.audit.workers.sf_open") as mock_open:
        mock_engine = Mock()
        mock_open.return_value = mock_engine

        # Mock engine methods
        mock_engine.analyse.return_value = [{"pv": [chess.Move.from_uci("e7e5")]}]

        def mock_extract(b):
            return {"feat1": 1.0}

        with patch("chess_ai.audit.workers.sf_eval", return_value=10.0), patch(
            "chess_ai.audit.workers.sf_top_moves",
            return_value=[(chess.Move.from_uci("e2e4"), 10.0)],
        ):
            results = _audit_worker_process(
                board.fen(), cfg, mock_extract, INTERACTION_PAIRS
            )

        assert len(results) > 0
        assert "feat1" in results[0][0]
        mock_engine.quit.assert_called_once()


# Add more focused worker tests here...
