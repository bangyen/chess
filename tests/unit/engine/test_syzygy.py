"""Tests for Syzygy tablebase manager."""

from unittest.mock import Mock, patch

import chess

from chess_ai.engine.syzygy import SyzygyManager


def test_syzygy_manager_init():
    """Test SyzygyManager initialization."""
    with patch("chess_ai.engine.syzygy.SyzygyManager._initialize_syzygy"):
        manager = SyzygyManager("/fake/path")
        assert manager.syzygy_path == "/fake/path"


def test_syzygy_manager_no_path():
    """Test SyzygyManager with no path."""
    manager = SyzygyManager(None)
    assert manager.syzygy is None
    assert manager.get_syzygy_data(chess.Board()) == {}


def test_syzygy_manager_get_data():
    """Test getting Syzygy data."""
    manager = SyzygyManager("/fake/path")
    mock_syzygy = Mock()
    manager.syzygy = mock_syzygy

    mock_syzygy.probe_wdl.return_value = 2
    mock_syzygy.probe_dtz.return_value = 10

    board = chess.Board("k7/8/8/8/8/8/8/K7 w - - 0 1")
    data = manager.get_syzygy_data(board)

    assert data["wdl"] == 2
    assert data["dtz"] == 10


def test_syzygy_manager_get_reason():
    """Test Syzygy reasoning."""
    manager = SyzygyManager("/fake/path")

    def mock_get_data(board):
        if board.fen().startswith("k7"):  # Win
            return {"wdl": 2}
        return {"wdl": 0}  # Draw

    with patch.object(manager, "get_syzygy_data", side_effect=mock_get_data):
        board_before = chess.Board("k7/8/8/8/8/8/8/K7 w - - 0 1")
        board_after = chess.Board("k7/8/8/8/8/8/8/K7 b - - 0 1")

        # Win -> Opposite Turn Win (which means mock returns 2 for after)
        # Wait, get_syzygy_reason: wdl_after = -data_after.get("wdl")
        # So if we want wdl_after = 2, data_after.get("wdl") must be -2.

        def mock_get_data_v2(board):
            if board.turn == chess.WHITE:
                return {"wdl": 2}  # Win
            return {"wdl": -2}  # Opponent is lost

        with patch.object(manager, "get_syzygy_data", side_effect=mock_get_data_v2):
            reason = manager.get_syzygy_reason(board_before, board_after)
            assert reason is not None
            assert "Maintains" in reason[2]
