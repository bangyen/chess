"""Tests for engine interface functions."""

from unittest.mock import Mock, patch

import chess

from src.chess_ai.engine.config import SFConfig
from src.chess_ai.engine.interface import sf_eval, sf_open, sf_top_moves


class TestEngineInterface:
    """Test Stockfish engine interface functions."""

    def test_sf_open(self):
        """Test opening a Stockfish engine."""
        config = SFConfig(engine_path="/path/to/stockfish", threads=2)

        with patch("chess.engine.SimpleEngine.popen_uci") as mock_popen:
            mock_engine = Mock()
            mock_popen.return_value = mock_engine

            engine = sf_open(config)

            # Check that engine was opened with correct path
            mock_popen.assert_called_once_with("/path/to/stockfish")

            # Check that engine was configured with correct threads
            mock_engine.configure.assert_called_once_with({"Threads": 2})

            assert engine == mock_engine

    def test_sf_eval_with_depth(self):
        """Test getting evaluation with depth limit."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=16, movetime=0)
        board = chess.Board()

        # Mock engine and analysis result
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 50  # 50 centipawns
        mock_info = {"score": mock_score}
        mock_engine.analyse.return_value = mock_info

        result = sf_eval(mock_engine, board, config)

        # Check that analyse was called with correct parameters
        mock_engine.analyse.assert_called_once()
        call_args = mock_engine.analyse.call_args
        assert call_args[0][0] == board  # board argument
        assert call_args[1]["limit"].depth == 16  # depth limit
        assert call_args[1]["multipv"] == 1  # single line

        # Check result
        assert result == 50.0

    def test_sf_eval_with_movetime(self):
        """Test getting evaluation with time limit."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=0, movetime=5000)
        board = chess.Board()

        # Mock engine and analysis result
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = -25  # -25 centipawns
        mock_info = {"score": mock_score}
        mock_engine.analyse.return_value = mock_info

        result = sf_eval(mock_engine, board, config)

        # Check that analyse was called with time limit
        call_args = mock_engine.analyse.call_args
        assert call_args[1]["limit"].time == 5.0  # 5000ms = 5.0s

        # Check result
        assert result == -25.0

    def test_sf_eval_with_list_result(self):
        """Test getting evaluation when engine returns list instead of dict."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=16)
        board = chess.Board()

        # Mock engine returning list (when multipv > 1)
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 100
        mock_info = {"score": mock_score}
        mock_engine.analyse.return_value = [mock_info]  # List format

        result = sf_eval(mock_engine, board, config)

        assert result == 100.0

    def test_sf_eval_mate_score_clipping(self):
        """Test that mate scores are clipped to prevent instability."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=16)
        board = chess.Board()

        # Mock engine with extreme mate score
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 50000  # Very high mate score
        mock_info = {"score": mock_score}
        mock_engine.analyse.return_value = mock_info

        result = sf_eval(mock_engine, board, config)

        # Should be clipped to 1000
        assert result == 1000.0

    def test_sf_eval_negative_mate_score_clipping(self):
        """Test that negative mate scores are clipped."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=16)
        board = chess.Board()

        # Mock engine with extreme negative mate score
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = (
            -50000
        )  # Very negative mate score
        mock_info = {"score": mock_score}
        mock_engine.analyse.return_value = mock_info

        result = sf_eval(mock_engine, board, config)

        # Should be clipped to -1000
        assert result == -1000.0

    def test_sf_top_moves_with_depth(self):
        """Test getting top moves with depth limit."""
        config = SFConfig(
            engine_path="/path/to/stockfish", depth=16, movetime=0, multipv=3
        )
        board = chess.Board()

        # Mock engine and analysis results
        mock_engine = Mock()
        mock_move1 = chess.Move.from_uci("e2e4")
        mock_move2 = chess.Move.from_uci("d2d4")
        mock_move3 = chess.Move.from_uci("g1f3")

        mock_info1 = {"pv": [mock_move1], "score": Mock()}
        mock_info1["score"].pov.return_value.score.return_value = 50
        mock_info2 = {"pv": [mock_move2], "score": Mock()}
        mock_info2["score"].pov.return_value.score.return_value = 25
        mock_info3 = {"pv": [mock_move3], "score": Mock()}
        mock_info3["score"].pov.return_value.score.return_value = 10

        mock_engine.analyse.return_value = [mock_info1, mock_info2, mock_info3]

        result = sf_top_moves(mock_engine, board, config)

        # Check that analyse was called with correct parameters
        call_args = mock_engine.analyse.call_args
        assert call_args[0][0] == board
        assert call_args[1]["limit"].depth == 16
        assert call_args[1]["multipv"] == 3

        # Check result format
        assert len(result) == 3
        assert result[0] == (mock_move1, 50.0)
        assert result[1] == (mock_move2, 25.0)
        assert result[2] == (mock_move3, 10.0)

    def test_sf_top_moves_with_movetime(self):
        """Test getting top moves with time limit."""
        config = SFConfig(
            engine_path="/path/to/stockfish", depth=0, movetime=3000, multipv=2
        )
        board = chess.Board()

        # Mock engine
        mock_engine = Mock()
        mock_move1 = chess.Move.from_uci("e2e4")
        mock_move2 = chess.Move.from_uci("d2d4")

        mock_info1 = {"pv": [mock_move1], "score": Mock()}
        mock_info1["score"].pov.return_value.score.return_value = 30
        mock_info2 = {"pv": [mock_move2], "score": Mock()}
        mock_info2["score"].pov.return_value.score.return_value = 20

        mock_engine.analyse.return_value = [mock_info1, mock_info2]

        result = sf_top_moves(mock_engine, board, config)

        # Check time limit was used
        call_args = mock_engine.analyse.call_args
        assert call_args[1]["limit"].time == 3.0  # 3000ms = 3.0s

        assert len(result) == 2

    def test_sf_top_moves_empty_pv(self):
        """Test handling of empty principal variation."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=16, multipv=2)
        board = chess.Board()

        # Mock engine with empty PV
        mock_engine = Mock()
        mock_info1 = {"pv": [], "score": Mock()}  # Empty PV
        mock_info1["score"].pov.return_value.score.return_value = 50
        mock_info2 = {"pv": [chess.Move.from_uci("e2e4")], "score": Mock()}
        mock_info2["score"].pov.return_value.score.return_value = 25

        mock_engine.analyse.return_value = [mock_info1, mock_info2]

        result = sf_top_moves(mock_engine, board, config)

        # Should only return moves with non-empty PV
        assert len(result) == 1
        assert result[0][0] == chess.Move.from_uci("e2e4")

    def test_sf_top_moves_missing_pv(self):
        """Test handling of missing principal variation."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=16, multipv=2)
        board = chess.Board()

        # Mock engine with missing PV key
        mock_engine = Mock()
        mock_info1 = {"score": Mock()}  # Missing PV key
        mock_info1["score"].pov.return_value.score.return_value = 50
        mock_info2 = {"pv": [chess.Move.from_uci("e2e4")], "score": Mock()}
        mock_info2["score"].pov.return_value.score.return_value = 25

        mock_engine.analyse.return_value = [mock_info1, mock_info2]

        result = sf_top_moves(mock_engine, board, config)

        # Should only return moves with PV
        assert len(result) == 1
        assert result[0][0] == chess.Move.from_uci("e2e4")

    def test_sf_top_moves_mate_score_clipping(self):
        """Test that mate scores in top moves are clipped."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=16, multipv=2)
        board = chess.Board()

        # Mock engine with extreme mate score
        mock_engine = Mock()
        mock_move = chess.Move.from_uci("e2e4")
        mock_info = {"pv": [mock_move], "score": Mock()}
        mock_info["score"].pov.return_value.score.return_value = (
            50000  # Very high mate score
        )

        mock_engine.analyse.return_value = [mock_info]

        result = sf_top_moves(mock_engine, board, config)

        # Score should be clipped to 1000
        assert result[0][1] == 1000.0
