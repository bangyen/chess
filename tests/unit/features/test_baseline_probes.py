"""Tests for engine probe closures returned by baseline_extract_features.

Exercises the _engine_probes callables (sf_eval_shallow,
hanging_after_reply, best_forcing_swing) by calling them with a
mock engine, covering the closure bodies that normal feature
extraction tests never invoke.
"""

from unittest.mock import Mock, patch

import chess
import chess.engine

from chess_ai.features.baseline import baseline_extract_features


class TestEngineProbeClosures:
    """Call the engine probe closures returned in _engine_probes."""

    def _get_probes(self):
        """Extract probes from a starting-position feature dict."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        return feats["_engine_probes"], board

    def test_sf_eval_shallow_success(self):
        """sf_eval_shallow returns centipawn score from engine."""
        probes, board = self._get_probes()
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 42
        mock_engine.analyse.return_value = {"score": mock_score}

        result = probes["sf_eval_shallow"](mock_engine, board, depth=6)
        assert result == 42.0

    def test_sf_eval_shallow_exception(self):
        """sf_eval_shallow returns 0.0 on exception."""
        probes, board = self._get_probes()
        mock_engine = Mock()
        mock_engine.analyse.side_effect = RuntimeError("fail")

        result = probes["sf_eval_shallow"](mock_engine, board, depth=6)
        assert result == 0.0

    def test_sf_eval_shallow_list_response(self):
        """sf_eval_shallow handles analyse returning a list."""
        probes, board = self._get_probes()
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 100
        mock_engine.analyse.return_value = [{"score": mock_score}]

        result = probes["sf_eval_shallow"](mock_engine, board, depth=6)
        assert result == 100.0

    def test_sf_eval_shallow_clips_extreme(self):
        """sf_eval_shallow clips mate scores to [-1000, 1000]."""
        probes, board = self._get_probes()
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 99999
        mock_engine.analyse.return_value = {"score": mock_score}

        result = probes["sf_eval_shallow"](mock_engine, board, depth=6)
        assert result == 1000.0

    def test_hanging_after_reply_no_reply(self):
        """hanging_after_reply returns (0,0,0) when no reply found."""
        probes, board = self._get_probes()
        mock_engine = Mock()
        mock_engine.analyse.return_value = {"pv": []}

        result = probes["hanging_after_reply"](mock_engine, board, depth=6)
        assert result == (0, 0, 0)

    def test_hanging_after_reply_exception(self):
        """hanging_after_reply returns (0,0,0) on exception."""
        probes, board = self._get_probes()
        mock_engine = Mock()
        mock_engine.analyse.side_effect = RuntimeError("fail")

        result = probes["hanging_after_reply"](mock_engine, board, depth=6)
        assert result == (0, 0, 0)

    def test_hanging_after_reply_with_reply(self):
        """hanging_after_reply processes a reply move."""
        probes, board = self._get_probes()
        mock_engine = Mock()
        # Return e7e5 as reply (legal from starting position after e2e4)
        board.push(chess.Move.from_uci("e2e4"))
        reply_move = chess.Move.from_uci("e7e5")
        mock_engine.analyse.return_value = {"pv": [reply_move]}

        cnt, v_max, near_king = probes["hanging_after_reply"](
            mock_engine, board, depth=6
        )
        assert isinstance(cnt, int)
        assert isinstance(v_max, int)
        assert near_king in (0, 1)

    def test_best_forcing_swing_no_forcing(self):
        """best_forcing_swing returns 0.0 when no forcing moves exist."""
        _probes, _ = self._get_probes()
        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 0
        mock_engine.analyse.return_value = {"score": mock_score}

        # Use a position with no captures or checks
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        # Re-extract probes for this board
        feats = baseline_extract_features(board)
        probes2 = feats["_engine_probes"]

        result = probes2["best_forcing_swing"](mock_engine, board, d_base=6, k_max=12)
        assert isinstance(result, float)

    def test_best_forcing_swing_exception(self):
        """best_forcing_swing returns 0.0 on exception."""
        probes, board = self._get_probes()
        mock_engine = Mock()
        mock_engine.analyse.side_effect = RuntimeError("fail")

        result = probes["best_forcing_swing"](mock_engine, board, d_base=6, k_max=12)
        assert result == 0.0


class TestPythonFallbackProbeClosures:
    """Test probe closures from the Python fallback path."""

    @patch("chess_ai.features.baseline.RUST_AVAILABLE", False)
    def test_python_sf_eval_shallow(self):
        """Python-path sf_eval_shallow works with mock engine."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        probes = feats["_engine_probes"]

        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 25
        mock_engine.analyse.return_value = {"score": mock_score}

        result = probes["sf_eval_shallow"](mock_engine, board, depth=6)
        assert result == 25.0

    @patch("chess_ai.features.baseline.RUST_AVAILABLE", False)
    def test_python_hanging_after_reply(self):
        """Python-path hanging_after_reply returns tuple."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        feats = baseline_extract_features(board)
        probes = feats["_engine_probes"]

        mock_engine = Mock()
        reply = chess.Move.from_uci("e7e5")
        mock_engine.analyse.return_value = {"pv": [reply]}

        cnt, _v_max, _near_king = probes["hanging_after_reply"](
            mock_engine, board, depth=6
        )
        assert isinstance(cnt, int)

    @patch("chess_ai.features.baseline.RUST_AVAILABLE", False)
    def test_python_best_forcing_swing(self):
        """Python-path best_forcing_swing returns float."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        probes = feats["_engine_probes"]

        mock_engine = Mock()
        mock_score = Mock()
        mock_score.pov.return_value.score.return_value = 0
        mock_engine.analyse.return_value = {"score": mock_score}

        result = probes["best_forcing_swing"](mock_engine, board, d_base=6, k_max=12)
        assert isinstance(result, float)

    @patch("chess_ai.features.baseline.RUST_AVAILABLE", False)
    def test_python_sf_eval_shallow_exception(self):
        """Python-path sf_eval_shallow returns 0.0 on exception."""
        board = chess.Board()
        feats = baseline_extract_features(board)
        probes = feats["_engine_probes"]

        mock_engine = Mock()
        mock_engine.analyse.side_effect = Exception("boom")

        result = probes["sf_eval_shallow"](mock_engine, board, depth=6)
        assert result == 0.0

    @patch("chess_ai.features.baseline.RUST_AVAILABLE", False)
    def test_python_hanging_after_reply_no_reply(self):
        """Python-path hanging_after_reply with no reply returns zeros."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        feats = baseline_extract_features(board)
        probes = feats["_engine_probes"]

        mock_engine = Mock()
        mock_engine.analyse.return_value = {"pv": []}

        result = probes["hanging_after_reply"](mock_engine, board, depth=6)
        assert result == (0, 0, 0)
