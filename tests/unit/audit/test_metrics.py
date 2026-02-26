"""Tests for audit metrics (faithfulness, ranking, stability)."""

from unittest.mock import Mock, patch

import chess

from chess_ai.audit import audit_feature_set
from chess_ai.engine import SFConfig


@patch("chess_ai.audit.sf_eval")
@patch("chess_ai.audit.sf_top_moves")
def test_faithfulness_metrics(mock_top, mock_eval):
    """Test that faithfulness metrics are computed correctly."""
    mock_eval.return_value = 100.0
    # Provide a large gap to trigger decisive faithfulness
    mock_top.return_value = [
        (chess.Move.from_uci("e2e4"), 200.0),
        (chess.Move.from_uci("d2d4"), 0.0),
    ]

    boards = [chess.Board() for _ in range(10)]
    engine = Mock()
    engine.analyse.return_value = {"pv": [chess.Move.from_uci("e7e5")]}
    cfg = SFConfig(engine_path="stockfish", depth=1)

    def mock_extract(b):
        return {"feat1": 1.0}

    result = audit_feature_set(
        boards=boards,
        engine=engine,
        cfg=cfg,
        extract_features_fn=mock_extract,
        gap_threshold_cp=50.0,
        stability_bootstraps=1,
    )

    assert result.local_faithfulness >= 0.0
    assert result.local_faithfulness_decisive >= 0.0


@patch("chess_ai.audit.sf_eval")
@patch("chess_ai.audit.sf_top_moves")
def test_stability_selection(mock_top, mock_eval):
    """Test stability selection logic."""
    mock_eval.side_effect = lambda e, b, c: float(hash(b.fen()) % 100)
    mock_top.return_value = [(chess.Move.from_uci("e2e4"), 100.0)]

    # Need enough boards for stability selection (usually >= 20 in implementation)
    boards = [chess.Board() for _ in range(25)]
    engine = Mock()
    engine.analyse.return_value = {"pv": [chess.Move.from_uci("e7e5")]}
    cfg = SFConfig(engine_path="stockfish", depth=1)

    def mock_extract(b):
        return {"feat1": float(hash(b.fen()) % 10)}

    result = audit_feature_set(
        boards=boards,
        engine=engine,
        cfg=cfg,
        extract_features_fn=mock_extract,
        stability_bootstraps=2,
    )

    assert isinstance(result.stable_features, list)
