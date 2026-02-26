"""Core tests for audit functionality."""

from unittest.mock import Mock, patch

import chess

from chess_ai.audit import AuditResult, audit_feature_set
from chess_ai.engine import SFConfig


class TestAuditResult:
    """Test AuditResult dataclass."""

    def test_audit_result_creation(self):
        result = AuditResult(
            r2=0.75,
            tau_mean=0.6,
            tau_covered=10,
            n_tau=15,
            local_faithfulness=0.8,
            local_faithfulness_decisive=0.85,
            sparsity_mean=3.5,
            coverage_ratio=0.7,
            stable_features=[],
            top_features_by_coef=[],
        )
        assert result.r2 == 0.75
        assert result.tau_mean == 0.6


class TestAuditFlow:
    """Test the main audit_feature_set flow."""

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_basic(self, mock_top, mock_eval):
        mock_eval.return_value = 100.0
        mock_top.return_value = [(chess.Move.from_uci("e2e4"), 100.0)]

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
            stability_bootstraps=1,
        )
        assert isinstance(result, AuditResult)
