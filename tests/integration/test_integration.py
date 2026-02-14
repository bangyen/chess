"""Integration tests for the chess feature audit system."""

import warnings
from unittest.mock import Mock, patch

import chess
import pytest

from chess_ai import (
    AuditResult,
    SFConfig,
    audit_feature_set,
    baseline_extract_features,
    load_feature_module,
)

# Suppress sklearn convergence warnings in tests
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Import ConvergenceWarning for filtering
try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    ConvergenceWarning = UserWarning


class TestIntegration:
    """Integration tests for the complete system."""

    def test_package_imports(self):
        """Test that all package imports work correctly."""
        # Test that we can import all main components

    def test_baseline_features_integration(self):
        """Test baseline features with real chess positions."""
        # Test with initial position
        board = chess.Board()
        features = baseline_extract_features(board)

        # Should have all expected features
        expected_features = [
            "material_us",
            "material_them",
            "material_diff",
            "mobility_us",
            "mobility_them",
            "king_ring_pressure_us",
            "king_ring_pressure_them",
            "passed_us",
            "passed_them",
            "open_files_us",
            "semi_open_us",
            "open_files_them",
            "semi_open_them",
            "phase",
            "center_control_us",
            "center_control_them",
            "piece_activity_us",
            "piece_activity_them",
            "king_safety_us",
            "king_safety_them",
            "hanging_us",
            "hanging_them",
            "_engine_probes",
        ]

        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"

        # Test with a different position
        board.push(chess.Move.from_uci("e2e4"))
        features2 = baseline_extract_features(board)

        # Features should be different
        assert features != features2

    def test_metrics_integration(self):
        """Test metrics with real chess positions."""
        from chess_ai.metrics import (
            kendall_tau,
            passed_pawn_momentum_snapshot,
        )

        # Test Kendall tau
        rank_a = [1, 2, 3, 4, 5]
        rank_b = [1, 2, 3, 4, 5]
        tau = kendall_tau(rank_a, rank_b)
        assert tau == 1.0

        # Test passed pawn momentum
        board = chess.Board()
        result = passed_pawn_momentum_snapshot(board, chess.WHITE)
        assert isinstance(result, dict)
        assert "pp_count" in result
        assert "pp_min_dist" in result

    def test_engine_config_integration(self):
        """Test engine configuration integration."""
        config = SFConfig(
            engine_path="/path/to/stockfish", depth=16, movetime=0, multipv=3, threads=1
        )

        assert config.engine_path == "/path/to/stockfish"
        assert config.depth == 16
        assert config.multipv == 3

    def test_sampling_integration(self):
        """Test position sampling integration."""
        from chess_ai.utils import sample_random_positions

        with patch("chess_ai.utils.sampling.tqdm") as mock_tqdm:
            mock_tqdm.return_value = range(3)

            positions = sample_random_positions(3, max_random_plies=15)

            assert len(positions) == 3
            for pos in positions:
                assert isinstance(pos, chess.Board)
                assert not pos.is_game_over()

    def test_audit_result_integration(self):
        """Test AuditResult integration."""
        result = AuditResult(
            r2=0.75,
            tau_mean=0.6,
            tau_covered=10,
            n_tau=15,
            local_faithfulness=0.8,
            local_faithfulness_decisive=0.85,
            sparsity_mean=3.5,
            coverage_ratio=0.7,
            stable_features=["material_diff", "mobility_us"],
            top_features_by_coef=[("material_diff", 0.5), ("mobility_us", 0.3)],
        )

        assert result.r2 == 0.75
        assert result.tau_mean == 0.6
        assert len(result.stable_features) == 2
        assert len(result.top_features_by_coef) == 2

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_integration(self, mock_sf_top_moves, mock_sf_eval):
        """Test audit integration with mocked engine."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
        ]

        # Create test data with enough samples for stable convergence
        boards = [chess.Board() for _ in range(5)]
        engine = Mock()

        # Mock engine.analyse to return proper structure
        def mock_analyse(board, limit=None, multipv=1):
            # Return a legal move based on the current position
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return {"pv": [legal_moves[0]]}
            else:
                return {"pv": []}

        engine.analyse = mock_analyse

        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)

        # Run audit
        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=baseline_extract_features,
            multipv_for_ranking=2,
            test_size=0.4,
            stability_bootstraps=2,  # Ensure we have enough data for LassoCV
        )

        # Check result
        assert isinstance(result, AuditResult)
        assert hasattr(result, "r2")
        assert hasattr(result, "tau_mean")
        assert hasattr(result, "stable_features")
        assert hasattr(result, "top_features_by_coef")

    def test_feature_module_loading_integration(self):
        """Test feature module loading integration."""
        import os
        import tempfile

        # Create a temporary feature module
        feature_code = '''
def extract_features(board):
    """Extract features from a chess board."""
    return {
        "material": 0.0,
        "mobility": 0.0,
        "center_control": 0.0
    }
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(feature_code)
            temp_path = f.name

        try:
            module = load_feature_module(temp_path)

            # Test that the module was loaded correctly
            assert hasattr(module, "extract_features")
            assert callable(module.extract_features)

            # Test that the function works
            board = chess.Board()
            features = module.extract_features(board)
            assert isinstance(features, dict)
            assert "material" in features
            assert "mobility" in features
            assert "center_control" in features

        finally:
            os.unlink(temp_path)

    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with mocked components."""
        # This test simulates the complete workflow from feature extraction to audit

        # 1. Create test positions
        boards = [chess.Board() for _ in range(3)]

        # 2. Extract features
        features_list = [baseline_extract_features(board) for board in boards]

        # 3. Verify features
        for features in features_list:
            assert isinstance(features, dict)
            assert "material_us" in features
            assert "mobility_us" in features

        # 4. Test metrics
        from chess_ai.metrics import kendall_tau

        rank_a = [1, 2, 3]
        rank_b = [1, 2, 3]
        tau = kendall_tau(rank_a, rank_b)
        assert tau == 1.0

        # 5. Test audit result creation
        result = AuditResult(
            r2=0.75,
            tau_mean=0.6,
            tau_covered=2,
            n_tau=3,
            local_faithfulness=0.8,
            local_faithfulness_decisive=0.85,
            sparsity_mean=3.5,
            coverage_ratio=0.7,
            stable_features=["material_diff"],
            top_features_by_coef=[("material_diff", 0.5)],
        )

        assert isinstance(result, AuditResult)
        assert result.r2 == 0.75

    def test_error_handling_integration(self):
        """Test error handling across components."""
        # Test invalid feature module loading
        with pytest.raises((RuntimeError, FileNotFoundError)):
            load_feature_module("/nonexistent/path/features.py")

        # Test invalid rankings for Kendall tau
        with pytest.raises(ValueError, match="equal length"):  # noqa: PT012
            from chess_ai.metrics import kendall_tau

            kendall_tau([1, 2, 3], [1, 2])  # Different lengths

        # Test empty board list for audit
        with pytest.raises((ValueError, IndexError)):
            audit_feature_set(
                boards=[],
                engine=Mock(),
                cfg=SFConfig(engine_path="/path/to/stockfish"),
                extract_features_fn=baseline_extract_features,
            )

    def test_performance_characteristics(self):
        """Test basic performance characteristics."""
        import time

        # Test feature extraction performance
        board = chess.Board()
        start_time = time.time()
        features = baseline_extract_features(board)
        end_time = time.time()

        # Should complete quickly (less than 1 second)
        assert end_time - start_time < 1.0
        assert isinstance(features, dict)

        # Test metrics performance
        from chess_ai.metrics import kendall_tau

        start_time = time.time()
        tau = kendall_tau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        end_time = time.time()

        # Should complete very quickly
        assert end_time - start_time < 0.1
        assert tau == 1.0
