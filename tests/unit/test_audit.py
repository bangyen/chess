"""Tests for main audit functionality."""

import warnings
from unittest.mock import Mock, patch

import chess
import pytest

from src.chess_ai.audit import AuditResult, audit_feature_set
from src.chess_ai.engine.config import SFConfig

# Suppress sklearn convergence warnings in tests
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Import ConvergenceWarning for filtering
try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    ConvergenceWarning = UserWarning


class TestAuditResult:
    """Test AuditResult dataclass."""

    def test_audit_result_creation(self):
        """Test creating an AuditResult."""
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
        assert result.tau_covered == 10
        assert result.n_tau == 15
        assert result.local_faithfulness == 0.8
        assert result.local_faithfulness_decisive == 0.85
        assert result.sparsity_mean == 3.5
        assert result.coverage_ratio == 0.7
        assert result.stable_features == ["material_diff", "mobility_us"]
        assert result.top_features_by_coef == [
            ("material_diff", 0.5),
            ("mobility_us", 0.3),
        ]

    def test_audit_result_defaults(self):
        """Test AuditResult with default values."""
        result = AuditResult(
            r2=0.0,
            tau_mean=0.0,
            tau_covered=0,
            n_tau=0,
            local_faithfulness=0.0,
            local_faithfulness_decisive=0.0,
            sparsity_mean=0.0,
            coverage_ratio=0.0,
            stable_features=[],
            top_features_by_coef=[],
        )

        assert result.r2 == 0.0
        assert result.tau_mean == 0.0
        assert result.stable_features == []
        assert result.top_features_by_coef == []


class TestAuditFeatureSet:
    """Test the main audit_feature_set function."""

    def create_mock_engine(self):
        """Create a mock Stockfish engine."""
        mock_engine = Mock()

        # Mock engine.analyse to return consistent results
        def mock_analyse(board, limit=None, multipv=1):
            if multipv == 1:
                return {
                    "score": Mock(),
                    "pv": [chess.Move.from_uci("e7e5")],
                }  # Legal reply move
            else:
                return [
                    {"score": Mock(), "pv": [chess.Move.from_uci("e2e4")]},
                    {"score": Mock(), "pv": [chess.Move.from_uci("d2d4")]},
                    {"score": Mock(), "pv": [chess.Move.from_uci("g1f3")]},
                ]

        mock_engine.analyse = mock_analyse
        return mock_engine

    def create_mock_feature_extractor(self):
        """Create a mock feature extractor function."""

        def mock_extract_features(board):
            return {
                "material_us": 10.0,
                "material_them": 10.0,
                "material_diff": 0.0,
                "mobility_us": 20.0,
                "mobility_them": 20.0,
                "king_ring_pressure_us": 0.0,
                "king_ring_pressure_them": 0.0,
                "passed_us": 0.0,
                "passed_them": 0.0,
                "open_files_us": 0.0,
                "semi_open_us": 0.0,
                "open_files_them": 0.0,
                "semi_open_them": 0.0,
                "phase": 32.0,
                "center_control_us": 2.0,
                "center_control_them": 2.0,
                "piece_activity_us": 15.0,
                "piece_activity_them": 15.0,
                "king_safety_us": 3.0,
                "king_safety_them": 3.0,
                "hanging_us": 0.0,
                "hanging_them": 0.0,
                "_engine_probes": {
                    "hanging_after_reply": lambda engine, board, depth=6: (0, 0, 0),
                    "best_forcing_swing": lambda engine, board, d_base=6, k_max=12: 0.0,
                    "sf_eval_shallow": lambda engine, board, depth=6: 0.0,
                },
            }

        return mock_extract_features

    @patch("src.chess_ai.audit.sf_eval")
    @patch("src.chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_basic(self, mock_sf_top_moves, mock_sf_eval):
        """Test basic audit functionality."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0  # 50 centipawns
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
            (chess.Move.from_uci("g1f3"), 10.0),
        ]

        # Create test data
        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        # Run audit
        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=3,
            test_size=0.4,
            l1_alpha=0.01,
            gap_threshold_cp=50.0,
            attribution_topk=5,
            stability_bootstraps=5,
            stability_thresh=0.7,
        )

        # Check result structure
        assert isinstance(result, AuditResult)
        assert hasattr(result, "r2")
        assert hasattr(result, "tau_mean")
        assert hasattr(result, "tau_covered")
        assert hasattr(result, "n_tau")
        assert hasattr(result, "local_faithfulness")
        assert hasattr(result, "local_faithfulness_decisive")
        assert hasattr(result, "sparsity_mean")
        assert hasattr(result, "coverage_ratio")
        assert hasattr(result, "stable_features")
        assert hasattr(result, "top_features_by_coef")

        # Check that values are reasonable
        assert 0.0 <= result.r2 <= 1.0
        assert -1.0 <= result.tau_mean <= 1.0
        assert result.tau_covered >= 0
        assert result.n_tau >= 0
        assert 0.0 <= result.local_faithfulness <= 1.0
        assert 0.0 <= result.local_faithfulness_decisive <= 1.0
        assert result.sparsity_mean >= 0.0
        assert 0.0 <= result.coverage_ratio <= 1.0
        assert isinstance(result.stable_features, list)
        assert isinstance(result.top_features_by_coef, list)

    @patch("src.chess_ai.audit.sf_eval")
    @patch("src.chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_small_dataset(self, mock_sf_top_moves, mock_sf_eval):
        """Test audit with very small dataset."""
        # Setup mocks
        mock_sf_eval.return_value = 25.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 25.0),
            (chess.Move.from_uci("d2d4"), 20.0),
        ]

        # Create test data with enough samples for stable convergence
        boards = [
            chess.Board() for _ in range(10)
        ]  # More samples for better convergence
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        # Run audit with minimal parameters
        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=2,
            test_size=0.5,
            stability_bootstraps=2,
        )

        # Should still return valid result
        assert isinstance(result, AuditResult)
        assert result.n_tau >= 0

    @patch("src.chess_ai.audit.sf_eval")
    @patch("src.chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_different_parameters(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test audit with different parameter values."""
        # Setup mocks
        mock_sf_eval.return_value = 75.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 75.0),
            (chess.Move.from_uci("d2d4"), 50.0),
            (chess.Move.from_uci("g1f3"), 25.0),
            (chess.Move.from_uci("c2c4"), 10.0),
        ]

        boards = [chess.Board() for _ in range(8)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=20, multipv=4)
        extract_fn = self.create_mock_feature_extractor()

        # Run audit with different parameters
        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=4,
            test_size=0.3,
            l1_alpha=0.1,
            gap_threshold_cp=100.0,
            attribution_topk=10,
            stability_bootstraps=10,
            stability_thresh=0.8,
        )

        # Should handle different parameters gracefully
        assert isinstance(result, AuditResult)
        assert result.tau_covered >= 0

    def test_audit_feature_set_empty_boards(self):
        """Test audit with empty board list."""
        boards = []
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        # Should handle empty boards gracefully
        with pytest.raises((ValueError, IndexError)):
            audit_feature_set(
                boards=boards, engine=engine, cfg=cfg, extract_features_fn=extract_fn
            )

    @patch("src.chess_ai.audit.sf_eval")
    @patch("src.chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_engine_errors(self, mock_sf_top_moves, mock_sf_eval):
        """Test audit when engine returns errors."""
        # Setup mocks to simulate engine errors
        mock_sf_eval.side_effect = Exception("Engine error")
        mock_sf_top_moves.return_value = []

        boards = [chess.Board() for _ in range(10)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        # Should handle engine errors gracefully
        with pytest.raises(Exception, match="Engine error"):
            audit_feature_set(
                boards=boards, engine=engine, cfg=cfg, extract_features_fn=extract_fn
            )

    @patch("src.chess_ai.audit.sf_eval")
    @patch("src.chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_feature_extraction_errors(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test audit when feature extraction fails."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [(chess.Move.from_uci("e2e4"), 50.0)]

        # Create feature extractor that raises errors
        def failing_extract_features(board):
            raise Exception("Feature extraction failed")

        boards = [chess.Board() for _ in range(10)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)

        # Should handle feature extraction errors
        with pytest.raises(Exception, match="Feature extraction failed"):
            audit_feature_set(
                boards=boards,
                engine=engine,
                cfg=cfg,
                extract_features_fn=failing_extract_features,
            )

    @patch("src.chess_ai.audit.sf_eval")
    @patch("src.chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_stability_selection(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test stability selection with sufficient data."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
        ]

        # Create larger dataset for stability selection
        boards = [chess.Board() for _ in range(25)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            stability_bootstraps=5,
            stability_thresh=0.7,
        )

        # Should have stability selection results
        assert isinstance(result.stable_features, list)
        assert isinstance(result.top_features_by_coef, list)

    def test_audit_feature_set_parameter_validation(self):
        """Test parameter validation."""
        boards = [chess.Board() for _ in range(10)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        # Test with invalid test_size
        with pytest.raises((ValueError, AssertionError, TypeError)):
            audit_feature_set(
                boards=boards,
                engine=engine,
                cfg=cfg,
                extract_features_fn=extract_fn,
                test_size=1.5,  # Invalid: > 1.0
            )

        # Test with invalid test_size
        with pytest.raises((ValueError, AssertionError, TypeError)):
            audit_feature_set(
                boards=boards,
                engine=engine,
                cfg=cfg,
                extract_features_fn=extract_fn,
                test_size=-0.1,  # Invalid: < 0.0
            )
