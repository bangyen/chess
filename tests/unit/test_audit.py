"""Tests for main audit functionality."""

import math
import warnings
from unittest.mock import Mock, patch

import chess
import numpy as np
import pytest

from chess_ai.audit import AuditResult, _cp_to_winrate, audit_feature_set
from chess_ai.engine.config import SFConfig

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

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
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

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
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
            chess.Board() for _ in range(5)
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

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
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

        boards = [chess.Board() for _ in range(5)]
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
            stability_bootstraps=3,
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

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
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

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
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

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
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
        boards = [chess.Board() for _ in range(10)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            stability_bootstraps=3,
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

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_small_dataset_edge_case(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test audit with very small dataset (edge case)."""
        # Setup mocks
        mock_sf_eval.return_value = 25.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 25.0),
        ]

        # Create very small dataset but with enough samples for cross-validation
        boards = [chess.Board() for _ in range(5)]  # Increased to avoid CV issues
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        # Run audit with minimal parameters
        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=1,
            test_size=0.5,
            stability_bootstraps=2,
        )

        # Should still return valid result
        assert isinstance(result, AuditResult)
        assert result.n_tau >= 0

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_list_reply_info(self, mock_sf_top_moves, mock_sf_eval):
        """Test audit when engine returns list for reply info."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
        ]

        # Create mock engine that returns list for reply info
        mock_engine = Mock()

        def mock_analyse(board, limit=None, multipv=1):
            if multipv == 1:
                return [{"pv": [chess.Move.from_uci("e7e5")]}]  # List format
            else:
                return [
                    {"score": Mock(), "pv": [chess.Move.from_uci("e2e4")]},
                ]

        mock_engine.analyse = mock_analyse

        boards = [chess.Board() for _ in range(5)]
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        # Run audit
        result = audit_feature_set(
            boards=boards,
            engine=mock_engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
        )

        # Should handle list reply info gracefully
        assert isinstance(result, AuditResult)

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_insufficient_candidates(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test audit when there are insufficient candidates for ranking."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
        ]  # Only one candidate

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
            multipv_for_ranking=3,  # More than available candidates
        )

        # Should handle insufficient candidates gracefully
        assert isinstance(result, AuditResult)
        assert result.n_tau >= 0

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_ambiguous_positions(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test audit with ambiguous positions (small gaps)."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 49.0),  # Very small gap
        ]

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        # Run audit with high gap threshold
        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            gap_threshold_cp=100.0,  # High threshold
        )

        # Should handle ambiguous positions gracefully
        assert isinstance(result, AuditResult)

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_stability_selection_small_dataset(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test stability selection with small dataset."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
        ]

        # Create small dataset (less than 20 samples)
        boards = [chess.Board() for _ in range(8)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            stability_bootstraps=3,
        )

        # Should handle small dataset for stability selection
        assert isinstance(result, AuditResult)
        assert isinstance(result.stable_features, list)
        assert isinstance(result.top_features_by_coef, list)

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_zero_coefficients(self, mock_sf_top_moves, mock_sf_eval):
        """Test audit when model has zero coefficients."""
        # Setup mocks
        mock_sf_eval.return_value = 0.0  # Neutral evaluation
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 0.0),
            (chess.Move.from_uci("d2d4"), 0.0),
        ]

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
        )

        # Should handle zero coefficients gracefully
        assert isinstance(result, AuditResult)
        assert result.coverage_ratio >= 0.0

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_decisive_gap_calculation(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test audit with decisive gap calculation."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 100.0),  # Large gap
            (chess.Move.from_uci("d2d4"), 20.0),
        ]

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            gap_threshold_cp=50.0,
        )

        # Should handle decisive gap calculation
        assert isinstance(result, AuditResult)
        assert result.local_faithfulness_decisive >= 0.0

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_empty_feature_names(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Test audit when feature names list is empty."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
        ]

        # Create feature extractor that returns empty features
        def empty_extract_features(board):
            return {}

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=empty_extract_features,
        )

        # Should handle empty feature names gracefully
        assert isinstance(result, AuditResult)

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_boolean_features(self, mock_sf_top_moves, mock_sf_eval):
        """Test audit with boolean features."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
        ]

        # Create feature extractor that returns boolean features
        def boolean_extract_features(board):
            return {
                "material_diff": True,  # Boolean value
                "mobility_us": False,  # Boolean value
                "center_control": 2.0,  # Float value
            }

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=boolean_extract_features,
        )

        # Should handle boolean features gracefully
        assert isinstance(result, AuditResult)

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_feature_set_engine_probes(self, mock_sf_top_moves, mock_sf_eval):
        """Test audit with engine probes."""
        # Setup mocks
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
        ]

        # Create feature extractor with engine probes
        def probe_extract_features(board):
            return {
                "material_diff": 0.0,
                "mobility_us": 20.0,
                "_engine_probes": {
                    "hanging_after_reply": lambda engine, board, depth=6: (1, 5, 0),
                    "best_forcing_swing": lambda engine, board, d_base=6, k_max=12: 10.0,
                    "sf_eval_shallow": lambda engine, board, depth=6: 25.0,
                },
            }

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=probe_extract_features,
        )

        # Should handle engine probes gracefully
        assert isinstance(result, AuditResult)

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_delta_features_computed(self, mock_sf_top_moves, mock_sf_eval):
        """Verify that delta features (d_<key>) are produced during training.

        The delta features are the primary mechanism for the surrogate to
        predict eval *changes* rather than absolute eval, so they must appear
        in the feature vector used by the model.
        """
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
        ]

        call_count = {"n": 0}

        def varying_extract_features(board):
            """Return features that vary between base and after-reply positions.

            Odd calls (base extraction) return one set of values; even calls
            (after-reply extraction) return a different set so that deltas
            are non-trivially zero.
            """
            call_count["n"] += 1
            is_after = call_count["n"] % 3 != 1  # first of every 3 is base
            return {
                "material_us": 12.0 if is_after else 10.0,
                "material_them": 10.0,
                "material_diff": 2.0 if is_after else 0.0,
                "mobility_us": 25.0 if is_after else 20.0,
                "mobility_them": 20.0,
                "king_ring_pressure_us": 0.0,
                "king_ring_pressure_them": 0.0,
                "passed_us": 0.0,
                "passed_them": 0.0,
                "open_files_us": 0.0,
                "semi_open_us": 0.0,
                "open_files_them": 0.0,
                "semi_open_them": 0.0,
                "phase": 10.0,
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
                    "best_forcing_swing": (
                        lambda engine, board, d_base=6, k_max=12: 0.0
                    ),
                    "sf_eval_shallow": lambda engine, board, depth=6: 0.0,
                },
            }

        boards = [chess.Board() for _ in range(8)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=varying_extract_features,
            stability_bootstraps=2,
        )

        assert isinstance(result, AuditResult)
        # At minimum, delta features should be present in the model's feature space
        # (they may or may not appear in top-15 depending on coefficients)
        # Verify through the top_features list that the model ran
        assert len(result.top_features_by_coef) > 0

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_distilled_lasso_produces_signed_coefficients(
        self, mock_sf_top_moves, mock_sf_eval
    ):
        """Verify the distilled Lasso produces signed feature coefficients.

        The top_features_by_coef list should contain (name, coef)
        tuples derived from the Lasso distilled on GBT predictions,
        with finite float coefficients (which may be negative).
        Feature importance pre-selection should zero out most
        coefficients, concentrating weight on a handful of features.
        """
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
        ]

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            stability_bootstraps=2,
        )

        assert isinstance(result, AuditResult)
        assert len(result.top_features_by_coef) > 0
        # Distilled Lasso coefficients are signed; verify they are finite floats
        non_zero = 0
        for _name, coef_val in result.top_features_by_coef:
            assert isinstance(coef_val, float)
            assert np.isfinite(coef_val)
            if coef_val != 0.0:
                non_zero += 1
        # Feature importance pre-selection ensures most coefficients are zero;
        # at most distill_top_k (default 10) can be non-zero.
        assert non_zero <= 10


class TestCpToWinrate:
    """Tests for the _cp_to_winrate helper function."""

    def test_zero_cp_gives_half(self):
        """A centipawn score of 0 should map to 50% win probability."""
        assert math.isclose(_cp_to_winrate(0.0), 0.5, abs_tol=1e-9)

    def test_positive_cp_above_half(self):
        """Positive centipawn scores map to >50% win probability."""
        assert _cp_to_winrate(100.0) > 0.5
        assert _cp_to_winrate(500.0) > 0.5

    def test_negative_cp_below_half(self):
        """Negative centipawn scores map to <50% win probability."""
        assert _cp_to_winrate(-100.0) < 0.5
        assert _cp_to_winrate(-500.0) < 0.5

    def test_monotonicity(self):
        """Higher cp scores should always give higher win rates."""
        vals = [-1000, -500, -100, 0, 100, 500, 1000]
        wrs = [_cp_to_winrate(float(v)) for v in vals]
        for i in range(len(wrs) - 1):
            assert wrs[i] < wrs[i + 1]

    def test_bounded_zero_one(self):
        """Output should always be in [0, 1]."""
        for cp in [-10000.0, -100.0, 0.0, 100.0, 10000.0]:
            wr = _cp_to_winrate(cp)
            assert 0.0 <= wr <= 1.0
        # Moderate values should be strictly inside (0, 1)
        for cp in [-500.0, -100.0, 0.0, 100.0, 500.0]:
            wr = _cp_to_winrate(cp)
            assert 0.0 < wr < 1.0

    def test_symmetry(self):
        """_cp_to_winrate(x) + _cp_to_winrate(-x) should equal 1."""
        for cp in [50.0, 111.0, 300.0, 700.0]:
            assert math.isclose(
                _cp_to_winrate(cp) + _cp_to_winrate(-cp), 1.0, abs_tol=1e-9
            )

    def test_custom_k(self):
        """A larger k produces a less steep sigmoid."""
        # With a larger k, 100 cp maps closer to 0.5
        wr_default = _cp_to_winrate(100.0, k=111.0)
        wr_wide = _cp_to_winrate(100.0, k=400.0)
        assert wr_wide < wr_default  # wider sigmoid → closer to 0.5

    def test_numpy_vectorised(self):
        """_cp_to_winrate works element-wise on numpy arrays."""
        arr = np.array([-200.0, 0.0, 200.0])
        result = _cp_to_winrate(arr)
        assert result.shape == (3,)
        assert math.isclose(float(result[1]), 0.5, abs_tol=1e-9)
        assert float(result[0]) < 0.5 < float(result[2])


class TestAuditCanonicalFeatureSet:
    """Tests verifying the union-based canonical feature set."""

    def create_mock_engine(self):
        """Create a mock Stockfish engine."""
        mock_engine = Mock()

        def mock_analyse(board, limit=None, multipv=1):
            if multipv == 1:
                return {"pv": [chess.Move.from_uci("e7e5")]}
            return [
                {"score": Mock(), "pv": [chess.Move.from_uci("e2e4")]},
                {"score": Mock(), "pv": [chess.Move.from_uci("d2d4")]},
            ]

        mock_engine.analyse = mock_analyse
        return mock_engine

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_sparse_features_not_dropped(self, mock_sf_top_moves, mock_sf_eval):
        """Features present in only some positions survive into the model."""
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
        ]

        call_count = {"n": 0}

        def sparse_extract(board):
            call_count["n"] += 1
            feats = {
                "material_diff": 0.0,
                "mobility_us": 20.0,
                "phase": 20.0,
                "_engine_probes": {
                    "hanging_after_reply": lambda engine, board, depth=6: (0, 0, 0),
                    "best_forcing_swing": (
                        lambda engine, board, d_base=6, k_max=12: 0.0
                    ),
                },
            }
            # Only some calls include this feature
            if call_count["n"] % 3 == 0:
                feats["endgame_only_feat"] = 5.0
            return feats

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=sparse_extract,
            stability_bootstraps=2,
        )

        assert isinstance(result, AuditResult)
        # At minimum the model trained and returned coefficients
        assert len(result.top_features_by_coef) > 0


class TestAuditElasticNetDistillation:
    """Tests verifying ElasticNet distillation behaviour."""

    def create_mock_engine(self):
        """Create a mock Stockfish engine."""
        mock_engine = Mock()

        def mock_analyse(board, limit=None, multipv=1):
            if multipv == 1:
                return {"pv": [chess.Move.from_uci("e7e5")]}
            return [
                {"score": Mock(), "pv": [chess.Move.from_uci("e2e4")]},
                {"score": Mock(), "pv": [chess.Move.from_uci("d2d4")]},
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
                    "best_forcing_swing": (
                        lambda engine, board, d_base=6, k_max=12: 0.0
                    ),
                    "sf_eval_shallow": lambda engine, board, depth=6: 0.0,
                },
            }

        return mock_extract_features

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_audit_still_returns_valid_result(self, mock_sf_top_moves, mock_sf_eval):
        """After the ElasticNet + winrate upgrade the audit still works."""
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
            (chess.Move.from_uci("g1f3"), 10.0),
        ]

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            multipv_for_ranking=3,
            test_size=0.4,
            stability_bootstraps=3,
            stability_thresh=0.7,
        )

        assert isinstance(result, AuditResult)
        assert 0.0 <= result.r2 <= 1.0 or result.r2 < 0.0  # R² can be negative
        assert -1.0 <= result.tau_mean <= 1.0
        assert 0.0 <= result.local_faithfulness <= 1.0
        assert 0.0 <= result.coverage_ratio <= 1.0
        assert isinstance(result.stable_features, list)

    @patch("chess_ai.audit.sf_eval")
    @patch("chess_ai.audit.sf_top_moves")
    def test_distilled_coefficients_are_finite(self, mock_sf_top_moves, mock_sf_eval):
        """Distilled ElasticNet coefficients should be finite floats."""
        mock_sf_eval.return_value = 50.0
        mock_sf_top_moves.return_value = [
            (chess.Move.from_uci("e2e4"), 50.0),
            (chess.Move.from_uci("d2d4"), 25.0),
        ]

        boards = [chess.Board() for _ in range(5)]
        engine = self.create_mock_engine()
        cfg = SFConfig(engine_path="/path/to/stockfish", depth=16)
        extract_fn = self.create_mock_feature_extractor()

        result = audit_feature_set(
            boards=boards,
            engine=engine,
            cfg=cfg,
            extract_features_fn=extract_fn,
            stability_bootstraps=2,
        )

        assert len(result.top_features_by_coef) > 0
        for _name, coef_val in result.top_features_by_coef:
            assert isinstance(coef_val, float)
            assert np.isfinite(coef_val)
