"""Tests for baseline feature extraction."""

import chess

from src.chess_ai.features.baseline import baseline_extract_features


class TestBaselineFeatures:
    """Test baseline feature extraction functionality."""

    def test_initial_position_features(self):
        """Test feature extraction on initial chess position."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Check that we get expected features
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
            "king_ring_pressure_us",
            "king_ring_pressure_them",
            "_engine_probes",
        ]

        for feature in expected_features:
            assert feature in features, f"Missing feature: {feature}"

        # Initial position specific checks
        assert features["material_us"] == features["material_them"]  # Equal material
        assert features["material_diff"] == 0.0  # No material difference
        assert features["phase"] == 14.0  # All pieces present
        assert features["hanging_us"] == 0  # No hanging pieces in initial position
        assert features["hanging_them"] == 0

        # Check engine probes are present
        assert "_engine_probes" in features
        assert isinstance(features["_engine_probes"], dict)
        expected_probes = [
            "hanging_after_reply",
            "best_forcing_swing",
            "sf_eval_shallow",
        ]
        for probe in expected_probes:
            assert probe in features["_engine_probes"]

    def test_material_calculation(self):
        """Test material calculation accuracy."""
        board = chess.Board()

        # Test initial position material
        features = baseline_extract_features(board)
        # White and black should have equal material
        assert features["material_us"] == features["material_them"]

        # Test after capturing a pawn
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))
        board.push(chess.Move.from_uci("d2d4"))
        board.push(chess.Move.from_uci("e5d4"))  # Black captures white pawn

        features = baseline_extract_features(board)
        # White should have 1 point less material
        assert features["material_us"] == features["material_them"] - 1.0
        assert features["material_diff"] == -1.0

    def test_mobility_calculation(self):
        """Test mobility calculation."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Initial position should have reasonable mobility
        assert features["mobility_us"] > 0
        assert features["mobility_them"] > 0
        assert features["mobility_us"] <= 40  # Capped at 40
        assert features["mobility_them"] <= 40

    def test_king_ring_pressure(self):
        """Test king ring pressure calculation."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Initial position should have no king ring pressure
        assert features["king_ring_pressure_us"] == 0.0
        assert features["king_ring_pressure_them"] == 0.0

    def test_passed_pawns(self):
        """Test passed pawn detection."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Initial position should have no passed pawns
        assert features["passed_us"] == 0
        assert features["passed_them"] == 0

    def test_file_state(self):
        """Test open/semi-open file detection."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Initial position should have no open files
        assert features["open_files_us"] == 0
        assert features["open_files_them"] == 0
        assert features["semi_open_us"] == 0
        assert features["semi_open_them"] == 0

    def test_center_control(self):
        """Test center control calculation."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Initial position should have some center control
        assert features["center_control_us"] >= 0
        assert features["center_control_them"] >= 0

    def test_piece_activity(self):
        """Test piece activity calculation."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Initial position should have some piece activity
        assert features["piece_activity_us"] > 0
        assert features["piece_activity_them"] > 0

    def test_king_safety(self):
        """Test king safety calculation."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Initial position should have some king safety
        assert features["king_safety_us"] >= 0
        assert features["king_safety_them"] >= 0

    def test_hanging_pieces(self):
        """Test hanging piece detection."""
        board = chess.Board()
        features = baseline_extract_features(board)

        # Initial position should have no hanging pieces
        assert features["hanging_us"] == 0
        assert features["hanging_them"] == 0

    def test_feature_types(self):
        """Test that all features are numeric (except _engine_probes)."""
        board = chess.Board()
        features = baseline_extract_features(board)

        for key, value in features.items():
            if key != "_engine_probes":
                assert isinstance(
                    value, (int, float)
                ), f"Feature {key} should be numeric, got {type(value)}"

    def test_engine_probes_structure(self):
        """Test that engine probes are properly structured."""
        board = chess.Board()
        features = baseline_extract_features(board)

        probes = features["_engine_probes"]
        assert isinstance(probes, dict)
        assert "hanging_after_reply" in probes
        assert "best_forcing_swing" in probes
        assert "sf_eval_shallow" in probes

        # All should be callable functions
        for probe_name, probe_func in probes.items():
            assert callable(probe_func), f"Probe {probe_name} should be callable"
