"""Tests for baseline feature extraction."""

import math

import chess

from chess_ai.features.baseline import baseline_extract_features


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
            "see_advantage_us",
            "see_advantage_them",
            "see_vulnerability_us",
            "see_vulnerability_them",
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

    # ── SEE feature tests ────────────────────────────────────────────

    def test_see_features_present(self):
        """SEE features should always be present in the feature dict."""
        board = chess.Board()
        features = baseline_extract_features(board)
        for key in [
            "see_advantage_us",
            "see_advantage_them",
            "see_vulnerability_us",
            "see_vulnerability_them",
        ]:
            assert key in features, f"Missing SEE feature: {key}"
            assert isinstance(features[key], (int, float))

    def test_see_initial_position_zero(self):
        """No side can win material in the starting position, so SEE
        advantage should be zero for both sides.
        """
        board = chess.Board()
        features = baseline_extract_features(board)
        assert math.isclose(features["see_advantage_us"], 0.0, abs_tol=1e-6)
        assert math.isclose(features["see_advantage_them"], 0.0, abs_tol=1e-6)
        assert math.isclose(features["see_vulnerability_us"], 0.0, abs_tol=1e-6)
        assert math.isclose(
            features["see_vulnerability_them"], 0.0, abs_tol=1e-6
        )

    def test_see_undefended_piece(self):
        """An undefended knight should give the opponent positive SEE
        advantage and show up as a vulnerability.
        """
        # Black knight on e5 undefended, White pawn on d4 attacks it.
        board = chess.Board(
            "rnbqkb1r/pppppppp/8/4n3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1"
        )
        features = baseline_extract_features(board)

        # White (us) can profitably capture the knight.
        assert features["see_advantage_us"] > 0, (
            "White should have positive SEE advantage (can capture knight)"
        )
        # Black's knight is vulnerable.
        assert features["see_vulnerability_them"] >= 1.0, (
            "Black should have at least 1 vulnerable piece"
        )

    def test_see_defended_piece_no_advantage(self):
        """A pawn-defended knight should yield no SEE advantage for
        the attacker because Nxe5 dxe5 is an equal trade.
        """
        # Black knight on e5 defended by pawn on d6.
        board = chess.Board(
            "rnbqkb1r/ppp1pppp/3p4/4n3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
        )
        features = baseline_extract_features(board)
        # SEE of Nf3xe5 dxe5 is ~0 (knight for knight equivalent), not
        # positive, so see_advantage should remain 0.
        assert math.isclose(features["see_advantage_us"], 0.0, abs_tol=0.15)

    # ── Phase-interpolated PST tests ─────────────────────────────────

    def test_pst_features_present(self):
        """PST features should exist for both sides."""
        board = chess.Board()
        features = baseline_extract_features(board)
        assert "pst_us" in features
        assert "pst_them" in features

    def test_pst_symmetric_initial_position(self):
        """In the starting position the PST score should be equal for
        both sides because the position is symmetric.
        """
        board = chess.Board()
        features = baseline_extract_features(board)
        assert math.isclose(
            features["pst_us"], features["pst_them"], abs_tol=0.01
        ), f"PST should be equal: us={features['pst_us']}, them={features['pst_them']}"

    def test_pst_phase_interpolation(self):
        """With very few pieces (endgame), the PST should reflect
        endgame king values — a centralized king should score higher
        than a cornered king.
        """
        # King + pawn endgame: king on e4 (centralized) vs king on a1
        # phase ~ 0 (pure endgame)
        board_central = chess.Board("8/8/8/8/4K3/8/P7/k7 w - - 0 1")
        board_corner = chess.Board("8/8/8/8/8/8/P7/K6k b - - 0 1")

        feats_central = baseline_extract_features(board_central)
        feats_corner = baseline_extract_features(board_corner)

        # In endgame PST, a centralized king scores higher
        assert feats_central["pst_us"] > feats_corner["pst_us"], (
            "Centralized king should have higher PST than corner king in endgame"
        )

    # ── Pawn structure cache consistency tests ───────────────────────

    def test_pawn_features_consistent(self):
        """Pawn structure features should be consistent between
        repeated calls on the same position (cache correctness).
        """
        board = chess.Board()
        f1 = baseline_extract_features(board)
        f2 = baseline_extract_features(board)

        pawn_keys = [
            "isolated_pawns_us",
            "isolated_pawns_them",
            "doubled_pawns_us",
            "doubled_pawns_them",
            "backward_pawns_us",
            "backward_pawns_them",
            "passed_us",
            "passed_them",
            "pawn_chain_us",
            "pawn_chain_them",
        ]
        for key in pawn_keys:
            assert math.isclose(f1[key], f2[key], abs_tol=1e-6), (
                f"{key} changed between calls: {f1[key]} vs {f2[key]}"
            )
