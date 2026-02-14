"""Tests for CLI functionality."""

import contextlib
from io import StringIO
from unittest.mock import Mock, mock_open, patch

import chess
import pytest

from chess_ai.cli.audit import main


class TestCLI:
    """Test command-line interface functionality."""

    def test_cli_help(self):
        """Test CLI help output."""
        with patch("sys.argv", ["cli.py", "--help"]), pytest.raises(
            SystemExit
        ) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_cli_no_engine(self):
        """Test CLI without engine path."""
        with patch("sys.argv", ["cli.py", "--baseline_features"]), patch(
            "os.environ.get", return_value=""
        ), pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_cli_with_engine_env_var(self):
        """Test CLI with engine from environment variable."""
        with patch(
            "sys.argv", ["cli.py", "--baseline_features", "--positions", "2"]
        ), patch("os.environ.get", return_value="/path/to/stockfish"), patch(
            "chess_ai.cli.audit.sf_open"
        ) as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Mock the audit function to prevent it from running
            mock_audit.side_effect = lambda *args, **kwargs: mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock(), Mock()]

            # Should not raise SystemExit
            try:
                main()
            except SystemExit:
                pytest.fail("CLI should not exit with SystemExit")

    def test_cli_with_engine_argument(self):
        """Test CLI with engine as command line argument."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/custom/stockfish",
                "--baseline_features",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock(), Mock()]

            # Should not raise SystemExit
            try:
                main()
            except SystemExit:
                pytest.fail("CLI should not exit with SystemExit")

    def test_cli_baseline_features(self):
        """Test CLI with baseline features."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock(), Mock()]

            try:
                main()
            except SystemExit:
                pytest.fail("CLI should not exit with SystemExit")

    def test_cli_features_module(self):
        """Test CLI with features module."""
        import os
        import tempfile

        # Create a temporary features file
        features_code = """
def extract_features(board):
    return {"test_feature": 1.0, "_engine_probes": {}}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(features_code)
            temp_path = f.name

        try:
            with patch(
                "sys.argv",
                [
                    "cli.py",
                    "--engine",
                    "/path/to/stockfish",
                    "--features_module",
                    temp_path,
                    "--positions",
                    "2",
                ],
            ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
                "chess_ai.cli.audit.audit_feature_set"
            ) as mock_audit, patch(
                "chess_ai.utils.sampling.sample_random_positions"
            ) as mock_sample:
                # Mock the audit function to return a proper result without running the actual audit
                class MockResult:
                    def __init__(self):
                        self.r2 = 0.75
                        self.tau_mean = 0.6
                        self.tau_covered = 10
                        self.n_tau = 15
                        self.local_faithfulness = 0.8
                        self.local_faithfulness_decisive = 0.85
                        self.sparsity_mean = 3.5
                        self.coverage_ratio = 0.7
                        self.top_features_by_coef = [("material_diff", 0.5)]
                        self.stable_features = ["material_diff"]

                mock_result = MockResult()
                mock_audit.return_value = mock_result
                # Mock the audit function to prevent it from running
                mock_audit.side_effect = lambda *args, **kwargs: mock_result
                # Setup mocks
                mock_engine = Mock()
                # Mock the analyse method to return proper structure
                mock_score = Mock()
                mock_score.pov.return_value = mock_score
                mock_score.score.return_value = 0.5
                # Mock a proper move for the PV
                mock_move = chess.Move.from_uci("e2e4")
                mock_engine.analyse.return_value = [
                    {"score": mock_score, "pv": [mock_move]},
                    {"score": mock_score, "pv": [mock_move]},
                    {"score": mock_score, "pv": [mock_move]},
                ]
                mock_sf_open.return_value = mock_engine
                mock_sample.return_value = [Mock(), Mock()]

                try:
                    main()
                except SystemExit:
                    pytest.fail("CLI should not exit with SystemExit")
        finally:
            os.unlink(temp_path)

    def test_cli_no_features_specified(self):
        """Test CLI without specifying features."""
        with patch(
            "sys.argv", ["cli.py", "--engine", "/path/to/stockfish"]
        ), pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_cli_with_pgn_file(self):
        """Test CLI with PGN file."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--pgn",
                "/path/to/games.pgn",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "builtins.open", mock_open(read_data='[Event "Test"]\n1. e4 e5')
        ), patch(
            "chess_ai.utils.sampling.sample_positions_from_pgn"
        ) as mock_sample_pgn, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample_random:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample_pgn.return_value = [Mock(), Mock()]
            mock_sample_random.return_value = []

            try:
                main()
            except SystemExit:
                pytest.fail("CLI should not exit with SystemExit")

    def test_cli_custom_parameters(self):
        """Test CLI with custom parameters."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--positions",
                "10",
                "--depth",
                "20",
                "--threads",
                "4",
                "--multipv",
                "5",
                "--test_size",
                "0.3",
                "--alpha",
                "0.05",
                "--gap",
                "100.0",
                "--seed",
                "123",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock() for _ in range(10)]

            try:
                main()
            except SystemExit:
                pytest.fail("CLI should not exit with SystemExit")

    def test_cli_movetime_instead_of_depth(self):
        """Test CLI with movetime instead of depth."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--depth",
                "0",
                "--movetime",
                "5000",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock(), Mock()]

            try:
                main()
            except SystemExit:
                pytest.fail("CLI should not exit with SystemExit")

    def test_cli_ply_skip_parameter(self):
        """Test CLI with ply_skip parameter."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--pgn",
                "/path/to/games.pgn",
                "--ply-skip",
                "4",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "builtins.open", mock_open(read_data='[Event "Test"]\n1. e4 e5')
        ), patch(
            "chess_ai.utils.sampling.sample_positions_from_pgn"
        ) as mock_sample_pgn, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample_random:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample_pgn.return_value = [Mock(), Mock()]
            mock_sample_random.return_value = []

            try:
                main()
            except SystemExit:
                pytest.fail("CLI should not exit with SystemExit")

    def test_cli_engine_quit_called(self):
        """Test that engine.quit() is called."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock(), Mock()]

            with contextlib.suppress(SystemExit):
                main()

            # Check that engine.quit() was called
            mock_engine.quit.assert_called_once()

    def test_cli_engine_quit_called_on_exception(self):
        """Test that engine.quit() is called even when an exception occurs."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock(), Mock()]
            mock_audit.side_effect = Exception("Audit failed")

            try:
                main()
            except SystemExit:
                pass
            except Exception:  # noqa: S110
                # Expected due to mock exception
                pass

            # Check that engine.quit() was called even after exception
            mock_engine.quit.assert_called_once()

    def test_cli_output_formatting(self):
        """Test CLI output formatting."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample, patch(
            "sys.stdout", new_callable=StringIO
        ) as mock_stdout:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock(), Mock()]

            with contextlib.suppress(SystemExit):
                main()

            # Check that output was printed
            output = mock_stdout.getvalue()
            assert "Explainability Audit Report" in output
            assert "Fidelity (Delta-R^2)" in output
            assert "Move ranking (Kendall tau)" in output
            assert "Local faithfulness" in output
            assert "Sparsity" in output
            assert "Coverage" in output
            assert "Top features by |coef|" in output
            assert "Guidance:" in output

    def test_cli_random_seed_setting(self):
        """Test that random seed is set correctly."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--baseline_features",
                "--seed",
                "42",
                "--positions",
                "2",
            ],
        ), patch("chess_ai.cli.audit.sf_open") as mock_sf_open, patch(
            "chess_ai.cli.audit.audit_feature_set"
        ) as mock_audit, patch(
            "random.seed"
        ) as mock_random_seed, patch(
            "chess_ai.cli.audit.np.random.seed"
        ) as mock_numpy_seed, patch(
            "chess_ai.utils.sampling.sample_random_positions"
        ) as mock_sample:
            # Mock the audit function to return a proper result without running the actual audit
            class MockResult:
                def __init__(self):
                    self.r2 = 0.75
                    self.tau_mean = 0.6
                    self.tau_covered = 10
                    self.n_tau = 15
                    self.local_faithfulness = 0.8
                    self.local_faithfulness_decisive = 0.85
                    self.sparsity_mean = 3.5
                    self.coverage_ratio = 0.7
                    self.top_features_by_coef = [("material_diff", 0.5)]
                    self.stable_features = ["material_diff"]

            mock_result = MockResult()
            mock_audit.return_value = mock_result
            # Mock the seed functions to be called
            mock_random_seed.return_value = None
            mock_numpy_seed.return_value = None
            # Setup mocks
            mock_engine = Mock()
            # Mock the analyse method to return proper structure
            mock_score = Mock()
            mock_score.pov.return_value = mock_score
            mock_score.score.return_value = 0.5
            # Mock a proper move for the PV
            mock_move = chess.Move.from_uci("e2e4")
            mock_engine.analyse.return_value = [
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
                {"score": mock_score, "pv": [mock_move]},
            ]
            mock_sf_open.return_value = mock_engine
            mock_sample.return_value = [Mock(), Mock()]

            with contextlib.suppress(SystemExit):
                main()

            # Check that seeds were set
            mock_random_seed.assert_called_once_with(42)
            mock_numpy_seed.assert_called_once_with(42)
