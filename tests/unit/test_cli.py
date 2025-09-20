"""Tests for CLI functionality."""

from io import StringIO
from unittest.mock import Mock, patch

import pytest

from chess_feature_audit.cli import main


class TestCLI:
    """Test command-line interface functionality."""

    def test_cli_help(self):
        """Test CLI help output."""
        with patch("sys.argv", ["cli.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_cli_no_engine(self):
        """Test CLI without engine path."""
        with patch("sys.argv", ["cli.py", "--baseline_features"]):
            with patch("os.environ.get", return_value=""):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_cli_with_engine_env_var(self):
        """Test CLI with engine from environment variable."""
        with patch("sys.argv", ["cli.py", "--baseline_features", "--positions", "2"]):
            with patch("os.environ.get", return_value="/path/to/stockfish"):
                with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                    with patch(
                        "chess_feature_audit.cli.audit_feature_set"
                    ) as mock_audit:
                        with patch(
                            "chess_feature_audit.cli.sample_random_positions"
                        ) as mock_sample:
                            # Setup mocks
                            mock_engine = Mock()
                            mock_sf_open.return_value = mock_engine
                            mock_sample.return_value = [Mock(), Mock()]
                            mock_audit.return_value = Mock()

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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        # Setup mocks
                        mock_engine = Mock()
                        mock_sf_open.return_value = mock_engine
                        mock_sample.return_value = [Mock(), Mock()]
                        mock_audit.return_value = Mock()

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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        # Setup mocks
                        mock_engine = Mock()
                        mock_sf_open.return_value = mock_engine
                        mock_sample.return_value = [Mock(), Mock()]
                        mock_audit.return_value = Mock()

                        try:
                            main()
                        except SystemExit:
                            pytest.fail("CLI should not exit with SystemExit")

    def test_cli_features_module(self):
        """Test CLI with features module."""
        with patch(
            "sys.argv",
            [
                "cli.py",
                "--engine",
                "/path/to/stockfish",
                "--features_module",
                "/path/to/features.py",
                "--positions",
                "2",
            ],
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        with patch(
                            "chess_feature_audit.cli.load_feature_module"
                        ) as mock_load:
                            # Setup mocks
                            mock_engine = Mock()
                            mock_sf_open.return_value = mock_engine
                            mock_sample.return_value = [Mock(), Mock()]
                            mock_audit.return_value = Mock()
                            mock_module = Mock()
                            mock_module.extract_features = Mock()
                            mock_load.return_value = mock_module

                            try:
                                main()
                            except SystemExit:
                                pytest.fail("CLI should not exit with SystemExit")

    def test_cli_no_features_specified(self):
        """Test CLI without specifying features."""
        with patch("sys.argv", ["cli.py", "--engine", "/path/to/stockfish"]):
            with pytest.raises(SystemExit) as exc_info:
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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_positions_from_pgn"
                    ) as mock_sample_pgn:
                        with patch(
                            "chess_feature_audit.cli.sample_random_positions"
                        ) as mock_sample_random:
                            # Setup mocks
                            mock_engine = Mock()
                            mock_sf_open.return_value = mock_engine
                            mock_sample_pgn.return_value = [Mock(), Mock()]
                            mock_sample_random.return_value = []
                            mock_audit.return_value = Mock()

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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        # Setup mocks
                        mock_engine = Mock()
                        mock_sf_open.return_value = mock_engine
                        mock_sample.return_value = [Mock() for _ in range(10)]
                        mock_audit.return_value = Mock()

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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        # Setup mocks
                        mock_engine = Mock()
                        mock_sf_open.return_value = mock_engine
                        mock_sample.return_value = [Mock(), Mock()]
                        mock_audit.return_value = Mock()

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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_positions_from_pgn"
                    ) as mock_sample_pgn:
                        with patch(
                            "chess_feature_audit.cli.sample_random_positions"
                        ) as mock_sample_random:
                            # Setup mocks
                            mock_engine = Mock()
                            mock_sf_open.return_value = mock_engine
                            mock_sample_pgn.return_value = [Mock(), Mock()]
                            mock_sample_random.return_value = []
                            mock_audit.return_value = Mock()

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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        # Setup mocks
                        mock_engine = Mock()
                        mock_sf_open.return_value = mock_engine
                        mock_sample.return_value = [Mock(), Mock()]
                        mock_audit.return_value = Mock()

                        try:
                            main()
                        except SystemExit:
                            pass

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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        # Setup mocks
                        mock_engine = Mock()
                        mock_sf_open.return_value = mock_engine
                        mock_sample.return_value = [Mock(), Mock()]
                        mock_audit.side_effect = Exception("Audit failed")

                        try:
                            main()
                        except SystemExit:
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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                            # Setup mocks
                            mock_engine = Mock()
                            mock_sf_open.return_value = mock_engine
                            mock_sample.return_value = [Mock(), Mock()]

                            # Create mock audit result
                            mock_result = Mock()
                            mock_result.r2 = 0.75
                            mock_result.tau_mean = 0.6
                            mock_result.tau_covered = 10
                            mock_result.n_tau = 15
                            mock_result.local_faithfulness = 0.8
                            mock_result.local_faithfulness_decisive = 0.85
                            mock_result.sparsity_mean = 3.5
                            mock_result.coverage_ratio = 0.7
                            mock_result.top_features_by_coef = [
                                ("material_diff", 0.5),
                                ("mobility_us", 0.3),
                            ]
                            mock_result.stable_features = ["material_diff"]
                            mock_audit.return_value = mock_result

                            try:
                                main()
                            except SystemExit:
                                pass

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
        ):
            with patch("chess_feature_audit.cli.sf_open") as mock_sf_open:
                with patch("chess_feature_audit.cli.audit_feature_set") as mock_audit:
                    with patch(
                        "chess_feature_audit.cli.sample_random_positions"
                    ) as mock_sample:
                        with patch("random.seed") as mock_random_seed:
                            with patch("numpy.random.seed") as mock_numpy_seed:
                                # Setup mocks
                                mock_engine = Mock()
                                mock_sf_open.return_value = mock_engine
                                mock_sample.return_value = [Mock(), Mock()]
                                mock_audit.return_value = Mock()

                                try:
                                    main()
                                except SystemExit:
                                    pass

                                # Check that seeds were set
                                mock_random_seed.assert_called_once_with(42)
                                mock_numpy_seed.assert_called_once_with(42)
