"""Tests for CLI explainable module."""

from unittest.mock import Mock, patch

import pytest

from chess_ai.cli.explainable import find_stockfish, main


class TestFindStockfish:
    """Test find_stockfish function."""

    @patch("os.path.isfile")
    @patch("os.access")
    def test_find_stockfish_common_paths(self, mock_access, mock_isfile):
        """Test finding Stockfish in common paths."""
        # Mock first path to exist and be executable
        mock_isfile.side_effect = lambda path: path == "/opt/homebrew/bin/stockfish"
        mock_access.side_effect = (
            lambda path, mode: path == "/opt/homebrew/bin/stockfish"
        )

        result = find_stockfish()

        assert result == "/opt/homebrew/bin/stockfish"

    @patch("os.path.isfile")
    @patch("os.access")
    @patch("shutil.which")
    def test_find_stockfish_in_path(self, mock_which, mock_access, mock_isfile):
        """Test finding Stockfish in PATH."""
        # Mock all common paths to not exist
        mock_isfile.return_value = False
        mock_access.return_value = False
        mock_which.return_value = "/usr/local/bin/stockfish"

        result = find_stockfish()

        assert result == "/usr/local/bin/stockfish"
        mock_which.assert_called_once_with("stockfish")

    @patch("os.path.isfile")
    @patch("os.access")
    @patch("shutil.which")
    def test_find_stockfish_not_found(self, mock_which, mock_access, mock_isfile):
        """Test when Stockfish is not found."""
        # Mock all paths to not exist
        mock_isfile.return_value = False
        mock_access.return_value = False
        mock_which.return_value = None

        with pytest.raises(FileNotFoundError, match="Stockfish not found"):
            find_stockfish()

    @patch("os.path.isfile")
    @patch("os.access")
    def test_find_stockfish_not_executable(self, mock_access, mock_isfile):
        """Test when Stockfish exists but is not executable."""
        # Mock file to exist but not be executable
        mock_isfile.return_value = True
        mock_access.return_value = False

        with pytest.raises(FileNotFoundError, match="Stockfish not found"):
            find_stockfish()


class TestMain:
    """Test main CLI function."""

    @patch("chess_ai.cli.explainable.find_stockfish")
    @patch("chess_ai.cli.explainable.ExplainableChessEngine")
    def test_main_default_args(self, mock_engine_class, mock_find_stockfish):
        """Test main with default arguments."""
        mock_find_stockfish.return_value = "/path/to/stockfish"
        mock_engine = Mock()
        mock_engine_class.return_value.__enter__.return_value = mock_engine
        mock_engine_class.return_value.__exit__.return_value = None

        # Mock sys.argv
        with patch("sys.argv", ["explainable"]):
            main()

        mock_find_stockfish.assert_called_once()
        mock_engine_class.assert_called_once_with("/path/to/stockfish", 16, "beginner")
        mock_engine.play_interactive_game.assert_called_once()

    @patch("chess_ai.cli.explainable.find_stockfish")
    @patch("chess_ai.cli.explainable.ExplainableChessEngine")
    def test_main_custom_args(self, mock_engine_class, mock_find_stockfish):
        """Test main with custom arguments."""
        mock_find_stockfish.return_value = "/path/to/stockfish"
        mock_engine = Mock()
        mock_engine_class.return_value.__enter__.return_value = mock_engine
        mock_engine_class.return_value.__exit__.return_value = None

        # Mock sys.argv with custom arguments
        with patch(
            "sys.argv",
            [
                "explainable",
                "--engine",
                "/custom/stockfish",
                "--depth",
                "20",
                "--strength",
                "expert",
            ],
        ):
            main()

        mock_engine_class.assert_called_once_with("/custom/stockfish", 20, "expert")

    @patch("chess_ai.cli.explainable.find_stockfish")
    def test_main_stockfish_not_found(self, mock_find_stockfish):
        """Test main when Stockfish is not found."""
        mock_find_stockfish.side_effect = FileNotFoundError("Stockfish not found")

        # Mock sys.argv
        with patch("sys.argv", ["explainable"]):
            with pytest.raises(SystemExit):
                main()

    @patch("chess_ai.cli.explainable.find_stockfish")
    @patch("chess_ai.cli.explainable.ExplainableChessEngine")
    def test_main_engine_error(self, mock_engine_class, mock_find_stockfish):
        """Test main when engine raises error."""
        mock_find_stockfish.return_value = "/path/to/stockfish"
        mock_engine_class.side_effect = FileNotFoundError("Engine not found")

        # Mock sys.argv
        with patch("sys.argv", ["explainable"]):
            with pytest.raises(SystemExit):
                main()

    @patch("chess_ai.cli.explainable.find_stockfish")
    @patch("chess_ai.cli.explainable.ExplainableChessEngine")
    def test_main_keyboard_interrupt(self, mock_engine_class, mock_find_stockfish):
        """Test main with keyboard interrupt."""
        mock_find_stockfish.return_value = "/path/to/stockfish"
        mock_engine = Mock()
        mock_engine_class.return_value.__enter__.return_value = mock_engine
        mock_engine_class.return_value.__exit__.return_value = None
        mock_engine.play_interactive_game.side_effect = KeyboardInterrupt()

        # Mock sys.argv
        with patch("sys.argv", ["explainable"]):
            # Should not raise SystemExit for KeyboardInterrupt
            main()

    @patch("chess_ai.cli.explainable.find_stockfish")
    @patch("chess_ai.cli.explainable.ExplainableChessEngine")
    def test_main_general_exception(self, mock_engine_class, mock_find_stockfish):
        """Test main with general exception."""
        mock_find_stockfish.return_value = "/path/to/stockfish"
        mock_engine_class.side_effect = Exception("General error")

        # Mock sys.argv
        with patch("sys.argv", ["explainable"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_help(self, capsys):
        """Test main help output."""
        # Mock sys.argv with help
        with patch("sys.argv", ["explainable", "--help"]):
            with pytest.raises(SystemExit):
                main()

        # Note: argparse help is printed to stderr, so we can't easily test it here
        # But we can verify that SystemExit is raised (which is expected for --help)

    def test_main_invalid_strength(self):
        """Test main with invalid strength argument."""
        # Mock sys.argv with invalid strength
        with patch("sys.argv", ["explainable", "--strength", "invalid"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_invalid_depth(self):
        """Test main with invalid depth argument."""
        # Mock sys.argv with invalid depth
        with patch("sys.argv", ["explainable", "--depth", "invalid"]):
            with pytest.raises(SystemExit):
                main()

    @patch("chess_ai.cli.explainable.find_stockfish")
    @patch("chess_ai.cli.explainable.ExplainableChessEngine")
    def test_main_all_strength_levels(self, mock_engine_class, mock_find_stockfish):
        """Test main with all strength levels."""
        mock_find_stockfish.return_value = "/path/to/stockfish"
        mock_engine = Mock()
        mock_engine_class.return_value.__enter__.return_value = mock_engine
        mock_engine_class.return_value.__exit__.return_value = None

        strength_levels = ["beginner", "novice", "intermediate", "advanced", "expert"]

        for strength in strength_levels:
            # Mock sys.argv with different strength
            with patch("sys.argv", ["explainable", "--strength", strength]):
                main()

            mock_engine_class.assert_called_with("/path/to/stockfish", 16, strength)
            mock_engine_class.reset_mock()

    @patch("chess_ai.cli.explainable.find_stockfish")
    @patch("chess_ai.cli.explainable.ExplainableChessEngine")
    def test_main_different_depths(self, mock_engine_class, mock_find_stockfish):
        """Test main with different depth values."""
        mock_find_stockfish.return_value = "/path/to/stockfish"
        mock_engine = Mock()
        mock_engine_class.return_value.__enter__.return_value = mock_engine
        mock_engine_class.return_value.__exit__.return_value = None

        depths = [1, 5, 10, 20, 30]

        for depth in depths:
            # Mock sys.argv with different depth
            with patch("sys.argv", ["explainable", "--depth", str(depth)]):
                main()

            mock_engine_class.assert_called_with(
                "/path/to/stockfish", depth, "beginner"
            )
            mock_engine_class.reset_mock()

    @patch("chess_ai.cli.explainable.find_stockfish")
    @patch("chess_ai.cli.explainable.ExplainableChessEngine")
    def test_main_combined_args(self, mock_engine_class, mock_find_stockfish):
        """Test main with combined custom arguments."""
        mock_find_stockfish.return_value = "/path/to/stockfish"
        mock_engine = Mock()
        mock_engine_class.return_value.__enter__.return_value = mock_engine
        mock_engine_class.return_value.__exit__.return_value = None

        # Mock sys.argv with combined arguments
        with patch(
            "sys.argv",
            [
                "explainable",
                "--engine",
                "/custom/stockfish",
                "--depth",
                "25",
                "--strength",
                "advanced",
            ],
        ):
            main()

        mock_engine_class.assert_called_once_with("/custom/stockfish", 25, "advanced")
