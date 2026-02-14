"""Tests for CLI main module."""

import sys
from unittest.mock import patch

import pytest

from chess_ai.cli import main

# Get the actual module object (not the function exported from __init__.py)
main_module = sys.modules["chess_ai.cli.main"]


class TestMain:
    """Test main CLI dispatcher function."""

    def test_main_no_args(self, capsys):
        """Test main with no arguments."""
        with patch("sys.argv", ["chess-ai"]):
            main()

        captured = capsys.readouterr()
        assert "Chess AI Tools" in captured.out
        assert "Available commands:" in captured.out
        assert "audit" in captured.out
        assert "play" in captured.out
        assert "help" in captured.out

    def test_main_help_command(self, capsys):
        """Test main with help command."""
        with patch("sys.argv", ["chess-ai", "help"]):
            main()

        captured = capsys.readouterr()
        assert "Chess AI Tools" in captured.out
        assert "Available commands:" in captured.out

    def test_main_h_flag(self, capsys):
        """Test main with -h flag."""
        with patch("sys.argv", ["chess-ai", "-h"]):
            main()

        captured = capsys.readouterr()
        assert "Chess AI Tools" in captured.out

    def test_main_help_flag(self, capsys):
        """Test main with --help flag."""
        with patch("sys.argv", ["chess-ai", "--help"]):
            main()

        captured = capsys.readouterr()
        assert "Chess AI Tools" in captured.out

    @patch.object(main_module, "audit_main")
    def test_main_audit_command(self, mock_audit_main):
        """Test main with audit command."""
        with patch("sys.argv", ["chess-ai", "audit", "--positions", "100"]):
            main()

        mock_audit_main.assert_called_once()

    @patch.object(main_module, "explainable_main")
    def test_main_play_command(self, mock_explainable_main):
        """Test main with play command."""
        with patch("sys.argv", ["chess-ai", "play", "--strength", "intermediate"]):
            main()

        mock_explainable_main.assert_called_once()

    @patch.object(main_module, "audit_main")
    def test_main_audit_command_with_args(self, mock_audit_main):
        """Test main with audit command and arguments."""
        with patch(
            "sys.argv",
            ["chess-ai", "audit", "--baseline_features", "--positions", "50"],
        ):
            main()

        mock_audit_main.assert_called_once()

    @patch.object(main_module, "explainable_main")
    def test_main_play_command_with_args(self, mock_explainable_main):
        """Test main with play command and arguments."""
        with patch(
            "sys.argv", ["chess-ai", "play", "--depth", "20", "--strength", "expert"]
        ):
            main()

        mock_explainable_main.assert_called_once()

    def test_main_unknown_command(self, capsys):
        """Test main with unknown command."""
        with patch("sys.argv", ["chess-ai", "unknown"]), pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "Unknown command: unknown" in captured.out
        assert "Available commands: audit, play, help" in captured.out

    @patch.object(main_module, "audit_main")
    def test_main_audit_command_sys_argv_restoration(self, mock_audit_main):
        """Test that sys.argv is properly restored after audit command."""
        original_argv = sys.argv.copy()

        with patch("sys.argv", ["chess-ai", "audit", "--positions", "100"]):
            main()

        # sys.argv should be restored to original
        assert sys.argv == original_argv
        mock_audit_main.assert_called_once()

    @patch.object(main_module, "explainable_main")
    def test_main_play_command_sys_argv_restoration(self, mock_explainable_main):
        """Test that sys.argv is properly restored after play command."""
        original_argv = sys.argv.copy()

        with patch("sys.argv", ["chess-ai", "play", "--strength", "intermediate"]):
            main()

        # sys.argv should be restored to original
        assert sys.argv == original_argv
        mock_explainable_main.assert_called_once()

    @patch.object(main_module, "audit_main")
    def test_main_audit_command_exception_handling(self, mock_audit_main):
        """Test that exceptions in audit command are properly handled."""
        mock_audit_main.side_effect = Exception("Audit error")

        with patch("sys.argv", ["chess-ai", "audit"]), pytest.raises(
            Exception, match="Audit error"
        ):
            main()

    @patch.object(main_module, "explainable_main")
    def test_main_play_command_exception_handling(self, mock_explainable_main):
        """Test that exceptions in play command are properly handled."""
        mock_explainable_main.side_effect = Exception("Play error")

        with patch("sys.argv", ["chess-ai", "play"]), pytest.raises(
            Exception, match="Play error"
        ):
            main()

    def test_main_with_args_parameter(self):
        """Test main with explicit args parameter."""
        with patch.object(main_module, "audit_main") as mock_audit_main:
            main(["audit", "--positions", "100"])

        mock_audit_main.assert_called_once()

    def test_main_with_args_parameter_play(self):
        """Test main with explicit args parameter for play command."""
        with patch.object(main_module, "explainable_main") as mock_explainable_main:
            main(["play", "--strength", "intermediate"])

        mock_explainable_main.assert_called_once()

    def test_main_with_args_parameter_help(self, capsys):
        """Test main with explicit args parameter for help."""
        main(["help"])

        captured = capsys.readouterr()
        assert "Chess AI Tools" in captured.out

    def test_main_with_args_parameter_unknown(self, capsys):
        """Test main with explicit args parameter for unknown command."""
        with pytest.raises(SystemExit):
            main(["unknown"])

        captured = capsys.readouterr()
        assert "Unknown command: unknown" in captured.out

    def test_main_empty_args(self, capsys):
        """Test main with empty args list."""
        main([])

        captured = capsys.readouterr()
        assert "Chess AI Tools" in captured.out

    def test_main_args_none(self, capsys):
        """Test main with args=None."""
        with patch("sys.argv", ["chess-ai"]):
            main(None)

        captured = capsys.readouterr()
        assert "Chess AI Tools" in captured.out

    @patch.object(main_module, "audit_main")
    def test_main_audit_command_multiple_args(self, mock_audit_main):
        """Test main with audit command and multiple arguments."""
        with patch(
            "sys.argv",
            [
                "chess-ai",
                "audit",
                "--baseline_features",
                "--positions",
                "100",
                "--depth",
                "20",
            ],
        ):
            main()

        mock_audit_main.assert_called_once()

    @patch.object(main_module, "explainable_main")
    def test_main_play_command_multiple_args(self, mock_explainable_main):
        """Test main with play command and multiple arguments."""
        with patch(
            "sys.argv",
            [
                "chess-ai",
                "play",
                "--depth",
                "20",
                "--strength",
                "expert",
                "--engine",
                "/path/to/stockfish",
            ],
        ):
            main()

        mock_explainable_main.assert_called_once()

    def test_main_command_case_sensitivity(self, capsys):
        """Test that commands are case sensitive."""
        with patch("sys.argv", ["chess-ai", "AUDIT"]), pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "Unknown command: AUDIT" in captured.out

    def test_main_command_case_sensitivity_play(self, capsys):
        """Test that play command is case sensitive."""
        with patch("sys.argv", ["chess-ai", "PLAY"]), pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "Unknown command: PLAY" in captured.out

    @patch.object(main_module, "audit_main")
    def test_main_audit_command_no_additional_args(self, mock_audit_main):
        """Test main with audit command and no additional arguments."""
        with patch("sys.argv", ["chess-ai", "audit"]):
            main()

        mock_audit_main.assert_called_once()

    @patch.object(main_module, "explainable_main")
    def test_main_play_command_no_additional_args(self, mock_explainable_main):
        """Test main with play command and no additional arguments."""
        with patch("sys.argv", ["chess-ai", "play"]):
            main()

        mock_explainable_main.assert_called_once()

    def test_main_help_examples_in_output(self, capsys):
        """Test that help output includes examples."""
        with patch("sys.argv", ["chess-ai"]):
            main()

        captured = capsys.readouterr()
        assert "Examples:" in captured.out
        assert "chess-ai audit" in captured.out
        assert "chess-ai play" in captured.out

    def test_main_help_description(self, capsys):
        """Test that help output includes proper description."""
        with patch("sys.argv", ["chess-ai"]):
            main()

        captured = capsys.readouterr()
        assert "Feature analysis and explainable gameplay" in captured.out
