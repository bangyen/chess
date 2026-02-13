"""Extended tests for cli/audit.py to increase coverage.

Targets uncovered lines: _parse_phase_weights validation, main() with
missing engine, missing features module, and basic argument handling.
"""

import argparse
import json
from unittest.mock import patch

import pytest

from chess_ai.cli.audit import _parse_phase_weights


class TestParsePhaseWeights:
    """Tests for the _parse_phase_weights argument parser helper."""

    def test_valid_json(self):
        """Parses valid JSON with known phases."""
        raw = json.dumps({"opening": 0.25, "middlegame": 0.50, "endgame": 0.25})
        result = _parse_phase_weights(raw)
        assert result == {"opening": 0.25, "middlegame": 0.50, "endgame": 0.25}

    def test_invalid_json(self):
        """Raises ArgumentTypeError for malformed JSON."""
        with pytest.raises(argparse.ArgumentTypeError, match="valid JSON"):
            _parse_phase_weights("{not valid}")

    def test_unknown_phase(self):
        """Raises ArgumentTypeError for unknown phase names."""
        raw = json.dumps({"opening": 0.5, "lategame": 0.5})
        with pytest.raises(argparse.ArgumentTypeError, match="Unknown phase"):
            _parse_phase_weights(raw)

    def test_negative_weight(self):
        """Raises ArgumentTypeError for non-positive weights."""
        raw = json.dumps({"opening": -1.0})
        with pytest.raises(argparse.ArgumentTypeError, match="positive number"):
            _parse_phase_weights(raw)

    def test_zero_weight(self):
        """Raises ArgumentTypeError for zero weight."""
        raw = json.dumps({"opening": 0})
        with pytest.raises(argparse.ArgumentTypeError, match="positive number"):
            _parse_phase_weights(raw)

    def test_string_weight(self):
        """Raises ArgumentTypeError for non-numeric weight."""
        raw = json.dumps({"opening": "high"})
        with pytest.raises(argparse.ArgumentTypeError, match="positive number"):
            _parse_phase_weights(raw)

    def test_partial_phases(self):
        """Allows specifying only a subset of phases."""
        raw = json.dumps({"opening": 1.0})
        result = _parse_phase_weights(raw)
        assert result == {"opening": 1.0}


class TestMainExitCases:
    """Test main() early exit branches."""

    @patch("sys.argv", ["audit", "--engine", ""])
    def test_no_engine_exits(self):
        """main() exits with code 1 when no engine is provided."""
        from chess_ai.cli.audit import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch(
        "sys.argv",
        ["audit", "--engine", "/some/engine"],
    )
    def test_no_features_module_exits(self):
        """main() exits when neither --baseline_features nor --features_module is given."""
        from chess_ai.cli.audit import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
