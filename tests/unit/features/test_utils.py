"""Tests for feature utilities."""

import os
import tempfile

import pytest

from chess_feature_audit.features.utils import load_feature_module


class TestFeatureUtils:
    """Test feature utility functions."""

    def test_load_feature_module_success(self):
        """Test successful loading of a feature module."""
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

            # Check that the module was loaded correctly
            assert hasattr(module, "extract_features")
            assert callable(module.extract_features)

            # Test that the function works
            import chess

            board = chess.Board()
            features = module.extract_features(board)
            assert isinstance(features, dict)
            assert "material" in features
            assert "mobility" in features
            assert "center_control" in features

        finally:
            os.unlink(temp_path)

    def test_load_feature_module_file_not_found(self):
        """Test loading a non-existent feature module."""
        with pytest.raises((RuntimeError, FileNotFoundError)):
            load_feature_module("/nonexistent/path/features.py")

    def test_load_feature_module_missing_extract_features(self):
        """Test loading a module without extract_features function."""
        # Create a temporary module without extract_features
        feature_code = """
def some_other_function():
    return "not extract_features"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(feature_code)
            temp_path = f.name

        try:
            with pytest.raises(
                RuntimeError, match="Feature module must define extract_features"
            ):
                load_feature_module(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_feature_module_invalid_python(self):
        """Test loading a module with invalid Python syntax."""
        # Create a temporary module with invalid syntax
        feature_code = """
def extract_features(board):
    invalid syntax here
    return {}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(feature_code)
            temp_path = f.name

        try:
            with pytest.raises(
                (SyntaxError, TypeError)
            ):  # Should raise syntax or type error
                load_feature_module(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_feature_module_with_imports(self):
        """Test loading a feature module that imports other modules."""
        feature_code = '''
import chess

def extract_features(board):
    """Extract features using chess library."""
    return {
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "turn": board.turn
    }
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(feature_code)
            temp_path = f.name

        try:
            module = load_feature_module(temp_path)

            # Test that the function works with chess imports
            import chess

            board = chess.Board()
            features = module.extract_features(board)

            assert isinstance(features, dict)
            assert "is_check" in features
            assert "is_checkmate" in features
            assert "is_stalemate" in features
            assert "turn" in features
            assert isinstance(features["turn"], bool)

        finally:
            os.unlink(temp_path)

    def test_load_feature_module_complex_features(self):
        """Test loading a feature module with complex feature extraction."""
        feature_code = '''
def extract_features(board):
    """Extract complex features from a chess board."""
    features = {}

    # Material count
    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # pawn, knight, bishop, rook, queen, king
    for color in [True, False]:
        material = 0
        for piece_type, value in piece_values.items():
            material += len(board.pieces(piece_type, color)) * value
        features[f"material_{color}"] = material

    # Mobility
    features["mobility"] = len(list(board.legal_moves))

    # Game state
    features["is_check"] = board.is_check()
    features["is_checkmate"] = board.is_checkmate()
    features["is_stalemate"] = board.is_stalemate()
    features["is_game_over"] = board.is_game_over()

    return features
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(feature_code)
            temp_path = f.name

        try:
            module = load_feature_module(temp_path)

            import chess

            board = chess.Board()
            features = module.extract_features(board)

            # Check complex features
            assert "material_True" in features
            assert "material_False" in features
            assert "mobility" in features
            assert "is_check" in features
            assert "is_checkmate" in features
            assert "is_stalemate" in features
            assert "is_game_over" in features

            # Check that material is equal for both sides in initial position
            assert features["material_True"] == features["material_False"]

        finally:
            os.unlink(temp_path)
