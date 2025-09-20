"""Tests for engine configuration."""

from chess_feature_audit.engine.config import SFConfig


class TestSFConfig:
    """Test Stockfish configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SFConfig(engine_path="/path/to/stockfish")

        assert config.engine_path == "/path/to/stockfish"
        assert config.depth == 16
        assert config.movetime == 0
        assert config.multipv == 3
        assert config.threads == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SFConfig(
            engine_path="/custom/path/stockfish",
            depth=20,
            movetime=5000,
            multipv=5,
            threads=4,
        )

        assert config.engine_path == "/custom/path/stockfish"
        assert config.depth == 20
        assert config.movetime == 5000
        assert config.multipv == 5
        assert config.threads == 4

    def test_config_immutability(self):
        """Test that config fields can be modified (dataclass allows this)."""
        config = SFConfig(engine_path="/path/to/stockfish")

        # Dataclasses are mutable by default
        config.depth = 20
        assert config.depth == 20

        config.multipv = 5
        assert config.multipv == 5

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = SFConfig(engine_path="/path/to/stockfish", depth=16)
        config2 = SFConfig(engine_path="/path/to/stockfish", depth=16)
        config3 = SFConfig(engine_path="/path/to/stockfish", depth=20)

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        """Test config string representation."""
        config = SFConfig(engine_path="/path/to/stockfish", depth=16)
        repr_str = repr(config)

        assert "SFConfig" in repr_str
        assert "/path/to/stockfish" in repr_str
        assert "depth=16" in repr_str

    def test_config_with_zero_values(self):
        """Test config with zero values."""
        config = SFConfig(
            engine_path="/path/to/stockfish", depth=0, movetime=0, multipv=0, threads=0
        )

        assert config.depth == 0
        assert config.movetime == 0
        assert config.multipv == 0
        assert config.threads == 0

    def test_config_with_negative_values(self):
        """Test config with negative values (should be allowed)."""
        config = SFConfig(
            engine_path="/path/to/stockfish",
            depth=-1,
            movetime=-1000,
            multipv=-1,
            threads=-1,
        )

        assert config.depth == -1
        assert config.movetime == -1000
        assert config.multipv == -1
        assert config.threads == -1

    def test_config_with_large_values(self):
        """Test config with large values."""
        config = SFConfig(
            engine_path="/path/to/stockfish",
            depth=100,
            movetime=60000,
            multipv=20,
            threads=32,
        )

        assert config.depth == 100
        assert config.movetime == 60000
        assert config.multipv == 20
        assert config.threads == 32
