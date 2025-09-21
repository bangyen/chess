"""Stockfish engine configuration."""

from dataclasses import dataclass


@dataclass
class SFConfig:
    """Configuration for Stockfish engine."""

    engine_path: str
    depth: int = 16
    movetime: int = 0
    multipv: int = 3
    threads: int = 1
