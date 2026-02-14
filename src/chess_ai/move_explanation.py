"""Data model for move explanations.

Separating the dataclass here avoids circular imports between the
engine module and the explanation-generation helpers.
"""

from dataclasses import dataclass

import chess


@dataclass
class MoveExplanation:
    """Explanation for why a move is good or bad."""

    move: chess.Move
    score: float
    reasons: list[tuple[str, float, str]]  # (feature_name, contribution, explanation)
    overall_explanation: str
