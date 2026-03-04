"""Surrogate model-based explanations for chess moves."""

import logging
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np

from .model_trainer import PhaseEnsemble

logger = logging.getLogger(__name__)


class SurrogateExplainer:
    """Converts surrogate model outputs to human-readable move explanations.

    Uses trained PhaseEnsemble model to calculate feature contributions in centipawns,
    then maps them to interpretable descriptions.
    """

    # Feature name to explanation template mapping
    FEATURE_TEMPLATES: ClassVar[dict[str, str]] = {
        "material_diff": "Gains a material advantage ({:+.0f} cp)",
        "mobility_us": "Increases piece activity and mobility ({:+.0f} cp)",
        "mobility_them": "Restricts opponent's piece activity ({:+.0f} cp)",
        "king_ring_pressure_us": "Increases attacking pressure near the opponent's king ({:+.0f} cp)",
        "king_ring_pressure_them": "Reduces attacking pressure on our own king ({:+.0f} cp)",
        "batteries_us": "Forms a powerful battery arrangement ({:+.0f} cp)",
        "outposts_us": "Establishes a strong knight outpost ({:+.0f} cp)",
        "bishop_pair_us": "Maintains the bishop pair advantage ({:+.0f} cp)",
        "bishop_pair_them": "Eliminates the opponent's bishop pair ({:+.0f} cp)",
        "passed_us": "Creates a dangerous passed pawn ({:+.0f} cp)",
        "passed_them": "Successfully blocks or stops an opponent's passed pawn ({:+.0f} cp)",
        "isolated_pawns_us": "Avoids creating pawn weaknesses ({:+.0f} cp)",
        "isolated_pawns_them": "Forces a pawn weakness (isolated pawn) for the opponent ({:+.0f} cp)",
        "center_control_us": "Improves control over the critical central squares ({:+.0f} cp)",
        "center_control_them": "Challenges and reduces opponent's central control ({:+.0f} cp)",
        "safe_mobility_us": "Safely activates pieces to better squares ({:+.0f} cp)",
        "rook_open_file_us": "Positions a rook effectively on an open file ({:+.0f} cp)",
        "backward_pawns_us": "Solidifies the pawn structure by fixing a weakness ({:+.0f} cp)",
        "backward_pawns_them": "Induces a backward pawn weakness in the opponent's camp ({:+.0f} cp)",
        "pst_us": "Optimizes piece placement on the board ({:+.0f} cp)",
        "pst_them": "Forces opponent pieces to suboptimal squares ({:+.0f} cp)",
        "pinned_us": "Successfully escapes an annoying pin ({:+.0f} cp)",
        "pinned_them": "Pins an opponent's piece to create tactical opportunities ({:+.0f} cp)",
        "phase": "Strategic move appropriate for the current game phase ({:+.0f} cp)",
    }

    def __init__(
        self,
        model: PhaseEnsemble,
        scaler: Any,
        feature_names: List[str],
    ) -> None:
        """Initialize explainer with trained model.

        Args:
            model: Trained PhaseEnsemble model
            scaler: Fitted StandardScaler for features
            feature_names: List of feature names
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def calculate_contributions(
        self,
        features_before: Dict[str, float],
        features_after: Dict[str, float],
        top_k: int = 5,
        min_cp: float = 5.0,
    ) -> List[Tuple[str, float, str]]:
        """Calculate feature contributions for a move.

        Args:
            features_before: Features before move (from moving side's perspective)
            features_after: Features after move (from opponent's perspective, already flipped)
            top_k: Number of top contributions to return
            min_cp: Minimum contribution magnitude to include (centipawns)

        Returns:
            List of (feature_name, cp_contribution, human_readable_text) tuples
        """
        reasons = []

        try:
            # Calculate feature delta
            # Delta features are already difference measurements (e.g., from audit.py)
            delta_features = {}

            for fname in self.feature_names:
                # For delta features, use raw values as they represent change
                delta_features[fname] = float(features_after.get(fname, 0.0))

            # Convert to array in same order as feature_names
            delta_vec = np.array(
                [delta_features.get(fname, 0.0) for fname in self.feature_names],
                dtype=float,
            )

            # Scale using the fitted scaler
            delta_vec_scaled = self.scaler.transform(delta_vec.reshape(1, -1))[0]

            # Get per-feature contributions from model
            contributions = self.model.get_contributions(delta_vec_scaled)

            # Filter out negligible contributions
            significant = []
            for _i, (fname, contrib) in enumerate(
                zip(self.feature_names, contributions)
            ):
                cp_value = float(contrib)
                if abs(cp_value) >= min_cp:
                    significant.append((fname, cp_value))

            # Sort by magnitude
            significant.sort(key=lambda x: -abs(x[1]))

            # Generate explanations for top k
            for fname, cp_value in significant[:top_k]:
                template = self.FEATURE_TEMPLATES.get(fname, f"{fname} ({{:+.0f}} cp)")
                try:
                    explanation = template.format(cp_value)
                except (KeyError, TypeError):
                    explanation = f"{fname} ({cp_value:+.0f} cp)"

                reasons.append((fname, cp_value, explanation))

        except Exception:
            # Gracefully handle errors and return empty list
            logger.warning("Failed to calculate contributions", exc_info=True)
            return []

        return reasons
