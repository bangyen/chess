"""Feature enrichment helpers for the audit pipeline.

Extracted from ``audit.py`` so the enrichment logic can be tested
independently and the main audit function stays focused on
orchestration.
"""

from typing import Any, Callable, Dict, List, Tuple, cast

import chess

from .metrics.positional import (
    checkability_now,
    confinement_delta,
    passed_pawn_momentum_delta,
)


def enrich_features(
    board: chess.Board,
    base_board: chess.Board,
    base_feats_raw: Dict[str, float],
    extract_fn: Callable[..., Dict[str, Any]],
    engine: Any,
    interaction_pairs: List[Tuple[str, str]],
    hanging_cache: Dict[str, Tuple[int, float, int]],
    swing_cache: Dict[str, float],
) -> Dict[str, float]:
    """Extract features and add probes, deltas, and interactions.

    Centralises the feature-enrichment pipeline that was previously
    duplicated in every loop of the audit.
    """
    # Attempt to use the batched Rust implementation for a massive speedup
    try:
        from chess_ai.rust_utils import extract_features_delta_rust

        if extract_fn.__name__ == "baseline_extract_features" and base_board:
            # Cast to Dict[str, float] to satisfy mypy, as the Rust extension
            # return type is seen as Any.
            # We pass depth=6 for search probes (forcing swing, hanging after reply)
            rust_feats = cast(
                Dict[str, float],
                extract_features_delta_rust(base_board.fen(), board.fen(), 6),
            )

            # Map the Rust probe names to the names expected by the Python audit
            # and interaction terms if necessary.
            if "hanging_cnt_after_reply" in rust_feats:
                # Keep them as individual floats for the surrogate model
                pass

            # d_forcing_swing is used in INTERACTION_PAIRS
            if "best_forcing_swing" in rust_feats:
                rust_feats["d_forcing_swing"] = rust_feats["best_forcing_swing"]

            # Even with the Rust batch, we still need to apply interaction terms
            _apply_interactions(rust_feats, interaction_pairs)
            return rust_feats
    except (ImportError, AttributeError):
        pass

    # Fallback to Python-orchestrated enrichment
    feats = extract_fn(board)
    probes = feats.pop("_engine_probes", {})
    feats = {
        k: (1.0 if isinstance(v, bool) and v else float(v)) for k, v in feats.items()
    }

    # Engine-based probe features (cached by FEN to skip
    # redundant Rust searches on repeated positions).
    if probes:
        fen_key = board.fen()
        if fen_key in hanging_cache:
            hang_cnt, hang_max_val, hang_near_king = hanging_cache[fen_key]
        else:
            hang_cnt, hang_max_val, hang_near_king = probes["hanging_after_reply"](
                engine, board, depth=6
            )
            hanging_cache[fen_key] = (hang_cnt, hang_max_val, hang_near_king)
        feats["hang_cnt"] = hang_cnt
        feats["hang_max_val"] = hang_max_val
        feats["hang_near_king"] = hang_near_king

        if fen_key in swing_cache:
            forcing_swing = swing_cache[fen_key]
        else:
            forcing_swing = probes["best_forcing_swing"](
                engine, board, d_base=6, k_max=12
            )
            swing_cache[fen_key] = forcing_swing
        feats["forcing_swing"] = forcing_swing

    # Passed-pawn momentum delta
    pp_delta = passed_pawn_momentum_delta(base_board, board)
    feats.update(pp_delta)

    # Checkability delta
    base_check = checkability_now(base_board)
    after_check = checkability_now(board)
    check_delta = {
        "d_quiet_checks": after_check["d_quiet_checks"] - base_check["d_quiet_checks"],
        "d_capture_checks": after_check["d_capture_checks"]
        - base_check["d_capture_checks"],
    }
    feats.update(check_delta)

    # Confinement delta
    conf_delta = confinement_delta(base_board, board)
    feats.update(conf_delta)

    # Compute delta features: d_<key> = after - base
    for k in base_feats_raw:
        if k in feats:
            feats[f"d_{k}"] = feats[k] - base_feats_raw[k]

    # Interaction terms
    _apply_interactions(feats, interaction_pairs)
    return feats


def extract_base_feats(
    extract_fn: Callable[..., Dict[str, Any]], board: chess.Board
) -> Dict[str, float]:
    """Extract and clean base features for delta computation."""
    raw = extract_fn(board)
    raw.pop("_engine_probes", None)
    return {k: (1.0 if isinstance(v, bool) and v else float(v)) for k, v in raw.items()}


def _apply_interactions(
    feats: Dict[str, float],
    interaction_pairs: List[Tuple[str, str]],
) -> None:
    """Add pairwise interaction features in-place."""
    for p1, p2 in interaction_pairs:
        if p1 in feats and p2 in feats:
            feats[f"{p1}_x_{p2}"] = float(feats[p1]) * float(feats[p2])
