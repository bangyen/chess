"""Baseline feature extraction functions."""

import os
from typing import Any, Dict, Optional, cast

import chess
from chess.syzygy import Tablebase as SyzygyTablebase

from chess_ai.features.engine_probes import (
    best_forcing_swing,
    hanging_after_reply,
    sf_eval_shallow,
)

try:
    from chess_ai.rust_utils import (
        SyzygyTablebase as RustSyzygyTablebase,
    )
    from chess_ai.rust_utils import (
        extract_features_rust as extract_features_rust_base,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustSyzygyTablebase = None  # type: ignore
    extract_features_rust_base = None  # type: ignore

_SYZYGY_TB: Optional[Any] = None

# Minimal fallback features when Rust fails (e.g. for illegal boards in tests)
FALLBACK_FEAT_KEYS = [
    "material_diff",
    "king_ring_pressure_us",
    "king_ring_pressure_them",
    "passed_us",
    "passed_them",
    "open_files_us",
    "semi_open_us",
    "open_files_them",
    "semi_open_them",
    "center_control_us",
    "center_control_them",
    "piece_activity_us",
    "piece_activity_them",
    "king_safety_us",
    "king_safety_them",
    "hanging_us",
    "hanging_them",
    "bishop_pair_us",
    "bishop_pair_them",
    "rook_on_7th_us",
    "rook_on_7th_them",
    "king_pawn_shield_us",
    "king_pawn_shield_them",
    "outposts_us",
    "outposts_them",
    "batteries_us",
    "batteries_them",
    "isolated_pawns_us",
    "isolated_pawns_them",
    "safe_mobility_us",
    "safe_mobility_them",
    "rook_open_file_us",
    "rook_open_file_them",
    "backward_pawns_us",
    "backward_pawns_them",
    "connected_rooks_us",
    "connected_rooks_them",
    "pst_us",
    "pst_them",
    "pinned_us",
    "pinned_them",
    "threats_us",
    "threats_them",
    "doubled_pawns_us",
    "doubled_pawns_them",
    "space_us",
    "space_them",
    "king_tropism_us",
    "king_tropism_them",
    "pawn_chain_us",
    "pawn_chain_them",
    "see_advantage_us",
    "see_advantage_them",
    "see_vulnerability_us",
    "see_vulnerability_them",
    "syzygy_wdl",
    "syzygy_dtz",
]


def _inject_kings(board: "chess.Board") -> "chess.Board":
    """Inject kings into safe corners for illegal board positions."""
    temp_board = board.copy()
    if temp_board.king(chess.WHITE) is None:
        for sq in [chess.H1, chess.G1, chess.F1, chess.E1, chess.A1]:
            if temp_board.piece_at(sq) is None:
                temp_board.set_piece_at(sq, chess.Piece(chess.KING, chess.WHITE))
                break
    if temp_board.king(chess.BLACK) is None:
        for sq in [chess.A8, chess.B8, chess.C8, chess.D8, chess.H8]:
            if temp_board.piece_at(sq) is None:
                temp_board.set_piece_at(sq, chess.Piece(chess.KING, chess.BLACK))
                break
    return temp_board


def _minimal_python_fallback(board: "chess.Board") -> Dict[str, float]:
    """Minimal Python implementation for essential features."""
    feats = {
        "material_us": sum(
            len(board.pieces(p, board.turn)) * v
            for p, v in {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3.1,
                chess.ROOK: 5,
                chess.QUEEN: 9,
            }.items()
        ),
        "material_them": sum(
            len(board.pieces(p, not board.turn)) * v
            for p, v in {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3.1,
                chess.ROOK: 5,
                chess.QUEEN: 9,
            }.items()
        ),
        "mobility_us": float(min(board.legal_moves.count(), 40)),
        "phase": float(
            sum(
                len(board.pieces(pt, True)) + len(board.pieces(pt, False))
                for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            )
        ),
    }
    for f in FALLBACK_FEAT_KEYS:
        if f not in feats:
            feats[f] = 0.0
    feats["material_diff"] = feats["material_us"] - feats["material_them"]
    return feats


def baseline_extract_features(board: "chess.Board") -> Dict[str, Any]:
    """Small, fast, interpretable baseline feature set.

    Args:
        board: The chess board position to extract features from

    Returns:
        Dictionary of feature names to values
    """
    global _SYZYGY_TB

    if not RUST_AVAILABLE:
        raise ImportError(
            "Rust utilities not available. Please ensure the Rust extension is built."
        )

    # Try Rust first
    try:
        feats = extract_features_rust_base(board.fen())
    except ValueError:
        # Position might be illegal (e.g., missing kings in unit tests).
        # Try to inject kings into safe corners to satisfy strict Rust parsers.
        temp_board = _inject_kings(board)
        try:
            feats = extract_features_rust_base(temp_board.fen())
        except ValueError:
            # Final minimal fallback
            feats = _minimal_python_fallback(board)
        except Exception:
            # Catch other potential errors from Rust extraction with injected kings
            feats = _minimal_python_fallback(board)

    # Add Syzygy tablebase features (if available)
    syzygy_path = os.environ.get("SYZYGY_PATH")
    if syzygy_path:
        try:
            if not _SYZYGY_TB:
                if RustSyzygyTablebase:
                    _SYZYGY_TB = RustSyzygyTablebase()
                    _SYZYGY_TB.add_directory(syzygy_path)
                else:
                    _SYZYGY_TB = SyzygyTablebase()
                    _SYZYGY_TB.add_directory(syzygy_path)

            if len(board.piece_map()) <= 7:
                wdl = _SYZYGY_TB.probe_wdl(board)
                dtz = _SYZYGY_TB.probe_dtz(board)
                if wdl is not None:
                    feats["syzygy_wdl"] = float(wdl) / 2.0
                if dtz is not None:
                    feats["syzygy_dtz"] = float(dtz) / 100.0
        except Exception:  # noqa: S110
            pass

    feats["_engine_probes"] = {
        "hanging_after_reply": hanging_after_reply,
        "best_forcing_swing": best_forcing_swing,
        "sf_eval_shallow": sf_eval_shallow,
    }
    return cast(Dict[str, Any], feats)
