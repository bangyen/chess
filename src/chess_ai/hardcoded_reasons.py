"""Threshold-based move-explanation generators.

These standalone functions produce human-readable positional reasons
for a move based on feature deltas.  They serve as a fallback when the
surrogate model is unavailable and are also used by
``explain_move_with_board`` for board-specific explanations.
"""

from __future__ import annotations

import contextlib

import chess


def generate_hardcoded_reasons(  # noqa: C901
    feats_before: dict, feats_after: dict
) -> list[tuple[str, float, str]]:
    """Generate threshold-based reasons (fallback when model unavailable).

    Compares feature values before and after a move (with us/them
    perspective swap) and emits reasons for significant deltas.
    """
    reasons: list[tuple[str, float, str]] = []

    def get_delta(feature_name: str) -> float:
        val_before = feats_before.get(f"{feature_name}_us", 0.0)
        val_after = feats_after.get(f"{feature_name}_them", 0.0)
        return float(val_after - val_before)

    def get_opp_delta(feature_name: str) -> float:
        val_before = feats_before.get(f"{feature_name}_them", 0.0)
        val_after = feats_after.get(f"{feature_name}_us", 0.0)
        return float(val_after - val_before)

    # Batteries
    delta = get_delta("batteries")
    if delta > 0.5:
        reasons.append(("batteries_us", 20.0, "Forms a battery arrangement (+20 cp)"))

    # Outposts
    delta = get_delta("outposts")
    if delta > 0.5:
        reasons.append(("outposts_us", 30.0, "Establishes a knight outpost (+30 cp)"))

    # King Ring Pressure
    delta = get_delta("king_ring_pressure")
    if delta > 0.5:
        reasons.append(
            ("king_pressure", 25.0, "Increases pressure on enemy king (+25 cp)")
        )

    # Bishop Pair
    delta = get_delta("bishop_pair")
    if delta > 0.5:
        reasons.append(("bishop_pair", 20.0, "Secures the bishop pair (+20 cp)"))

    # Passed Pawns
    delta = get_delta("passed")
    if delta > 0.5:
        reasons.append(("passed_pawns", 30.0, "Creates a passed pawn (+30 cp)"))

    # Isolated Pawns
    delta_opp = get_opp_delta("isolated_pawns")
    if delta_opp > 0.5:
        reasons.append(
            (
                "structure_damage",
                15.0,
                "Creates an isolated pawn for opponent (+15 cp)",
            )
        )

    # Center Control
    delta = get_delta("center_control")
    if delta > 0.5:
        reasons.append(("center_control", 15.0, "Improves central control (+15 cp)"))

    # Safe Mobility
    delta = get_delta("safe_mobility")
    if delta > 1.5:
        reasons.append(
            ("safe_mobility", 15.0, "Increases safe piece activity (+15 cp)")
        )

    # Rook on Open File
    delta = get_delta("rook_open_file")
    if delta > 0.4:
        reasons.append(
            (
                "rook_activity",
                25.0,
                "Places rook on an open or semi-open file (+25 cp)",
            )
        )

    # Backward Pawns
    delta_opp = get_opp_delta("backward_pawns")
    if delta_opp > 0.5:
        reasons.append(
            (
                "structure_damage",
                15.0,
                "Creates a backward pawn weakness for opponent (+15 cp)",
            )
        )

    delta = get_delta("backward_pawns")
    if delta < -0.5:
        reasons.append(
            ("structure_repair", 15.0, "Fixes a backward pawn weakness (+15 cp)")
        )

    # PST Improvement
    delta = get_delta("pst")
    if delta > 0.4:
        reasons.append(
            ("piece_quality", 15.0, "Improves piece placement quality (+15 cp)")
        )

    # Pins
    delta_opp = get_opp_delta("pinned")
    if delta_opp > 0.5:
        reasons.append(("pin_creation", 25.0, "Pins an opponent's piece (+25 cp)"))

    delta = get_delta("pinned")
    if delta < -0.5:
        reasons.append(("pin_escape", 25.0, "Escapes a pin (+25 cp)"))

    return reasons


def generate_move_reasons_with_board(
    move: chess.Move, board: chess.Board, move_history: list[chess.Move]
) -> list[tuple[str, float, str]]:
    """Generate specific reasons why a move is good or bad using a
    specific board state.

    Produces heuristic-based reasons (captures, checks, development,
    centre control, etc.) without requiring a surrogate model.
    """
    reasons: list[tuple[str, float, str]] = []

    try:
        with contextlib.suppress(Exception):
            board.san(move)

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                piece_value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}.get(
                    captured_piece.symbol().upper(), 0
                )
                reasons.append(
                    (
                        "capture",
                        float(piece_value),
                        f"Captures {captured_piece.symbol()} (worth {piece_value} points)",
                    )
                )

        if board.gives_check(move):
            reasons.append(("check", 2.0, "Gives check to opponent's king"))

        if board.is_capture(move) or board.gives_check(move):
            reasons.append(("tactical", 1.0, "Tactical move (capture or check)"))

        piece = board.piece_at(move.from_square)
        if (
            piece
            and piece.piece_type == chess.PAWN
            and (move.from_square < 16 or move.from_square > 47)
        ):
            reasons.append(("development", 1.0, "Develops pawn from starting position"))
        elif (
            piece
            and piece.piece_type in [chess.KNIGHT, chess.BISHOP]
            and (move.from_square < 16 or move.from_square > 47)
        ):
            reasons.append(
                (
                    "development",
                    2.0,
                    "Develops minor piece from starting position",
                )
            )

        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        if move.to_square in center_squares:
            reasons.append(("center_control", 1.0, "Controls central squares"))

        if piece and piece.piece_type == chess.KING and len(move_history) < 20:
            reasons.append(
                (
                    "king_safety",
                    -1,
                    "Moves king in opening (reduces castling options)",
                )
            )

        if board.is_castling(move):
            reasons.append(("castling", 3, "Castles to improve king safety"))

        if board.is_en_passant(move):
            reasons.append(("en_passant", 1, "En passant capture"))

    except Exception:  # noqa: S110
        pass

    return reasons
