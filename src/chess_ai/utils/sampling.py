"""Position sampling utilities.

Provides functions for generating or extracting diverse chess positions
used to train and evaluate surrogate explainability models.  The key
entry-points are :func:`sample_random_positions` (quick, uniform),
:func:`sample_stratified_positions` (phase-aware), and
:func:`sample_positions_from_pgn` (from real games).
"""

import logging
import random
from typing import Dict, List, Optional

import chess
import chess.pgn

try:
    from tqdm import tqdm
except Exception:
    raise

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

#: Default phase distribution for stratified sampling.
DEFAULT_PHASE_WEIGHTS: Dict[str, float] = {
    "opening": 0.25,
    "middlegame": 0.50,
    "endgame": 0.25,
}

#: Ply ranges used when *generating* random positions for each phase.
#: The generator plays this many random half-moves from the start
#: position, then checks whether the resulting phase matches.
_PHASE_PLY_RANGES: Dict[str, tuple] = {
    "opening": (4, 14),
    "middlegame": (15, 35),
    "endgame": (36, 60),
}

#: Maximum number of generation attempts per position before giving
#: up on a specific phase bucket.  Prevents infinite loops when a
#: phase is hard to reach with random play.
_MAX_ATTEMPTS_PER_POSITION = 50


def _board_phase_value(board: chess.Board) -> int:
    """Count non-pawn, non-king pieces on the board.

    This mirrors the ``phase`` feature in ``features/baseline.py`` and
    is used to classify a position into opening / middlegame / endgame.
    Higher values indicate more material (opening-like).
    """
    return sum(
        len(board.pieces(pt, True)) + len(board.pieces(pt, False))
        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    )


def classify_phase(board: chess.Board) -> str:
    """Classify a board position into a game-phase bucket.

    Uses the non-pawn/non-king piece count (the same ``phase`` feature
    used elsewhere in the project) to decide the phase so that
    stratified sampling targets are consistent with model features.

    The piece count ranges from 0 (bare kings) to 14 (starting
    position).  Thresholds are chosen to give a roughly equal spread:

    * **Opening** (>= 12): nearly all minor/major pieces remain.
    * **Middlegame** (6-11): a mix of pieces has been traded.
    * **Endgame** (< 6): most pieces gone.

    Returns:
        One of ``"opening"``, ``"middlegame"``, or ``"endgame"``.
    """
    phase = _board_phase_value(board)
    if phase >= 12:
        return "opening"
    if phase >= 6:
        return "middlegame"
    return "endgame"


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------


def sample_positions_from_pgn(
    path: str,
    max_positions: int,
    ply_skip: int = 8,
    phase_weights: Optional[Dict[str, float]] = None,
) -> List["chess.Board"]:
    """Sample positions from a PGN file.

    When *phase_weights* is provided the sampler fills each phase
    bucket proportionally rather than taking every Nth ply uniformly,
    giving the surrogate model a more balanced training distribution.

    Args:
        path: Path to the PGN file.
        max_positions: Maximum number of positions to sample.
        ply_skip: Keep every Nth ply when sampling (ignored when
            *phase_weights* is given).
        phase_weights: Optional mapping ``{phase: weight}`` with
            weights summing to ~1.  When provided, positions are
            collected until each bucket reaches its target count.

    Returns:
        List of chess board positions.
    """
    if phase_weights is not None:
        return _sample_pgn_stratified(path, max_positions, phase_weights)

    # ── Uniform (legacy) path ──
    boards: List[chess.Board] = []
    logger.info("Sampling positions from PGN file: %s", path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for plies, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                if plies % ply_skip == 0 and not board.is_game_over():
                    boards.append(board.copy(stack=False))
                    if len(boards) >= max_positions:
                        logger.info("Sampled %d positions from PGN", len(boards))
                        return boards
    logger.info("Sampled %d positions from PGN", len(boards))
    return boards[:max_positions]


def _sample_pgn_stratified(
    path: str,
    max_positions: int,
    phase_weights: Dict[str, float],
) -> List[chess.Board]:
    """Sample from a PGN file filling phase buckets proportionally.

    Iterates through games and collects positions until each bucket
    reaches its target.  Positions are taken at every ply (not
    skipped) so that rare phases (especially endgame) are not missed.
    """
    total_weight = sum(phase_weights.values())
    targets: Dict[str, int] = {}
    for phase, w in phase_weights.items():
        targets[phase] = max(1, round(max_positions * w / total_weight))

    buckets: Dict[str, List[chess.Board]] = {p: [] for p in targets}

    def _all_full() -> bool:
        return all(len(buckets[p]) >= targets[p] for p in targets)

    logger.info("Sampling stratified positions from PGN file: %s", path)
    with open(path, encoding="utf-8", errors="ignore") as f:
        while not _all_full():
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.is_game_over():
                    continue
                phase = classify_phase(board)
                if phase in buckets and len(buckets[phase]) < targets.get(phase, 0):
                    buckets[phase].append(board.copy(stack=False))
                if _all_full():
                    break

    result: List[chess.Board] = []
    for phase in targets:
        result.extend(buckets[phase])
    random.shuffle(result)
    logger.info("Sampled %d stratified positions from PGN", len(result))
    return result[:max_positions]


def sample_random_positions(n: int, max_random_plies: int = 24) -> List["chess.Board"]:
    """Generate random chess positions by playing random legal moves.

    Produces uniform (non-stratified) positions that tend toward
    chaotic middlegame states.  Prefer :func:`sample_stratified_positions`
    when training-data diversity matters.

    Args:
        n: Number of positions to generate.
        max_random_plies: Maximum number of random plies to play.

    Returns:
        List of chess board positions.
    """
    boards: List[chess.Board] = []
    logger.info("Generating %d random positions...", n)
    for _ in tqdm(range(n), desc="Generating positions"):
        b = chess.Board()
        # play random but legal moves to get middlegame-ish positions
        plies = random.randint(10, max_random_plies)  # noqa: S311
        for __ in range(plies):
            if b.is_game_over():
                break
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(random.choice(moves))  # noqa: S311
        if not b.is_game_over():
            boards.append(b.copy(stack=False))
    logger.info("Generated %d valid positions", len(boards))
    return boards


def sample_stratified_positions(
    n: int,
    phase_weights: Optional[Dict[str, float]] = None,
) -> List["chess.Board"]:
    """Generate random positions stratified by game phase.

    Unlike :func:`sample_random_positions`, this function targets a
    specific distribution across opening, middlegame, and endgame
    buckets so that surrogate models train on a representative sample
    of each regime.

    For endgame positions the generator biases move selection toward
    captures so that material is removed from the board, making low
    piece-count positions reachable within a reasonable number of
    attempts.

    Args:
        n: Total number of positions to generate.
        phase_weights: Mapping ``{phase: weight}`` controlling the
            proportion of each phase.  Weights are normalised
            internally.  Defaults to ``DEFAULT_PHASE_WEIGHTS``
            (25% opening, 50% middlegame, 25% endgame).

    Returns:
        List of chess board positions, shuffled.

    Raises:
        ValueError: If *n* is negative or *phase_weights* contains
            unknown phase names.
    """
    if n <= 0:
        return []

    if phase_weights is None:
        phase_weights = DEFAULT_PHASE_WEIGHTS

    valid_phases = {"opening", "middlegame", "endgame"}
    unknown = set(phase_weights.keys()) - valid_phases
    if unknown:
        raise ValueError(
            f"Unknown phase names: {unknown}. " f"Valid phases are {valid_phases}."
        )

    total_weight = sum(phase_weights.values())
    if total_weight <= 0:
        raise ValueError("Phase weights must sum to a positive number.")

    targets: Dict[str, int] = {}
    for phase, w in phase_weights.items():
        targets[phase] = max(1, round(n * w / total_weight))

    # Adjust rounding so we don't overshoot or undershoot the total.
    _adjust_targets(targets, n)

    buckets: Dict[str, List[chess.Board]] = {p: [] for p in targets}

    logger.info("Generating %d stratified positions...", n)
    for phase, target in targets.items():
        lo, hi = _PHASE_PLY_RANGES[phase]
        generated = 0
        attempts = 0
        max_attempts = target * _MAX_ATTEMPTS_PER_POSITION
        pbar = tqdm(total=target, desc=f"  {phase}")
        while generated < target and attempts < max_attempts:
            attempts += 1
            board = _generate_candidate(phase, lo, hi)
            if board is None:
                continue
            if classify_phase(board) == phase:
                buckets[phase].append(board)
                generated += 1
                pbar.update(1)
        pbar.close()

    result: List[chess.Board] = []
    for phase in targets:
        result.extend(buckets[phase])
    random.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _adjust_targets(targets: Dict[str, int], n: int) -> None:
    """Adjust rounded bucket targets so they sum exactly to *n*.

    Modifies *targets* in-place by distributing the rounding remainder
    to the largest bucket first.
    """
    diff = n - sum(targets.values())
    if diff == 0:
        return
    # Sort phases by target size descending so adjustments go to the
    # biggest bucket (least relative impact).
    ordered = sorted(targets, key=lambda p: targets[p], reverse=True)
    idx = 0
    while diff != 0:
        phase = ordered[idx % len(ordered)]
        step = 1 if diff > 0 else -1
        targets[phase] = max(1, targets[phase] + step)
        diff -= step
        idx += 1


def _generate_candidate(
    phase: str, min_plies: int, max_plies: int
) -> Optional[chess.Board]:
    """Play random moves and return a non-game-over board, or *None*.

    For the endgame phase, moves are biased toward captures so that
    material is removed and a low piece-count is more likely.
    """
    b = chess.Board()
    plies = random.randint(min_plies, max_plies)  # noqa: S311
    bias_captures = phase == "endgame"

    for _ in range(plies):
        if b.is_game_over():
            return None
        moves = list(b.legal_moves)
        if not moves:
            return None

        if bias_captures:
            captures = [m for m in moves if b.is_capture(m)]
            if captures and random.random() < 0.6:  # noqa: S311
                b.push(random.choice(captures))  # noqa: S311
                continue

        b.push(random.choice(moves))  # noqa: S311

    if b.is_game_over():
        return None
    return b.copy(stack=False)
