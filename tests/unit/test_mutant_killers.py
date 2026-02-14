"""Targeted tests to kill surviving mutmut mutants.

Each test is designed to detect specific mutations in the core logic
modules: positional metrics, feature utils, and surrogate explainer.
"""

import math
import os
import tempfile
import warnings
from unittest.mock import MagicMock, patch

import chess
import numpy as np
import pytest

from chess_ai.features.utils import load_feature_module
from chess_ai.metrics.positional import (
    _blockaded,
    _rook_behind_passer,
    _runner_clear_path,
    _stoppers_in_path,
)
from chess_ai.model_trainer import PhaseEnsemble
from chess_ai.surrogate_explainer import SurrogateExplainer

# ---------------------------------------------------------------------------
# Suppress sklearn warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass


# ===================================================================
# helpers
# ===================================================================


def _empty_board_with(*pieces: tuple) -> chess.Board:
    """Create an empty board and place the listed pieces.

    Each *piece* is (square, piece_type, color).
    """
    board = chess.Board()
    board.clear()
    for sq, pt, color in pieces:
        board.set_piece_at(sq, chess.Piece(pt, color))
    return board


def _make_trained_ensemble(feature_names):
    """Build a small PhaseEnsemble fitted on synthetic data.

    Seeds the RNG to guarantee reproducible coefficients so that the
    tests are deterministic.
    """
    np.random.seed(42)
    n, d = 60, len(feature_names)
    X = np.random.randn(n, d)
    # Strong signal on the first feature so contributions are significant
    y = X[:, 0] * 50.0 + np.random.randn(n) * 0.1

    alphas = np.logspace(-2, 2, 10).tolist()
    ens = PhaseEnsemble(feature_names, alphas, cv_folds=2, max_iter=10000)
    ens.fit(X, y)
    return ens


def _make_scaler(feature_names):
    """Build a fitted StandardScaler matching *feature_names*."""
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    X = np.random.randn(60, len(feature_names))
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ===================================================================
# _rook_behind_passer
# ===================================================================


class TestRookBehindPasserMutants:
    """Kill mutants in _rook_behind_passer."""

    def test_white_rook_immediately_behind(self):
        """White rook on rank directly below the pawn should be detected.

        Kills mutant_12 (range start -2 instead of -1): if we skip the
        immediately-behind rank, this test fails.
        """
        # White pawn on e4 (rank 3), white rook on e3 (rank 2)
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.E3, chess.ROOK, chess.WHITE),
        )
        assert _rook_behind_passer(board, chess.E4, chess.WHITE) == 1

    def test_white_rook_range_step_minus_one(self):
        """Rook at the very bottom rank (e1) behind a pawn on e3.

        Kills mutant_16 (step -2 instead of -1): with step=-2 the range
        [rank-1, ..., 0] becomes [rank-1, rank-3, ...] and e2 is skipped.
        We put a blocking piece on e2 and rook on e1 -- with step=-2,
        the loop would see e2 first and stop; with step=-1, it also sees
        e2 first. Instead, put rook on e2 directly -- with step=-2
        the range(1, -1, -2) = [1] which is e2, BUT range(1, -1, -2) is
        actually just [1]. Let me think again...

        Actually range(rank-1, -1, -1) vs range(rank-1, -1, -2):
        Pawn on e3 (rank 2): range(1, -1, -1) = [1, 0] vs range(1, -1, -2) = [1]
        So put rook on e1 (rank 0) and the mutant would miss it.
        """
        board = _empty_board_with(
            (chess.E3, chess.PAWN, chess.WHITE),
            (chess.E1, chess.ROOK, chess.WHITE),
        )
        assert _rook_behind_passer(board, chess.E3, chess.WHITE) == 1

    def test_white_rook_range_stop_boundary(self):
        """Kills mutant_14 (stop -2 instead of -1).

        range(rank-1, -1, -1) vs range(rank-1, -2, -1):
        For pawn on e2 (rank 1): range(0, -1, -1) = [0] vs range(0, -2, -1) = [0, -1].
        The -1 square would be invalid. Both produce [0]. Need different approach.
        Actually this mutant is equivalent for most cases, but let's ensure
        the basic path is covered -- rank 0 is still visited.
        """
        board = _empty_board_with(
            (chess.E2, chess.PAWN, chess.WHITE),
            (chess.E1, chess.ROOK, chess.WHITE),
        )
        assert _rook_behind_passer(board, chess.E2, chess.WHITE) == 1

    def test_black_rook_behind_passer(self):
        """Black rook behind a black passed pawn.

        Kills mutants 20-24 which mutate the BLACK range:
        - mutant_20: range(8) instead of range(rank+1, 8) -- would check wrong ranks
        - mutant_21: range(rank+1,) which is range(rank+1) -- different semantics
        - mutant_22: range(rank-1, 8) instead of range(rank+1, 8)
        - mutant_24: range(rank+2, 8) instead of range(rank+1, 8) -- skips immediate
        """
        # Black pawn on e5 (rank 4), black rook on e6 (rank 5)
        # "behind" for black means higher rank (closer to rank 7)
        board = _empty_board_with(
            (chess.E5, chess.PAWN, chess.BLACK),
            (chess.E6, chess.ROOK, chess.BLACK),
        )
        assert _rook_behind_passer(board, chess.E5, chess.BLACK) == 1

    def test_black_rook_behind_passer_at_rank7(self):
        """Black pawn on rank 4, rook on rank 7 -- far behind.

        Ensures the full range is traversed for BLACK side.
        """
        board = _empty_board_with(
            (chess.E5, chess.PAWN, chess.BLACK),
            (chess.E8, chess.ROOK, chess.BLACK),  # rank 7
        )
        # Note: E8 is rank 7
        assert _rook_behind_passer(board, chess.E5, chess.BLACK) == 1

    def test_wrong_piece_type_behind(self):
        """A non-rook own piece behind should return 0.

        Kills mutant_34 (and→or): with `or`, any own piece would return 1.
        """
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.E3, chess.BISHOP, chess.WHITE),
        )
        assert _rook_behind_passer(board, chess.E4, chess.WHITE) == 0

    def test_opponent_rook_behind(self):
        """An opponent's rook behind should return 0.

        Also helps kill mutant_34 (and→or).
        """
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.E3, chess.ROOK, chess.BLACK),
        )
        assert _rook_behind_passer(board, chess.E4, chess.WHITE) == 0


# ===================================================================
# _stoppers_in_path
# ===================================================================


class TestStoppersInPathMutants:
    """Kill mutants in _stoppers_in_path."""

    def test_black_pawn_stopper_direction(self):
        """Black pawn should look backward (negative direction) for stoppers.

        Kills mutant_9 (else +1 instead of -1) and mutant_10 (else -2).
        """
        # Black pawn on e5 (rank 4), white pawn on e4 (rank 3) = 1 rank behind
        board = _empty_board_with(
            (chess.E5, chess.PAWN, chess.BLACK),
            (chess.E4, chess.PAWN, chess.WHITE),
        )
        # For BLACK, dirs = -1, so check ranks 3 and 2. e4 is rank 3 -> found.
        assert _stoppers_in_path(board, chess.E5, chess.BLACK) == 1

    def test_stopper_on_left_adjacent_file(self):
        """Stopper on file to the left.

        Kills mutant_13 ((-1,0,1) → (+1,0,1)) since +1 would never look left.
        Kills mutant_14 ((-1,0,1) → (-2,0,1)) since -2 is out of range via boundary check.
        """
        # White pawn on e4 (file 4), black pawn on d5 (file 3, rank 4)
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.D5, chess.PAWN, chess.BLACK),
        )
        assert _stoppers_in_path(board, chess.E4, chess.WHITE) == 1

    def test_stopper_on_right_adjacent_file(self):
        """Stopper on file to the right.

        Kills mutant_16 ((-1,0,1) → (-1,0,2)) since 2 would be 2 files away.
        """
        # White pawn on e4 (file 4), black pawn on f5 (file 5, rank 4)
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.F5, chess.PAWN, chess.BLACK),
        )
        assert _stoppers_in_path(board, chess.E4, chess.WHITE) == 1

    def test_nf_minus_df_symmetry_break(self):
        """Test that f+df is used, not f-df.

        Kills mutant_18 (f+df → f-df). With a pawn on e4 (file=4),
        df=-1 gives nf=3 (d-file). With f-df, nf=5 (f-file).
        Put a stopper on d5 and none on f5.
        """
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.D5, chess.PAWN, chess.BLACK),
        )
        count = _stoppers_in_path(board, chess.E4, chess.WHITE)
        assert count == 1  # only d5 stopper found

        # Now verify f5 alone is found (would fail with f-df when df=-1)
        board2 = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.F5, chess.PAWN, chess.BLACK),
        )
        count2 = _stoppers_in_path(board2, chess.E4, chess.WHITE)
        assert count2 == 1

    def test_boundary_file_zero_check(self):
        """Pawn on a-file: left adjacent (file -1) should be skipped.

        Kills mutant_19 (or → and), mutant_20 (< 0 → <= 0),
        mutant_21 (< 0 → < 1).
        """
        # White pawn on a4 (file 0). With df=-1, nf=-1 should be skipped.
        # Put a stopper at b5 (file 1, adjacent right) to verify it still counts.
        board = _empty_board_with(
            (chess.A4, chess.PAWN, chess.WHITE),
            (chess.B5, chess.PAWN, chess.BLACK),
        )
        result = _stoppers_in_path(board, chess.A4, chess.WHITE)
        assert result == 1

        # With mutant_21 (nf < 1), the a-file itself (nf=0 when df=0) would
        # also be skipped. Put a stopper on a5 to confirm it IS checked.
        board2 = _empty_board_with(
            (chess.A4, chess.PAWN, chess.WHITE),
            (chess.A5, chess.PAWN, chess.BLACK),
        )
        result2 = _stoppers_in_path(board2, chess.A4, chess.WHITE)
        assert result2 == 1

    def test_oob_file_alias_not_counted(self):
        """Out-of-bounds file must NOT alias to a valid square.

        Kills mutant_19 (or → and): the condition `nf < 0 and nf > 7`
        is always False, so nf=-1 wouldn't be skipped. chess.square(-1, r)
        aliases to a real square and could falsely count a pawn there.

        Also kills mutant_23 (>7 → >8): nf=8 wouldn't be skipped and
        chess.square(8, r) aliases to a real square.
        """
        # --- mutant_19 (or→and): nf=-1 aliases to a square on the board ---
        # White pawn on a4 (file=0, rank=3). df=-1 → nf=-1.
        # chess.square(-1, 4) = 4*8 + (-1) = 31 = h4 (file=7, rank=3).
        # chess.square(-1, 5) = 5*8 + (-1) = 39 = h5 (file=7, rank=4).
        # Put a black pawn on h5 so the mutation falsely finds a "stopper".
        board = _empty_board_with(
            (chess.A4, chess.PAWN, chess.WHITE),
            (chess.H5, chess.PAWN, chess.BLACK),  # alias target for sq(-1, 5)
        )
        # Correct: nf=-1 is skipped, only files 0 and 1 checked → 0 stoppers.
        # mutant_19: nf=-1 NOT skipped, checks alias sq 39 = h5 → falsely counts 1.
        assert _stoppers_in_path(board, chess.A4, chess.WHITE) == 0

        # --- mutant_23 (>7→>8): nf=8 aliases to a square on the board ---
        # White pawn on h4 (file=7, rank=3). df=+1 → nf=8.
        # chess.square(8, 4) = 4*8 + 8 = 40 = a6 (file=0, rank=5).
        # chess.square(8, 5) = 5*8 + 8 = 48 = a7 (file=0, rank=6).
        # Put a black pawn on a6 so the mutation falsely finds a "stopper".
        board2 = _empty_board_with(
            (chess.H4, chess.PAWN, chess.WHITE),
            (chess.A6, chess.PAWN, chess.BLACK),  # alias target for sq(8, 4)
        )
        # Correct: nf=8 is skipped → 0 stoppers.
        # mutant_23: nf=8 NOT skipped → falsely counts 1.
        assert _stoppers_in_path(board2, chess.H4, chess.WHITE) == 0

    def test_boundary_file_seven_check(self):
        """Pawn on h-file: right adjacent (file 8) should be skipped.

        Kills mutant_22 (> 7 → >= 7) and mutant_23 (> 7 → > 8).
        """
        # White pawn on h4 (file 7). With df=+1, nf=8 should be skipped.
        # With >= 7, even nf=7 (the h-file itself when df=0) would be skipped.
        # Put a stopper on h5 to verify h-file IS checked.
        board = _empty_board_with(
            (chess.H4, chess.PAWN, chess.WHITE),
            (chess.H5, chess.PAWN, chess.BLACK),
        )
        result = _stoppers_in_path(board, chess.H4, chess.WHITE)
        assert result == 1

        # Also put a stopper on g5 (file 6) -- should still count
        board2 = _empty_board_with(
            (chess.H4, chess.PAWN, chess.WHITE),
            (chess.G5, chess.PAWN, chess.BLACK),
        )
        assert _stoppers_in_path(board2, chess.H4, chess.WHITE) == 1

    def test_continue_vs_break_outer(self):
        """Outer continue→break: second file iteration should still happen.

        Kills mutant_24 (continue → break). Put pawn on a4 (file 0).
        df=-1 gives nf=-1 (out of bounds → continue). If break instead,
        df=0 and df=1 never run. Put stopper on b5 (df=+1, nf=1).
        """
        board = _empty_board_with(
            (chess.A4, chess.PAWN, chess.WHITE),
            (chess.B5, chess.PAWN, chess.BLACK),  # file 1, rank 4
        )
        result = _stoppers_in_path(board, chess.A4, chess.WHITE)
        assert result == 1

    def test_two_ranks_ahead_stopper(self):
        """Stopper exactly 2 ranks ahead should be counted.

        Kills mutant_26 (2*dirs → 3*dirs).
        """
        # White pawn on e4 (rank 3). 2 ranks ahead = rank 5 = e6.
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.E6, chess.PAWN, chess.BLACK),  # rank 5
        )
        assert _stoppers_in_path(board, chess.E4, chess.WHITE) == 1

    def test_rank_boundary_inner_loop(self):
        """Inner rank boundary: rank 0 and rank 7 edge cases.

        Kills mutants 30 (< 0 → <= 0), 31 (< 0 → < 1), 32 (> 7 → >= 7).
        """
        # White pawn on e2 (rank 1). dirs=+1, dr=2*dirs=2, nr=1+2=3 (valid).
        # Also dr=dirs=1, nr=1+1=2 (valid). No boundary issue.
        # For boundary: black pawn on e7 (rank 6). dirs=-1.
        # dr=-1, nr=5 (ok). dr=-2, nr=4 (ok). No boundary issue either.
        # For rank 0 boundary: white pawn on a1 wouldn't happen... but
        # let's test black pawn on e2 (rank 1). dirs=-1.
        # dr=-1, nr=0 -- should be valid (>= 0).
        # With mutant_30 (<=0) or mutant_31 (<1), nr=0 would be skipped.
        board = _empty_board_with(
            (chess.E2, chess.PAWN, chess.BLACK),
            (chess.E1, chess.PAWN, chess.WHITE),  # rank 0
        )
        assert _stoppers_in_path(board, chess.E2, chess.BLACK) == 1

        # For rank 7 boundary: white pawn on e6 (rank 5). dirs=+1.
        # dr=+2, nr=7 -- should be valid (<= 7).
        # With mutant_32 (>=7), nr=7 would be skipped.
        board2 = _empty_board_with(
            (chess.E6, chess.PAWN, chess.WHITE),
            (chess.E8, chess.PAWN, chess.BLACK),  # rank 7
        )
        assert _stoppers_in_path(board2, chess.E6, chess.WHITE) == 1

    def test_continue_vs_break_inner(self):
        """Inner continue→break: second rank in dr loop should run.

        Kills mutant_34. Put stopper at 2*dirs but not at dirs.
        dr=(dirs, 2*dirs). If inner boundary triggers break on dirs,
        2*dirs never runs. Use pawn on e1 (rank 0) for BLACK (dirs=-1):
        dr=-1 → nr=-1 (out of bounds → continue or break).
        dr=-2 → nr=-2 (also out of bounds). Not helpful.

        Better: white pawn on e7 (rank 6), dirs=+1.
        dr=+1, nr=7 (valid). dr=+2, nr=8 (invalid → continue/break).
        Put stopper at rank 7 only -- both continue and break find it.

        Actually: I need dr=dirs to go OUT of bounds but dr=2*dirs in bounds.
        That can't happen since |2*dirs| > |dirs|.

        The break in the inner loop means: when one dr goes out of bounds,
        the other dr for the same file is skipped. Since dr iterates over
        (dirs, 2*dirs), if dirs goes OOB, 2*dirs also goes OOB. So this
        mutant is actually equivalent for valid chess positions.

        BUT: consider a position where dirs dr is in bounds (rank ok)
        but we want to test both. Actually... let me reconsider.
        The inner loop is: for dr in (dirs, 2*dirs): if nr OOB: continue.
        With break: if dirs goes OOB, 2*dirs never checked. But as noted,
        if dirs goes OOB, 2*dirs ALSO goes OOB. So the break is equivalent.

        HOWEVER, consider white pawn on rank 6 (e7): dirs=+1.
        dr=+1 → nr=7 (OK, in bounds). dr=+2 → nr=8 (OOB → continue/break).
        Continue: keeps going to next df. Break: also keeps going to next df
        since it only breaks inner loop. So break exits inner loop, continues
        outer loop. Same effect as continue for last item.

        For dr order (dirs, 2*dirs): dirs is checked first. If it's in
        bounds but there's no stopper, then 2*dirs is checked. If 2*dirs
        is OOB, continue → goes to next df. break → goes to next df.
        Same result.

        If dirs is OOB, 2*dirs is also OOB. So break = continue here.
        This appears to be an equivalent mutant. Let's focus on other mutants.
        """
        # This is likely an equivalent mutant, but let's at least ensure
        # the function works for the basic case with 2-rank-ahead stopper.
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.E5, chess.PAWN, chess.BLACK),  # 1 rank ahead
            (chess.E6, chess.PAWN, chess.BLACK),  # 2 ranks ahead
        )
        assert _stoppers_in_path(board, chess.E4, chess.WHITE) == 2

    def test_multiple_stoppers_count(self):
        """Multiple stoppers should all be counted.

        Kills mutant_46 (cnt += 1 → cnt = 1).
        """
        # White pawn on e4. Black pawns on d5, e5, f5 = 3 stoppers.
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.D5, chess.PAWN, chess.BLACK),
            (chess.E5, chess.PAWN, chess.BLACK),
            (chess.F5, chess.PAWN, chess.BLACK),
        )
        result = _stoppers_in_path(board, chess.E4, chess.WHITE)
        assert result == 3  # with cnt=1, result would be 1


# ===================================================================
# _blockaded
# ===================================================================


class TestBlockadedMutants:
    """Kill mutants in _blockaded."""

    def test_black_pawn_blockaded(self):
        """Black pawn should look backwards (step=-1) for the blocker.

        Kills mutant_9 (else +1) and mutant_10 (else -2).
        """
        # Black pawn on e5 (rank 4). Step=-1 → check e4 (rank 3).
        board = _empty_board_with(
            (chess.E5, chess.PAWN, chess.BLACK),
            (chess.E4, chess.PAWN, chess.WHITE),  # blocker
        )
        assert _blockaded(board, chess.E5, chess.BLACK) == 1

    def test_black_pawn_not_blockaded(self):
        """Black pawn with no blocker directly ahead (lower rank)."""
        board = _empty_board_with(
            (chess.E5, chess.PAWN, chess.BLACK),
        )
        assert _blockaded(board, chess.E5, chess.BLACK) == 0

    def test_boundary_rank_zero(self):
        """Pawn checking rank 0: nr=0 should be valid.

        Kills mutant_14 (nr <= 0) and mutant_15 (nr < 1):
        both would skip rank 0 as out-of-bounds.
        """
        # Black pawn on e2 (rank 1). step=-1 → nr=0.
        # Put a white piece on e1 (rank 0) to block it.
        board = _empty_board_with(
            (chess.E2, chess.PAWN, chess.BLACK),
            (chess.E1, chess.KNIGHT, chess.WHITE),  # blocker on rank 0
        )
        assert _blockaded(board, chess.E2, chess.BLACK) == 1

    def test_boundary_rank_seven(self):
        """White pawn on rank 6 checking nr=7: should be valid.

        Kills mutant_16 (nr >= 7 instead of nr > 7): would incorrectly
        skip rank 7.
        """
        # White pawn on e7 (rank 6). step=+1 → nr=7.
        board = _empty_board_with(
            (chess.E7, chess.PAWN, chess.WHITE),
            (chess.E8, chess.KNIGHT, chess.BLACK),  # blocker on rank 7
        )
        assert _blockaded(board, chess.E7, chess.WHITE) == 1

    def test_out_of_bounds_returns_zero(self):
        """Pawn at edge with nr out of range should return 0, not 1.

        Kills mutant_18 (return 0 → return 1).
        """
        # White pawn on e8 (rank 7). step=+1 → nr=8 (out of bounds).
        board = _empty_board_with(
            (chess.E8, chess.PAWN, chess.WHITE),
        )
        assert _blockaded(board, chess.E8, chess.WHITE) == 0

        # Black pawn on e1 (rank 0). step=-1 → nr=-1 (out of bounds).
        board2 = _empty_board_with(
            (chess.E1, chess.PAWN, chess.BLACK),
        )
        assert _blockaded(board2, chess.E1, chess.BLACK) == 0


# ===================================================================
# _runner_clear_path
# ===================================================================


class TestRunnerClearPathMutants:
    """Kill mutants in _runner_clear_path."""

    def test_stoppers_but_not_blockaded(self):
        """Stopper present but not directly blockaded → NOT a runner.

        Kills mutant_2 (and → or): with `or`, having blockaded==0 alone
        would make this return 1.
        """
        # White pawn on e4, black pawn on d5 (stopper on adjacent file)
        # but nothing on e5 (not blockaded).
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.D5, chess.PAWN, chess.BLACK),  # stopper, not blockade
        )
        assert _stoppers_in_path(board, chess.E4, chess.WHITE) > 0
        assert _blockaded(board, chess.E4, chess.WHITE) == 0
        # With `and`: 0 and 1 → 0. With `or`: 1 or 1 → 1.
        # Wait: _runner returns int(stoppers==0 AND blockaded==0)
        # stoppers=1 → stoppers==0 is False → result is 0
        # With `or`: stoppers==0 is False, blockaded==0 is True → True → 1
        assert _runner_clear_path(board, chess.E4, chess.WHITE) == 0

    def test_blockaded_but_no_stoppers(self):
        """Blockaded but no nearby stopper pawns → NOT a runner.

        Additional coverage for mutant_2 (and→or).
        """
        # White pawn on e4, black knight on e5 (blocks, but not a pawn stopper).
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.E5, chess.KNIGHT, chess.BLACK),  # blockade, not pawn stopper
        )
        assert _stoppers_in_path(board, chess.E4, chess.WHITE) == 0
        assert _blockaded(board, chess.E4, chess.WHITE) == 1
        assert _runner_clear_path(board, chess.E4, chess.WHITE) == 0

    def test_color_none_kills_stoppers(self):
        """Using actual color matters for stopper detection.

        Kills mutant_5 (color → None in _stoppers_in_path call).
        When color=None, WHITE==None is False, so dirs becomes -1 (else branch).
        For a white pawn, this checks the wrong direction.
        """
        # Put stopper only on adjacent file so blockade is 0.
        # With correct color=WHITE, dirs=+1, finds the stopper → runner=0.
        # With color=None, dirs=-1, checks wrong direction → runner could be 1.
        board2 = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.D5, chess.PAWN, chess.BLACK),
        )
        # stoppers=1, blockaded=0 → runner=0
        # With None: stoppers would check wrong direction → may find 0 → runner could be 1
        assert _runner_clear_path(board2, chess.E4, chess.WHITE) == 0

    def test_color_none_kills_blockaded(self):
        """Using actual color matters for blockade detection.

        Kills mutant_13 (color → None in _blockaded call).
        When color=None, step becomes +1 (None==WHITE is False → else -1...
        wait, _blockaded uses `+1 if color == chess.WHITE else -1`).
        So None → step=-1 for what should be a WHITE pawn (step=+1).
        """
        # White pawn on e4, black piece on e5 (rank 4 → rank 5, ahead for white).
        # With correct color=WHITE, step=+1, checks rank 5 → blocked.
        # With color=None, step=-1, checks rank 3 → not blocked.
        board = _empty_board_with(
            (chess.E4, chess.PAWN, chess.WHITE),
            (chess.E5, chess.KNIGHT, chess.BLACK),  # blocker ahead
        )
        # stoppers=0 (it's a knight), blockaded=1
        assert _runner_clear_path(board, chess.E4, chess.WHITE) == 0


# ===================================================================
# features/utils.py - load_feature_module
# ===================================================================


class TestLoadFeatureModuleMutants:
    """Kill mutants in load_feature_module."""

    def test_module_name_is_user_features(self):
        """The loaded module must have __name__ == 'user_features'.

        Kills mutant_6 (→ 'XXuser_featuresXX') and mutant_7 (→ 'USER_FEATURES').
        """
        feature_code = "def extract_features(board):\n" "    return {'x': 1.0}\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(feature_code)
            temp_path = f.name

        try:
            mod = load_feature_module(temp_path)
            assert mod.__name__ == "user_features"
        finally:
            os.unlink(temp_path)

    def test_error_message_exact_text(self):
        """The RuntimeError message for missing extract_features must match.

        Kills mutant_23 which prefixes/suffixes 'XX' to the error message.
        """
        feature_code = "x = 1\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(feature_code)
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError, match=r"^Feature module must"):
                load_feature_module(temp_path)
        finally:
            os.unlink(temp_path)


# ===================================================================
# SurrogateExplainer.calculate_contributions
# ===================================================================


class TestCalculateContributionsMutants:
    """Kill mutants in SurrogateExplainer.calculate_contributions."""

    @pytest.fixture()
    def setup(self):
        """Create a SurrogateExplainer with strong, deterministic signal.

        The model is trained with a very strong first feature coefficient,
        so contributions for `material_diff` will always be significant.
        """
        names = ["material_diff", "mobility_us"]
        model = _make_trained_ensemble(names)
        scaler = _make_scaler(names)
        explainer = SurrogateExplainer(model=model, scaler=scaler, feature_names=names)
        return explainer, names

    def test_default_top_k_is_5(self, setup):
        """Default top_k=5, not 6.

        Kills mutant_1 (top_k=5 → top_k=6).
        We create an explainer with 6+ features, give all strong signal,
        and call with default top_k.
        """
        names = [
            "material_diff",
            "mobility_us",
            "mobility_them",
            "king_ring_pressure_us",
            "passed_us",
            "center_control_us",
            "phase",
        ]
        model = _make_trained_ensemble(names)
        scaler = _make_scaler(names)
        explainer = SurrogateExplainer(model=model, scaler=scaler, feature_names=names)
        before = dict.fromkeys(names, 0.0)
        after = dict.fromkeys(names, 100.0)

        # Use default top_k (should be 5)
        result = explainer.calculate_contributions(before, after, min_cp=0.0)
        assert len(result) <= 5

    def test_default_min_cp_is_5(self, setup):
        """Default min_cp=5.0, not 6.0.

        Kills mutmut_2 (min_cp=5.0 → min_cp=6.0).
        Uses a mock model that returns a contribution of exactly 5.5 cp,
        which passes min_cp=5.0 but fails min_cp=6.0.
        """
        names = ["material_diff"]
        mock_model = MagicMock()
        mock_model.get_contributions.return_value = np.array([5.5])

        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: x

        explainer = SurrogateExplainer(
            model=mock_model, scaler=mock_scaler, feature_names=names
        )
        before = {"material_diff": 0.0}
        after = {"material_diff": 10.0}

        # Default min_cp=5.0: 5.5 >= 5.0 → included
        # Mutant min_cp=6.0: 5.5 < 6.0 → excluded
        result = explainer.calculate_contributions(before, after)
        assert len(result) == 1
        assert math.isclose(result[0][1], 5.5, abs_tol=0.01)

    def test_feature_values_change_output(self, setup):
        """Different feature_after values must produce different contributions.

        Kills mutant_7 (get(None,0.0)): always reads 0.0, output is constant.
        Kills mutant_17 (get(None,0.0) in comprehension): same effect.
        """
        explainer, names = setup
        before = dict.fromkeys(names, 0.0)

        after_high = {"material_diff": 200.0, "mobility_us": 0.0}
        after_zero = {"material_diff": 0.0, "mobility_us": 0.0}

        result_high = explainer.calculate_contributions(
            before, after_high, top_k=10, min_cp=0.0
        )
        result_zero = explainer.calculate_contributions(
            before, after_zero, top_k=10, min_cp=0.0
        )

        # At least one result should be non-empty and differ
        assert len(result_high) > 0

        # Extract material_diff contribution from each
        cp_high = next((r[1] for r in result_high if r[0] == "material_diff"), 0.0)
        cp_zero = next((r[1] for r in result_zero if r[0] == "material_diff"), 0.0)
        # Contributions must be different when inputs differ
        assert not math.isclose(cp_high, cp_zero, abs_tol=0.1)

    def test_missing_feature_defaults_to_zero(self, setup):
        """A missing key in features_after should default to 0.0 not error.

        Kills mutant_8 (default=None → float(None) → TypeError → []).
        Kills mutant_10 (get(fname) → None → float(None) → TypeError → []).
        Kills mutmut_11 (default=1.0 instead of 0.0 → different contribution).
        Uses a mock model where the second feature has a large coefficient,
        so a 0.0 vs 1.0 difference is clearly detectable.
        """
        names = ["feat_a", "feat_b"]
        mock_model = MagicMock()
        # Return contributions proportional to input: each value * 10
        mock_model.get_contributions.side_effect = lambda vec: vec * 10.0

        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: x

        explainer = SurrogateExplainer(
            model=mock_model, scaler=mock_scaler, feature_names=names
        )
        before = {"feat_a": 0.0, "feat_b": 0.0}

        # Only provide feat_a, leave feat_b missing → should default to 0.0
        after_partial = {"feat_a": 5.0}

        result = explainer.calculate_contributions(
            before, after_partial, top_k=10, min_cp=0.0
        )
        # Should NOT be empty (mutant_8, 10 would crash → [])
        assert len(result) > 0

        # feat_b contribution should be ~0.0 (0.0 * 10.0 = 0.0)
        # With mutant_11 (default=1.0), feat_b = 1.0 → contribution = 10.0
        feat_b_cp = next((r[1] for r in result if r[0] == "feat_b"), 0.0)
        assert math.isclose(feat_b_cp, 0.0, abs_tol=0.5)

    def test_scaler_transform_correct(self, setup):
        """The scaler.transform call must receive correctly shaped input.

        Kills mutant_30 (→reshape(1,-2) → ValueError → []).
        Also kills mutants 22, 23, 26, 28 (various reshape/None errors).
        """
        explainer, names = setup
        before = dict.fromkeys(names, 0.0)
        after = {"material_diff": 80.0, "mobility_us": 0.0}

        result = explainer.calculate_contributions(before, after, top_k=10, min_cp=0.0)
        # Must produce actual results, not empty (error) list
        assert len(result) > 0
        # Verify a specific contribution value to rule out error-path results
        assert any(r[0] == "material_diff" for r in result)

    def test_get_contributions_receives_correct_input(self, setup):
        """model.get_contributions must receive the scaled vector, not None.

        Kills mutant_33 (→get_contributions(None)).
        """
        explainer, names = setup
        before = dict.fromkeys(names, 0.0)
        after = {"material_diff": 60.0, "mobility_us": 0.0}

        result = explainer.calculate_contributions(before, after, top_k=10, min_cp=0.0)
        assert len(result) > 0

    def test_min_cp_boundary_gte(self, setup):
        """Contributions exactly equal to min_cp should be included (>=, not >).

        Kills mutant_43 (>= → >).
        """
        explainer, names = setup
        before = dict.fromkeys(names, 0.0)
        after = {"material_diff": 100.0, "mobility_us": 0.0}

        # Get the actual contribution values first
        result_all = explainer.calculate_contributions(
            before, after, top_k=10, min_cp=0.0
        )
        assert len(result_all) > 0

        # Use the exact cp value as min_cp -- should still be included
        exact_cp = abs(result_all[0][1])
        result_exact = explainer.calculate_contributions(
            before, after, top_k=10, min_cp=exact_cp
        )
        # With >= the contribution passes; with > it doesn't
        assert len(result_exact) >= 1
        assert math.isclose(abs(result_exact[0][1]), exact_cp, rel_tol=1e-9)

    def test_error_print_message(self, setup):
        """When an error occurs, the warning message includes the error text.

        Kills mutant_59 (print(f"Warning: ...") → print(None)).
        """
        names = ["material_diff"]
        mock_model = MagicMock()
        mock_model.get_contributions.side_effect = ValueError("test error")

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[1.0]])

        explainer = SurrogateExplainer(
            model=mock_model, scaler=mock_scaler, feature_names=names
        )

        with patch("builtins.print") as mock_print:
            result = explainer.calculate_contributions(
                {"material_diff": 0.0}, {"material_diff": 5.0}
            )
            assert result == []
            # Verify the print was called with a string containing "Warning"
            mock_print.assert_called_once()
            printed_arg = mock_print.call_args[0][0]
            assert isinstance(printed_arg, str)
            assert "Warning" in printed_arg
            assert "test error" in printed_arg
