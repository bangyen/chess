"""Syzygy tablebase integration for chess AI."""

from typing import Any, Dict

import chess


class SyzygyManager:
    """Manages Syzygy tablebase interactions."""

    def __init__(self, syzygy_path: str | None = None) -> None:
        self.syzygy_path = syzygy_path
        self.syzygy: Any = None
        self._initialize_syzygy()

    def _initialize_syzygy(self) -> None:
        """Initialize Syzygy tablebases."""
        if self.syzygy_path:
            try:
                from ..rust_utils import SyzygyTablebase

                self.syzygy = SyzygyTablebase(self.syzygy_path)
            except Exception:
                self.syzygy = None

    def get_syzygy_data(self, board: chess.Board) -> Dict[str, int]:
        """Get Syzygy tablebase data for the position."""
        if not self.syzygy:
            return {}

        # Check piece count (Syzygy generally supports up to 7 pieces)
        if len(board.piece_map()) > 7:
            return {}

        try:
            fen = board.fen()
            wdl = self.syzygy.probe_wdl(fen)
            dtz = self.syzygy.probe_dtz(fen)

            result = {}
            if wdl is not None:
                result["wdl"] = wdl
            if dtz is not None:
                result["dtz"] = dtz
            return result
        except Exception:
            return {}

    def get_syzygy_reason(
        self, board_before: chess.Board, board_after: chess.Board
    ) -> tuple[str, float, str] | None:
        """Generate a reason based on Syzygy tablebase changes."""
        data_before = self.get_syzygy_data(board_before)
        if not data_before:
            return None

        data_after = self.get_syzygy_data(board_after)
        if not data_after:
            return None

        wdl_before = data_before.get("wdl", 0)
        # Invert wdl_after because it's from opponent's perspective
        wdl_after = -data_after.get("wdl", 0)

        # WDL: 2=Win, 1=MaybeWin, 0=Draw, -1=MaybeLoss, -2=Loss

        # Check for blunders (Win -> Draw/Loss, Draw -> Loss)
        if wdl_before == 2 and wdl_after < 2:
            return ("syzygy_blunder", -500.0, "Tablebase: Throws away a forced win")
        if wdl_before == 0 and wdl_after < 0:
            return ("syzygy_blunder", -300.0, "Tablebase: Blunders into a forced loss")

        # Check for good moves (maintaining win, finding win)
        if wdl_before == 2 and wdl_after == 2:
            return ("syzygy_good", 50.0, "Tablebase: Maintains forced win")

        if wdl_before < 2 and wdl_after == 2:
            return ("syzygy_brilliant", 500.0, "Tablebase: Finds a forced win!")

        if wdl_before == -2 and wdl_after == 0:
            return ("syzygy_save", 300.0, "Tablebase: Salvages a draw from a loss")

        return None
