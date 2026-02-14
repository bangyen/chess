"""Engine-based dynamic probe functions for feature enrichment.

These functions provide deeper positional signals (e.g. hanging pieces
after a reply, forcing swing) by calling into either the Rust extension
or Stockfish as a fallback.  They are attached as callables to the
feature dictionary under the ``_engine_probes`` key so the audit pipeline
can invoke them lazily.
"""

from typing import Any, Tuple

import chess
import chess.engine

try:
    from chess_ai.rust_utils import (
        calculate_forcing_swing,
        find_best_reply,
    )

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def sf_eval_shallow(engine: Any, board: chess.Board, depth: int = 6) -> float:
    """Shallow engine evaluation with depth limit.

    Returns a centipawn score clipped to [-1000, 1000] so that mate
    evaluations don't dominate surrogate model training.
    """
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
        if isinstance(info, list):
            info = info[0]
        score = info["score"].pov(board.turn)
        cp = score.score(mate_score=100000)
        return float(max(-1000, min(1000, cp)))
    except Exception:
        return 0.0


def hanging_after_reply(
    engine: Any, board: chess.Board, depth: int = 6
) -> Tuple[int, int, int]:
    """Count hanging (attacked-but-undefended) enemy pieces after the
    opponent's best reply.

    Uses the Rust search engine when available (faster TT + pruning),
    falling back to Stockfish analysis otherwise.

    Returns ``(count, max_value, near_king)`` where *count* is the
    number of hanging pieces, *max_value* is the value of the most
    valuable one, and *near_king* is 1 if any hanging piece is within
    one square of the opponent king.
    """
    try:
        reply = None
        if _RUST_AVAILABLE:
            try:
                rust_depth = min(depth, 8)
                uci = find_best_reply(board.fen(), rust_depth)
                if uci:
                    reply = chess.Move.from_uci(uci)
            except Exception:  # noqa: S110
                pass
        if reply is None:
            info = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=1,
            )
            if isinstance(info, list):
                info = info[0]
            reply = info.get("pv", [None])[0]
        if reply is None:
            return 0, 0, 0
        board.push(reply)
        side = board.turn
        cnt, v_max, near_king = 0, 0, 0
        opp = not side
        opp_king_sq = board.king(opp)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == opp:
                defenders = board.attackers(opp, sq)
                attackers = board.attackers(side, sq)
                if not defenders and attackers:
                    cnt += 1
                    pv = {
                        chess.PAWN: 1,
                        chess.KNIGHT: 3,
                        chess.BISHOP: 3,
                        chess.ROOK: 5,
                        chess.QUEEN: 9,
                    }.get(piece.piece_type, 0)
                    v_max = max(v_max, pv)
                    if (
                        opp_king_sq is not None
                        and chess.square_distance(sq, opp_king_sq) <= 1
                    ):
                        near_king = 1
        board.pop()
        return cnt, v_max, near_king
    except Exception:
        return 0, 0, 0


def best_forcing_swing(
    engine: Any, board: chess.Board, d_base: int = 6, k_max: int = 12
) -> float:
    """Largest evaluation swing from a forcing (capture/check) move.

    Uses the Rust engine when available for faster computation,
    falling back to Stockfish otherwise.
    """
    if _RUST_AVAILABLE:
        try:
            rust_depth = min(d_base, 8)
            return float(calculate_forcing_swing(board.fen(), rust_depth))
        except Exception:  # noqa: S110
            pass
    try:
        base = sf_eval_shallow(engine, board, d_base)
        forcing = [
            m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)
        ]
        swings = []
        for m in forcing[:k_max]:
            board.push(m)
            v = sf_eval_shallow(engine, board, d_base - 1)
            board.pop()
            swings.append(v - base)
        return max(swings) if swings else 0.0
    except Exception:
        return 0.0
