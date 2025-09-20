"""Stockfish engine interface functions."""

from typing import List, Tuple

import chess
import chess.engine

from .config import SFConfig


def sf_open(cfg: SFConfig):
    """Open a Stockfish engine with the given configuration.

    Args:
        cfg: Stockfish configuration

    Returns:
        The opened engine
    """
    engine = chess.engine.SimpleEngine.popen_uci(cfg.engine_path)
    engine.configure({"Threads": cfg.threads})
    return engine


def sf_eval(engine, board: "chess.Board", cfg: SFConfig) -> float:
    """Get Stockfish evaluation for a position.

    Args:
        engine: The Stockfish engine
        board: The chess board position
        cfg: Stockfish configuration

    Returns:
        Evaluation in centipawns
    """
    limit = (
        chess.engine.Limit(depth=cfg.depth)
        if cfg.movetime == 0
        else chess.engine.Limit(time=cfg.movetime / 1000.0)
    )
    info = engine.analyse(board, limit=limit, multipv=1)
    # engine.analyse returns a list when multipv > 1, but a dict when multipv=1
    if isinstance(info, list):
        info = info[0]
    score = info["score"].pov(board.turn)
    cp = score.score(mate_score=100000)
    # Clip mate scores to prevent instability
    return float(max(-1000, min(1000, cp)))


def sf_top_moves(
    engine, board: "chess.Board", cfg: SFConfig
) -> List[Tuple[chess.Move, float]]:
    """Get top moves from Stockfish for a position.

    Args:
        engine: The Stockfish engine
        board: The chess board position
        cfg: Stockfish configuration

    Returns:
        List of (move, score) tuples
    """
    limit = (
        chess.engine.Limit(depth=cfg.depth)
        if cfg.movetime == 0
        else chess.engine.Limit(time=cfg.movetime / 1000.0)
    )
    infos = engine.analyse(board, limit=limit, multipv=cfg.multipv)
    out = []
    for d in infos:
        pv = d.get("pv", [])
        if not pv:
            continue
        move = pv[0]
        score = d["score"].pov(board.turn).score(mate_score=100000)
        out.append((move, float(score)))
    return out
