use pyo3::prelude::*;
use shakmaty::{
    attacks, Bitboard, CastlingMode, Chess, Color, Move, Position, Role, Square,
};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{Tablebase, AmbiguousWdl, MaybeRounded};
use std::collections::BTreeMap;
use std::sync::OnceLock;

// Material values in centipawns.
fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 100,
        Role::Knight => 320,
        Role::Bishop => 330,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 20000,
    }
}

// ── Piece-square tables (from White's perspective, rank-8 = index 0) ──

static EVAL_PST_PAWN: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

static EVAL_PST_KNIGHT: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

static EVAL_PST_BISHOP: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

static EVAL_PST_ROOK: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

static EVAL_PST_QUEEN: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

static EVAL_PST_KING_MG: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

static EVAL_PST_KING_EG: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

/// Count non-pawn, non-king pieces for game-phase detection.
fn count_phase(board: &shakmaty::Board) -> i32 {
    let mut phase = 0i32;
    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            if piece.role != Role::Pawn && piece.role != Role::King {
                phase += 1;
            }
        }
    }
    phase
}

/// PST index for a square, flipped for Black so tables are always
/// written from White's perspective.
#[inline]
fn pst_index(sq: Square, color: Color) -> usize {
    let vis_r = if color == Color::White {
        7 - sq.rank() as usize
    } else {
        sq.rank() as usize
    };
    vis_r * 8 + sq.file() as usize
}

/// Positional evaluation: material + piece-square tables + bishop-pair
/// bonus.  Returns score from the side-to-move's perspective so
/// negamax works directly.
fn evaluate(pos: &Chess) -> i32 {
    let board = pos.board();
    let phase = count_phase(board);
    let is_endgame = phase < 10;

    let mut score = 0i32;

    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let mat = piece_value(piece.role);
            let idx = pst_index(sq, piece.color);
            let pst = match piece.role {
                Role::Pawn   => EVAL_PST_PAWN[idx],
                Role::Knight => EVAL_PST_KNIGHT[idx],
                Role::Bishop => EVAL_PST_BISHOP[idx],
                Role::Rook   => EVAL_PST_ROOK[idx],
                Role::Queen  => EVAL_PST_QUEEN[idx],
                Role::King   => if is_endgame { EVAL_PST_KING_EG[idx] }
                                else          { EVAL_PST_KING_MG[idx] },
            };
            let val = mat + pst;
            if piece.color == Color::White {
                score += val;
            } else {
                score -= val;
            }
        }
    }

    // Bishop-pair bonus (+30 cp)
    let white_bishops = (board.by_role(Role::Bishop) & board.by_color(Color::White)).count();
    let black_bishops = (board.by_role(Role::Bishop) & board.by_color(Color::Black)).count();
    if white_bishops >= 2 { score += 30; }
    if black_bishops >= 2 { score -= 30; }

    if pos.turn() == Color::White { score } else { -score }
}

// ── Zobrist Hashing ──────────────────────────────────────────────────

/// Maps a (color, role) pair to a 0..11 index for Zobrist table lookup.
/// White pieces occupy indices 0..5, Black pieces 6..11.
#[inline]
fn piece_index(color: Color, role: Role) -> usize {
    let c = if color == Color::White { 0 } else { 6 };
    let r = match role {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    };
    c + r
}

/// Pre-computed random keys for Zobrist hashing, generated
/// deterministically from a fixed seed so values are stable across runs.
struct ZobristKeys {
    pieces: [[u64; 12]; 64],
    side_to_move: u64,
    castling_sq: [u64; 64],
    en_passant: [u64; 8],
}

/// Simple xorshift64 PRNG for deterministic Zobrist key generation.
/// The state must never be zero.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Lazily initialised global Zobrist key table so that every call to
/// `zobrist_hash` uses the same random constants.
fn zobrist_keys() -> &'static ZobristKeys {
    static KEYS: OnceLock<ZobristKeys> = OnceLock::new();
    KEYS.get_or_init(|| {
        let mut rng: u64 = 0x123456789ABCDEF0;
        let mut pieces = [[0u64; 12]; 64];
        for sq in pieces.iter_mut() {
            for pc in sq.iter_mut() {
                *pc = xorshift64(&mut rng);
            }
        }
        let side_to_move = xorshift64(&mut rng);
        let mut castling_sq = [0u64; 64];
        for v in castling_sq.iter_mut() {
            *v = xorshift64(&mut rng);
        }
        let mut en_passant = [0u64; 8];
        for v in en_passant.iter_mut() {
            *v = xorshift64(&mut rng);
        }
        ZobristKeys { pieces, side_to_move, castling_sq, en_passant }
    })
}

/// Compute a full Zobrist hash for a chess position, incorporating
/// piece placement, side to move, and castling rights.
fn zobrist_hash(pos: &Chess) -> u64 {
    let keys = zobrist_keys();
    let board = pos.board();
    let mut hash = 0u64;

    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let pi = piece_index(piece.color, piece.role);
            hash ^= keys.pieces[sq as usize][pi];
        }
    }
    if pos.turn() == Color::Black {
        hash ^= keys.side_to_move;
    }
    for sq in pos.castles().castling_rights() {
        hash ^= keys.castling_sq[sq as usize];
    }
    if let Some(ep) = pos.maybe_ep_square() {
        hash ^= keys.en_passant[ep.file() as usize];
    }
    hash
}

// ── Transposition Table & Search Context ─────────────────────────────

const MAX_PLY: usize = 64;
const TT_SIZE: usize = 1 << 20;

/// Whether a transposition-table score is exact, a lower bound
/// (failed high), or an upper bound (failed low).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

/// A single entry in the transposition table, storing the best
/// information discovered for one position at a given search depth.
#[derive(Clone)]
struct TTEntry {
    key: u64,
    depth: u8,
    score: i32,
    flag: TTFlag,
    best_move: Option<Move>,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry { key: 0, depth: 0, score: 0, flag: TTFlag::Exact, best_move: None }
    }
}

/// Mutable state threaded through the search: transposition table,
/// killer-move slots, and history heuristic counters.
struct SearchContext {
    tt: Vec<TTEntry>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[i32; 64]; 12],
    ply: usize,
    allow_null: bool,
}

impl SearchContext {
    /// Allocate a fresh search context with an empty TT.
    fn new() -> Self {
        SearchContext {
            tt: vec![TTEntry::default(); TT_SIZE],
            killers: [[None; 2]; MAX_PLY],
            history: [[0; 64]; 12],
            ply: 0,
            allow_null: true,
        }
    }

    /// Probe the TT for a matching entry.  Returns `None` when the
    /// slot is empty or holds a different position.
    fn tt_probe(&self, key: u64) -> Option<&TTEntry> {
        let idx = (key as usize) % TT_SIZE;
        let entry = &self.tt[idx];
        if entry.key == key { Some(entry) } else { None }
    }

    /// Store search results in the TT (always-replace policy).
    fn tt_store(
        &mut self, key: u64, depth: u8, score: i32,
        flag: TTFlag, best_move: Option<Move>,
    ) {
        let idx = (key as usize) % TT_SIZE;
        self.tt[idx] = TTEntry { key, depth, score, flag, best_move };
    }

    /// Record a killer move for the current ply (quiet moves that
    /// caused a beta cutoff).
    fn store_killer(&mut self, ply: usize, m: Move) {
        if ply < MAX_PLY && self.killers[ply][0] != Some(m) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(m);
        }
    }

    /// Bump the history counter for a quiet move that caused a beta
    /// cutoff, aging all counters when they grow too large.
    fn update_history(&mut self, pos: &Chess, m: &Move, depth: u8) {
        if let Some(from_sq) = m.from() {
            if let Some(piece) = pos.board().piece_at(from_sq) {
                let pi = piece_index(piece.color, piece.role);
                let to = m.to() as usize;
                self.history[pi][to] += (depth as i32) * (depth as i32);
                if self.history[pi][to] > 1_000_000 {
                    for p in self.history.iter_mut() {
                        for s in p.iter_mut() {
                            *s /= 2;
                        }
                    }
                }
            }
        }
    }
}

/// Assign a priority score to each legal move so the most promising
/// moves are searched first.  Ordering: TT move > promotions >
/// captures (MVV-LVA) > killers > history heuristic.
fn score_moves(
    pos: &Chess,
    moves: &[Move],
    ctx: &SearchContext,
    tt_move: Option<&Move>,
) -> Vec<(i32, Move)> {
    moves
        .iter()
        .map(|m| {
            let score;
            if tt_move.map_or(false, |tm| tm == m) {
                score = 30000;
            } else if m.is_promotion() {
                score = 20000;
            } else if m.is_capture() {
                let board = pos.board();
                let victim = board.piece_at(m.to()).map(|p| p.role).unwrap_or(Role::Pawn);
                let attacker = m
                    .from()
                    .and_then(|sq| board.piece_at(sq))
                    .map(|p| p.role)
                    .unwrap_or(Role::Pawn);
                score = 10000 + piece_value(victim) - piece_value(attacker);
            } else if ctx.ply < MAX_PLY && ctx.killers[ctx.ply][0] == Some(*m) {
                score = 9000;
            } else if ctx.ply < MAX_PLY && ctx.killers[ctx.ply][1] == Some(*m) {
                score = 8999;
            } else {
                score = m
                    .from()
                    .and_then(|sq| pos.board().piece_at(sq))
                    .map(|p| {
                        let pi = piece_index(p.color, p.role);
                        ctx.history[pi][m.to() as usize].min(8998)
                    })
                    .unwrap_or(0);
            }
            (score, *m)
        })
        .collect()
}

/// Quiescence search with delta pruning: keeps searching tactical
/// moves (captures / promotions) until the position is quiet, while
/// pruning captures that cannot possibly raise alpha.
fn quiesce(pos: &Chess, mut alpha: i32, beta: i32) -> i32 {
    let stand_pat = evaluate(pos);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let moves = pos.legal_moves();
    let mut tactical: Vec<(i32, Move)> = moves
        .into_iter()
        .filter(|m| m.is_capture() || m.is_promotion())
        .map(|m| {
            let mut score = 0i32;
            if m.is_capture() {
                let victim = pos.board().piece_at(m.to())
                    .map(|p| p.role).unwrap_or(Role::Pawn);
                let attacker = pos.board().piece_at(m.from().unwrap())
                    .map(|p| p.role).unwrap_or(Role::Pawn);
                score = 10000 + piece_value(victim) - piece_value(attacker);
            }
            if m.is_promotion() {
                score += 20000;
            }
            (score, m)
        })
        .collect();
    tactical.sort_by(|a, b| b.0.cmp(&a.0));

    for (_, m) in tactical {
        // Delta pruning: skip captures whose best-case gain cannot
        // reach alpha (promotions are always searched).
        if !m.is_promotion() && m.is_capture() {
            let victim_val = pos.board().piece_at(m.to())
                .map(|p| piece_value(p.role))
                .unwrap_or(100);
            if stand_pat + victim_val + 200 < alpha {
                continue;
            }
        }

        let mut new_pos = pos.clone();
        new_pos.play_unchecked(m);
        let score = -quiesce(&new_pos, -beta, -alpha);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }
    alpha
}

/// Alpha-beta search enhanced with transposition table lookups,
/// null-move pruning, and late move reductions.  Falls back to
/// quiescence search at the leaves.
fn alpha_beta(
    pos: &Chess, mut alpha: i32, beta: i32, depth: u8, ctx: &mut SearchContext,
) -> i32 {
    // ── Terminal checks ──────────────────────────────────────────────
    if pos.is_game_over() {
        if pos.is_checkmate() {
            return -30000 + ctx.ply as i32;
        }
        return 0;
    }
    if depth == 0 {
        return quiesce(pos, alpha, beta);
    }

    let hash = zobrist_hash(pos);
    let orig_alpha = alpha;

    // ── TT probe ─────────────────────────────────────────────────────
    let tt_move: Option<Move>;
    if let Some(entry) = ctx.tt_probe(hash) {
        tt_move = entry.best_move;
        if entry.depth >= depth {
            match entry.flag {
                TTFlag::Exact => return entry.score,
                TTFlag::LowerBound => {
                    if entry.score >= beta { return entry.score; }
                }
                TTFlag::UpperBound => {
                    if entry.score <= alpha { return entry.score; }
                }
            }
        }
    } else {
        tt_move = None;
    }

    let in_check = pos.is_check();

    // ── Null-move pruning ────────────────────────────────────────────
    if ctx.allow_null && !in_check && depth >= 3 {
        let board = pos.board();
        let us = pos.turn();
        let has_pieces =
            !(board.by_color(us) & !board.by_role(Role::Pawn) & !board.by_role(Role::King))
                .is_empty();
        if has_pieces {
            if let Ok(null_pos) = pos.clone().swap_turn() {
                let r: u8 = if depth >= 6 { 3 } else { 2 };
                let prev_null = ctx.allow_null;
                ctx.allow_null = false;
                ctx.ply += 1;
                let null_score =
                    -alpha_beta(&null_pos, -beta, -beta + 1, depth - 1 - r, ctx);
                ctx.ply -= 1;
                ctx.allow_null = prev_null;
                if null_score >= beta {
                    return beta;
                }
            }
        }
    }

    // ── Move generation & ordering ───────────────────────────────────
    let moves = pos.legal_moves();
    if moves.is_empty() {
        return if in_check { -30000 + ctx.ply as i32 } else { 0 };
    }

    let move_list: Vec<Move> = moves.into_iter().collect();
    let mut scored = score_moves(pos, &move_list, ctx, tt_move.as_ref());
    scored.sort_by(|a, b| b.0.cmp(&a.0));

    let mut best_score = -50000i32;
    let mut best_move: Option<Move> = None;
    let ply = ctx.ply;

    for (i, (_, m)) in scored.iter().enumerate() {
        let mut new_pos = pos.clone();
        new_pos.play_unchecked(*m);
        let gives_check = new_pos.is_check();

        // ── Late Move Reductions (LMR) ──────────────────────────────
        let score;
        if i >= 4
            && depth >= 3
            && !m.is_capture()
            && !m.is_promotion()
            && !gives_check
            && !in_check
        {
            let reduction: u8 = if depth >= 6 && i >= 8 { 2 } else { 1 };
            let reduced_depth = depth - 1 - reduction;
            ctx.ply += 1;
            let lmr_score =
                -alpha_beta(&new_pos, -alpha - 1, -alpha, reduced_depth, ctx);
            ctx.ply -= 1;

            if lmr_score > alpha {
                // Re-search at full depth to verify.
                ctx.ply += 1;
                score = -alpha_beta(&new_pos, -beta, -alpha, depth - 1, ctx);
                ctx.ply -= 1;
            } else {
                score = lmr_score;
            }
        } else {
            ctx.ply += 1;
            score = -alpha_beta(&new_pos, -beta, -alpha, depth - 1, ctx);
            ctx.ply -= 1;
        }

        if score > best_score {
            best_score = score;
            best_move = Some(*m);
        }
        if score >= beta {
            if !m.is_capture() && !m.is_promotion() {
                ctx.store_killer(ply, *m);
                ctx.update_history(pos, m, depth);
            }
            ctx.tt_store(hash, depth, beta, TTFlag::LowerBound, Some(*m));
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    let flag = if best_score <= orig_alpha {
        TTFlag::UpperBound
    } else {
        TTFlag::Exact
    };
    ctx.tt_store(hash, depth, best_score, flag, best_move);
    best_score
}

/// Core iterative-deepening search with aspiration windows.  The
/// transposition table persists across iterations so each deeper
/// search benefits from earlier results.
fn find_best_reply_impl(pos: &Chess, depth: u8) -> Option<String> {
    let moves = pos.legal_moves();
    if moves.is_empty() {
        return None;
    }

    let mut ctx = SearchContext::new();
    let mut best_move: Option<String> = None;
    let mut prev_score = 0i32;

    for d in 1..=depth {
        // Aspiration windows: narrow search around the previous score
        // for d >= 2, falling back to a full window on fail.
        let score;
        if d <= 1 {
            score = alpha_beta(pos, -50000, 50000, d, &mut ctx);
        } else {
            let a = prev_score - 25;
            let b = prev_score + 25;
            let s = alpha_beta(pos, a, b, d, &mut ctx);
            if s <= a || s >= b {
                // Window failed -- re-search with full bounds.
                score = alpha_beta(pos, -50000, 50000, d, &mut ctx);
            } else {
                score = s;
            }
        }
        prev_score = score;

        // Retrieve the best move for the root position from the TT.
        let hash = zobrist_hash(pos);
        if let Some(entry) = ctx.tt_probe(hash) {
            if let Some(m) = entry.best_move {
                best_move = Some(m.to_uci(CastlingMode::Standard).to_string());
            }
        }
    }

    // Fallback: return the first legal move if TT lookup failed.
    if best_move.is_none() {
        let first = pos.legal_moves().into_iter().next();
        if let Some(m) = first {
            best_move = Some(m.to_uci(CastlingMode::Standard).to_string());
        }
    }

    best_move
}

/// Python-facing wrapper around `find_best_reply_impl` that parses
/// the FEN string and converts the result to a `PyResult`.
#[pyfunction]
fn find_best_reply(fen: &str, depth: u8) -> PyResult<Option<String>> {
    let setup: Fen = fen
        .parse()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
    let pos: Chess = setup
        .into_position(CastlingMode::Standard)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;
    Ok(find_best_reply_impl(&pos, depth))
}

/// Core forcing-swing calculation: evaluates the largest evaluation
/// swing obtainable from a forcing move (capture or check).
fn calculate_forcing_swing_impl(pos: &Chess, depth: u8) -> f32 {
    let mut ctx = SearchContext::new();
    let base_eval = alpha_beta(pos, -50000, 50000, depth, &mut ctx);

    let moves = pos.legal_moves();
    let mut max_swing: f32 = 0.0;

    for m in moves {
        let is_capture = m.is_capture();
        let gives_check = {
            let mut test_pos = pos.clone();
            test_pos.play_unchecked(m);
            test_pos.is_check()
        };

        if is_capture || gives_check {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(m);
            let ev_after =
                alpha_beta(&new_pos, -50000, 50000, depth.saturating_sub(1), &mut ctx);
            let score_for_us = -ev_after;
            let swing = (score_for_us - base_eval) as f32;
            if swing > max_swing {
                max_swing = swing;
            }
        }
    }

    max_swing
}

/// Python-facing wrapper around `calculate_forcing_swing_impl` that
/// parses the FEN string and converts the result to a `PyResult`.
#[pyfunction]
fn calculate_forcing_swing(fen: &str, depth: u8) -> PyResult<f32> {
    let setup: Fen = fen
        .parse()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
    let pos: Chess = setup
        .into_position(CastlingMode::Standard)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;
    Ok(calculate_forcing_swing_impl(&pos, depth))
}

#[pyclass]
struct SyzygyTablebase {
    tb: Tablebase<Chess>,
}

#[pymethods]
impl SyzygyTablebase {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let mut tb = Tablebase::new();
        tb.add_directory(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(SyzygyTablebase { tb })
    }

    fn probe_wdl(&self, fen: &str) -> PyResult<Option<i32>> {
        let setup: Fen = fen.parse().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
        let pos: Chess = setup.into_position(CastlingMode::Standard).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;

        match self.tb.probe_wdl(&pos) {
            Ok(wdl) => match wdl {
                AmbiguousWdl::Win => Ok(Some(2)),
                AmbiguousWdl::Loss => Ok(Some(-2)),
                AmbiguousWdl::Draw => Ok(Some(0)),
                AmbiguousWdl::BlessedLoss => Ok(Some(-1)),
                AmbiguousWdl::CursedWin => Ok(Some(1)),
                AmbiguousWdl::MaybeWin => Ok(Some(1)),
                AmbiguousWdl::MaybeLoss => Ok(Some(-1)),
            },
            Err(_) => Ok(None),
        }
    }

    fn probe_dtz(&self, fen: &str) -> PyResult<Option<i32>> {
        let setup: Fen = fen.parse().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
        let pos: Chess = setup.into_position(CastlingMode::Standard).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;

        match self.tb.probe_dtz(&pos) {
            Ok(maybe_rounded) => {
                let dtz = match maybe_rounded {
                    MaybeRounded::Precise(d) => d,
                    MaybeRounded::Rounded(d) => d,
                };
                Ok(Some(dtz.0 as i32))
            },
            Err(_) => Ok(None),
        }
    }
}

#[pyfunction]
fn extract_features_rust(fen: &str) -> PyResult<BTreeMap<String, f32>> {
    let setup: Fen = fen.parse().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
    let pos: Chess = setup.into_position(CastlingMode::Standard).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;

    let mut feats = BTreeMap::new();
    let turn = pos.turn();
    let opp = turn.other();

    // 1. Material & Phase
    let board = pos.board();
    let mut mat_us = 0.0;
    let mut mat_them = 0.0;
    let mut phase = 0;

    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let val = match piece.role {
                Role::Pawn => 1.0,
                Role::Knight => 3.0,
                Role::Bishop => 3.1,
                Role::Rook => 5.0,
                Role::Queen => 9.0,
                Role::King => 0.0,
            };
            if piece.color == turn {
                mat_us += val;
            } else {
                mat_them += val;
            }

            if piece.role != Role::Pawn && piece.role != Role::King {
                phase += 1;
            }
        }
    }
    feats.insert("material_us".to_string(), mat_us);
    feats.insert("material_them".to_string(), mat_them);
    feats.insert("material_diff".to_string(), mat_us - mat_them);
    feats.insert("phase".to_string(), phase as f32);

    // 2. Mobility
    let moves = pos.legal_moves();
    feats.insert("mobility_us".to_string(), (moves.len() as f32).min(40.0));

    let opp_pos = pos.clone().swap_turn().unwrap_or_else(|_| pos.clone());
    feats.insert("mobility_them".to_string(), (opp_pos.legal_moves().len() as f32).min(40.0));

    // 3. King Ring Pressure
    let get_king_ring = |side: Color| {
        let mut ring = Bitboard::EMPTY;
        if let Some(ksq) = board.king_of(side) {
            for s in Square::ALL {
                if ksq.distance(s) <= 1 {
                    ring |= Bitboard::from_square(s);
                }
            }
        }
        ring
    };

    let weight_pressure = |role: Role| match role {
        Role::Pawn => 1.0,
        Role::Knight => 3.0f32.powf(0.7),
        Role::Bishop => 3.1f32.powf(0.7),
        Role::Rook => 5.0f32.powf(0.7),
        Role::Queen => 9.0f32.powf(0.7),
        _ => 0.0,
    };

    let calc_pressure = |attacking_side: Color| {
        let ring = get_king_ring(attacking_side.other());
        if ring.is_empty() { return 0.0; }
        let mut s = 0.0;
        let occupied = board.occupied();
        for sq in ring {
            let attackers = board.attacks_to(sq, attacking_side, occupied);
            if !attackers.is_empty() {
                // Strongest attacker weight
                let mut max_w = 0.0;
                for a_sq in attackers {
                    if let Some(p) = board.piece_at(a_sq) {
                        let w = weight_pressure(p.role);
                        if w > max_w { max_w = w; }
                    }
                }
                s += max_w;
            }
        }
        s / (phase as f32).max(6.0)
    };

    feats.insert("king_ring_pressure_us".to_string(), calc_pressure(turn));
    feats.insert("king_ring_pressure_them".to_string(), calc_pressure(opp));

    // 4. Passed Pawns
    let is_passed = |sq: Square, side: Color| {
        let file = sq.file();
        let rank = sq.rank();
        let enemy_pawns = board.by_role(Role::Pawn) & board.by_color(side.other());
        
        for f_off in -1..=1 {
            let f = file as i8 + f_off;
            if f < 0 || f > 7 { continue; }
            let check_file = shakmaty::File::new(f as u32);
            let file_bb = Bitboard::from_file(check_file);
            let ahead_bb = match side {
                Color::White => {
                    let mut bb = Bitboard::EMPTY;
                    for r in (rank as usize + 1)..8 {
                        bb |= Bitboard::from_rank(shakmaty::Rank::new(r as u32));
                    }
                    bb
                }
                Color::Black => {
                    let mut bb = Bitboard::EMPTY;
                    for r in 0..(rank as usize) {
                        bb |= Bitboard::from_rank(shakmaty::Rank::new(r as u32));
                    }
                    bb
                }
            };
            if !(enemy_pawns & file_bb & ahead_bb).is_empty() {
                return false;
            }
        }
        true
    };

    let count_passed = |side: Color| {
        let mut count = 0;
        let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
        for sq in my_pawns {
            if is_passed(sq, side) {
                count += 1;
            }
        }
        count as f32
    };

    feats.insert("passed_us".to_string(), count_passed(turn));
    feats.insert("passed_them".to_string(), count_passed(opp));

    // 5. File State
    let get_file_state = |side: Color| {
        let mut open = 0;
        let mut semi_open = 0;
        for f in 0..8 {
            let file_bb = Bitboard::from_file(shakmaty::File::new(f as u32));
            let pawns_on_file = board.by_role(Role::Pawn) & file_bb;
            let my_pawns = pawns_on_file & board.by_color(side);
            let opp_pawns = pawns_on_file & board.by_color(side.other());
            
            if pawns_on_file.is_empty() {
                open += 1;
            } else if opp_pawns.is_empty() && !my_pawns.is_empty() {
                // In Python: elif not pawns_opp: semi_open += 1
                // Wait, if not pawns_opp, it means only OUR pawns or NO pawns. 
                // But open handles NO pawns. So it's ONLY our pawns.
                semi_open += 1;
            } else if my_pawns.is_empty() && !opp_pawns.is_empty() {
                // The python code for file_state(side) returns:
                // if not pawns_side and not pawns_opp: open += 1
                // elif not pawns_opp: semi_open += 1
                // Wait, if not pawns_opp but there ARE pawns_side, it's semi_open?
                // Yes. 
                semi_open += 1;
            }
        }
        (open as f32, semi_open as f32)
    };
    // Re-check Python logic:
    // if not pawns_side and not pawns_opp: open_files += 1
    // elif not pawns_opp: semi_open += 1
    // This is weird. semi_open is if WE have pawns but they don't?
    // Standard def: Semi-open for US means NO OWN pawns, but enemy might have.
    // Python code: `semi_open` is incremented if `not pawns_opp`.
    // So if NO enemy pawns, it's semi_open.
    // This seems flipped from standard, but I'll follow Python.
    
    let (of_us, sof_us) = get_file_state(turn);
    let (of_them, sof_them) = get_file_state(opp);
    feats.insert("open_files_us".to_string(), of_us);
    feats.insert("semi_open_us".to_string(), sof_us);
    feats.insert("open_files_them".to_string(), of_them);
    feats.insert("semi_open_them".to_string(), sof_them);

    // 6. Center Control & Piece Activity
    let center_squares = Bitboard::from_square(Square::D4) | Bitboard::from_square(Square::D5) | 
                         Bitboard::from_square(Square::E4) | Bitboard::from_square(Square::E5);
    
    let center_us = (board.occupied() & board.by_color(turn) & center_squares).count() as f32;
    let center_them = (board.occupied() & board.by_color(opp) & center_squares).count() as f32;
    feats.insert("center_control_us".to_string(), center_us);
    feats.insert("center_control_them".to_string(), center_them);

    let calc_activity = |side: Color| {
        let mut attacks = Bitboard::EMPTY;
        for sq in board.by_color(side) {
            attacks |= board.attacks_from(sq);
        }
        attacks.count() as f32
    };
    feats.insert("piece_activity_us".to_string(), calc_activity(turn));
    feats.insert("piece_activity_them".to_string(), calc_activity(opp));

    // 7. King Safety
    let king_safety = |side: Color| {
        if let Some(ksq) = board.king_of(side) {
            let mut safety = 0.0;
            let _occupied = board.occupied();
            // In shakmaty, king attacks don't need occupied? 
            // Actually board.attacks_from(Role::King, ksq, EMPTY) works.
            for sq in board.attacks_from(ksq) {
                if let Some(p) = board.piece_at(sq) {
                    if p.color == side {
                        safety += 1.0;
                    }
                }
            }
            safety
        } else {
            0.0
        }
    };
    feats.insert("king_safety_us".to_string(), king_safety(turn));
    feats.insert("king_safety_them".to_string(), king_safety(opp));

    // 8. Tactical
    let count_hanging = |side: Color| {
        let mut count = 0;
        let my_pieces = board.by_color(side);
        let occupied = board.occupied();
        for sq in my_pieces {
            let attacked = board.attacks_to(sq, side.other(), occupied);
            let defended = board.attacks_to(sq, side, occupied);
            if !attacked.is_empty() && defended.is_empty() {
                count += 1;
            }
        }
        count as f32
    };
    feats.insert("hanging_us".to_string(), count_hanging(turn));
    feats.insert("hanging_them".to_string(), count_hanging(opp));

    let has_bishop_pair = |side: Color| {
        if (board.by_role(Role::Bishop) & board.by_color(side)).count() >= 2 { 1.0 } else { 0.0 }
    };
    feats.insert("bishop_pair_us".to_string(), has_bishop_pair(turn));
    feats.insert("bishop_pair_them".to_string(), has_bishop_pair(opp));

    let count_rook_7th = |side: Color| {
        let target_rank = match side {
            Color::White => shakmaty::Rank::Seventh, // 0-indexed: Seventh rank is index 6? 
            // Wait, Rank::Seventh is index 6. 
            // In shakmaty, Rank::First is 0. Rank::Seventh is 6.
            Color::Black => shakmaty::Rank::Second, // index 1
        };
        (board.by_role(Role::Rook) & board.by_color(side) & Bitboard::from_rank(target_rank)).count() as f32
    };
    feats.insert("rook_on_7th_us".to_string(), count_rook_7th(turn));
    feats.insert("rook_on_7th_them".to_string(), count_rook_7th(opp));

    let king_pawn_shield = |side: Color| {
        if let Some(ksq) = board.king_of(side) {
            let file = ksq.file();
            let rank = ksq.rank();
            let mut count = 0;
            
            let shield_ranks = match side {
                Color::White => [rank.offset(1), rank.offset(2)],
                Color::Black => [rank.offset(-1), rank.offset(-2)],
            };
            
            let mut shield_files = Vec::new();
            if let Some(f) = file.offset(-1) { shield_files.push(f); }
            shield_files.push(file);
            if let Some(f) = file.offset(1) { shield_files.push(f); }
            
            for &f in &shield_files {
                for &r_opt in &shield_ranks {
                    if let Some(r) = r_opt {
                        let sq = Square::from_coords(f, r);
                        if let Some(p) = board.piece_at(sq) {
                            if p.role == Role::Pawn && p.color == side {
                                count += 1;
                            }
                        }
                    }
                }
            }
            count as f32
        } else {
            0.0
        }
    };
    feats.insert("king_pawn_shield_us".to_string(), king_pawn_shield(turn));
    feats.insert("king_pawn_shield_them".to_string(), king_pawn_shield(opp));

    // 9. Outposts
    let count_outposts = |side: Color| {
        let mut count = 0;
        let knights = board.by_role(Role::Knight) & board.by_color(side);
        let occupied = board.occupied();
        for sq in knights {
            let rank: shakmaty::Rank = sq.rank();
            let rel_rank = if side == Color::White { rank as usize } else { 7 - rank as usize };
            if rel_rank < 3 || rel_rank > 5 { continue; }
            
            // Pawn support
            let mut is_supported = false;
            for attacker_sq in board.attacks_to(sq, side, occupied) {
                if let Some(p) = board.piece_at(attacker_sq) {
                    if p.role == Role::Pawn {
                        is_supported = true;
                        break;
                    }
                }
            }
            if !is_supported { continue; }
            
            // Attacked by enemy pawn?
            let mut attacked_by_pawn = false;
            for attacker_sq in board.attacks_to(sq, side.other(), occupied) {
                if let Some(p) = board.piece_at(attacker_sq) {
                    if p.role == Role::Pawn {
                        attacked_by_pawn = true;
                        break;
                    }
                }
            }
            if attacked_by_pawn { continue; }
            
            count += 1;
        }
        count as f32
    };
    feats.insert("outposts_us".to_string(), count_outposts(turn));
    feats.insert("outposts_them".to_string(), count_outposts(opp));

    // 10. Batteries
    let count_batteries = |side: Color| {
        let mut count = 0;
        
        // Files and Ranks
        for i in 0..8 {
            // File
            let mut file_pieces = 0;
            for r in 0..8 {
                let sq = Square::from_coords(shakmaty::File::new(i as u32), shakmaty::Rank::new(r as u32));
                if let Some(p) = board.piece_at(sq) {
                    if p.color == side && (p.role == Role::Rook || p.role == Role::Queen) {
                        file_pieces += 1;
                    }
                }
            }
            if file_pieces >= 2 { count += 1; }

            // Rank
            let mut rank_pieces = 0;
            for f in 0..8 {
                let sq = Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new(i as u32));
                if let Some(p) = board.piece_at(sq) {
                    if p.color == side && (p.role == Role::Rook || p.role == Role::Queen) {
                        rank_pieces += 1;
                    }
                }
            }
            if rank_pieces >= 2 { count += 1; }
        }

        // Diagonals (B+Q)
        // Simplification: just check if multiple B/Q on any diagonal
        // Using bitboards for diagonals...
        // Positive diagonals: f + r = s (0..14)
        for s in 0..15 {
            let mut diag_pieces = 0;
            for f in 0..8 {
                let r = s as i32 - f as i32;
                if r >= 0 && r < 8 {
                    let sq = Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new(r as u32));
                    if let Some(p) = board.piece_at(sq) {
                        if p.color == side && (p.role == Role::Bishop || p.role == Role::Queen) {
                            diag_pieces += 1;
                        }
                    }
                }
            }
            if diag_pieces >= 2 { count += 1; }
        }
        // Negative diagonals: f - r = d (-7..7)
        for d in -7..8 {
            let mut diag_pieces = 0;
            for f in 0..8 {
                let r = f as i32 - d;
                if r >= 0 && r < 8 {
                    let sq = Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new(r as u32));
                    if let Some(p) = board.piece_at(sq) {
                        if p.color == side && (p.role == Role::Bishop || p.role == Role::Queen) {
                            diag_pieces += 1;
                        }
                    }
                }
            }
            if diag_pieces >= 2 { count += 1; }
        }

        count as f32
    };
    feats.insert("batteries_us".to_string(), count_batteries(turn));
    feats.insert("batteries_them".to_string(), count_batteries(opp));

    // 11. Isolated Pawns
    let count_isolated = |side: Color| {
        let mut count = 0;
        let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
        for sq in my_pawns {
            let file: shakmaty::File = sq.file();
            let mut has_neighbor = false;
            for f_off in [-1, 1] {
                let f = file as i32 + f_off;
                if f >= 0 && f < 8 {
                    let adj_file = shakmaty::File::new(f as u32);
                    let adj_bb = Bitboard::from_file(adj_file);
                    if !(board.by_role(Role::Pawn) & board.by_color(side) & adj_bb).is_empty() {
                        has_neighbor = true;
                        break;
                    }
                }
            }
            if !has_neighbor { count += 1; }
        }
        count as f32
    };
    feats.insert("isolated_pawns_us".to_string(), count_isolated(turn));
    feats.insert("isolated_pawns_them".to_string(), count_isolated(opp));

    // 12. Safe Mobility
    // Moves that don't land on squares attacked by enemy pawns
    let get_safe_mobility = |pos_in: &Chess| {
        let side = pos_in.turn();
        let opp = side.other();
        let mut enemy_pawn_attacks = Bitboard::EMPTY;
        let _occupied = pos_in.board().occupied();
        for sq in pos_in.board().by_role(Role::Pawn) & pos_in.board().by_color(opp) {
            enemy_pawn_attacks |= shakmaty::attacks::pawn_attacks(opp, sq);
        }
        
        let mut safe_count = 0;
        for m in pos_in.legal_moves() {
            if !(enemy_pawn_attacks & Bitboard::from_square(m.to())).is_empty() {
                continue;
            }
            safe_count += 1;
        }
        (safe_count as f32).min(40.0)
    };
    feats.insert("safe_mobility_us".to_string(), get_safe_mobility(&pos));
    
    // For opponent:
    feats.insert("safe_mobility_them".to_string(), get_safe_mobility(&opp_pos));

    // 13. Rook on Open File (standard definition)
    let rook_on_open = |side: Color| {
        let mut count = 0.0;
        let rooks = board.by_role(Role::Rook) & board.by_color(side);
        for sq in rooks {
            let sq_file: shakmaty::File = sq.file();
            let file_bb = Bitboard::from_file(sq_file);
            let pawns_on_file = board.by_role(Role::Pawn) & file_bb;
            if pawns_on_file.is_empty() {
                count += 1.0;
            } else if (pawns_on_file & board.by_color(side)).is_empty() {
                count += 0.5; // Semi-open (no own pawns)
            }
        }
        count
    };
    feats.insert("rook_open_file_us".to_string(), rook_on_open(turn));
    feats.insert("rook_open_file_them".to_string(), rook_on_open(opp));

    // 13b. Connected Rooks (same rank, no pieces between)
    let connected_rooks = |side: Color| -> f32 {
        let rooks: Vec<Square> = (board.by_role(Role::Rook) & board.by_color(side)).into_iter().collect();
        if rooks.len() < 2 {
            return 0.0;
        }
        let (r0, r1) = (rooks[0], rooks[1]);
        if r0.rank() != r1.rank() {
            return 0.0;
        }
        // Check no pieces between the two rooks on their shared rank
        let between = attacks::between(r0, r1) & board.occupied();
        if between.is_empty() { 1.0 } else { 0.0 }
    };
    feats.insert("connected_rooks_us".to_string(), connected_rooks(turn));
    feats.insert("connected_rooks_them".to_string(), connected_rooks(opp));

    // 14. Backward Pawns
    let count_backward = |side: Color| {
        let mut count = 0;
        let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
        let opp = side.other();
        let mut enemy_pawn_attacks = Bitboard::EMPTY;
        for sq in board.by_role(Role::Pawn) & board.by_color(opp) {
            enemy_pawn_attacks |= shakmaty::attacks::pawn_attacks(opp, sq); 
        }

        for sq in my_pawns {
            let file: shakmaty::File = sq.file();
            let rank: shakmaty::Rank = sq.rank();
            let rank_usize = rank as usize;

            // 1. Is supported?
            let mut is_supported = false;
            for f_off in [-1, 1] {
                let f = file as i32 + f_off;
                if f >= 0 && f < 8 {
                    let adj_file = shakmaty::File::new(f as u32);
                    let adj_bb = Bitboard::from_file(adj_file);
                    let adj_pawns = board.by_role(Role::Pawn) & board.by_color(side) & adj_bb;
                    for p_sq in adj_pawns {
                        let p_rank: shakmaty::Rank = p_sq.rank();
                        if (side == Color::White && (p_rank as usize) <= rank_usize) || (side == Color::Black && (p_rank as usize) >= rank_usize) {
                            is_supported = true;
                            break;
                        }
                    }
                }
                if is_supported { break; }
            }
            if is_supported { continue; }

            // 2. Is stop square controlled by enemy pawn?
            let stop_rank = if side == Color::White { sq.rank().offset(1) } else { sq.rank().offset(-1) };
            if let Some(r) = stop_rank {
                let stop_sq = Square::from_coords(file, r);
                if !(enemy_pawn_attacks & Bitboard::from_square(stop_sq)).is_empty() {
                    count += 1;
                }
            }
        }
        count as f32
    };
    feats.insert("backward_pawns_us".to_string(), count_backward(turn));
    feats.insert("backward_pawns_them".to_string(), count_backward(opp));

    let count_pinned = |side: Color| {
        let mut count = 0;
        if let Some(king) = board.king_of(side) {
            let enemy_side = side.other();
            let snipers = (attacks::rook_attacks(king, Bitboard::EMPTY) & board.rooks_and_queens())
                | (attacks::bishop_attacks(king, Bitboard::EMPTY) & board.bishops_and_queens());

            let mut blockers = Bitboard::EMPTY;
            for sniper in snipers & board.by_color(enemy_side) {
                let b = attacks::between(king, sniper) & board.occupied();
                if !b.more_than_one() && !b.is_empty() {
                    blockers |= b;
                }
            }
            count = (blockers & board.by_color(side)).count();
        }
        count as f32
    };
    feats.insert("pinned_us".to_string(), count_pinned(turn));
    feats.insert("pinned_them".to_string(), count_pinned(opp));

    // 16. Threats: attacks on higher-value enemy pieces
    let count_threats = |side: Color| {
        let mut count = 0.0;
        let occupied = board.occupied();
        let them = side.other();
        for sq in board.by_color(them) {
            let victim = board.piece_at(sq).unwrap();
            if victim.role == Role::King { continue; }
            let attackers = board.attacks_to(sq, side, occupied);
            for a_sq in attackers {
                if let Some(attacker) = board.piece_at(a_sq) {
                    if piece_value(attacker.role) < piece_value(victim.role) {
                        count += 1.0;
                    }
                }
            }
        }
        count
    };
    feats.insert("threats_us".to_string(), count_threats(turn));
    feats.insert("threats_them".to_string(), count_threats(opp));

    // 17. Doubled pawns: files with 2+ own pawns
    let count_doubled = |side: Color| {
        let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
        let mut count = 0.0;
        for f in 0..8 {
            let file_bb = Bitboard::from_file(shakmaty::File::new(f as u32));
            let pawns_on_file = (my_pawns & file_bb).count();
            if pawns_on_file >= 2 {
                count += (pawns_on_file - 1) as f32;
            }
        }
        count
    };
    feats.insert("doubled_pawns_us".to_string(), count_doubled(turn));
    feats.insert("doubled_pawns_them".to_string(), count_doubled(opp));

    // 18. Space: squares controlled in the opponent's half
    let count_space = |side: Color| {
        let mut controlled = Bitboard::EMPTY;
        for sq in board.by_color(side) {
            controlled |= board.attacks_from(sq);
        }
        // Opponent's half: ranks 5-8 for White, ranks 1-4 for Black
        let opp_half = if side == Color::White {
            Bitboard::from_rank(shakmaty::Rank::Fifth)
                | Bitboard::from_rank(shakmaty::Rank::Sixth)
                | Bitboard::from_rank(shakmaty::Rank::Seventh)
                | Bitboard::from_rank(shakmaty::Rank::Eighth)
        } else {
            Bitboard::from_rank(shakmaty::Rank::First)
                | Bitboard::from_rank(shakmaty::Rank::Second)
                | Bitboard::from_rank(shakmaty::Rank::Third)
                | Bitboard::from_rank(shakmaty::Rank::Fourth)
        };
        (controlled & opp_half).count() as f32
    };
    feats.insert("space_us".to_string(), count_space(turn));
    feats.insert("space_them".to_string(), count_space(opp));

    // 19. King tropism: sum of (7 - distance) for attacking pieces to
    //     the opponent king.  Higher = pieces are closer to enemy king.
    let king_tropism = |side: Color| {
        let them = side.other();
        let enemy_king = board.king_of(them);
        if enemy_king.is_none() { return 0.0; }
        let ksq = enemy_king.unwrap();
        let mut tropism = 0.0;
        for sq in board.by_color(side) {
            if let Some(piece) = board.piece_at(sq) {
                if piece.role == Role::King || piece.role == Role::Pawn {
                    continue;
                }
                let dist = ksq.distance(sq) as f32;
                tropism += 7.0 - dist;
            }
        }
        tropism
    };
    feats.insert("king_tropism_us".to_string(), king_tropism(turn));
    feats.insert("king_tropism_them".to_string(), king_tropism(opp));

    // 20. Pawn chain: pawns defended by another pawn
    let pawn_chain = |side: Color| {
        let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
        let mut count = 0.0;
        for sq in my_pawns {
            // Check if any friendly pawn attacks this square
            let pawn_attackers = attacks::pawn_attacks(side.other(), sq)
                & my_pawns;
            if !pawn_attackers.is_empty() {
                count += 1.0;
            }
        }
        count
    };
    feats.insert("pawn_chain_us".to_string(), pawn_chain(turn));
    feats.insert("pawn_chain_them".to_string(), pawn_chain(opp));

    // 21. PST (Piece-Square Tables)
    const PST_PAWN: [i16; 64] = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ];
    const PST_KNIGHT: [i16; 64] = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ];
    const PST_BISHOP: [i16; 64] = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ];
    const PST_ROOK: [i16; 64] = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ];
    const PST_QUEEN: [i16; 64] = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ];
    const PST_KING_MG: [i16; 64] = [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ];
    const PST_KING_EG: [i16; 64] = [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ];

    let calc_pst = |side: Color| {
        let mut score = 0.0;
        let is_endgame = phase < 10;
        
        for sq in board.by_color(side) {
            let piece = board.piece_at(sq).unwrap();
            let table = match piece.role {
                Role::Pawn => &PST_PAWN,
                Role::Knight => &PST_KNIGHT,
                Role::Bishop => &PST_BISHOP,
                Role::Rook => &PST_ROOK,
                Role::Queen => &PST_QUEEN,
                Role::King => if is_endgame { &PST_KING_EG } else { &PST_KING_MG },
            };
            
            let vis_r = if side == Color::White { 7 - sq.rank() as usize } else { sq.rank() as usize };
            let vis_c = sq.file() as usize;
            score += table[vis_r * 8 + vis_c] as f32;
        }
        score / 100.0
    };

    feats.insert("pst_us".to_string(), calc_pst(turn));
    feats.insert("pst_them".to_string(), calc_pst(opp));

    Ok(feats)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _chess_ai_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_best_reply, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_forcing_swing, m)?)?;
    m.add_function(wrap_pyfunction!(extract_features_rust, m)?)?;
    m.add_class::<SyzygyTablebase>()?;
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Chess};

    /// Helper: parse a FEN into a `Chess` position, panicking on failure.
    fn pos_from_fen(fen: &str) -> Chess {
        let setup: Fen = fen.parse().unwrap();
        setup.into_position(CastlingMode::Standard).unwrap()
    }

    #[test]
    fn test_tt_store_and_retrieve() {
        let mut ctx = SearchContext::new();
        let pos =
            pos_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let hash = zobrist_hash(&pos);

        ctx.tt_store(hash, 5, 42, TTFlag::Exact, None);

        let entry = ctx.tt_probe(hash).unwrap();
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.score, 42);
        assert_eq!(entry.flag, TTFlag::Exact);
    }

    #[test]
    fn test_find_best_reply_mate_in_one() {
        // Scholar's mate setup: White plays Qxf7# (h5f7).
        let pos = pos_from_fen(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        );
        let result = find_best_reply_impl(&pos, 4);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "h5f7");
    }

    #[test]
    fn test_find_best_reply_obvious_capture() {
        // White queen captures undefended black queen on e5.
        let pos =
            pos_from_fen("rnb1kbnr/pppppppp/8/4q3/4Q3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1");
        let result = find_best_reply_impl(&pos, 4);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "e4e5");
    }

    #[test]
    fn test_quiesce_reasonable_score() {
        // Early game position: score should be near zero.
        let pos = pos_from_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        );
        let score = quiesce(&pos, -50000, 50000);
        assert!(
            score > -1000 && score < 1000,
            "Unexpected quiesce score: {score}"
        );
    }

    #[test]
    fn test_aspiration_returns_legal_move() {
        // Verify aspiration windows don't produce an illegal move.
        let pos = pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        );
        let result = find_best_reply_impl(&pos, 5);
        assert!(result.is_some());
        let legal: Vec<String> = pos
            .legal_moves()
            .into_iter()
            .map(|m| m.to_uci(CastlingMode::Standard).to_string())
            .collect();
        assert!(legal.contains(&result.unwrap()));
    }

    #[test]
    fn test_depth_returns_valid_moves() {
        // At every search depth 1..6 the engine must return a legal move.
        let pos = pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        );
        for d in 1..=6u8 {
            let result = find_best_reply_impl(&pos, d);
            assert!(result.is_some(), "No move at depth {d}");
            let legal: Vec<String> = pos
                .legal_moves()
                .into_iter()
                .map(|m| m.to_uci(CastlingMode::Standard).to_string())
                .collect();
            assert!(
                legal.contains(&result.clone().unwrap()),
                "Illegal move {} at depth {d}",
                result.unwrap()
            );
        }
    }
}
