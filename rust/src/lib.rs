use pyo3::prelude::*;
use shakmaty::{
    fen::Fen, CastlingMode, Chess, Color, Move, Position, Role, Square,
};
use shakmaty_syzygy::{Tablebase, Wdl, Dtz, Syzygy, AmbiguousWdl, MaybeRounded};
use std::path::Path;
use std::str::FromStr; // Needed for parse()

// Simple material values
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

fn evaluate(pos: &Chess) -> i32 {
    let board = pos.board();
    let mut score = 0;
    
    // Iterate over all squares to avoid iterator issues with &Board
    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let val = piece_value(piece.role);
            if piece.color == Color::White {
                score += val;
            } else {
                score -= val;
            }
        }
    }
    
    if pos.turn() == Color::White {
        score
    } else {
        -score
    }
}

// Simple alpha-beta search
// Simple alpha-beta search
fn alpha_beta(pos: &Chess, mut alpha: i32, beta: i32, depth: u8) -> i32 {
    if depth == 0 || pos.is_game_over() {
        return evaluate(pos);
    }

    let moves = pos.legal_moves();
    if moves.is_empty() {
        if pos.is_check() {
            return -30000; // Checkmate
        } else {
            return 0; // Stalemate
        }
    }

    // Move ordering: MVV-LVA for captures, Promotions high priority
    let mut move_scores: Vec<(i32, Move)> = moves.into_iter().map(|m| {
        let mut score = 0;
        if m.is_capture() {
            let board = pos.board();
            let victim = board.piece_at(m.to()).map(|p| p.role).unwrap_or(Role::Pawn);
            let attacker = board.piece_at(m.from().unwrap()).map(|p| p.role).unwrap_or(Role::Pawn);
            score = 10000 + piece_value(victim) - piece_value(attacker);
        }
        if m.is_promotion() {
            score += 20000;
        }
        (score, m)
    }).collect();

    // Sort descending by score
    move_scores.sort_by(|a, b| b.0.cmp(&a.0));

    for (_, m) in move_scores {
        let mut new_pos = pos.clone();
        new_pos.play_unchecked(m.clone());
        
        let score = -alpha_beta(&new_pos, -beta, -alpha, depth - 1);
        
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }
    alpha
}

#[pyfunction]
fn find_best_reply(fen: &str, depth: u8) -> PyResult<Option<String>> {
    let setup: Fen = fen.parse().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
    let pos: Chess = setup.into_position(CastlingMode::Standard).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;

    let moves = pos.legal_moves();
    if moves.is_empty() {
        return Ok(None);
    }

    // Initial move ordering (captures/promotions)
    let mut root_moves: Vec<(i32, Move)> = moves.into_iter().map(|m| {
        let mut score = 0;
        if m.is_capture() {
            let board = pos.board();
            let victim = board.piece_at(m.to()).map(|p| p.role).unwrap_or(Role::Pawn);
            let attacker = board.piece_at(m.from().unwrap()).map(|p| p.role).unwrap_or(Role::Pawn);
            score = 10000 + piece_value(victim) - piece_value(attacker);
        }
        if m.is_promotion() {
            score += 20000;
        }
        (score, m)
    }).collect();

    // Sort descending
    root_moves.sort_by(|a, b| b.0.cmp(&a.0));

    let mut best_move: Option<String> = None;
    
    // Iterative Deepening
    // For depth 1 to requested_depth
    for d in 1..=depth {
        let mut alpha = -50000;
        let beta = 50000;
        let mut best_score_at_depth = -50000;
        
        // We will re-sort root_moves based on scores from this iteration
        // to improve ordering for the next hydration.
        // Actually, preventing re-allocation is better, but for simplicity we'll just update scores.
        
        for (score_ref, m) in root_moves.iter_mut() {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(m.clone());
            
            let score = -alpha_beta(&new_pos, -beta, -alpha, d - 1);
            
            // Update the score associated with this move (for sorting next iteration)
            *score_ref = score;
            
            if score > best_score_at_depth {
                best_score_at_depth = score;
                if d == depth {
                    best_move = Some(m.to_uci(CastlingMode::Standard).to_string());
                }
            }
            if score > alpha {
                alpha = score;
            }
        }
        
        // Sort for next iteration (best moves first)
        root_moves.sort_by(|a, b| b.0.cmp(&a.0));
    }
    
    // If we didn't complete the loop (unlikely with this logic unless depth=0), best_move might be None if depth=0.
    // But legal_moves is not empty.
    // If depth passed is 0, we return None or just any move? 
    // The loop 1..=0 is empty.
    // Let's handle depth 0 or just ensure we return the first move if loop doesn't run.
    if best_move.is_none() && !root_moves.is_empty() {
        best_move = Some(root_moves[0].1.to_uci(CastlingMode::Standard).to_string());
    }

    Ok(best_move)
}

#[pyfunction]
fn calculate_forcing_swing(fen: &str, depth: u8) -> PyResult<f32> {
    let setup: Fen = fen.parse().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
    let pos: Chess = setup.into_position(CastlingMode::Standard).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;

    // Base evaluation at current depth
    let base_eval = alpha_beta(&pos, -50000, 50000, depth);

    // Generate forcing moves (captures and checks)
    let moves = pos.legal_moves();
    let mut max_swing = 0.0;

    for m in moves {
        // Is it forcing?
        let is_capture = m.is_capture();
        let gives_check = {
            let mut test_pos = pos.clone();
            test_pos.play_unchecked(m.clone());
            test_pos.is_check()
        };

        if is_capture || gives_check {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(m.clone());
            
            // Evaluate position after forcing move with reduced depth
            // We search to depth - 1
            // Note: alpha_beta returns score from side-to-move perspective.
            // pos.turn() is us. new_pos.turn() is them.
            // ev_after is from their perspective. 
            // So score for us is -ev_after.
            
            let ev_after = alpha_beta(&new_pos, -50000, 50000, depth.saturating_sub(1));
            let score_for_us = -ev_after;
            
            // Swing = New Score - Base Score
            let swing = (score_for_us - base_eval) as f32;
            
            // Convert to centipawns (approx)
            // our eval is 100 per pawn. Stockfish is 100 per pawn.
            // So we can return it directly.
            
            if swing > max_swing {
                max_swing = swing;
            }
        }
    }

    Ok(max_swing)
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
            Err(_) => Ok(None), // Position not in tablebase
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
                // Dtz(i32) usually? Or u32?
                // Let's assume it implements Into<i32> or has .0
                // I'll try .0 assuming it's a tuple struct.
                // If not, I'll error and fix.
                Ok(Some(dtz.0 as i32))
            },
            Err(_) => Ok(None),
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _chess_ai_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_best_reply, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_forcing_swing, m)?)?;
    m.add_class::<SyzygyTablebase>()?;
    Ok(())
}
