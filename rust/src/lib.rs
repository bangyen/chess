use pyo3::prelude::*;
use shakmaty::{
    fen::Fen, CastlingMode, Chess, Color, Move, Position, Role, Square,
    uci::Uci
};
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

    // Move ordering would go here, but omitted for simplicity
    for m in moves {
        let mut new_pos = pos.clone();
        new_pos.play_unchecked(&m);
        
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

    let mut best_move: Option<String> = None;
    let mut best_score = -50000;
    let mut alpha = -50000;
    let beta = 50000;

    for m in moves {
        let mut new_pos = pos.clone();
        new_pos.play_unchecked(&m);
        
        let score = -alpha_beta(&new_pos, -beta, -alpha, depth - 1);
        
        if score > best_score {
            best_score = score;
            // Convert to UCI string
            best_move = Some(Uci::from_move(&m, CastlingMode::Standard).to_string());
        }
        if score > alpha {
            alpha = score;
        }
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
            test_pos.play_unchecked(&m);
            test_pos.is_check()
        };

        if is_capture || gives_check {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(&m);
            
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

/// A Python module implemented in Rust.
#[pymodule]
fn rust_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_best_reply, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_forcing_swing, m)?)?;
    Ok(())
}
