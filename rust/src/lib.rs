use pyo3::prelude::*;
use shakmaty::{
    attacks, Bitboard, CastlingMode, Chess, Color, Move, Position, Role, Square,
};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{Tablebase, AmbiguousWdl, MaybeRounded};
use std::collections::BTreeMap;

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
    let mut root_moves: Vec<(i32, Move)> = moves.into_iter().map(|m: Move| {
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
            new_pos.play_unchecked(*m);
            
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
        let is_capture: bool = m.is_capture();
        let gives_check = {
            let mut test_pos = pos.clone();
            test_pos.play_unchecked(m);
            test_pos.is_check()
        };

        if is_capture || gives_check {
            let mut new_pos = pos.clone();
            new_pos.play_unchecked(m);
            
            // Evaluate position after forcing move with reduced depth
            let ev_after = alpha_beta(&new_pos, -50000, 50000, depth.saturating_sub(1));
            let score_for_us = -ev_after;
            
            let swing = (score_for_us - base_eval) as f32;
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

    // 16. PST (Piece-Square Tables)
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
