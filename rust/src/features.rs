use pyo3::prelude::*;
use shakmaty::{
    attacks, Bitboard, CastlingMode, Chess, Color, Position, Role, Square,
};
use shakmaty::fen::Fen;
use std::collections::BTreeMap;

use crate::eval::{phase_factor, piece_value};
use crate::pawn_cache::{pawn_cache, pawn_zobrist, PawnCacheEntry, PAWN_CACHE_SIZE};
use crate::see::{least_valuable_attacker, see};

#[pyfunction]
pub fn extract_features_rust(fen: &str) -> PyResult<BTreeMap<String, f32>> {
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

    // NOTE: passed pawn values are overwritten later by the pawn
    // structure cache; we insert placeholders here to keep feature
    // ordering deterministic, then replace them below.
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
                semi_open += 1;
            } else if my_pawns.is_empty() && !opp_pawns.is_empty() {
                semi_open += 1;
            }
        }
        (open as f32, semi_open as f32)
    };

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
            Color::White => shakmaty::Rank::Seventh,
            Color::Black => shakmaty::Rank::Second,
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

        for i in 0..8 {
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

    // 11. Pawn structure features (cached by pawn Zobrist hash)
    let pawn_hash = pawn_zobrist(board);
    let pawn_key = pawn_hash ^ if turn == Color::White { 0 } else { 0xAAAA_AAAA_AAAA_AAAA };

    let pawn_feats = {
        let cache = pawn_cache().lock().unwrap();
        let idx = (pawn_key as usize) % PAWN_CACHE_SIZE;
        if cache[idx].key == pawn_key {
            Some(cache[idx].clone())
        } else {
            None
        }
    };

    let pawn_feats = pawn_feats.unwrap_or_else(|| {
        let count_isolated_fn = |side: Color| -> f32 {
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

        let count_doubled_fn = |side: Color| -> f32 {
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

        let count_backward_fn = |side: Color| -> f32 {
            let mut count = 0;
            let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
            let opp_side = side.other();
            let mut enemy_pawn_attacks = Bitboard::EMPTY;
            for sq in board.by_role(Role::Pawn) & board.by_color(opp_side) {
                enemy_pawn_attacks |= attacks::pawn_attacks(opp_side, sq);
            }
            for sq in my_pawns {
                let file: shakmaty::File = sq.file();
                let rank: shakmaty::Rank = sq.rank();
                let rank_usize = rank as usize;
                let mut is_supported = false;
                for f_off in [-1, 1] {
                    let f = file as i32 + f_off;
                    if f >= 0 && f < 8 {
                        let adj_file = shakmaty::File::new(f as u32);
                        let adj_bb = Bitboard::from_file(adj_file);
                        let adj_pawns = board.by_role(Role::Pawn) & board.by_color(side) & adj_bb;
                        for p_sq in adj_pawns {
                            let p_rank = p_sq.rank();
                            if (side == Color::White && (p_rank as usize) <= rank_usize)
                                || (side == Color::Black && (p_rank as usize) >= rank_usize) {
                                is_supported = true;
                                break;
                            }
                        }
                    }
                    if is_supported { break; }
                }
                if is_supported { continue; }
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

        let count_passed_fn = |side: Color| -> f32 {
            let mut count = 0;
            let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
            for sq in my_pawns {
                if is_passed(sq, side) {
                    count += 1;
                }
            }
            count as f32
        };

        let pawn_chain_fn = |side: Color| -> f32 {
            let my_pawns = board.by_role(Role::Pawn) & board.by_color(side);
            let mut count = 0.0;
            for sq in my_pawns {
                let pawn_attackers = attacks::pawn_attacks(side.other(), sq) & my_pawns;
                if !pawn_attackers.is_empty() {
                    count += 1.0;
                }
            }
            count
        };

        let entry = PawnCacheEntry {
            key: pawn_key,
            isolated_us: count_isolated_fn(turn),
            isolated_them: count_isolated_fn(opp),
            doubled_us: count_doubled_fn(turn),
            doubled_them: count_doubled_fn(opp),
            backward_us: count_backward_fn(turn),
            backward_them: count_backward_fn(opp),
            passed_us: count_passed_fn(turn),
            passed_them: count_passed_fn(opp),
            pawn_chain_us: pawn_chain_fn(turn),
            pawn_chain_them: pawn_chain_fn(opp),
        };

        if let Ok(mut cache) = pawn_cache().lock() {
            let idx = (pawn_key as usize) % PAWN_CACHE_SIZE;
            cache[idx] = entry.clone();
        }
        entry
    });

    feats.insert("isolated_pawns_us".to_string(), pawn_feats.isolated_us);
    feats.insert("isolated_pawns_them".to_string(), pawn_feats.isolated_them);

    // 12. Safe Mobility
    let get_safe_mobility = |pos_in: &Chess| {
        let side = pos_in.turn();
        let opp = side.other();
        let mut enemy_pawn_attacks = Bitboard::EMPTY;
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
    feats.insert("safe_mobility_them".to_string(), get_safe_mobility(&opp_pos));

    // 13. Rook on Open File
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
                count += 0.5;
            }
        }
        count
    };
    feats.insert("rook_open_file_us".to_string(), rook_on_open(turn));
    feats.insert("rook_open_file_them".to_string(), rook_on_open(opp));

    // 13b. Connected Rooks
    let connected_rooks = |side: Color| -> f32 {
        let rooks: Vec<Square> = (board.by_role(Role::Rook) & board.by_color(side)).into_iter().collect();
        if rooks.len() < 2 {
            return 0.0;
        }
        let (r0, r1) = (rooks[0], rooks[1]);
        if r0.rank() != r1.rank() {
            return 0.0;
        }
        let between = attacks::between(r0, r1) & board.occupied();
        if between.is_empty() { 1.0 } else { 0.0 }
    };
    feats.insert("connected_rooks_us".to_string(), connected_rooks(turn));
    feats.insert("connected_rooks_them".to_string(), connected_rooks(opp));

    // 14. Backward Pawns (from pawn cache)
    feats.insert("backward_pawns_us".to_string(), pawn_feats.backward_us);
    feats.insert("backward_pawns_them".to_string(), pawn_feats.backward_them);

    // Overwrite passed pawn placeholders with cached values
    feats.insert("passed_us".to_string(), pawn_feats.passed_us);
    feats.insert("passed_them".to_string(), pawn_feats.passed_them);

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

    // 16. Threats
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

    // 17. Doubled pawns (from pawn cache)
    feats.insert("doubled_pawns_us".to_string(), pawn_feats.doubled_us);
    feats.insert("doubled_pawns_them".to_string(), pawn_feats.doubled_them);

    // 18. Space
    let count_space = |side: Color| {
        let mut controlled = Bitboard::EMPTY;
        for sq in board.by_color(side) {
            controlled |= board.attacks_from(sq);
        }
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

    // 19. King tropism
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

    // 20. Pawn chain (from pawn cache)
    feats.insert("pawn_chain_us".to_string(), pawn_feats.pawn_chain_us);
    feats.insert("pawn_chain_them".to_string(), pawn_feats.pawn_chain_them);

    // 21. PST with continuous phase interpolation
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

    let pf = phase_factor(phase);
    let calc_pst = |side: Color| {
        let mut score = 0.0;

        for sq in board.by_color(side) {
            let piece = board.piece_at(sq).unwrap();
            let vis_r = if side == Color::White { 7 - sq.rank() as usize } else { sq.rank() as usize };
            let vis_c = sq.file() as usize;
            let idx = vis_r * 8 + vis_c;

            let pst_val = match piece.role {
                Role::Pawn => PST_PAWN[idx] as f32,
                Role::Knight => PST_KNIGHT[idx] as f32,
                Role::Bishop => PST_BISHOP[idx] as f32,
                Role::Rook => PST_ROOK[idx] as f32,
                Role::Queen => PST_QUEEN[idx] as f32,
                Role::King => {
                    let mg = PST_KING_MG[idx] as f32;
                    let eg = PST_KING_EG[idx] as f32;
                    pf * mg + (1.0 - pf) * eg
                }
            };
            score += pst_val;
        }
        score / 100.0
    };

    feats.insert("pst_us".to_string(), calc_pst(turn));
    feats.insert("pst_them".to_string(), calc_pst(opp));

    // 22. SEE features
    let calc_see_features = |side: Color| -> (f32, f32) {
        let them = side.other();
        let occupied = board.occupied();
        let mut advantage = 0.0f32;
        let mut vulnerability = 0.0f32;

        for target_sq in board.by_color(them) {
            let victim = match board.piece_at(target_sq) {
                Some(p) => p,
                None => continue,
            };
            if victim.role == Role::King { continue; }

            if let Some((attacker_sq, _role)) = least_valuable_attacker(board, target_sq, side, occupied) {
                let see_val = see(board, target_sq, attacker_sq);
                if see_val > 0 {
                    advantage += see_val as f32 / 100.0;
                }
            }
        }

        for our_sq in board.by_color(side) {
            let piece = match board.piece_at(our_sq) {
                Some(p) => p,
                None => continue,
            };
            if piece.role == Role::King { continue; }

            if let Some((attacker_sq, _role)) = least_valuable_attacker(board, our_sq, them, occupied) {
                let see_val = see(board, our_sq, attacker_sq);
                if see_val > 0 {
                    vulnerability += 1.0;
                }
            }
        }

        (advantage, vulnerability)
    };

    let (see_adv_us, see_vuln_us) = calc_see_features(turn);
    let (see_adv_them, see_vuln_them) = calc_see_features(opp);
    feats.insert("see_advantage_us".to_string(), see_adv_us);
    feats.insert("see_advantage_them".to_string(), see_adv_them);
    feats.insert("see_vulnerability_us".to_string(), see_vuln_us);
    feats.insert("see_vulnerability_them".to_string(), see_vuln_them);

    Ok(feats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_features_includes_see() {
        let feats = extract_features_rust(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        assert!(feats.contains_key("see_advantage_us"), "Missing see_advantage_us");
        assert!(feats.contains_key("see_advantage_them"), "Missing see_advantage_them");
        assert!(feats.contains_key("see_vulnerability_us"), "Missing see_vulnerability_us");
        assert!(feats.contains_key("see_vulnerability_them"), "Missing see_vulnerability_them");
    }

    #[test]
    fn test_extract_features_initial_position_no_see_advantage() {
        let feats = extract_features_rust(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ).unwrap();
        assert!(
            feats["see_advantage_us"].abs() < f32::EPSILON,
            "Initial pos should have 0 SEE advantage for us"
        );
        assert!(
            feats["see_advantage_them"].abs() < f32::EPSILON,
            "Initial pos should have 0 SEE advantage for them"
        );
    }
}
