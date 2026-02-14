use shakmaty::Role;
use std::sync::{Mutex, OnceLock};

use crate::zobrist::{piece_index, zobrist_keys};

/// Compute a Zobrist hash of only the pawn positions so that pawn
/// structure features can be cached and reused when only pieces move.
pub fn pawn_zobrist(board: &shakmaty::Board) -> u64 {
    let keys = zobrist_keys();
    let mut hash = 0u64;
    let pawns = board.by_role(Role::Pawn);
    for sq in pawns {
        if let Some(piece) = board.piece_at(sq) {
            let pi = piece_index(piece.color, piece.role);
            hash ^= keys.pieces[sq as usize][pi];
        }
    }
    hash
}

/// Cached pawn structure features for one side: isolated, doubled,
/// backward, passed, and pawn chain counts.  Keyed by a combined
/// hash of both colours' pawn placements plus the perspective side.
#[derive(Clone)]
pub struct PawnCacheEntry {
    pub key: u64,
    pub isolated_us: f32,
    pub isolated_them: f32,
    pub doubled_us: f32,
    pub doubled_them: f32,
    pub backward_us: f32,
    pub backward_them: f32,
    pub passed_us: f32,
    pub passed_them: f32,
    pub pawn_chain_us: f32,
    pub pawn_chain_them: f32,
}

pub const PAWN_CACHE_SIZE: usize = 1 << 16; // 65536 entries

/// Global pawn structure cache shared across feature extraction calls.
/// The cache is protected by a `Mutex` for thread-safety.
pub fn pawn_cache() -> &'static Mutex<Vec<PawnCacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<PawnCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let entry = PawnCacheEntry {
            key: 0,
            isolated_us: 0.0,
            isolated_them: 0.0,
            doubled_us: 0.0,
            doubled_them: 0.0,
            backward_us: 0.0,
            backward_them: 0.0,
            passed_us: 0.0,
            passed_them: 0.0,
            pawn_chain_us: 0.0,
            pawn_chain_them: 0.0,
        };
        Mutex::new(vec![entry; PAWN_CACHE_SIZE])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Chess, Position};

    fn pos_from_fen(fen: &str) -> Chess {
        let setup: Fen = fen.parse().unwrap();
        setup.into_position(CastlingMode::Standard).unwrap()
    }

    #[test]
    fn test_pawn_hash_same_position() {
        let pos1 = pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        );
        let pos2 = pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        );
        assert_eq!(pawn_zobrist(pos1.board()), pawn_zobrist(pos2.board()));
    }

    #[test]
    fn test_pawn_hash_different_on_pawn_move() {
        let pos1 = pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        );
        let pos2 = pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        );
        assert_ne!(pawn_zobrist(pos1.board()), pawn_zobrist(pos2.board()));
    }

    #[test]
    fn test_pawn_hash_same_after_piece_move() {
        let pos1 = pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        );
        let pos2 = pos_from_fen(
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        );
        assert_eq!(pawn_zobrist(pos1.board()), pawn_zobrist(pos2.board()));
    }
}
