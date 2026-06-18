//! Game of the Amazons on a 10×10 board, **bitboard** implementation. No-move-loses.
//!
//! Each turn: pick one of your four queens, queen-move it to an empty square, then
//! shoot an arrow (also a queen-move) from the new position to any other empty
//! square (the queen's origin counts as empty for the arrow). Arrows are permanent.
//!
//! The board is three `u128` bitboards (`first`, `second`, `arrows`), bit
//! `r*10 + c`. Move generation and `apply` are pure bit ops (shift-mask rays, a few
//! XORs) so the CPU never bottlenecks the GPU during self-play at ~2000 moves/pos.
//! Mirrors go9's `u128` bitboard approach.

use playground_core::{Game, Outcome, Player, SearchOutcome, Solver};

pub const N: usize = 10;
pub const NN: usize = N * N; // 100 squares
pub const CELLS: usize = NN;
/// Input planes for the AZ net: my queens, opp queens, arrows, bias(=1).
pub const PLANES: usize = 4;

pub const BOARD_MASK: u128 = (1u128 << NN) - 1;

/// Column-0 and column-9 bit masks, to stop sideways/diagonal shifts wrapping rows.
const FILE_0: u128 = {
    let mut m = 0u128;
    let mut r = 0;
    while r < N {
        m |= 1u128 << (r * N);
        r += 1;
    }
    m
};
const FILE_LAST: u128 = FILE_0 << (N - 1);

#[inline]
fn idx(r: usize, c: usize) -> usize {
    r * N + c
}

/// Shift a bitboard one step in direction `d` (0..8), masking off-board wraps.
#[inline]
fn shift(b: u128, d: usize) -> u128 {
    match d {
        0 => (b << N) & BOARD_MASK,                 // N  (r+1)
        1 => b >> N,                                // S  (r-1)
        2 => ((b & !FILE_LAST) << 1) & BOARD_MASK,  // E  (c+1)
        3 => (b & !FILE_0) >> 1,                     // W  (c-1)
        4 => ((b & !FILE_LAST) << (N + 1)) & BOARD_MASK, // NE (r+1,c+1)
        5 => ((b & !FILE_0) << (N - 1)) & BOARD_MASK,    // NW (r+1,c-1)
        6 => (b & !FILE_LAST) >> (N - 1),            // SE (r-1,c+1)
        _ => (b & !FILE_0) >> (N + 1),               // SW (r-1,c-1)
    }
}

/// Empty squares reachable by a queen-slide from square `src` over `!occ`.
#[inline]
pub fn ray_reach(occ: u128, src: usize) -> u128 {
    let s = 1u128 << src;
    let mut reach = 0u128;
    for d in 0..8 {
        let mut b = s;
        loop {
            b = shift(b, d);
            if b == 0 || (b & occ) != 0 {
                break;
            }
            reach |= b;
        }
    }
    reach
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AmazonsState {
    pub first: u128,
    pub second: u128,
    pub arrows: u128,
    pub to_move: u8, // 1 or 2
    pub ply: u16,
}

/// Standard 10×10 opening: First (1) at (0,3)(0,6)(3,0)(3,9); Second (2) mirrored.
fn initial_state() -> AmazonsState {
    let mut first = 0u128;
    let mut second = 0u128;
    for (r, c) in [(0, 3), (0, 6), (3, 0), (3, 9)] {
        first |= 1u128 << idx(r, c);
    }
    for (r, c) in [(6, 0), (6, 9), (9, 3), (9, 6)] {
        second |= 1u128 << idx(r, c);
    }
    AmazonsState { first, second, arrows: 0, to_move: 1, ply: 0 }
}

impl Default for AmazonsState {
    fn default() -> Self {
        initial_state()
    }
}

impl AmazonsState {
    #[inline]
    pub fn occupied(&self) -> u128 {
        self.first | self.second | self.arrows
    }
    /// Bitboard of the side-to-move's queens.
    #[inline]
    pub fn mine(&self) -> u128 {
        if self.to_move == 1 {
            self.first
        } else {
            self.second
        }
    }
}

/// A full Amazons turn: queen moves from→to, then fires arrow to arrow square.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AmazonsMove {
    pub from_r: u8,
    pub from_c: u8,
    pub to_r: u8,
    pub to_c: u8,
    pub arrow_r: u8,
    pub arrow_c: u8,
}

pub struct Amazons;

/// Is there any legal move for the side to move? A queen with any reachable empty
/// can always shoot back along its path (the origin is empty for the arrow), so a
/// reachable target ⟺ a full legal move exists.
pub fn has_any_move(state: &AmazonsState) -> bool {
    let occ = state.occupied();
    let mut q = state.mine();
    while q != 0 {
        let from = q.trailing_zeros() as usize;
        q &= q - 1;
        if ray_reach(occ, from) != 0 {
            return true;
        }
    }
    false
}

/// All legal moves (empty iff terminal). Bitboard enumeration. The from/to
/// coordinates are hoisted out of the hot per-arrow loop (computed per queen / per
/// target, not per move), and the output is pre-reserved to avoid regrowth.
pub fn legal_moves(state: &AmazonsState) -> Vec<AmazonsMove> {
    let occ = state.occupied();
    let mut out = Vec::with_capacity(1024);
    let mut q = state.mine();
    while q != 0 {
        let from = q.trailing_zeros() as usize;
        q &= q - 1;
        let from_r = (from / N) as u8;
        let from_c = (from % N) as u8;
        let occ_wo_from = occ & !(1u128 << from);
        let mut targets = ray_reach(occ, from);
        while targets != 0 {
            let to = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            let to_r = (to / N) as u8;
            let to_c = (to % N) as u8;
            // After the queen moves, its origin is empty and `to` is occupied.
            let mut shots = ray_reach(occ_wo_from | (1u128 << to), to);
            while shots != 0 {
                let ar = shots.trailing_zeros() as usize;
                shots &= shots - 1;
                out.push(AmazonsMove {
                    from_r,
                    from_c,
                    to_r,
                    to_c,
                    arrow_r: (ar / N) as u8,
                    arrow_c: (ar % N) as u8,
                });
            }
        }
    }
    out
}

/// Apply a (legal) move: move the queen, drop the arrow, flip side. Pure bit ops.
pub fn apply(state: &AmazonsState, mv: &AmazonsMove) -> AmazonsState {
    let from = 1u128 << idx(mv.from_r as usize, mv.from_c as usize);
    let to = 1u128 << idx(mv.to_r as usize, mv.to_c as usize);
    let arrow = 1u128 << idx(mv.arrow_r as usize, mv.arrow_c as usize);
    let mut next = state.clone();
    if state.to_move == 1 {
        next.first = (next.first & !from) | to;
    } else {
        next.second = (next.second & !from) | to;
    }
    next.arrows |= arrow;
    next.to_move = 3 - state.to_move;
    next.ply += 1;
    next
}

/// Fill the 4 canonical AZ planes (my queens, opp queens, arrows, bias) for `state`
/// into `out` (length `PLANES*CELLS`), row-major per plane. Single source of truth
/// shared by the candle net, the ORT path, and `MoveGame::encode_planes`.
pub fn fill_planes_into(state: &AmazonsState, out: &mut [f32]) {
    let (mine, opp) = if state.to_move == 1 {
        (state.first, state.second)
    } else {
        (state.second, state.first)
    };
    for (plane, bits) in [mine, opp, state.arrows].into_iter().enumerate() {
        let base = plane * CELLS;
        for sq in 0..NN {
            out[base + sq] = if (bits >> sq) & 1 == 1 { 1.0 } else { 0.0 };
        }
    }
    for x in &mut out[3 * CELLS..4 * CELLS] {
        *x = 1.0; // bias plane
    }
}

/// Position identity hash (boards + side); excludes ply. (For future eval cache.)
#[allow(dead_code)]
pub fn hash_key(state: &AmazonsState) -> u128 {
    let mut h = state.first.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31)
        ^ state.second.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= state.arrows.rotate_left(17).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= state.to_move as u128;
    h
}

impl Game for Amazons {
    type State = AmazonsState;
    type Move = AmazonsMove;

    fn id() -> u32 {
        3
    }

    fn initial_state() -> Self::State {
        AmazonsState::default()
    }

    fn to_move(state: &Self::State) -> Player {
        if state.to_move == 1 {
            Player::First
        } else {
            Player::Second
        }
    }

    fn outcome(state: &Self::State) -> Outcome {
        if has_any_move(state) {
            Outcome::InProgress
        } else if state.to_move == 1 {
            Outcome::SecondWins
        } else {
            Outcome::FirstWins
        }
    }

    fn legal_moves(state: &Self::State) -> Vec<Self::Move> {
        legal_moves(state)
    }

    fn apply(state: &Self::State, mv: &AmazonsMove) -> Self::State {
        apply(state, mv)
    }

    fn ply(state: &Self::State) -> u32 {
        state.ply as u32
    }

    fn to_features(state: &Self::State) -> Vec<u8> {
        // Absolute first|second|arrows|side planes (generic/serving path).
        let mut out = Vec::with_capacity(4 * NN);
        for bits in [state.first, state.second, state.arrows] {
            for sq in 0..NN {
                out.push(((bits >> sq) & 1) as u8);
            }
        }
        let side: u8 = if state.to_move == 1 { 1 } else { 0 };
        for _ in 0..NN {
            out.push(side);
        }
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 * 3 + 3);
        out.extend_from_slice(&state.first.to_le_bytes());
        out.extend_from_slice(&state.second.to_le_bytes());
        out.extend_from_slice(&state.arrows.to_le_bytes());
        out.push(state.to_move);
        out.extend_from_slice(&state.ply.to_le_bytes());
        out
    }

    fn decode_state(bytes: &[u8]) -> anyhow::Result<Self::State> {
        let expected = 16 * 3 + 3;
        if bytes.len() != expected {
            anyhow::bail!("amazons state: expected {expected} bytes, got {}", bytes.len());
        }
        let rd = |o: usize| {
            let mut b = [0u8; 16];
            b.copy_from_slice(&bytes[o..o + 16]);
            u128::from_le_bytes(b)
        };
        Ok(AmazonsState {
            first: rd(0),
            second: rd(16),
            arrows: rd(32),
            to_move: bytes[48],
            ply: u16::from_le_bytes([bytes[49], bytes[50]]),
        })
    }

    fn encode_move(mv: &Self::Move) -> Vec<u8> {
        vec![mv.from_r, mv.from_c, mv.to_r, mv.to_c, mv.arrow_r, mv.arrow_c]
    }

    fn decode_move(bytes: &[u8]) -> anyhow::Result<Self::Move> {
        if bytes.len() != 6 {
            anyhow::bail!("amazons move: expected 6 bytes, got {}", bytes.len());
        }
        Ok(AmazonsMove {
            from_r: bytes[0],
            from_c: bytes[1],
            to_r: bytes[2],
            to_c: bytes[3],
            arrow_r: bytes[4],
            arrow_c: bytes[5],
        })
    }
}

// Placeholder built-in engine: first legal move. No strong Amazons solver yet.
impl Solver for Amazons {
    fn choose_move(state: &Self::State) -> SearchOutcome<Self::Move> {
        let best_move = legal_moves(state)
            .into_iter()
            .next()
            .expect("non-terminal position has a legal move");
        SearchOutcome { best_move, value: 0.0, nodes: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_board_in_progress() {
        let s = Amazons::initial_state();
        assert_eq!(Amazons::outcome(&s), Outcome::InProgress);
        assert_eq!(s.first.count_ones(), 4);
        assert_eq!(s.second.count_ones(), 4);
    }

    #[test]
    fn initial_moves_nonempty() {
        let s = Amazons::initial_state();
        // Each of 4 queens has open lines on the empty opening board.
        assert!(!legal_moves(&s).is_empty());
    }

    #[test]
    fn initial_move_count_is_2176() {
        // The standard 10×10 Amazons opening has exactly 2176 legal moves — a strong
        // end-to-end check on the bitboard ray generation.
        let s = Amazons::initial_state();
        assert_eq!(legal_moves(&s).len(), 2176);
    }

    #[test]
    fn apply_move_places_pieces() {
        let s = Amazons::initial_state();
        let mv = legal_moves(&s)[0];
        let s2 = Amazons::apply(&s, &mv);
        assert_eq!(s2.to_move, 2);
        let from = 1u128 << idx(mv.from_r as usize, mv.from_c as usize);
        let to = 1u128 << idx(mv.to_r as usize, mv.to_c as usize);
        let arrow = 1u128 << idx(mv.arrow_r as usize, mv.arrow_c as usize);
        assert_eq!(s2.first & from, 0); // vacated
        assert_ne!(s2.first & to, 0); // queen moved here
        assert_ne!(s2.arrows & arrow, 0); // arrow dropped
    }

    #[test]
    fn no_wrap_on_edges() {
        // A queen at column 0 must not slide "west" into column 9 of the row below.
        let mut s = AmazonsState { first: 0, second: 0, arrows: 0, to_move: 1, ply: 0 };
        s.first = 1u128 << idx(5, 0);
        s.second = 1u128 << idx(0, 0); // keep it non-terminal-ish
        let reach = ray_reach(s.occupied(), idx(5, 0));
        // Reachable west set must be empty (already at col 0); check no col-9 bits in row 4.
        assert_eq!(reach & (1u128 << idx(4, 9)), 0);
    }

    #[test]
    fn state_roundtrip() {
        let s = Amazons::initial_state();
        let mv = legal_moves(&s)[0];
        let s2 = Amazons::apply(&s, &mv);
        let bytes = Amazons::encode_state(&s2);
        let s3 = Amazons::decode_state(&bytes).unwrap();
        assert_eq!(s2, s3);
    }
}
