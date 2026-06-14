//! 9×9 Go. Chinese rules, komi 7.5, positional ko (one-step lookback).
//!
//! The board is two **bitboards** (`black`, `white`) over the 81 points, bit
//! `i = r*9 + c`. This makes the rules hot path cheap: group flood-fill, liberty
//! and capture checks are `u128` shift/mask ops (no per-cell array scans or heap
//! allocation), and a position's identity (for the eval cache) is just the pair
//! of bitboards rather than a 162-cell hash. Outcome is decided on two passes;
//! scoring is Chinese area scoring (stones + surrounded empty points).

use playground_core::{Game, Outcome, Player, SearchOutcome, Solver};

pub const N: usize = 9;
pub const KOMI: f64 = 7.5;
const NN: usize = N * N; // 81 points
/// Low 81 bits set — the valid board area within a u128.
pub const BOARD_MASK: u128 = (1u128 << NN) - 1;

const fn col_mask(c: usize) -> u128 {
    let mut m = 0u128;
    let mut r = 0;
    while r < N {
        m |= 1u128 << (r * N + c);
        r += 1;
    }
    m
}
const COL0: u128 = col_mask(0); // leftmost column (c == 0)
const COL8: u128 = col_mask(N - 1); // rightmost column (c == 8)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Go9State {
    /// Bit `r*9+c` set = a black stone there.
    pub black: u128,
    pub white: u128,
    /// The (black, white) bitboards before the last move, for positional ko.
    pub prev: Option<(u128, u128)>,
    /// 1 = black's turn, 2 = white's turn.
    pub to_move: u8,
    pub consecutive_passes: u8,
    pub ply: u16,
}

impl Default for Go9State {
    fn default() -> Self {
        Self { black: 0, white: 0, prev: None, to_move: 1, consecutive_passes: 0, ply: 0 }
    }
}

impl Go9State {
    /// Stone at point `i` (0..81): 0 empty, 1 black, 2 white.
    #[inline]
    pub fn cell_at(&self, i: usize) -> u8 {
        if (self.black >> i) & 1 == 1 {
            1
        } else if (self.white >> i) & 1 == 1 {
            2
        } else {
            0
        }
    }
    /// Stones of the side to move / the opponent, as bitboards.
    #[inline]
    pub fn own_opp(&self) -> (u128, u128) {
        if self.to_move == 1 { (self.black, self.white) } else { (self.white, self.black) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Go9Move {
    Pass,
    Play { r: u8, c: u8 },
}

pub struct Go9;

// ── Bitboard rules primitives ────────────────────────────────────────────────

/// All orthogonal neighbours of the set `x` (column masks stop left/right wrap).
#[inline]
fn expand(x: u128) -> u128 {
    ((x >> N) | (x << N) | ((x & !COL0) >> 1) | ((x & !COL8) << 1)) & BOARD_MASK
}

/// The maximal connected group of `color` stones reachable from `seed` (a single
/// stone bit), by flood-fill to a fixpoint.
#[inline]
fn group_of(color: u128, seed: u128) -> u128 {
    let mut g = seed & color;
    loop {
        let next = (g | (expand(g) & color)) & BOARD_MASK;
        if next == g {
            return g;
        }
        g = next;
    }
}

/// Does `group` touch at least one empty point?
#[inline]
fn has_lib(group: u128, empty: u128) -> bool {
    expand(group) & empty != 0
}

/// Place a stone (bit `b`) of the mover (`mine`) and resolve opponent captures.
/// Returns the post-capture (mine, opp) bitboards. Does *not* check suicide/ko.
#[inline]
fn place_with_captures(mine: u128, opp: u128, b: u128) -> (u128, u128) {
    let mine2 = mine | b;
    let mut opp2 = opp;
    let mut adj = expand(b) & opp2; // opponent stones touching the new stone
    while adj != 0 {
        let seed = adj & adj.wrapping_neg(); // lowest set bit
        let g = group_of(opp2, seed);
        let empty = !(mine2 | opp2) & BOARD_MASK;
        if !has_lib(g, empty) {
            opp2 &= !g; // capture: group has no liberties
        }
        adj &= !g; // done with this group (captured or not)
    }
    (mine2, opp2)
}

// ── Game trait ───────────────────────────────────────────────────────────────

impl Game for Go9 {
    type State = Go9State;
    type Move = Go9Move;

    fn id() -> u32 {
        2
    }

    fn initial_state() -> Self::State {
        Go9State::default()
    }

    fn to_move(state: &Self::State) -> Player {
        if state.to_move == 1 { Player::First } else { Player::Second }
    }

    fn outcome(state: &Self::State) -> Outcome {
        if state.consecutive_passes < 2 {
            return Outcome::InProgress;
        }
        area_outcome(state)
    }

    fn legal_moves(state: &Self::State) -> Vec<Self::Move> {
        if Self::outcome(state).is_terminal() {
            return vec![];
        }
        let mut moves = vec![Go9Move::Pass];
        let mut empty = !(state.black | state.white) & BOARD_MASK;
        while empty != 0 {
            let i = empty.trailing_zeros() as usize;
            empty &= empty - 1;
            if apply_play_checked(state, i / N, i % N).is_some() {
                moves.push(Go9Move::Play { r: (i / N) as u8, c: (i % N) as u8 });
            }
        }
        moves
    }

    fn apply(state: &Self::State, mv: &Self::Move) -> Self::State {
        let mut next = *state;
        let prev = (state.black, state.white);
        match mv {
            Go9Move::Pass => {
                next.consecutive_passes += 1;
            }
            Go9Move::Play { r, c } => {
                let b = 1u128 << (*r as usize * N + *c as usize);
                let (mine, opp) = state.own_opp();
                let (mut mine2, opp2) = place_with_captures(mine, opp, b);
                // Suicide is excluded by legality, but guard anyway (matches the
                // old array implementation): drop the placed group if libertyless.
                let empty = !(mine2 | opp2) & BOARD_MASK;
                if !has_lib(group_of(mine2, b), empty) {
                    mine2 &= !group_of(mine2, b);
                }
                if state.to_move == 1 {
                    next.black = mine2;
                    next.white = opp2;
                } else {
                    next.white = mine2;
                    next.black = opp2;
                }
                next.consecutive_passes = 0;
            }
        }
        next.prev = Some(prev);
        next.to_move = 3 - state.to_move;
        next.ply += 1;
        next
    }

    fn ply(state: &Self::State) -> u32 {
        state.ply as u32
    }

    fn to_features(state: &Self::State) -> Vec<u8> {
        // Planes: [own stones, opp stones, to-move constant].
        let (own, opp) = state.own_opp();
        let mut out = Vec::with_capacity(3 * NN);
        for &bb in &[own, opp] {
            for i in 0..NN {
                out.push(((bb >> i) & 1) as u8);
            }
        }
        let side_plane: u8 = if state.to_move == 1 { 1 } else { 0 };
        out.resize(3 * NN, side_plane);
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        // Wire format preserved: 81 cell bytes, optional prev (81 bytes), to_move,
        // passes, ply — so existing decoders keep working.
        let mut out = Vec::with_capacity(2 * NN + 5);
        for i in 0..NN {
            out.push(state.cell_at(i));
        }
        if let Some((pb, pw)) = state.prev {
            out.push(1);
            for i in 0..NN {
                out.push(if (pb >> i) & 1 == 1 { 1 } else if (pw >> i) & 1 == 1 { 2 } else { 0 });
            }
        } else {
            out.push(0);
        }
        out.push(state.to_move);
        out.push(state.consecutive_passes);
        out.extend_from_slice(&state.ply.to_le_bytes());
        out
    }

    fn decode_state(bytes: &[u8]) -> anyhow::Result<Self::State> {
        if bytes.len() < NN + 4 {
            anyhow::bail!("go9 state: too short ({} bytes)", bytes.len());
        }
        let mut state = Go9State::default();
        let mut i = 0;
        for p in 0..NN {
            match bytes[i] {
                1 => state.black |= 1u128 << p,
                2 => state.white |= 1u128 << p,
                _ => {}
            }
            i += 1;
        }
        let has_prev = bytes[i];
        i += 1;
        if has_prev == 1 {
            if bytes.len() < i + NN + 4 {
                anyhow::bail!("go9 state: too short for prev board");
            }
            let (mut pb, mut pw) = (0u128, 0u128);
            for p in 0..NN {
                match bytes[i] {
                    1 => pb |= 1u128 << p,
                    2 => pw |= 1u128 << p,
                    _ => {}
                }
                i += 1;
            }
            state.prev = Some((pb, pw));
        }
        if bytes.len() < i + 4 {
            anyhow::bail!("go9 state: missing tail");
        }
        state.to_move = bytes[i];
        state.consecutive_passes = bytes[i + 1];
        state.ply = u16::from_le_bytes([bytes[i + 2], bytes[i + 3]]);
        Ok(state)
    }

    fn encode_move(mv: &Self::Move) -> Vec<u8> {
        match mv {
            Go9Move::Pass => vec![0],
            Go9Move::Play { r, c } => vec![1, *r, *c],
        }
    }

    fn decode_move(bytes: &[u8]) -> anyhow::Result<Self::Move> {
        match bytes {
            [0] => Ok(Go9Move::Pass),
            [1, r, c] => Ok(Go9Move::Play { r: *r, c: *c }),
            _ => anyhow::bail!("invalid go9 move bytes (len={})", bytes.len()),
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Apply a stone play with full legality (the lazy check the batched search uses):
/// place the stone + resolve captures, returning `None` if the move is suicide or
/// repeats the previous board (positional ko). `r,c` must be an empty point.
pub fn apply_play_checked(state: &Go9State, r: usize, c: usize) -> Option<Go9State> {
    let i = r * N + c;
    if ((state.black | state.white) >> i) & 1 == 1 {
        return None; // occupied
    }
    let b = 1u128 << i;
    let (mine, opp) = state.own_opp();
    let (mine2, opp2) = place_with_captures(mine, opp, b);
    let empty = !(mine2 | opp2) & BOARD_MASK;
    if !has_lib(group_of(mine2, b), empty) {
        return None; // suicide
    }
    let (nb, nw) = if state.to_move == 1 { (mine2, opp2) } else { (opp2, mine2) };
    if let Some((pb, pw)) = state.prev {
        if nb == pb && nw == pw {
            return None; // positional ko
        }
    }
    Some(Go9State {
        black: nb,
        white: nw,
        prev: Some((state.black, state.white)),
        to_move: 3 - state.to_move,
        consecutive_passes: 0,
        ply: state.ply + 1,
    })
}

/// Write the 3×9×9 self-play planes (own, opp, side-to-move) into `out` (len 243).
/// Single source of truth for both the in-process net and the search.
pub fn fill_planes_into(state: &Go9State, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 3 * NN);
    let (own, opp) = state.own_opp();
    let side: f32 = if state.to_move == 1 { 1.0 } else { 0.0 };
    for i in 0..NN {
        out[i] = ((own >> i) & 1) as f32;
        out[NN + i] = ((opp >> i) & 1) as f32;
        out[2 * NN + i] = side;
    }
}

/// Area-score outcome (Chinese rules + komi) for a position, ignoring the pass
/// count. White gets the komi. Used to end a game on two passes and to adjudicate
/// a game stopped at a move cap.
pub fn area_outcome(state: &Go9State) -> Outcome {
    let (black, white_raw) = score_area(state.black, state.white);
    let white = white_raw as f64 + KOMI;
    match (black as f64).partial_cmp(&white) {
        Some(std::cmp::Ordering::Greater) => Outcome::FirstWins,
        Some(std::cmp::Ordering::Less) => Outcome::SecondWins,
        _ => Outcome::Draw,
    }
}

/// Chinese area score `(black, white)`: stones plus empty regions bordered by a
/// single colour. Bitboard flood-fill of the empty regions.
fn score_area(black: u128, white: u128) -> (u32, u32) {
    let mut b = black.count_ones();
    let mut w = white.count_ones();
    let occupied = black | white;
    let mut remaining = !occupied & BOARD_MASK;
    while remaining != 0 {
        let seed = remaining & remaining.wrapping_neg();
        let region = group_of(!occupied & BOARD_MASK, seed);
        remaining &= !region;
        let border = expand(region) & occupied;
        let touches_black = border & black != 0;
        let touches_white = border & white != 0;
        if touches_black && !touches_white {
            b += region.count_ones();
        } else if touches_white && !touches_black {
            w += region.count_ones();
        } // else dame — counts for neither
    }
    (b, w)
}

// Placeholder built-in engine: plays the first legal move. No strong Go solver
// yet — this only exists so the engine can answer RequestMove.
impl Solver for Go9 {
    fn choose_move(state: &Self::State) -> SearchOutcome<Self::Move> {
        let best_move = Self::legal_moves(state)
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
    fn empty_board_has_82_moves() {
        let s = Go9::initial_state();
        assert_eq!(Go9::legal_moves(&s).len(), 82); // 81 plays + pass
    }

    #[test]
    fn two_passes_end_game() {
        let mut s = Go9::initial_state();
        s = Go9::apply(&s, &Go9Move::Pass);
        s = Go9::apply(&s, &Go9Move::Pass);
        assert!(Go9::outcome(&s).is_terminal());
    }

    #[test]
    fn capture_removes_stone() {
        // Surround (0,0) black stone with white.
        let mut s = Go9::initial_state();
        s = Go9::apply(&s, &Go9Move::Play { r: 0, c: 0 }); // black
        s = Go9::apply(&s, &Go9Move::Play { r: 0, c: 1 }); // white
        s = Go9::apply(&s, &Go9Move::Pass); // black passes
        s = Go9::apply(&s, &Go9Move::Play { r: 1, c: 0 }); // white captures
        assert_eq!(s.cell_at(0), 0, "captured stone should be empty");
    }

    #[test]
    fn suicide_is_illegal() {
        // White surrounds the single empty point (0,0); black playing there is
        // suicide (no capture, no liberty) → illegal.
        let mut s = Go9::initial_state();
        s.white = (1u128 << 1) | (1u128 << N); // (0,1) and (1,0)
        s.to_move = 1;
        assert!(apply_play_checked(&s, 0, 0).is_none());
    }

    #[test]
    fn move_roundtrip() {
        for mv in [Go9Move::Pass, Go9Move::Play { r: 4, c: 4 }] {
            let bytes = Go9::encode_move(&mv);
            assert_eq!(Go9::decode_move(&bytes).unwrap(), mv);
        }
    }

    #[test]
    fn state_roundtrip() {
        let mut s = Go9::initial_state();
        s = Go9::apply(&s, &Go9Move::Play { r: 2, c: 3 });
        s = Go9::apply(&s, &Go9Move::Play { r: 4, c: 4 });
        let bytes = Go9::encode_state(&s);
        let back = Go9::decode_state(&bytes).unwrap();
        assert_eq!(back.black, s.black);
        assert_eq!(back.white, s.white);
        assert_eq!(back.to_move, s.to_move);
    }
}
