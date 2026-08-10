//! Hex on a 13×13 rhombus. **First = Black connects top↔bottom, Second = White
//! connects left↔right** — benzene/MoHex's convention (`VERTICAL_COLOR = BLACK`),
//! so our board and a MoHex subprocess agree without any re-mapping.
//!
//! Hex has no draws: when the board fills, exactly one player owns a crossing
//! chain. It also has no captures — a placed stone never moves — which makes the
//! rules trivial and the *connectivity test* the only real work. We keep two
//! 169-bit bitboards (see [`crate::bitboard`]) and, on each `apply`, flood-fill
//! the mover's stones from their home edge; the result is cached in
//! `HexState::winner` so `outcome()` (hit on every MCTS node) is O(1).
//!
//! Cell `i = r*13 + c`; neighbours on the hex grid are the six offsets
//! `(r-1,c) (r-1,c+1) (r,c-1) (r,c+1) (r+1,c-1) (r+1,c)` — i.e. the square-grid
//! 4-neighbourhood plus the NE/SW diagonal. Standard notation is a column
//! letter followed by a row number, `a1`..`m13` (letters run `a`..`m`, with no
//! skipped `i`, as benzene does it).
//!
//! **Swap (pie) rule**: off by default, see [`ALLOW_SWAP`].

use playground_core::{Game, Outcome, Player, SearchOutcome, Solver};

use crate::bitboard::{Bb, BOARD_MASK};

pub const N: usize = 13;
pub const NN: usize = N * N; // 169 cells
/// Input planes for the AZ net: own stones, opponent stones, side-to-move.
pub const PLANES: usize = 3;

/// Whether the swap (pie) rule is in force. When on, Second's *only* extra
/// option is [`HexMove::Swap`] as the reply to First's opening move: First's
/// stone at `(r,c)` becomes a Second stone at `(c,r)` — the reflection across
/// the long diagonal that maps a vertical connection to a horizontal one, which
/// is exactly benzene's `swap-pieces`.
///
/// Off by default: it keeps the action space at 169 (no swap logit) and keeps
/// self-play positions in one frame of reference. It costs balance — Hex is a
/// first-player win, and without swap self-play skews toward Black — so a
/// training run compensates with randomised openings instead. Flip this to
/// `true` for standard-rules play against MoHex.
pub const ALLOW_SWAP: bool = false;

const fn row_mask(r: usize) -> Bb {
    let mut w = [0u64; 3];
    let mut c = 0;
    while c < N {
        let i = r * N + c;
        w[i / 64] |= 1u64 << (i % 64);
        c += 1;
    }
    Bb(w)
}

const fn col_mask(c: usize) -> Bb {
    let mut w = [0u64; 3];
    let mut r = 0;
    while r < N {
        let i = r * N + c;
        w[i / 64] |= 1u64 << (i % 64);
        r += 1;
    }
    Bb(w)
}

/// Black's home edges (top and bottom rows).
pub const TOP: Bb = row_mask(0);
pub const BOTTOM: Bb = row_mask(N - 1);
/// White's home edges (left and right columns).
pub const LEFT: Bb = col_mask(0);
pub const RIGHT: Bb = col_mask(N - 1);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HexState {
    /// First player's stones (Black, connects top↔bottom).
    pub black: Bb,
    /// Second player's stones (White, connects left↔right).
    pub white: Bb,
    /// 1 = Black to move, 2 = White to move.
    pub to_move: u8,
    /// 0 = game in progress, 1 = Black has won, 2 = White has won. Cached on
    /// `apply` so `outcome()` costs nothing.
    pub winner: u8,
    pub ply: u16,
}

impl Default for HexState {
    fn default() -> Self {
        Self { black: Bb::ZERO, white: Bb::ZERO, to_move: 1, winner: 0, ply: 0 }
    }
}

impl HexState {
    /// Stone at cell `i` (0..169): 0 empty, 1 black, 2 white.
    #[inline]
    pub fn cell_at(&self, i: usize) -> u8 {
        if self.black.get(i) {
            1
        } else if self.white.get(i) {
            2
        } else {
            0
        }
    }

    #[inline]
    pub fn occupied(&self) -> Bb {
        self.black.or(self.white)
    }

    #[inline]
    pub fn empty(&self) -> Bb {
        self.occupied().complement()
    }

    /// Stones of the side to move, then the opponent's.
    #[inline]
    pub fn own_opp(&self) -> (Bb, Bb) {
        if self.to_move == 1 { (self.black, self.white) } else { (self.white, self.black) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum HexMove {
    Play { r: u8, c: u8 },
    /// The pie-rule reply (see [`ALLOW_SWAP`]).
    Swap,
}

pub struct Hex;

// ── Bitboard primitives ──────────────────────────────────────────────────────

/// The six hex neighbours of every cell in `x`. Column masks stop the ±1 and the
/// diagonal shifts from wrapping around a row edge.
#[inline]
pub fn expand(x: Bb) -> Bb {
    let no_left = x.andnot(LEFT); // safe to step to c-1
    let no_right = x.andnot(RIGHT); // safe to step to c+1
    x.shift_down(N as u32) // (r-1, c  )
        .or(no_right.shift_down((N - 1) as u32)) // (r-1, c+1)
        .or(no_left.shift_down(1)) // (r  , c-1)
        .or(no_right.shift_up(1)) // (r  , c+1)
        .or(no_left.shift_up((N - 1) as u32)) // (r+1, c-1)
        .or(x.shift_up(N as u32)) // (r+1, c  )
        .and(BOARD_MASK)
}

/// Does `stones` contain a chain joining `from` to `to`? Flood-fill from the
/// `from` edge to a fixpoint (≤ 169 steps, typically ~N).
#[inline]
pub fn connects(stones: Bb, from: Bb, to: Bb) -> bool {
    let mut reach = stones.and(from);
    if reach.is_empty() {
        return false;
    }
    loop {
        if reach.and(to).any() {
            return true;
        }
        let next = reach.or(expand(reach).and(stones));
        if next == reach {
            return false;
        }
        reach = next;
    }
}

/// Winner of a position from scratch (1 black, 2 white, 0 neither) — the
/// authority `apply` caches into [`HexState::winner`].
pub fn compute_winner(state: &HexState) -> u8 {
    if connects(state.black, TOP, BOTTOM) {
        1
    } else if connects(state.white, LEFT, RIGHT) {
        2
    } else {
        0
    }
}

// ── Notation ─────────────────────────────────────────────────────────────────

/// `(r, c)` → `"a1"`-style cell name (column letter, then 1-based row).
pub fn cell_name(r: usize, c: usize) -> String {
    format!("{}{}", (b'a' + c as u8) as char, r + 1)
}

/// Parse `"a1"`..`"m13"` into `(r, c)`. Case-insensitive.
pub fn parse_cell(s: &str) -> Option<(usize, usize)> {
    let s = s.trim();
    let mut chars = s.chars();
    let c = (chars.next()?.to_ascii_lowercase() as u8).checked_sub(b'a')? as usize;
    let r = chars.as_str().parse::<usize>().ok()?.checked_sub(1)?;
    (r < N && c < N).then_some((r, c))
}

/// The board as a staggered ASCII rhombus (black `X` top↔bottom, white `O`
/// left↔right). Debug aid for the MoHex bridge and tests.
pub fn render(state: &HexState) -> String {
    let mut out = String::new();
    out.push_str("  ");
    for c in 0..N {
        out.push((b'a' + c as u8) as char);
        out.push(' ');
    }
    out.push('\n');
    for r in 0..N {
        out.push_str(&" ".repeat(r));
        out.push_str(&format!("{:>2} ", r + 1));
        for c in 0..N {
            out.push(match state.cell_at(r * N + c) {
                1 => 'X',
                2 => 'O',
                _ => '.',
            });
            out.push(' ');
        }
        out.push_str(&format!("{}\n", r + 1));
    }
    out
}

// ── Symmetries ───────────────────────────────────────────────────────────────
//
// Hex has exactly two useful symmetries, both worth having before training:
//   * 180° rotation — same position, same colours (a plain data augmentation);
//   * transpose + colour swap — reflection across the long diagonal, which maps
//     a Black-to-move position to the mirrored White-to-move one. Canonicalising
//     with it lets a net always see "I am the vertical player".

/// Rotate a cell index by 180°.
#[inline]
pub fn rot180_index(i: usize) -> usize {
    NN - 1 - i
}

/// Transpose a cell index across the long diagonal (`(r,c) → (c,r)`).
#[inline]
pub fn transpose_index(i: usize) -> usize {
    (i % N) * N + i / N
}

fn map_bits(b: Bb, f: impl Fn(usize) -> usize) -> Bb {
    let mut out = Bb::ZERO;
    for i in b.iter() {
        out.set(f(i));
    }
    out
}

/// 180° rotation: an equivalent position for the same player.
pub fn rot180(state: &HexState) -> HexState {
    HexState {
        black: map_bits(state.black, rot180_index),
        white: map_bits(state.white, rot180_index),
        ..*state
    }
}

/// Transpose + colour swap: the same game with the two players' roles exchanged.
pub fn transpose_swap(state: &HexState) -> HexState {
    HexState {
        black: map_bits(state.white, transpose_index),
        white: map_bits(state.black, transpose_index),
        to_move: 3 - state.to_move,
        winner: match state.winner {
            0 => 0,
            w => 3 - w,
        },
        ply: state.ply,
    }
}

/// The position seen from the side to move: always Black-to-move, always
/// connecting top↔bottom. Apply [`transpose_index`] to map a move chosen in this
/// frame back to the real board (only needed when the original mover was White).
pub fn canonical(state: &HexState) -> HexState {
    if state.to_move == 1 { *state } else { transpose_swap(state) }
}

/// Write the `PLANES`×13×13 net input (own stones, opponent stones, side-to-move)
/// into `out` (len 507). Single source of truth for self-play and training.
pub fn fill_planes_into(state: &HexState, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PLANES * NN);
    let (own, opp) = state.own_opp();
    let side: f32 = if state.to_move == 1 { 1.0 } else { 0.0 };
    for i in 0..NN {
        out[i] = own.get(i) as u8 as f32;
        out[NN + i] = opp.get(i) as u8 as f32;
        out[2 * NN + i] = side;
    }
}

// ── Game trait ───────────────────────────────────────────────────────────────

impl Game for Hex {
    type State = HexState;
    type Move = HexMove;

    fn id() -> u32 {
        5
    }

    fn initial_state() -> Self::State {
        HexState::default()
    }

    fn to_move(state: &Self::State) -> Player {
        if state.to_move == 1 { Player::First } else { Player::Second }
    }

    fn outcome(state: &Self::State) -> Outcome {
        match state.winner {
            1 => Outcome::FirstWins,
            2 => Outcome::SecondWins,
            _ => Outcome::InProgress, // never a draw: a full board always has a winner
        }
    }

    fn legal_moves(state: &Self::State) -> Vec<Self::Move> {
        if state.winner != 0 {
            return vec![];
        }
        let mut moves = Vec::with_capacity(NN + 1);
        if swap_available(state) {
            moves.push(HexMove::Swap);
        }
        for i in state.empty().iter() {
            moves.push(HexMove::Play { r: (i / N) as u8, c: (i % N) as u8 });
        }
        moves
    }

    fn apply(state: &Self::State, mv: &Self::Move) -> Self::State {
        let mut next = *state;
        match mv {
            HexMove::Play { r, c } => {
                let i = *r as usize * N + *c as usize;
                debug_assert!(state.cell_at(i) == 0, "play on an occupied cell");
                if state.to_move == 1 {
                    next.black.set(i);
                    if connects(next.black, TOP, BOTTOM) {
                        next.winner = 1;
                    }
                } else {
                    next.white.set(i);
                    if connects(next.white, LEFT, RIGHT) {
                        next.winner = 2;
                    }
                }
            }
            HexMove::Swap => {
                // Structural precondition only (not [`ALLOW_SWAP`]): a peer
                // playing standard rules may send a swap even when our self-play
                // configuration doesn't generate one.
                debug_assert!(
                    state.ply == 1 && state.to_move == 2,
                    "swap is only the reply to First's opening move"
                );
                let i = state.black.iter().next().expect("swap follows exactly one black stone");
                next.black.clear(i);
                next.white.set(transpose_index(i));
            }
        }
        next.to_move = 3 - state.to_move;
        next.ply += 1;
        next
    }

    fn ply(state: &Self::State) -> u32 {
        state.ply as u32
    }

    fn to_features(state: &Self::State) -> Vec<u8> {
        let (own, opp) = state.own_opp();
        let mut out = Vec::with_capacity(PLANES * NN);
        for bb in [own, opp] {
            for i in 0..NN {
                out.push(bb.get(i) as u8);
            }
        }
        let side_plane: u8 = if state.to_move == 1 { 1 } else { 0 };
        out.resize(PLANES * NN, side_plane);
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        // 169 cell bytes, then to_move, winner, ply (LE u16).
        let mut out = Vec::with_capacity(NN + 4);
        for i in 0..NN {
            out.push(state.cell_at(i));
        }
        out.push(state.to_move);
        out.push(state.winner);
        out.extend_from_slice(&state.ply.to_le_bytes());
        out
    }

    fn decode_state(bytes: &[u8]) -> anyhow::Result<Self::State> {
        if bytes.len() < NN + 4 {
            anyhow::bail!("hex state: too short ({} bytes)", bytes.len());
        }
        let mut state = HexState::default();
        for (i, &b) in bytes[..NN].iter().enumerate() {
            match b {
                1 => state.black.set(i),
                2 => state.white.set(i),
                0 => {}
                other => anyhow::bail!("hex state: bad cell byte {other}"),
            }
        }
        state.to_move = bytes[NN];
        state.winner = bytes[NN + 1];
        state.ply = u16::from_le_bytes([bytes[NN + 2], bytes[NN + 3]]);
        if state.to_move != 1 && state.to_move != 2 {
            anyhow::bail!("hex state: bad to_move {}", state.to_move);
        }
        // Don't trust a peer's win flag — it decides terminality.
        state.winner = compute_winner(&state);
        Ok(state)
    }

    fn encode_move(mv: &Self::Move) -> Vec<u8> {
        match mv {
            HexMove::Swap => vec![0],
            HexMove::Play { r, c } => vec![1, *r, *c],
        }
    }

    fn decode_move(bytes: &[u8]) -> anyhow::Result<Self::Move> {
        match bytes {
            [0] => Ok(HexMove::Swap),
            [1, r, c] if (*r as usize) < N && (*c as usize) < N => {
                Ok(HexMove::Play { r: *r, c: *c })
            }
            _ => anyhow::bail!("invalid hex move bytes (len={})", bytes.len()),
        }
    }
}

/// Is [`HexMove::Swap`] legal now? Only for Second, in reply to First's opening
/// move, and only when [`ALLOW_SWAP`] is on.
#[inline]
pub fn swap_available(state: &HexState) -> bool {
    ALLOW_SWAP && state.ply == 1 && state.to_move == 2
}

/// Placeholder built-in engine: centre if free, else the first empty cell. Hex
/// gets its strength from the AZ net (and MoHex is the external reference
/// opponent) — this only exists so `RequestMove` can answer without a model.
impl Solver for Hex {
    fn choose_move(state: &Self::State) -> SearchOutcome<Self::Move> {
        let centre = (N / 2) * N + N / 2;
        let i = if state.cell_at(centre) == 0 {
            centre
        } else {
            state.empty().iter().next().expect("non-terminal position has an empty cell")
        };
        let best_move = HexMove::Play { r: (i / N) as u8, c: (i % N) as u8 };
        SearchOutcome { best_move, value: 0.0, nodes: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(state: &HexState, cell: &str) -> HexState {
        let (r, c) = parse_cell(cell).unwrap();
        Hex::apply(state, &HexMove::Play { r: r as u8, c: c as u8 })
    }

    #[test]
    fn empty_board_has_169_moves() {
        let s = Hex::initial_state();
        assert_eq!(Hex::legal_moves(&s).len(), NN);
        assert_eq!(Hex::outcome(&s), Outcome::InProgress);
    }

    #[test]
    fn neighbours_are_the_six_hex_directions() {
        // Interior cell: all six exist.
        let (r, c) = (6usize, 6usize);
        let nb = expand(Bb::cell(r, c));
        assert_eq!(nb.count(), 6);
        for (dr, dc) in [(-1i32, 0i32), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)] {
            let (nr, nc) = ((r as i32 + dr) as usize, (c as i32 + dc) as usize);
            assert!(nb.get(nr * N + nc), "missing neighbour ({nr},{nc})");
        }
        // The NW/SE diagonal is *not* adjacent on a hex grid.
        assert!(!nb.get((r - 1) * N + c - 1));
        assert!(!nb.get((r + 1) * N + c + 1));
        // Corners: the acute ones (a1, m13) have 2 neighbours, the obtuse ones
        // (m1, a13) have 3 — the rhombus, not a square.
        assert_eq!(expand(Bb::cell(0, 0)).count(), 2);
        assert_eq!(expand(Bb::cell(N - 1, N - 1)).count(), 2);
        assert_eq!(expand(Bb::cell(0, N - 1)).count(), 3);
        assert_eq!(expand(Bb::cell(N - 1, 0)).count(), 3);
        // No wrap: last cell of a row is not adjacent to the first of the next.
        assert!(!expand(Bb::cell(3, N - 1)).get(4 * N));
    }

    #[test]
    fn black_wins_a_straight_vertical_chain() {
        let mut s = Hex::initial_state();
        for r in 0..N {
            s = play(&s, &cell_name(r, 6)); // black climbs column g
            if r < N - 1 {
                s = play(&s, &cell_name(r, 0)); // white fills column a (no crossing)
            }
        }
        assert_eq!(Hex::outcome(&s), Outcome::FirstWins);
        assert!(Hex::legal_moves(&s).is_empty());
    }

    #[test]
    fn white_wins_a_zigzag_horizontal_chain() {
        // White snakes left→right, alternating the sideways step (r, c+1) with
        // the diagonal one (r-1, c+1) — the only two ways to advance a column.
        let mut white = Bb::ZERO;
        let mut r = 9usize;
        for c in 0..N {
            white.set(r * N + c);
            if c % 2 == 0 {
                r -= 1; // (r-1, c+1): the NE diagonal is adjacent on a hex grid
            }
        }
        let s = HexState { white, to_move: 1, ..Default::default() };
        assert_eq!(compute_winner(&s), 2);
        // Break one link (the chain's stone in the middle column) and the
        // connection is gone.
        let mut broken = s;
        let mid = s.white.iter().find(|i| i % N == N / 2).expect("chain crosses every column");
        broken.white.clear(mid);
        assert_eq!(compute_winner(&broken), 0);
    }

    #[test]
    fn a_full_board_always_has_exactly_one_winner() {
        // Fill the board by alternating pseudo-random empty cells; the first
        // crossing chain must appear before the board fills, and only one side
        // can ever have one.
        let mut seed = 12345u64;
        for _ in 0..50 {
            let mut s = Hex::initial_state();
            while s.winner == 0 {
                let moves = Hex::legal_moves(&s);
                assert!(!moves.is_empty(), "board filled with no winner");
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let mv = moves[(seed >> 33) as usize % moves.len()];
                s = Hex::apply(&s, &mv);
            }
            let black = connects(s.black, TOP, BOTTOM);
            let white = connects(s.white, LEFT, RIGHT);
            assert!(black ^ white, "exactly one player connects");
            assert_eq!(s.winner, if black { 1 } else { 2 });
            // The winner is always the player who just moved.
            assert_eq!(s.winner, 3 - s.to_move);
        }
    }

    #[test]
    fn notation_roundtrips() {
        assert_eq!(parse_cell("a1"), Some((0, 0)));
        assert_eq!(parse_cell("M13"), Some((12, 12)));
        assert_eq!(parse_cell("i5"), Some((4, 8))); // 'i' is a real column in Hex
        assert_eq!(parse_cell("n1"), None);
        assert_eq!(parse_cell("a14"), None);
        for i in 0..NN {
            let (r, c) = (i / N, i % N);
            assert_eq!(parse_cell(&cell_name(r, c)), Some((r, c)));
        }
    }

    #[test]
    fn symmetries_preserve_the_result() {
        let mut s = Hex::initial_state();
        for cell in ["g1", "a5", "g2", "b5", "g3", "c5"] {
            s = play(&s, cell);
        }
        // 180° rotation: same side to move, same winner-in-waiting.
        let r = rot180(&s);
        assert_eq!(r.to_move, s.to_move);
        assert_eq!(rot180(&r), s);
        assert_eq!(compute_winner(&r), compute_winner(&s));
        // Transpose + colour swap: roles exchanged, involutive.
        let t = transpose_swap(&s);
        assert_eq!(t.to_move, 3 - s.to_move);
        assert_eq!(t.black.count(), s.white.count());
        assert_eq!(transpose_swap(&t), s);
        // A black win maps to a white win.
        let mut won = Hex::initial_state();
        for r in 0..N {
            won.black.set(r * N + 6);
        }
        won.winner = compute_winner(&won);
        assert_eq!(won.winner, 1);
        assert_eq!(compute_winner(&transpose_swap(&won)), 2);
    }

    #[test]
    fn canonical_always_faces_black() {
        let s = play(&Hex::initial_state(), "g7");
        assert_eq!(s.to_move, 2);
        let c = canonical(&s);
        assert_eq!(c.to_move, 1);
        assert_eq!(canonical(&Hex::initial_state()), Hex::initial_state());
    }

    #[test]
    fn swap_is_gated_and_mirrors_the_stone() {
        let s = play(&Hex::initial_state(), "b3"); // (r,c) = (2,1)
        assert_eq!(swap_available(&s), ALLOW_SWAP);
        // Even with the rule off, applying Swap must do the right thing (the
        // move exists on the wire for standard-rules play against MoHex).
        let after = Hex::apply(&s, &HexMove::Swap);
        assert!(after.black.is_empty());
        assert_eq!(after.white, Bb::cell(1, 2)); // mirrored to (c,r)
        assert_eq!(after.to_move, 1);
    }

    #[test]
    fn move_roundtrip() {
        for mv in [HexMove::Swap, HexMove::Play { r: 12, c: 0 }, HexMove::Play { r: 6, c: 6 }] {
            assert_eq!(Hex::decode_move(&Hex::encode_move(&mv)).unwrap(), mv);
        }
        assert!(Hex::decode_move(&[1, 13, 0]).is_err());
    }

    #[test]
    fn state_roundtrip() {
        let mut s = Hex::initial_state();
        for cell in ["g7", "d4", "h7", "e4"] {
            s = play(&s, cell);
        }
        let back = Hex::decode_state(&Hex::encode_state(&s)).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn features_match_the_plane_helper() {
        let s = play(&Hex::initial_state(), "g7");
        let bytes = Hex::to_features(&s);
        let mut planes = vec![0f32; PLANES * NN];
        fill_planes_into(&s, &mut planes);
        assert_eq!(bytes.len(), PLANES * NN);
        for (b, p) in bytes.iter().zip(&planes) {
            assert_eq!(*b as f32, *p);
        }
        // White to move: own = white (empty), opp = black (one stone at g7).
        let (r, c) = parse_cell("g7").unwrap();
        assert_eq!(planes[r * N + c], 0.0);
        assert_eq!(planes[NN + r * N + c], 1.0);
        assert_eq!(planes[2 * NN], 0.0);
    }
}
