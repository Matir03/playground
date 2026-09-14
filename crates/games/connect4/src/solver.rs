//! Perfect Connect 4 solver.
//!
//! Bitboard negamax with alpha-beta, a transposition table, move ordering and
//! the standard "non-losing moves" pruning — the algorithm from Pascal Pons'
//! Connect 4 tutorial (<http://blog.gamesolver.org/>). It plays optimally:
//! it wins from every won position as fast as possible, holds every draw, and
//! delays every loss as long as possible.
//!
//! Bit layout: 7 bits per column (6 cells + 1 sentinel row), bit index
//! `col * 7 + row`, row 0 at the bottom. This matches the wire encoding in
//! `game.rs` (cells indexed `[row][col]`).

use playground_core::{SearchOutcome, Solver};

use crate::game::{Connect4, Connect4Move, Connect4State, C4_COLS, C4_ROWS};

const W: u32 = C4_COLS as u32; // 7
const H: u32 = C4_ROWS as u32; // 6
const BITS_PER_COL: u32 = C4_ROWS as u32 + 1; // 7

// Score range for a strong solve: ±(W*H)/2 around 0, trimmed by 3 (no win in
// the opening few plies). Used to encode TT bounds in a single byte.
const MIN_SCORE: i32 = -((W as i32 * H as i32) / 2) + 3;
const MAX_SCORE: i32 = (W as i32 * H as i32 + 1) / 2 - 3;

// Explore columns from the center out — better alpha-beta cutoffs.
const COLUMN_ORDER: [usize; 7] = [3, 2, 4, 1, 5, 0, 6];

#[inline]
fn bottom_mask_col(col: usize) -> u64 {
    1u64 << (col as u32 * BITS_PER_COL)
}
#[inline]
fn top_mask_col(col: usize) -> u64 {
    1u64 << ((H - 1) + col as u32 * BITS_PER_COL)
}
#[inline]
fn column_mask(col: usize) -> u64 {
    ((1u64 << H) - 1) << (col as u32 * BITS_PER_COL)
}

/// Full bottom row (one bit per column) and the full playable board, as
/// compile-time constants (hot-path: referenced millions of times per solve).
const fn bottom_row_const() -> u64 {
    let mut r = 0u64;
    let mut c = 0;
    while c < C4_COLS {
        r |= 1u64 << (c as u32 * BITS_PER_COL);
        c += 1;
    }
    r
}
const BOTTOM_ROW: u64 = bottom_row_const();
const BOARD_MASK: u64 = BOTTOM_ROW * ((1u64 << H) - 1);

/// A position as (current player's stones, all stones, ply).
#[derive(Clone, Copy)]
struct Position {
    current: u64,
    mask: u64,
    moves: u32,
}

impl Position {
    fn key(&self) -> u64 {
        self.current + self.mask
    }

    fn can_play(&self, col: usize) -> bool {
        (self.mask & top_mask_col(col)) == 0
    }

    fn play(&mut self, mv: u64) {
        self.current ^= self.mask;
        self.mask |= mv;
        self.moves += 1;
    }

    /// Bitmap of cells that would be played next in each column.
    fn possible(&self) -> u64 {
        (self.mask + BOTTOM_ROW) & BOARD_MASK
    }

    fn winning_position(&self) -> u64 {
        compute_winning(self.current, self.mask)
    }
    fn opponent_winning_position(&self) -> u64 {
        compute_winning(self.current ^ self.mask, self.mask)
    }
    fn can_win_next(&self) -> bool {
        (self.winning_position() & self.possible()) != 0
    }

    /// Playable moves that don't hand the opponent an immediate win.
    fn possible_non_losing_moves(&self) -> u64 {
        let mut possible = self.possible();
        let opponent_win = self.opponent_winning_position();
        let forced = possible & opponent_win;
        if forced != 0 {
            if forced & (forced - 1) != 0 {
                return 0; // two distinct threats — lost
            }
            possible = forced;
        }
        possible & !(opponent_win >> 1)
    }

    /// Heuristic move score: number of winning spots created.
    fn move_score(&self, mv: u64) -> u32 {
        compute_winning(self.current | mv, self.mask).count_ones()
    }

    /// Canonical base-3 key (mirror-folded) used by the opening book. Digits
    /// per column from the bottom up: 1 = current player, 2 = opponent, 0 = the
    /// empty cell above the stack. Identical to Pons' `Position::key3()`.
    fn key3(&self) -> u64 {
        let mut forward = 0u64;
        for col in 0..C4_COLS {
            self.partial_key3(&mut forward, col);
        }
        let mut reverse = 0u64;
        for col in (0..C4_COLS).rev() {
            self.partial_key3(&mut reverse, col);
        }
        // Smaller of the two mirror keys; the trailing digit is always 0.
        forward.min(reverse) / 3
    }

    fn partial_key3(&self, key: &mut u64, col: usize) {
        let mut pos = 1u64 << (col as u32 * BITS_PER_COL);
        while pos & self.mask != 0 {
            *key *= 3;
            *key += if pos & self.current != 0 { 1 } else { 2 };
            pos <<= 1;
        }
        *key *= 3;
    }
}

/// All cells where `position` already has (or would have) a 4-in-a-row,
/// restricted to currently empty cells.
fn compute_winning(position: u64, mask: u64) -> u64 {
    // vertical
    let mut r = (position << 1) & (position << 2) & (position << 3);

    // horizontal
    let mut p = (position << BITS_PER_COL) & (position << (2 * BITS_PER_COL));
    r |= p & (position << (3 * BITS_PER_COL));
    r |= p & (position >> BITS_PER_COL);
    p = (position >> BITS_PER_COL) & (position >> (2 * BITS_PER_COL));
    r |= p & (position << BITS_PER_COL);
    r |= p & (position >> (3 * BITS_PER_COL));

    // diagonal /
    p = (position << H) & (position << (2 * H));
    r |= p & (position << (3 * H));
    r |= p & (position >> H);
    p = (position >> H) & (position >> (2 * H));
    r |= p & (position << H);
    r |= p & (position >> (3 * H));

    // diagonal \
    let d = H + 2;
    p = (position << d) & (position << (2 * d));
    r |= p & (position << (3 * d));
    r |= p & (position >> d);
    p = (position >> d) & (position >> (2 * d));
    r |= p & (position << d);
    r |= p & (position >> (3 * d));

    r & (BOARD_MASK ^ mask)
}

// ── Transposition table ─────────────────────────────────────────────────────

/// Fixed-size table keyed by full position key (parallel arrays avoid the
/// padding of a packed struct). A stored byte of 0 means "empty".
struct TransTable {
    keys: Vec<u64>,
    vals: Vec<u8>,
}

const TT_SIZE: usize = 1 << 23; // ~8.4M entries → 72 MB. Plenty for a full solve.

impl TransTable {
    fn new() -> Self {
        Self {
            keys: vec![0; TT_SIZE],
            vals: vec![0; TT_SIZE],
        }
    }
    fn get(&self, key: u64) -> u8 {
        let i = (key as usize) % TT_SIZE;
        if self.keys[i] == key {
            self.vals[i]
        } else {
            0
        }
    }
    fn put(&mut self, key: u64, val: u8) {
        let i = (key as usize) % TT_SIZE;
        self.keys[i] = key;
        self.vals[i] = val;
    }
}

// ── Move ordering ───────────────────────────────────────────────────────────

/// Fixed-capacity move list kept sorted by score, highest first. Equal scores
/// keep insertion order, so `COLUMN_ORDER` (center-first) breaks ties.
struct MoveSorter {
    entries: [(u64, u32); 7],
    size: usize,
}

impl MoveSorter {
    fn new() -> Self {
        Self {
            entries: [(0, 0); 7],
            size: 0,
        }
    }
    fn add(&mut self, mv: u64, score: u32) {
        let mut i = self.size;
        self.size += 1;
        while i > 0 && self.entries[i - 1].1 < score {
            self.entries[i] = self.entries[i - 1];
            i -= 1;
        }
        self.entries[i] = (mv, score);
    }
}

// ── Solver ──────────────────────────────────────────────────────────────────

struct Engine {
    tt: TransTable,
    nodes: u64,
}

impl Engine {
    fn new() -> Self {
        Self {
            tt: TransTable::new(),
            nodes: 0,
        }
    }

    fn negamax(&mut self, pos: &Position, mut alpha: i32, mut beta: i32) -> i32 {
        debug_assert!(alpha < beta);
        self.nodes += 1;

        let possible = pos.possible_non_losing_moves();
        if possible == 0 {
            // Every move lets the opponent win next.
            return -((W as i32 * H as i32 - pos.moves as i32) / 2);
        }
        if pos.moves >= W * H - 2 {
            return 0; // board fills to a draw
        }

        let min = -((W as i32 * H as i32 - 2 - pos.moves as i32) / 2);
        if alpha < min {
            alpha = min;
            if alpha >= beta {
                return alpha;
            }
        }
        let mut max = (W as i32 * H as i32 - 1 - pos.moves as i32) / 2;
        if beta > max {
            beta = max;
            if alpha >= beta {
                return beta;
            }
        }

        let key = pos.key();
        let val = self.tt.get(key) as i32;
        if val != 0 {
            if val > MAX_SCORE - MIN_SCORE + 1 {
                // lower bound
                let lb = val + 2 * MIN_SCORE - MAX_SCORE - 2;
                if alpha < lb {
                    alpha = lb;
                    if alpha >= beta {
                        return alpha;
                    }
                }
            } else {
                // upper bound
                max = val + MIN_SCORE - 1;
                if beta > max {
                    beta = max;
                    if alpha >= beta {
                        return beta;
                    }
                }
            }
        }

        // Order candidate moves by heuristic score, center-first as tie-break.
        // Stack-allocated insertion sort — this runs once per node, so no heap.
        let mut sorter = MoveSorter::new();
        for &col in COLUMN_ORDER.iter() {
            let mv = possible & column_mask(col);
            if mv != 0 {
                sorter.add(mv, pos.move_score(mv));
            }
        }

        for k in 0..sorter.size {
            let mv = sorter.entries[k].0;
            let mut child = *pos;
            child.play(mv);
            let score = -self.negamax(&child, -beta, -alpha);
            if score >= beta {
                self.tt
                    .put(key, (score + MAX_SCORE - 2 * MIN_SCORE + 2) as u8); // lower bound
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        self.tt.put(key, (alpha - MIN_SCORE + 1) as u8); // upper bound
        alpha
    }

    /// Game-theoretic score for the side to move. `weak` restricts the window
    /// to {-1, 0, +1} (loss / draw / win) — far cheaper than the exact
    /// distance-to-result, and enough to play result-perfectly.
    fn solve(&mut self, pos: &Position, weak: bool) -> i32 {
        if pos.can_win_next() {
            return (W as i32 * H as i32 + 1 - pos.moves as i32) / 2;
        }
        let mut min = -((W as i32 * H as i32 - pos.moves as i32) / 2);
        let mut max = (W as i32 * H as i32 + 1 - pos.moves as i32) / 2;
        if weak {
            min = -1;
            max = 1;
        }
        // Iterative deepening with a null window — converges on the exact score.
        while min < max {
            let mut med = min + (max - min) / 2;
            if med <= 0 && min / 2 < med {
                med = min / 2;
            } else if med >= 0 && max / 2 > med {
                med = max / 2;
            }
            let r = self.negamax(pos, med, med + 1);
            if r <= med {
                max = r;
            } else {
                min = r;
            }
        }
        min
    }
}

/// Build a bitboard `Position` from the wire state.
fn to_position(state: &Connect4State) -> Position {
    let to_move_first = state.ply % 2 == 0;
    let me: u8 = if to_move_first { 1 } else { 2 };
    let mut current = 0u64;
    let mut mask = 0u64;
    for r in 0..C4_ROWS {
        for c in 0..C4_COLS {
            let v = state.cells[r][c];
            if v != 0 {
                let bit = 1u64 << (c as u32 * BITS_PER_COL + r as u32);
                mask |= bit;
                if v == me {
                    current |= bit;
                }
            }
        }
    }
    Position {
        current,
        mask,
        moves: state.ply as u32,
    }
}

/// One engine (transposition table) reused across moves and games. Warming it
/// up — especially on the expensive opening lines — makes every later move
/// cheaper. Serialized behind a mutex; interactive play is one solve at a time.
static ENGINE: std::sync::OnceLock<std::sync::Mutex<Engine>> = std::sync::OnceLock::new();

/// Opening book, loaded once. Covers shallow positions (≤ book depth) so the
/// engine plays the notoriously expensive opening instantly and perfectly;
/// deeper positions fall through to the live solver, which is fast by then.
static BOOK: std::sync::OnceLock<Option<crate::book::OpeningBook>> = std::sync::OnceLock::new();

fn book() -> Option<&'static crate::book::OpeningBook> {
    BOOK.get_or_init(|| {
        let path = std::env::var("CONNECT4_BOOK")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/data/7x6.book").to_string());
        match crate::book::OpeningBook::load(&path) {
            Ok(b) => {
                tracing::info!(%path, "loaded Connect 4 opening book");
                Some(b)
            }
            Err(e) => {
                tracing::warn!(%path, error = %e, "no opening book; opening play will be slow");
                None
            }
        }
    })
    .as_ref()
}

/// Exact/booked score for the side to move at `pos`: opening book first, then
/// the live weak solver. Returns the game value as -1 / 0 / +1 (with booked
/// scores keeping their magnitude so faster wins rank higher).
fn score_position(engine: &mut Engine, pos: &Position) -> i32 {
    if let Some(s) = book().and_then(|b| b.get(pos.key3(), pos.moves)) {
        s
    } else {
        engine.solve(pos, /* weak */ true)
    }
}

impl Solver for Connect4 {
    fn choose_move(state: &Connect4State) -> SearchOutcome<Connect4Move> {
        // Opening: center is provably optimal and skips the most expensive
        // full-board solve.
        if state.ply == 0 {
            return SearchOutcome {
                best_move: Connect4Move { col: 3 },
                value: 1.0,
                nodes: 0,
            };
        }

        let pos = to_position(state);

        // Immediate win available? Take it.
        for &col in COLUMN_ORDER.iter() {
            if pos.can_play(col) {
                let mv = pos.possible() & column_mask(col);
                if (pos.winning_position() & mv) != 0 {
                    return SearchOutcome {
                        best_move: Connect4Move { col: col as u8 },
                        value: 1.0,
                        nodes: 0,
                    };
                }
            }
        }

        // Result-perfect selection via weak solve. Only consider moves that
        // don't hand the opponent an immediate win (unless every move does, in
        // which case the position is lost and any legal move is equally fine).
        let non_losing = pos.possible_non_losing_moves();

        let mut engine = ENGINE
            .get_or_init(|| std::sync::Mutex::new(Engine::new()))
            .lock()
            .unwrap();
        let start_nodes = engine.nodes;

        let mut best_col: Option<usize> = None;
        let mut best_score = i32::MIN;
        let mut fallback: Option<usize> = None;
        for &col in COLUMN_ORDER.iter() {
            if !pos.can_play(col) {
                continue;
            }
            let mv = pos.possible() & column_mask(col);
            fallback.get_or_insert(col);
            if non_losing != 0 && (mv & non_losing) == 0 {
                continue; // a strictly better (non-losing) move exists
            }
            let mut child = pos;
            child.play(mv);
            let score = -score_position(&mut engine, &child);
            if score > best_score {
                best_score = score;
                best_col = Some(col);
            }
        }

        // best_col is None only when the position is lost (non_losing == 0 and
        // all moves were skipped) — fall back to the first legal column.
        let (col, value) = match best_col {
            Some(c) => (c, best_score.clamp(-1, 1) as f64),
            None => (
                fallback.expect("non-terminal position has a legal move"),
                -1.0,
            ),
        };

        SearchOutcome {
            best_move: Connect4Move { col: col as u8 },
            value,
            nodes: engine.nodes - start_nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use playground_core::{Game, Outcome};

    fn play(state: &Connect4State, col: u8) -> Connect4State {
        Connect4::apply(state, &Connect4Move { col })
    }

    #[test]
    fn takes_immediate_win() {
        // First has three in a row across cols 0,1,2 on the bottom; col 3 wins.
        let mut s = Connect4::initial_state();
        s = play(&s, 0); // F
        s = play(&s, 0); // S
        s = play(&s, 1); // F
        s = play(&s, 1); // S
        s = play(&s, 2); // F
        s = play(&s, 2); // S
                         // First to move; col 3 completes four across the bottom row.
        let out = Connect4::choose_move(&s);
        assert_eq!(out.best_move.col, 3);
        let next = play(&s, out.best_move.col);
        assert_eq!(Connect4::outcome(&next), Outcome::FirstWins);
    }

    #[test]
    fn blocks_immediate_threat() {
        // Second must block First's three-in-a-row.
        let mut s = Connect4::initial_state();
        s = play(&s, 0); // F col0
        s = play(&s, 6); // S col6
        s = play(&s, 1); // F col1
        s = play(&s, 6); // S col6
        s = play(&s, 2); // F col2  -> threat to win at col3
                         // Second to move; must play col3 to block.
        let out = Connect4::choose_move(&s);
        assert_eq!(out.best_move.col, 3, "engine must block the open three");
    }

    #[test]
    fn first_move_is_center() {
        let s = Connect4::initial_state();
        assert_eq!(Connect4::choose_move(&s).best_move.col, 3);
    }
}

/// No analysable net: this game has no sampled search or auxiliary heads, so
/// the default `analyze` (which reports "not configured") is the honest answer.
impl playground_core::Analyzable for Connect4 {}
