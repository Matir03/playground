//! 100-cell hex bitboards in a single `u128`.
//!
//! The board is exactly 100 cells, so a `u128` holds one bit per hex with 28 to
//! spare — the same trick as the reference engine's
//! `spooky/src/core/board/bitboards.rs`, which this is a direct port of: same
//! 10×10 rhombus, same six directions, same `i = y*10 + x` indexing, so the
//! shift/mask constants transfer unchanged.
//!
//! A dilation is six shift-and-mask ORs. Each direction's mask is the set of
//! *destination* cells that direction can legally reach, which is what stops a
//! shift from wrapping around a row edge: stepping W from `x = 0` would land on
//! `x = 9` of the row below, and that cell is not a valid W destination (W
//! destinations are `x ≤ 8`), so the mask drops it.

use crate::map::{N, NUM_CELLS};

pub type Bb = u128;

/// Every on-board cell set.
pub const FULL: Bb = (1u128 << NUM_CELLS) - 1;

#[inline]
pub const fn bit(i: u8) -> Bb {
    1u128 << i
}

#[inline]
pub const fn has(b: Bb, i: u8) -> bool {
    b & bit(i) != 0
}

/// The set of cells reachable as the destination of a single `(dx, dy)` step.
const fn dest_mask(dx: i32, dy: i32) -> Bb {
    let mut m: Bb = 0;
    let mut i = 0;
    while i < NUM_CELLS {
        let x = (i % N) as i32;
        let y = (i / N) as i32;
        let (nx, ny) = (x + dx, y + dy);
        if nx >= 0 && nx < N as i32 && ny >= 0 && ny < N as i32 {
            m |= 1u128 << (ny * N as i32 + nx);
        }
        i += 1;
    }
    m
}

// Directions in the same order as `map::NEIGHBORS`: W NW NE E SE SW.
// Index shifts are `dy*10 + dx`: -1, +9, +10, +1, -9, -10.
const M_W: Bb = dest_mask(-1, 0);
const M_NW: Bb = dest_mask(-1, 1);
const M_NE: Bb = dest_mask(0, 1);
const M_E: Bb = dest_mask(1, 0);
const M_SE: Bb = dest_mask(1, -1);
const M_SW: Bb = dest_mask(0, -1);

/// Every cell one hex step from a cell in `b` (not including `b` itself).
#[inline]
pub const fn dilate(b: Bb) -> Bb {
    ((b >> 1) & M_W)
        | ((b << 9) & M_NW)
        | ((b << 10) & M_NE)
        | ((b << 1) & M_E)
        | ((b >> 9) & M_SE)
        | ((b >> 10) & M_SW)
}

/// Cells reachable from `from` in at most `speed` steps, moving only through
/// cells in `pass`. The source stays in the result regardless of `pass`.
///
/// Bounded flood fill — `speed ≤ 3` here, so this is at most three dilations.
#[inline]
pub fn reachable(from: Bb, speed: u8, pass: Bb) -> Bb {
    let mut r = from;
    for _ in 0..speed {
        let next = r | (dilate(r) & pass);
        if next == r {
            break; // fully enclosed; further steps cannot help
        }
        r = next;
    }
    r
}

/// Iterate set cells, lowest index first.
#[inline]
pub fn cells(mut b: Bb) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if b == 0 {
            return None;
        }
        let i = b.trailing_zeros() as u8;
        b &= b - 1;
        Some(i)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{dist, idx, NEIGHBORS, NO_CELL};

    #[test]
    fn dilate_matches_the_neighbour_table() {
        for i in 0..NUM_CELLS as u8 {
            let got = dilate(bit(i));
            let mut want: Bb = 0;
            for &n in &NEIGHBORS[i as usize] {
                if n != NO_CELL {
                    want |= bit(n);
                }
            }
            assert_eq!(got, want, "cell {i}");
        }
    }

    #[test]
    fn dilate_never_wraps_off_an_edge() {
        // Every dilation result stays on-board and one step away.
        for i in 0..NUM_CELLS as u8 {
            let d = dilate(bit(i));
            assert_eq!(d & !FULL, 0, "cell {i} wrapped out of bounds");
            for c in cells(d) {
                assert_eq!(dist(i, c), 1, "{i} -> {c}");
            }
        }
    }

    #[test]
    fn reachable_on_an_open_board_is_the_distance_ball() {
        let start = idx(5, 5);
        for speed in 0..=3u8 {
            let r = reachable(bit(start), speed, FULL);
            let want = (0..NUM_CELLS as u8).filter(|&c| dist(start, c) <= speed as u32);
            assert_eq!(cells(r).count(), want.clone().count());
            for c in want {
                assert!(has(r, c), "speed {speed} should reach {c}");
            }
        }
        // 37 hexes within distance 3 — the size of the move head.
        assert_eq!(cells(reachable(bit(start), 3, FULL)).count(), 37);
    }

    #[test]
    fn blockers_cut_the_flood_fill() {
        let start = idx(5, 2);
        let corridor = idx(5, 3);
        let behind = idx(5, 4);
        let pass = FULL & !bit(corridor);
        let r = reachable(bit(start), 2, pass);
        assert!(!has(r, behind), "the only 2-step route runs through the corridor");
        assert!(has(r, start));
    }

    #[test]
    fn a_fully_enclosed_cell_reaches_only_itself() {
        let start = idx(5, 5);
        let pass = FULL & !dilate(bit(start));
        assert_eq!(reachable(bit(start), 3, pass), bit(start));
    }
}
