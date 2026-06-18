//! Factored action encoding for the two-head policy.
//!
//! A compound Amazons move is scored by two spatial heads: a **queen** head
//! (which queen moves where) and an **arrow** head (where it then shoots). Each is
//! `N*N*72` logits — per square, 72 move-types = 8 directions × `N-1` distances.
//! The per-square index layout is `(r*N + c)*72 + (dir*(N-1) + (dist-1))`, which
//! must match the Python conv heads' `(B,72,N,N) -> permute (B,N,N,72) -> (B,7200)`
//! reshape (square-major, move-type-minor). The full policy vector is the queen
//! head ‖ arrow head, so `POLICY_DIM = 2*PER_HEAD`.

use crate::game::{AmazonsMove, N};

pub const DIRS: usize = 8;
pub const DISTS: usize = N - 1; // 9 on a 10×10 board
pub const MOVE_TYPES: usize = DIRS * DISTS; // 72
pub const PER_HEAD: usize = N * N * MOVE_TYPES; // 7200
pub const POLICY_DIM: usize = 2 * PER_HEAD; // 14400 (queen ‖ arrow)

/// (dr, dc) unit step per direction index. Order is shared with `game::shift`.
/// The source of truth for `DIR_LUT` (verified by a test).
#[allow(dead_code)]
const DELTAS: [(i32, i32); DIRS] = [
    (1, 0),   // 0 N
    (-1, 0),  // 1 S
    (0, 1),   // 2 E
    (0, -1),  // 3 W
    (1, 1),   // 4 NE
    (1, -1),  // 5 NW
    (-1, 1),  // 6 SE
    (-1, -1), // 7 SW
];

/// (sdr+1)*3 + (sdc+1) -> direction index, matching `DELTAS` (center unused).
const DIR_LUT: [usize; 9] = [7, 1, 6, 3, 0, 2, 5, 0, 4];

#[inline]
fn dir_index(sdr: i32, sdc: i32) -> usize {
    DIR_LUT[((sdr + 1) * 3 + (sdc + 1)) as usize]
}

/// Move-type index for a queen-slide `(fr,fc) -> (tr,tc)` (must be a straight line).
#[inline]
pub fn move_type(fr: u8, fc: u8, tr: u8, tc: u8) -> usize {
    let dr = tr as i32 - fr as i32;
    let dc = tc as i32 - fc as i32;
    let dist = dr.abs().max(dc.abs());
    debug_assert!(dist >= 1 && dist <= DISTS as i32, "bad slide distance {dist}");
    let dir = dir_index(dr.signum(), dc.signum());
    dir * DISTS + (dist as usize - 1)
}

/// Index into the queen head for the queen-move part of `mv`.
#[inline]
pub fn queen_index(mv: &AmazonsMove) -> usize {
    let sq = mv.from_r as usize * N + mv.from_c as usize;
    sq * MOVE_TYPES + move_type(mv.from_r, mv.from_c, mv.to_r, mv.to_c)
}

/// Index into the arrow head for the arrow-shot part of `mv` (from the landing sq).
#[inline]
pub fn arrow_index(mv: &AmazonsMove) -> usize {
    let sq = mv.to_r as usize * N + mv.to_c as usize;
    sq * MOVE_TYPES + move_type(mv.to_r, mv.to_c, mv.arrow_r, mv.arrow_c)
}

/// (queen-head index, arrow-head index) for `mv`.
#[inline]
pub fn move_to_indices(mv: &AmazonsMove) -> (usize, usize) {
    (queen_index(mv), arrow_index(mv))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_in_range() {
        let mv = AmazonsMove { from_r: 0, from_c: 3, to_r: 0, to_c: 5, arrow_r: 4, arrow_c: 5 };
        let (q, a) = move_to_indices(&mv);
        assert!(q < PER_HEAD);
        assert!(a < PER_HEAD);
    }

    #[test]
    fn dir_lut_matches_deltas() {
        for (i, &(dr, dc)) in DELTAS.iter().enumerate() {
            assert_eq!(dir_index(dr, dc), i);
        }
    }

    #[test]
    fn move_type_roundtrip_examples() {
        // East 2: dir 2, dist 2 -> 2*9 + 1 = 19.
        assert_eq!(move_type(0, 3, 0, 5), 2 * DISTS + 1);
        // North 3: dir 0, dist 3 -> 2.
        assert_eq!(move_type(2, 2, 5, 2), 0 * DISTS + 2);
        // SW 1: dir 7, dist 1 -> 7*9.
        assert_eq!(move_type(5, 5, 4, 4), 7 * DISTS);
    }
}
