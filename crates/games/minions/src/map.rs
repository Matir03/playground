//! The single 7×7 hex board.
//!
//! Geometry is the reference implementation's, scaled down: a rhombus of hexes
//! addressed by `(x, y)` with `x` running `a`..`g` and `y` running `0`..`6`,
//! cell index `i = y*7 + x`, and the six neighbour offsets
//! `W(-1,0) NW(-1,+1) NE(0,+1) E(+1,0) SE(+1,-1) SW(0,-1)`. Hex distance is
//! `max(|dx|, |dy|, |dx+dy|)`.
//!
//! **Simplification: the map is all land.** The full game's water tiles and
//! native terrain (flood / earthquake / whirlwind / firestorm) are gone, so
//! `flying` only ever means "moves through enemy units". What is kept is the
//! graveyards, because they drive both income and the second way to win.
//!
//! The layout is symmetric under the 180° rotation `(x,y) → (6-x, 6-y)`, which
//! maps each side's setup onto the other's, and no two graveyards are adjacent
//! — both invariants the full game requires of its maps. A 7×7 board keeps the
//! whole action space intact (max speed is 3 and 37 hexes lie within distance 3
//! of a cell, so the per-cell move head is unchanged) while cutting the cells
//! from 100 to 49.

pub const N: usize = 7;
pub const NUM_CELLS: usize = N * N;

/// Graveyards a side must occupy at the end of its opponent's turn to take the
/// board that way. Four of six, keeping the full game's "clear majority, not
/// all" shape.
pub const GRAVEYARDS_TO_WIN: u32 = 4;

/// The six graveyards, as `(x, y)`: three rotation-symmetric pairs, none
/// adjacent to another (minimum pairwise hex distance 2) and none on a starting
/// cell. Two flank the centre diagonal, two sit on the short flanks, and one
/// pair holds the board's opposite edges.
pub const GRAVEYARD_XY: [(u8, u8); 6] = [
    (1, 4), (5, 2),
    (4, 1), (2, 5),
    (0, 3), (6, 3),
];

/// Necromancer start cells: `b1` for the first player, `f5` for the second —
/// each a full hex in from its corner, so all six neighbours exist for the
/// opening zombies.
pub const NECRO_START: [u8; 2] = [idx(1, 1), idx(5, 5)];

#[inline]
pub const fn idx(x: u8, y: u8) -> u8 {
    y * N as u8 + x
}

#[inline]
pub const fn cell_x(i: u8) -> i32 {
    (i % N as u8) as i32
}

#[inline]
pub const fn cell_y(i: u8) -> i32 {
    (i / N as u8) as i32
}

/// Hex distance between two cells.
#[inline]
pub fn dist(a: u8, b: u8) -> u32 {
    let dx = cell_x(a) - cell_x(b);
    let dy = cell_y(a) - cell_y(b);
    let s = dx + dy;
    (dx.abs().max(dy.abs()).max(s.abs())) as u32
}

const DIRS: [(i32, i32); 6] = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)];

/// `NO_CELL` pads the neighbour table for cells on the rim.
pub const NO_CELL: u8 = u8::MAX;

/// Six neighbours per cell, `NO_CELL`-padded. Built at compile time so move
/// generation never recomputes hex arithmetic.
pub const NEIGHBORS: [[u8; 6]; NUM_CELLS] = build_neighbors();

const fn build_neighbors() -> [[u8; 6]; NUM_CELLS] {
    let mut table = [[NO_CELL; 6]; NUM_CELLS];
    let mut i = 0;
    while i < NUM_CELLS {
        let x = (i % N) as i32;
        let y = (i / N) as i32;
        let mut d = 0;
        while d < 6 {
            let nx = x + DIRS[d].0;
            let ny = y + DIRS[d].1;
            if nx >= 0 && nx < N as i32 && ny >= 0 && ny < N as i32 {
                table[i][d] = (ny * N as i32 + nx) as u8;
            }
            d += 1;
        }
        i += 1;
    }
    table
}

/// `true` at cells listed in [`GRAVEYARD_XY`].
pub const IS_GRAVEYARD: [bool; NUM_CELLS] = build_graveyards();

const fn build_graveyards() -> [bool; NUM_CELLS] {
    let mut mask = [false; NUM_CELLS];
    let mut k = 0;
    while k < GRAVEYARD_XY.len() {
        let (x, y) = GRAVEYARD_XY[k];
        mask[idx(x, y) as usize] = true;
        k += 1;
    }
    mask
}

/// The six cells each side's opening zombies occupy: the neighbours of that
/// side's necromancer, exactly as in the full game's starting position.
pub fn start_zombies(side: usize) -> [u8; 6] {
    NEIGHBORS[NECRO_START[side] as usize]
}

/// `a0`-style name for a cell, matching the reference implementation's
/// notation (column letter `a`..`g`, then row digit `0`..`6`).
pub fn cell_name(i: u8) -> String {
    format!("{}{}", (b'a' + (i % N as u8)) as char, i / N as u8)
}

/// Parse an `a0`-style cell name.
pub fn parse_cell(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let x = b[0].checked_sub(b'a')?;
    let y = b[1].checked_sub(b'0')?;
    if x < N as u8 && y < N as u8 {
        Some(idx(x, y))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_are_symmetric_and_unit_distance() {
        for i in 0..NUM_CELLS as u8 {
            for &n in &NEIGHBORS[i as usize] {
                if n == NO_CELL {
                    continue;
                }
                assert_eq!(dist(i, n), 1, "{} -> {}", cell_name(i), cell_name(n));
                assert!(NEIGHBORS[n as usize].contains(&i), "asymmetric edge");
            }
        }
    }

    #[test]
    fn interior_cells_have_six_neighbours() {
        let interior = idx(3, 3);
        assert_eq!(NEIGHBORS[interior as usize].iter().filter(|&&n| n != NO_CELL).count(), 6);
    }

    #[test]
    fn graveyards_are_symmetric_and_non_adjacent() {
        let gys: Vec<u8> = GRAVEYARD_XY.iter().map(|&(x, y)| idx(x, y)).collect();
        assert_eq!(gys.len(), GRAVEYARD_XY.len());
        assert!(
            GRAVEYARDS_TO_WIN as usize * 2 > GRAVEYARD_XY.len(),
            "the win threshold must be a strict majority of the graveyards"
        );
        let last = N as u8 - 1;
        for &(x, y) in &GRAVEYARD_XY {
            assert!(
                GRAVEYARD_XY.contains(&(last - x, last - y)),
                "{x},{y} has no rotation partner"
            );
        }
        for (i, &a) in gys.iter().enumerate() {
            for &b in &gys[i + 1..] {
                assert!(dist(a, b) >= 2, "{} and {} are adjacent", cell_name(a), cell_name(b));
            }
        }
    }

    #[test]
    fn setup_cells_are_clear_of_graveyards() {
        for side in 0..2 {
            assert!(!IS_GRAVEYARD[NECRO_START[side] as usize]);
            for z in start_zombies(side) {
                assert_ne!(z, NO_CELL);
                assert!(!IS_GRAVEYARD[z as usize]);
            }
        }
    }

    #[test]
    fn cell_names_round_trip() {
        assert_eq!(cell_name(NECRO_START[0]), "b1");
        assert_eq!(cell_name(NECRO_START[1]), "f5");
        for i in 0..NUM_CELLS as u8 {
            assert_eq!(parse_cell(&cell_name(i)), Some(i));
        }
    }
}

/// 180° rotation about the board centre, which on this rhombus is exactly index
/// reversal: `(x, y) -> (N-1-x, N-1-y)` and `y*N+x -> NUM_CELLS-1-(y*N+x)`.
///
/// This is a genuine automorphism of the position, not an approximation, and it
/// is what lets the net see one point of view instead of two:
///   - the six hex directions are closed under negation, so adjacency survives;
///   - the graveyards map onto each other — (1,4)<->(5,2), (4,1)<->(2,5),
///     (0,3)<->(6,3);
///   - the necromancer starts swap — (1,1)<->(5,5).
/// Each of those is asserted in `rotation_is_an_automorphism` below.
#[inline]
pub const fn rot(i: usize) -> usize {
    NUM_CELLS - 1 - i
}

#[cfg(test)]
mod rotation_tests {
    use super::*;

    #[test]
    fn rotation_is_an_automorphism() {
        // Self-inverse.
        for i in 0..NUM_CELLS {
            assert_eq!(rot(rot(i)), i);
        }
        // Adjacency is preserved: the rotated neighbours of a cell are exactly
        // the neighbours of the rotated cell.
        for i in 0..NUM_CELLS {
            let mut a: Vec<u8> = NEIGHBORS[i]
                .iter()
                .filter(|&&n| n != NO_CELL)
                .map(|&n| rot(n as usize) as u8)
                .collect();
            let mut b: Vec<u8> =
                NEIGHBORS[rot(i)].iter().filter(|&&n| n != NO_CELL).copied().collect();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "adjacency broken at cell {i}");
        }
        // The graveyards are setwise invariant.
        for i in 0..NUM_CELLS {
            assert_eq!(IS_GRAVEYARD[i], IS_GRAVEYARD[rot(i)], "graveyards asymmetric at {i}");
        }
        // The two necromancers start on rotated squares.
        assert_eq!(rot(NECRO_START[0] as usize), NECRO_START[1] as usize);
    }
}
