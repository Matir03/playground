//! The single 10×10 hex board.
//!
//! Geometry is the reference implementation's: a rhombus of hexes addressed by
//! `(x, y)` with `x` running `a`..`j` and `y` running `0`..`9`, cell index
//! `i = y*10 + x`, and the six neighbour offsets
//! `W(-1,0) NW(-1,+1) NE(0,+1) E(+1,0) SE(+1,-1) SW(0,-1)`. Hex distance is
//! `max(|dx|, |dy|, |dx+dy|)`.
//!
//! **Simplification: the map is all land.** The full game's water tiles and
//! native terrain (flood / earthquake / whirlwind / firestorm) are gone, so
//! `flying` only ever means "moves through enemy units". What is kept is the
//! ten graveyards, because they drive both income and the second way to win.
//!
//! The layout is symmetric under the 180° rotation `(x,y) → (9-x, 9-y)`, which
//! maps each side's setup onto the other's, and no two graveyards are adjacent
//! — both invariants the full game requires of its maps.

pub const N: usize = 10;
pub const NUM_CELLS: usize = N * N;

/// Graveyards a side must occupy at the end of its opponent's turn to take the
/// board that way. Same threshold as the full game.
pub const GRAVEYARDS_TO_WIN: u32 = 8;

/// The ten graveyards, as `(x, y)`. Five rotation-symmetric pairs: two near
/// each side's home, one pair either side of the centre, and two contested
/// pairs on the long diagonal's flanks.
pub const GRAVEYARD_XY: [(u8, u8); 10] = [
    (4, 4), (5, 5),
    (2, 6), (7, 3),
    (6, 2), (3, 7),
    (0, 4), (9, 5),
    (4, 0), (5, 9),
];

/// Necromancer start cells: `c2` for the first player, `h7` for the second.
pub const NECRO_START: [u8; 2] = [idx(2, 2), idx(7, 7)];

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
/// notation (column letter `a`..`j`, then row digit `0`..`9`).
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
        let interior = idx(5, 5);
        assert_eq!(NEIGHBORS[interior as usize].iter().filter(|&&n| n != NO_CELL).count(), 6);
    }

    #[test]
    fn graveyards_are_symmetric_and_non_adjacent() {
        let gys: Vec<u8> = GRAVEYARD_XY.iter().map(|&(x, y)| idx(x, y)).collect();
        assert_eq!(gys.len(), 10);
        for &(x, y) in &GRAVEYARD_XY {
            assert!(GRAVEYARD_XY.contains(&(9 - x, 9 - y)), "{x},{y} has no rotation partner");
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
        assert_eq!(cell_name(NECRO_START[0]), "c2");
        assert_eq!(cell_name(NECRO_START[1]), "h7");
        for i in 0..NUM_CELLS as u8 {
            assert_eq!(parse_cell(&cell_name(i)), Some(i));
        }
    }
}
