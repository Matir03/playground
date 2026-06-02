//! Connect 4: 7 columns × 6 rows. First to line up four in a row wins.

use playground_core::{Game, Outcome, Player};

pub const C4_ROWS: usize = 6;
pub const C4_COLS: usize = 7;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Connect4State {
    // cells[row][col]: 0 empty, 1 first, 2 second. row 0 is the bottom.
    pub cells: [[u8; C4_COLS]; C4_ROWS],
    pub ply: u16,
}

impl Default for Connect4State {
    fn default() -> Self {
        Self {
            cells: [[0; C4_COLS]; C4_ROWS],
            ply: 0,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Connect4Move {
    pub col: u8,
}

pub struct Connect4;

impl Game for Connect4 {
    type State = Connect4State;
    type Move = Connect4Move;

    fn id() -> u32 {
        1
    }

    fn initial_state() -> Self::State {
        Connect4State::default()
    }

    fn to_move(state: &Self::State) -> Player {
        if state.ply % 2 == 0 {
            Player::First
        } else {
            Player::Second
        }
    }

    fn outcome(state: &Self::State) -> Outcome {
        match winner(state) {
            Some(1) => Outcome::FirstWins,
            Some(2) => Outcome::SecondWins,
            _ if state.ply as usize >= C4_ROWS * C4_COLS => Outcome::Draw,
            _ => Outcome::InProgress,
        }
    }

    fn legal_moves(state: &Self::State) -> Vec<Self::Move> {
        if Self::outcome(state).is_terminal() {
            return vec![];
        }
        (0..C4_COLS as u8)
            .filter(|&c| state.cells[C4_ROWS - 1][c as usize] == 0)
            .map(|col| Connect4Move { col })
            .collect()
    }

    fn apply(state: &Self::State, mv: &Self::Move) -> Self::State {
        let mut next = state.clone();
        let piece = if state.ply % 2 == 0 { 1u8 } else { 2 };
        let col = mv.col as usize;
        for r in 0..C4_ROWS {
            if next.cells[r][col] == 0 {
                next.cells[r][col] = piece;
                next.ply += 1;
                return next;
            }
        }
        panic!("illegal Connect4 move: column {col} is full");
    }

    fn ply(state: &Self::State) -> u32 {
        state.ply as u32
    }

    fn to_features(state: &Self::State) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 * C4_ROWS * C4_COLS);
        for piece in [1u8, 2] {
            for r in 0..C4_ROWS {
                for c in 0..C4_COLS {
                    out.push(if state.cells[r][c] == piece { 1 } else { 0 });
                }
            }
        }
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        let mut out = Vec::with_capacity(C4_ROWS * C4_COLS + 2);
        for r in 0..C4_ROWS {
            for c in 0..C4_COLS {
                out.push(state.cells[r][c]);
            }
        }
        out.extend_from_slice(&state.ply.to_le_bytes());
        out
    }

    fn decode_state(bytes: &[u8]) -> anyhow::Result<Self::State> {
        let expected = C4_ROWS * C4_COLS + 2;
        if bytes.len() != expected {
            anyhow::bail!("connect4 state: expected {expected} bytes, got {}", bytes.len());
        }
        let mut state = Connect4State::default();
        let mut i = 0;
        for r in 0..C4_ROWS {
            for c in 0..C4_COLS {
                state.cells[r][c] = bytes[i];
                i += 1;
            }
        }
        state.ply = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        Ok(state)
    }

    fn encode_move(mv: &Self::Move) -> Vec<u8> {
        vec![mv.col]
    }

    fn decode_move(bytes: &[u8]) -> anyhow::Result<Self::Move> {
        if bytes.len() != 1 {
            anyhow::bail!("connect4 move: expected 1 byte, got {}", bytes.len());
        }
        Ok(Connect4Move { col: bytes[0] })
    }
}

fn winner(state: &Connect4State) -> Option<u8> {
    const DIRS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    for r in 0..C4_ROWS as i32 {
        for c in 0..C4_COLS as i32 {
            let p = state.cells[r as usize][c as usize];
            if p == 0 {
                continue;
            }
            for (dr, dc) in DIRS {
                let mut count = 1;
                for k in 1..4 {
                    let rr = r + dr * k;
                    let cc = c + dc * k;
                    if rr < 0 || rr >= C4_ROWS as i32 || cc < 0 || cc >= C4_COLS as i32 {
                        break;
                    }
                    if state.cells[rr as usize][cc as usize] != p {
                        break;
                    }
                    count += 1;
                }
                if count >= 4 {
                    return Some(p);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_board_has_seven_moves() {
        let s = Connect4::initial_state();
        assert_eq!(Connect4::legal_moves(&s).len(), 7);
        assert_eq!(Connect4::outcome(&s), Outcome::InProgress);
    }

    #[test]
    fn vertical_four_wins() {
        let mut s = Connect4::initial_state();
        // First drops 4 in col 0, Second alternates in col 1.
        for _ in 0..3 {
            s = Connect4::apply(&s, &Connect4Move { col: 0 });
            s = Connect4::apply(&s, &Connect4Move { col: 1 });
        }
        s = Connect4::apply(&s, &Connect4Move { col: 0 });
        assert_eq!(Connect4::outcome(&s), Outcome::FirstWins);
    }

    #[test]
    fn state_roundtrip() {
        let mut s = Connect4::initial_state();
        s = Connect4::apply(&s, &Connect4Move { col: 3 });
        s = Connect4::apply(&s, &Connect4Move { col: 3 });
        let bytes = Connect4::encode_state(&s);
        let s2 = Connect4::decode_state(&bytes).unwrap();
        assert_eq!(s, s2);
    }
}
