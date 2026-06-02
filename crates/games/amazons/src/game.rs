//! Game of the Amazons on a 10×10 board. No-move-loses.
//!
//! Each turn: pick one of your four queens, queen-move it to an empty square,
//! then shoot an arrow (also a queen-move) from the new position to any other
//! empty square (the queen's origin square counts as empty for arrow purposes).
//! Arrows are permanent blockers.

use playground_core::{Game, Outcome, Player};

pub const N: usize = 10;

const EMPTY: u8 = 0;
const FIRST: u8 = 1;
const SECOND: u8 = 2;
const ARROW: u8 = 3;

type Board = [[u8; N]; N];

/// Standard opening position (row 0 = bottom for first player).
///   First  (1): (0,3) (0,6) (3,0) (3,9)
///   Second (2): (6,0) (6,9) (9,3) (9,6)
fn initial_board() -> Board {
    let mut b = [[EMPTY; N]; N];
    for (r, c) in [(0, 3), (0, 6), (3, 0), (3, 9)] {
        b[r][c] = FIRST;
    }
    for (r, c) in [(6, 0), (6, 9), (9, 3), (9, 6)] {
        b[r][c] = SECOND;
    }
    b
}

#[derive(Clone, Debug)]
pub struct AmazonsState {
    pub board: Board,
    pub to_move: u8, // 1 or 2
    pub ply: u16,
}

impl Default for AmazonsState {
    fn default() -> Self {
        Self {
            board: initial_board(),
            to_move: 1,
            ply: 0,
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

const DIRS: [(i32, i32); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

/// Squares reachable by a queen-slide from (r, c), treating `blank` as empty.
fn queen_reach(board: &Board, r: usize, c: usize, blank: Option<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(20);
    for (dr, dc) in DIRS {
        let mut rr = r as i32 + dr;
        let mut cc = c as i32 + dc;
        while rr >= 0 && rr < N as i32 && cc >= 0 && cc < N as i32 {
            let sq = (rr as usize, cc as usize);
            let cell = board[sq.0][sq.1];
            if cell != EMPTY && blank != Some(sq) {
                break;
            }
            out.push(sq);
            rr += dr;
            cc += dc;
        }
    }
    out
}

fn has_any_move(state: &AmazonsState) -> bool {
    let me = state.to_move;
    for r in 0..N {
        for c in 0..N {
            if state.board[r][c] != me {
                continue;
            }
            let queen_moves = queen_reach(&state.board, r, c, None);
            for (tr, tc) in queen_moves {
                let mut b = state.board;
                b[r][c] = EMPTY;
                b[tr][tc] = me;
                if !queen_reach(&b, tr, tc, None).is_empty() {
                    return true;
                }
            }
        }
    }
    false
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
        if !has_any_move(state) {
            return vec![];
        }
        let me = state.to_move;
        let mut out = Vec::new();
        for r in 0..N {
            for c in 0..N {
                if state.board[r][c] != me {
                    continue;
                }
                for (tr, tc) in queen_reach(&state.board, r, c, None) {
                    // After queen moves, origin (r,c) is blank for arrow purposes.
                    for (ar, ac) in queen_reach(&state.board, tr, tc, Some((r, c))) {
                        out.push(AmazonsMove {
                            from_r: r as u8,
                            from_c: c as u8,
                            to_r: tr as u8,
                            to_c: tc as u8,
                            arrow_r: ar as u8,
                            arrow_c: ac as u8,
                        });
                    }
                }
            }
        }
        out
    }

    fn apply(state: &Self::State, mv: &AmazonsMove) -> Self::State {
        let mut next = state.clone();
        let piece = state.to_move;
        next.board[mv.from_r as usize][mv.from_c as usize] = EMPTY;
        next.board[mv.to_r as usize][mv.to_c as usize] = piece;
        next.board[mv.arrow_r as usize][mv.arrow_c as usize] = ARROW;
        next.to_move = 3 - state.to_move;
        next.ply += 1;
        next
    }

    fn ply(state: &Self::State) -> u32 {
        state.ply as u32
    }

    fn to_features(state: &Self::State) -> Vec<u8> {
        // Planes: first queens | second queens | arrows | side-to-move constant
        let mut out = Vec::with_capacity(4 * N * N);
        for plane in [FIRST, SECOND, ARROW] {
            for r in 0..N {
                for c in 0..N {
                    out.push(if state.board[r][c] == plane { 1 } else { 0 });
                }
            }
        }
        let side: u8 = if state.to_move == 1 { 1 } else { 0 };
        for _ in 0..N * N {
            out.push(side);
        }
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        let mut out = Vec::with_capacity(N * N + 3);
        for r in 0..N {
            for c in 0..N {
                out.push(state.board[r][c]);
            }
        }
        out.push(state.to_move);
        out.extend_from_slice(&state.ply.to_le_bytes());
        out
    }

    fn decode_state(bytes: &[u8]) -> anyhow::Result<Self::State> {
        let expected = N * N + 3;
        if bytes.len() != expected {
            anyhow::bail!("amazons state: expected {expected} bytes, got {}", bytes.len());
        }
        let mut state = AmazonsState {
            board: [[EMPTY; N]; N],
            to_move: 1,
            ply: 0,
        };
        let mut i = 0;
        for r in 0..N {
            for c in 0..N {
                state.board[r][c] = bytes[i];
                i += 1;
            }
        }
        state.to_move = bytes[i];
        i += 1;
        state.ply = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        Ok(state)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_board_in_progress() {
        let s = Amazons::initial_state();
        assert_eq!(Amazons::outcome(&s), Outcome::InProgress);
    }

    #[test]
    fn initial_moves_nonempty() {
        let s = Amazons::initial_state();
        assert!(!Amazons::legal_moves(&s).is_empty());
    }

    #[test]
    fn apply_move_places_pieces() {
        let s = Amazons::initial_state();
        let mv = Amazons::legal_moves(&s)[0];
        let s2 = Amazons::apply(&s, &mv);
        assert_eq!(s2.to_move, 2);
        assert_eq!(s2.board[mv.from_r as usize][mv.from_c as usize], EMPTY);
        assert_eq!(s2.board[mv.to_r as usize][mv.to_c as usize], FIRST);
        assert_eq!(s2.board[mv.arrow_r as usize][mv.arrow_c as usize], ARROW);
    }

    #[test]
    fn state_roundtrip() {
        let s = Amazons::initial_state();
        let mv = Amazons::legal_moves(&s)[0];
        let s2 = Amazons::apply(&s, &mv);
        let bytes = Amazons::encode_state(&s2);
        let s3 = Amazons::decode_state(&bytes).unwrap();
        assert_eq!(s2.ply, s3.ply);
        assert_eq!(s2.to_move, s3.to_move);
        assert_eq!(s2.board, s3.board);
    }
}
