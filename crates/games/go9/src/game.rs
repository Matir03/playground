//! 9×9 Go. Chinese rules, komi 7.5, positional ko (one-step lookback).
//!
//! Outcome is determined only on two consecutive passes; territory scoring
//! uses flood-fill area counting (stones + empty points surrounded by one
//! colour).

use playground_core::{Game, Outcome, Player};

pub const N: usize = 9;
pub const KOMI: f64 = 7.5;

type Board = [[u8; N]; N]; // 0 empty, 1 black (first), 2 white (second)

#[derive(Clone, Debug)]
pub struct Go9State {
    pub board: Board,
    /// The board position immediately before the last move, for ko detection.
    pub prev_board: Option<Board>,
    /// 1 = black's turn, 2 = white's turn.
    pub to_move: u8,
    pub consecutive_passes: u8,
    pub ply: u16,
}

impl Default for Go9State {
    fn default() -> Self {
        Self {
            board: [[0; N]; N],
            prev_board: None,
            to_move: 1,
            consecutive_passes: 0,
            ply: 0,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Go9Move {
    Pass,
    Play { r: u8, c: u8 },
}

pub struct Go9;

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
        if state.to_move == 1 {
            Player::First
        } else {
            Player::Second
        }
    }

    fn outcome(state: &Self::State) -> Outcome {
        if state.consecutive_passes < 2 {
            return Outcome::InProgress;
        }
        let (black, white_raw) = score_area(&state.board);
        let white = white_raw as f64 + KOMI;
        match (black as f64).partial_cmp(&white) {
            Some(std::cmp::Ordering::Greater) => Outcome::FirstWins,
            Some(std::cmp::Ordering::Less) => Outcome::SecondWins,
            _ => Outcome::Draw,
        }
    }

    fn legal_moves(state: &Self::State) -> Vec<Self::Move> {
        if Self::outcome(state).is_terminal() {
            return vec![];
        }
        let mut moves = vec![Go9Move::Pass];
        for r in 0..N as u8 {
            for c in 0..N as u8 {
                if is_legal_play(state, r, c) {
                    moves.push(Go9Move::Play { r, c });
                }
            }
        }
        moves
    }

    fn apply(state: &Self::State, mv: &Self::Move) -> Self::State {
        let mut next = state.clone();
        let prev_board = state.board;
        match mv {
            Go9Move::Pass => {
                next.consecutive_passes += 1;
                next.prev_board = Some(prev_board);
            }
            Go9Move::Play { r, c } => {
                let (r, c) = (*r as usize, *c as usize);
                let color = next.to_move;
                next.board[r][c] = color;
                // Capture any opponent groups that now have no liberties.
                let opp = 3 - color;
                for (nr, nc) in neighbors(r, c) {
                    if next.board[nr][nc] == opp && !has_liberty(&next.board, nr, nc) {
                        remove_group(&mut next.board, nr, nc);
                    }
                }
                // Suicide is excluded by is_legal_play; this is a safety guard.
                if !has_liberty(&next.board, r, c) {
                    remove_group(&mut next.board, r, c);
                }
                next.consecutive_passes = 0;
                next.prev_board = Some(prev_board);
            }
        }
        next.to_move = 3 - state.to_move;
        next.ply += 1;
        next
    }

    fn to_features(state: &Self::State) -> Vec<u8> {
        // Planes: [own stones, opp stones, to-move constant]
        let own = state.to_move;
        let opp = 3 - own;
        let mut out = Vec::with_capacity(3 * N * N);
        for &color in &[own, opp] {
            for r in 0..N {
                for c in 0..N {
                    out.push(if state.board[r][c] == color { 1 } else { 0 });
                }
            }
        }
        let side_plane: u8 = if state.to_move == 1 { 1 } else { 0 };
        for _ in 0..N * N {
            out.push(side_plane);
        }
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        let board_bytes = N * N;
        let mut out = Vec::with_capacity(2 * board_bytes + 5);
        for r in 0..N {
            for c in 0..N {
                out.push(state.board[r][c]);
            }
        }
        if let Some(prev) = state.prev_board {
            out.push(1);
            for r in 0..N {
                for c in 0..N {
                    out.push(prev[r][c]);
                }
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
        let board_bytes = N * N;
        if bytes.len() < board_bytes + 4 {
            anyhow::bail!("go9 state: too short ({} bytes)", bytes.len());
        }
        let mut state = Go9State::default();
        let mut i = 0;
        for r in 0..N {
            for c in 0..N {
                state.board[r][c] = bytes[i];
                i += 1;
            }
        }
        let has_prev = bytes[i];
        i += 1;
        if has_prev == 1 {
            if bytes.len() < i + board_bytes + 4 {
                anyhow::bail!("go9 state: too short for prev board");
            }
            let mut prev = [[0u8; N]; N];
            for r in 0..N {
                for c in 0..N {
                    prev[r][c] = bytes[i];
                    i += 1;
                }
            }
            state.prev_board = Some(prev);
        }
        if bytes.len() < i + 4 {
            anyhow::bail!("go9 state: missing tail");
        }
        state.to_move = bytes[i];
        i += 1;
        state.consecutive_passes = bytes[i];
        i += 1;
        state.ply = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
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

fn neighbors(r: usize, c: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut v: [Option<(usize, usize)>; 4] = [None; 4];
    let mut idx = 0;
    if r > 0 {
        v[idx] = Some((r - 1, c));
        idx += 1;
    }
    if r + 1 < N {
        v[idx] = Some((r + 1, c));
        idx += 1;
    }
    if c > 0 {
        v[idx] = Some((r, c - 1));
        idx += 1;
    }
    if c + 1 < N {
        v[idx] = Some((r, c + 1));
        let _ = idx;
    }
    v.into_iter().flatten()
}

/// Returns (stones, liberties) for the connected group containing (r, c).
fn group_info(board: &Board, r: usize, c: usize) -> (Vec<(usize, usize)>, bool) {
    let color = board[r][c];
    let mut seen = [[false; N]; N];
    let mut stack = vec![(r, c)];
    let mut stones = vec![];
    let mut has_lib = false;
    while let Some((rr, cc)) = stack.pop() {
        if seen[rr][cc] {
            continue;
        }
        seen[rr][cc] = true;
        stones.push((rr, cc));
        for (nr, nc) in neighbors(rr, cc) {
            match board[nr][nc] {
                0 => has_lib = true,
                c if c == color && !seen[nr][nc] => stack.push((nr, nc)),
                _ => {}
            }
        }
    }
    (stones, has_lib)
}

fn has_liberty(board: &Board, r: usize, c: usize) -> bool {
    group_info(board, r, c).1
}

fn remove_group(board: &mut Board, r: usize, c: usize) {
    let (stones, _) = group_info(board, r, c);
    for (sr, sc) in stones {
        board[sr][sc] = 0;
    }
}

fn is_legal_play(state: &Go9State, r: u8, c: u8) -> bool {
    let (r, c) = (r as usize, c as usize);
    if state.board[r][c] != 0 {
        return false;
    }
    let color = state.to_move;
    let opp = 3 - color;

    // Simulate the placement.
    let mut b = state.board;
    b[r][c] = color;
    for (nr, nc) in neighbors(r, c) {
        if b[nr][nc] == opp && !has_liberty(&b, nr, nc) {
            remove_group(&mut b, nr, nc);
        }
    }
    // Suicide is illegal.
    if !has_liberty(&b, r, c) {
        return false;
    }
    // Positional ko: no immediate repetition of the previous board.
    if let Some(prev) = state.prev_board {
        if b == prev {
            return false;
        }
    }
    true
}

fn score_area(board: &Board) -> (u32, u32) {
    let mut black = 0u32;
    let mut white = 0u32;
    let mut visited = [[false; N]; N];

    // Count stones.
    for r in 0..N {
        for c in 0..N {
            match board[r][c] {
                1 => black += 1,
                2 => white += 1,
                _ => {}
            }
        }
    }

    // Flood-fill empty regions and assign territory.
    for start_r in 0..N {
        for start_c in 0..N {
            if board[start_r][start_c] != 0 || visited[start_r][start_c] {
                continue;
            }
            let mut stack = vec![(start_r, start_c)];
            let mut region = vec![];
            let mut touches = [false, false]; // [black, white]
            while let Some((rr, cc)) = stack.pop() {
                if visited[rr][cc] {
                    continue;
                }
                visited[rr][cc] = true;
                region.push((rr, cc));
                for (nr, nc) in neighbors(rr, cc) {
                    match board[nr][nc] {
                        0 if !visited[nr][nc] => stack.push((nr, nc)),
                        1 => touches[0] = true,
                        2 => touches[1] = true,
                        _ => {}
                    }
                }
            }
            let count = region.len() as u32;
            match touches {
                [true, false] => black += count,
                [false, true] => white += count,
                _ => {} // dame (contested) — counts for neither
            }
        }
    }
    (black, white)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_has_82_moves() {
        let s = Go9::initial_state();
        // 81 play moves + 1 pass
        assert_eq!(Go9::legal_moves(&s).len(), 82);
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
        assert_eq!(s.board[0][0], 0, "captured stone should be empty");
    }

    #[test]
    fn move_roundtrip() {
        for mv in [Go9Move::Pass, Go9Move::Play { r: 4, c: 4 }] {
            let bytes = Go9::encode_move(&mv);
            assert_eq!(Go9::decode_move(&bytes).unwrap(), mv);
        }
    }
}
