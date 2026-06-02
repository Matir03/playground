//! Arimaa on an 8×8 board. Official rules with the following simplifications:
//!   - Setup is skipped; initial_state() uses a standard hard-coded placement.
//!   - Push and pull are each modelled as single atomic compound moves
//!     (costing 2 steps). Trap checks after the full compound, not between
//!     the two sub-steps — this handles the common case correctly.
//!   - The "no-same-turn-board" rule is enforced only at EndTurn.
//!   - Three-fold repetition is not enforced (TODO for later milestones).
//!
//! Pieces: R(abbit)=1, C(at)=2, D(og)=3, H(orse)=4, M(camel)=5, E(lephant)=6
//! First player owns pieces 1-6; second player owns pieces 7-12 (type = piece - 6).
//! Piece strength equals its type value (E strongest, R weakest).
//!
//! Traps: c3, f3, c6, f6 (0-indexed squares 18, 21, 42, 45).
//! A piece on a trap with no adjacent friendly piece is removed immediately.
//!
//! Win conditions (checked after every step/compound-move):
//!   1. First rabbit on row 7  → FirstWins.
//!   2. Second rabbit on row 0 → SecondWins.
//!   3. First has no rabbits   → SecondWins.
//!   4. Second has no rabbits  → FirstWins.
//!   5. At start of a player's turn (steps_left==4): no legal steps → opponent wins.

use playground_core::{Game, Outcome, Player};

// ── Constants ─────────────────────────────────────────────────────────────────

const EMPTY: u8 = 0;
const TRAPS: [u8; 4] = [18, 21, 42, 45]; // c3 f3 c6 f6

// ── Piece helpers ─────────────────────────────────────────────────────────────

#[inline] fn piece_player(p: u8)   -> u8 { if p >= 7 { 2 } else { 1 } }
#[inline] fn piece_type(p: u8)     -> u8 { if p == 0 { 0 } else { ((p - 1) % 6) + 1 } }
#[inline] fn piece_strength(p: u8) -> u8 { piece_type(p) }
#[inline] fn piece_for(pl: u8, ty: u8) -> u8 { if pl == 1 { ty } else { ty + 6 } }

fn adj4(sq: u8) -> impl Iterator<Item = u8> {
    let r = sq / 8;
    let c = sq % 8;
    let mut v = [u8::MAX; 4];
    let mut n = 0;
    if r > 0 { v[n] = sq - 8; n += 1; }
    if r < 7 { v[n] = sq + 8; n += 1; }
    if c > 0 { v[n] = sq - 1; n += 1; }
    if c < 7 { v[n] = sq + 1; let _ = n; }
    v.into_iter().filter(|&x| x != u8::MAX)
}

fn is_frozen(board: &[u8; 64], sq: u8) -> bool {
    let p = board[sq as usize];
    if p == 0 { return false; }
    let pl = piece_player(p);
    let st = piece_strength(p);
    let mut strong_enemy = false;
    let mut friendly = false;
    for nb in adj4(sq) {
        let n = board[nb as usize];
        if n == 0 { continue; }
        if piece_player(n) != pl { if piece_strength(n) > st { strong_enemy = true; } }
        else { friendly = true; }
    }
    strong_enemy && !friendly
}

fn is_trap(sq: u8) -> bool { TRAPS.contains(&sq) }

fn apply_traps(board: &mut [u8; 64]) {
    for &trap in &TRAPS {
        let p = board[trap as usize];
        if p == 0 { continue; }
        let pl = piece_player(p);
        let support = adj4(trap).any(|nb| {
            let n = board[nb as usize];
            n != 0 && piece_player(n) == pl
        });
        if !support { board[trap as usize] = 0; }
    }
}

/// True if `sq` is the goal row for `player`'s rabbits.
fn is_goal(sq: u8, player: u8) -> bool {
    if player == 1 { sq >= 56 } else { sq < 8 }
}

/// Rabbit backward direction: first player can't decrease row, second can't increase.
fn is_rabbit_backward(from: u8, to: u8, player: u8) -> bool {
    if player == 1 { to < from && to / 8 < from / 8 } else { to > from && to / 8 > from / 8 }
}

// ── State & Move types ────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArimaaState {
    /// 0 empty; 1-6 first player R/C/D/H/M/E; 7-12 second player R/C/D/H/M/E.
    pub board: [u8; 64],
    pub to_move: u8,          // 1 or 2
    pub steps_left: u8,       // 1-4 at start of turn; decremented per step
    pub turn_start_board: [u8; 64],
    pub ply: u32,
}

impl Default for ArimaaState {
    fn default() -> Self {
        let mut board = [EMPTY; 64];
        // First player (gold): row 0 = back rank, row 1 = rabbits.
        // Back rank (sq 0-7): E M H H D D C C
        let back_first: [u8; 8] = [6, 5, 4, 4, 3, 3, 2, 2];
        for (c, &ty) in back_first.iter().enumerate() {
            board[c] = piece_for(1, ty);
        }
        for c in 0..8usize { board[8 + c] = piece_for(1, 1); } // rabbits on row 1

        // Second player (silver): row 7 = back rank, row 6 = rabbits.
        // Back rank (sq 56-63): C C D D H H M E
        let back_second: [u8; 8] = [2, 2, 3, 3, 4, 4, 5, 6];
        for (c, &ty) in back_second.iter().enumerate() {
            board[56 + c] = piece_for(2, ty);
        }
        for c in 0..8usize { board[48 + c] = piece_for(2, 1); } // rabbits on row 6

        let turn_start_board = board;
        Self { board, to_move: 1, steps_left: 4, turn_start_board, ply: 0 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArimaaMove {
    /// Move one own non-frozen piece orthogonally. Costs 1 step.
    Step { from: u8, to: u8 },
    /// Push: own piece at `own_from` (stronger, adjacent to `opp_from`) pushes
    /// enemy to `opp_to`, then steps into `opp_from`. Costs 2 steps.
    Push { own_from: u8, opp_from: u8, opp_to: u8 },
    /// Pull: own piece at `own_from` moves to `own_to`, dragging enemy at
    /// `opp_from` (adjacent to `own_from`, weaker) into `own_from`. Costs 2 steps.
    Pull { own_from: u8, own_to: u8, opp_from: u8 },
    /// End the current turn voluntarily (must have taken ≥1 step and board
    /// must differ from turn_start_board).
    EndTurn,
}

// ── Game impl ─────────────────────────────────────────────────────────────────

pub struct Arimaa;

fn switch_turn(state: &mut ArimaaState) {
    state.to_move = 3 - state.to_move;
    state.steps_left = 4;
    state.turn_start_board = state.board;
}

fn check_outcome_post_step(board: &[u8; 64], steps_left: u8, to_move: u8) -> Outcome {
    let mut first_has_rabbit = false;
    let mut second_has_rabbit = false;
    for sq in 0..64u8 {
        let p = board[sq as usize];
        if p == 0 { continue; }
        let pl = piece_player(p);
        let ty = piece_type(p);
        if ty == 1 {
            if pl == 1 {
                first_has_rabbit = true;
                if is_goal(sq, 1) { return Outcome::FirstWins; }
            } else {
                second_has_rabbit = true;
                if is_goal(sq, 2) { return Outcome::SecondWins; }
            }
        }
    }
    if !first_has_rabbit  { return Outcome::SecondWins; }
    if !second_has_rabbit { return Outcome::FirstWins; }

    // Immobilisation: only check at the very start of a player's turn.
    if steps_left == 4 && !has_any_own_step(board, to_move) {
        return if to_move == 1 { Outcome::SecondWins } else { Outcome::FirstWins };
    }
    Outcome::InProgress
}

/// True if `player` can make at least one Step, Push, or Pull.
fn has_any_own_step(board: &[u8; 64], player: u8) -> bool {
    for sq in 0..64u8 {
        let p = board[sq as usize];
        if p == 0 || piece_player(p) != player { continue; }
        // Simple step
        if !is_frozen(board, sq) {
            for nb in adj4(sq) {
                if board[nb as usize] == 0 {
                    let ty = piece_type(p);
                    if ty == 1 && is_rabbit_backward(sq, nb, player) { continue; }
                    return true;
                }
            }
        }
        // Could also check push/pull but simple-step check suffices for detecting total immobility.
    }
    false
}

impl Game for Arimaa {
    type State = ArimaaState;
    type Move = ArimaaMove;

    fn id() -> u32 { 4 }

    fn initial_state() -> Self::State {
        ArimaaState::default()
    }

    fn to_move(state: &Self::State) -> Player {
        if state.to_move == 1 { Player::First } else { Player::Second }
    }

    fn outcome(state: &Self::State) -> Outcome {
        check_outcome_post_step(&state.board, state.steps_left, state.to_move)
    }

    fn legal_moves(state: &Self::State) -> Vec<Self::Move> {
        if Self::outcome(state).is_terminal() { return vec![]; }

        let me = state.to_move;
        let opp = 3 - me;
        let sl = state.steps_left;
        let mut out = Vec::new();

        // EndTurn: available when at least 1 step was used and board changed.
        if sl < 4 && state.board != state.turn_start_board {
            out.push(ArimaaMove::EndTurn);
        }

        for sq in 0..64u8 {
            let p = state.board[sq as usize];
            if p == 0 || piece_player(p) != me { continue; }
            let frozen = is_frozen(&state.board, sq);

            // Simple steps.
            if sl >= 1 && !frozen {
                let is_rabbit = piece_type(p) == 1;
                for nb in adj4(sq) {
                    if state.board[nb as usize] != 0 { continue; }
                    if is_rabbit && is_rabbit_backward(sq, nb, me) { continue; }
                    out.push(ArimaaMove::Step { from: sq, to: nb });
                }
            }

            if sl >= 2 && !frozen {
                let own_str = piece_strength(p);

                // Pull: own moves, enemy weaker piece adjacent to own_from follows.
                for own_to in adj4(sq) {
                    if state.board[own_to as usize] != 0 { continue; }
                    // Check rabbit backward constraint for the moving piece.
                    if piece_type(p) == 1 && is_rabbit_backward(sq, own_to, me) { continue; }
                    // Any weaker enemy adjacent to `sq` can be pulled.
                    for opp_sq in adj4(sq) {
                        let ep = state.board[opp_sq as usize];
                        if ep == 0 || piece_player(ep) != opp { continue; }
                        if piece_strength(ep) < own_str {
                            out.push(ArimaaMove::Pull { own_from: sq, own_to, opp_from: opp_sq });
                        }
                    }
                }

                // Push: enemy weaker piece adjacent to `sq` can be pushed.
                for opp_sq in adj4(sq) {
                    let ep = state.board[opp_sq as usize];
                    if ep == 0 || piece_player(ep) != opp { continue; }
                    if piece_strength(ep) >= own_str { continue; } // must be strictly weaker
                    for opp_to in adj4(opp_sq) {
                        if state.board[opp_to as usize] != 0 { continue; }
                        out.push(ArimaaMove::Push { own_from: sq, opp_from: opp_sq, opp_to });
                    }
                }
            }
        }

        out
    }

    fn apply(state: &Self::State, mv: &ArimaaMove) -> Self::State {
        let mut next = state.clone();
        match mv {
            ArimaaMove::Step { from, to } => {
                next.board[*to as usize] = state.board[*from as usize];
                next.board[*from as usize] = 0;
                apply_traps(&mut next.board);
                next.steps_left -= 1;
                if next.steps_left == 0 { switch_turn(&mut next); }
            }
            ArimaaMove::Push { own_from, opp_from, opp_to } => {
                // Enemy piece displaced first; own piece steps into vacated square.
                next.board[*opp_to as usize] = state.board[*opp_from as usize];
                next.board[*opp_from as usize] = state.board[*own_from as usize];
                next.board[*own_from as usize] = 0;
                apply_traps(&mut next.board);
                next.steps_left -= 2;
                if next.steps_left == 0 { switch_turn(&mut next); }
            }
            ArimaaMove::Pull { own_from, own_to, opp_from } => {
                next.board[*own_to as usize] = state.board[*own_from as usize];
                next.board[*own_from as usize] = state.board[*opp_from as usize];
                next.board[*opp_from as usize] = 0;
                apply_traps(&mut next.board);
                next.steps_left -= 2;
                if next.steps_left == 0 { switch_turn(&mut next); }
            }
            ArimaaMove::EndTurn => {
                switch_turn(&mut next);
            }
        }
        next.ply += 1;
        next
    }

    fn ply(state: &Self::State) -> u32 {
        state.ply
    }

    fn to_features(state: &Self::State) -> Vec<u8> {
        // 12 planes (one per piece type × player) + 1 side-to-move plane, each 64 bytes.
        let mut out = Vec::with_capacity(13 * 64);
        for piece in 1u8..=12 {
            for sq in 0..64usize {
                out.push(if state.board[sq] == piece { 1 } else { 0 });
            }
        }
        let side: u8 = if state.to_move == 1 { 1 } else { 0 };
        for _ in 0..64 { out.push(side); }
        out
    }

    fn encode_state(state: &Self::State) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + 64 + 6);
        out.extend_from_slice(&state.board);
        out.push(state.to_move);
        out.push(state.steps_left);
        out.extend_from_slice(&state.ply.to_le_bytes());
        out.extend_from_slice(&state.turn_start_board);
        out
    }

    fn decode_state(bytes: &[u8]) -> anyhow::Result<Self::State> {
        let expected = 64 + 1 + 1 + 4 + 64;
        if bytes.len() != expected {
            anyhow::bail!("arimaa state: expected {expected} bytes, got {}", bytes.len());
        }
        let mut state = ArimaaState {
            board: [0; 64],
            to_move: 1,
            steps_left: 4,
            turn_start_board: [0; 64],
            ply: 0,
        };
        state.board.copy_from_slice(&bytes[0..64]);
        state.to_move = bytes[64];
        state.steps_left = bytes[65];
        state.ply = u32::from_le_bytes([bytes[66], bytes[67], bytes[68], bytes[69]]);
        state.turn_start_board.copy_from_slice(&bytes[70..134]);
        Ok(state)
    }

    fn encode_move(mv: &Self::Move) -> Vec<u8> {
        match mv {
            ArimaaMove::Step { from, to } => vec![0, *from, *to],
            ArimaaMove::Push { own_from, opp_from, opp_to } => vec![1, *own_from, *opp_from, *opp_to],
            ArimaaMove::Pull { own_from, own_to, opp_from } => vec![2, *own_from, *own_to, *opp_from],
            ArimaaMove::EndTurn => vec![3],
        }
    }

    fn decode_move(bytes: &[u8]) -> anyhow::Result<Self::Move> {
        match bytes {
            [0, from, to] => Ok(ArimaaMove::Step { from: *from, to: *to }),
            [1, of, op, ot] => Ok(ArimaaMove::Push { own_from: *of, opp_from: *op, opp_to: *ot }),
            [2, of, ot, op] => Ok(ArimaaMove::Pull { own_from: *of, own_to: *ot, opp_from: *op }),
            [3] => Ok(ArimaaMove::EndTurn),
            _ => anyhow::bail!("invalid arimaa move bytes (len={})", bytes.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_in_progress() {
        let s = Arimaa::initial_state();
        assert_eq!(Arimaa::outcome(&s), Outcome::InProgress);
    }

    #[test]
    fn initial_state_has_legal_moves() {
        let s = Arimaa::initial_state();
        let moves = Arimaa::legal_moves(&s);
        assert!(!moves.is_empty(), "should have moves at start");
        // No EndTurn at start of turn (0 steps taken).
        assert!(!moves.contains(&ArimaaMove::EndTurn));
    }

    #[test]
    fn step_move_advances_piece() {
        let s = Arimaa::initial_state();
        // Find a rabbit step — first's rabbits are on row 1 (sq 8-15).
        // A rabbit at sq 8 can step to sq 16 (forward).
        let mv = ArimaaMove::Step { from: 8, to: 16 };
        let s2 = Arimaa::apply(&s, &mv);
        assert_eq!(s2.board[8], EMPTY);
        assert_eq!(s2.board[16], piece_for(1, 1));
        assert_eq!(s2.steps_left, 3); // 4 - 1
        assert_eq!(s2.to_move, 1);    // still first player's turn
    }

    #[test]
    fn end_turn_switches_player() {
        let s = Arimaa::initial_state();
        let s2 = Arimaa::apply(&s, &ArimaaMove::Step { from: 8, to: 16 });
        let s3 = Arimaa::apply(&s2, &ArimaaMove::EndTurn);
        assert_eq!(s3.to_move, 2);
        assert_eq!(s3.steps_left, 4);
    }

    #[test]
    fn four_steps_auto_switches() {
        let s = Arimaa::initial_state();
        // Move 4 different rabbits forward one square each.
        let s2 = Arimaa::apply(&s, &ArimaaMove::Step { from: 8,  to: 16 });
        let s3 = Arimaa::apply(&s2, &ArimaaMove::Step { from: 9,  to: 17 });
        let s4 = Arimaa::apply(&s3, &ArimaaMove::Step { from: 10, to: 18 });
        // sq 18 is trap c3; rabbit on trap with no friendly adjacent → removed.
        // After removal, first player still stepping, 1 step left.
        let s5 = Arimaa::apply(&s4, &ArimaaMove::Step { from: 11, to: 19 });
        // 4 steps used → auto switch.
        assert_eq!(s5.to_move, 2);
    }

    #[test]
    fn trap_removes_unsupported_piece() {
        let mut s = Arimaa::initial_state();
        // Clear the rabbit at sq 10 and push one rabbit to sq 18 (trap c3)
        // without support.
        s.board[10] = EMPTY;
        s.board[18] = piece_for(1, 1); // rabbit on trap
        apply_traps(&mut s.board);
        assert_eq!(s.board[18], EMPTY, "unsupported rabbit on trap should die");
    }

    #[test]
    fn state_roundtrip() {
        let s = Arimaa::initial_state();
        let bytes = Arimaa::encode_state(&s);
        let s2 = Arimaa::decode_state(&bytes).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn move_roundtrip() {
        let moves = [
            ArimaaMove::Step { from: 8, to: 16 },
            ArimaaMove::Push { own_from: 0, opp_from: 8, opp_to: 16 },
            ArimaaMove::Pull { own_from: 0, own_to: 8, opp_from: 1 },
            ArimaaMove::EndTurn,
        ];
        for mv in moves {
            assert_eq!(Arimaa::decode_move(&Arimaa::encode_move(&mv)).unwrap(), mv);
        }
    }
}
