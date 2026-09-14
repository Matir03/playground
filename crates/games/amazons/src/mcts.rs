//! Amazons search glue: the `MoveGame` impl (move-gen + planes + terminal) for the
//! batched self-play/eval engine, plus the `Params` knobs for eval-match. Eval now
//! runs on the same pipeline as self-play (`batch_search::run_eval_moves`), so there
//! is no separate concrete MCTS here.

use playground_core::batch_search::MoveGame;

use crate::game::{self, AmazonsMove, AmazonsState, CELLS, PLANES as PLANE_COUNT};
use crate::moves::POLICY_DIM;

/// Floats per encoded position (4 planes × 100 cells).
pub const PLANES: usize = PLANE_COUNT * CELLS;

impl MoveGame for crate::game::Amazons {
    type State = AmazonsState;
    type Move = AmazonsMove;
    const PLANES: usize = PLANES;
    const POLICY_DIM: usize = POLICY_DIM;

    fn initial_state() -> AmazonsState {
        <crate::game::Amazons as playground_core::Game>::initial_state()
    }
    fn to_move(state: &AmazonsState) -> u8 {
        state.to_move
    }
    fn apply(state: &AmazonsState, mv: &AmazonsMove) -> AmazonsState {
        game::apply(state, mv)
    }
    fn terminal_value(state: &AmazonsState) -> Option<f32> {
        // No legal move ⇒ side to move loses.
        if game::has_any_move(state) {
            None
        } else {
            Some(-1.0)
        }
    }
    fn encode_planes(state: &AmazonsState, out: &mut [f32]) {
        game::fill_planes_into(state, out);
    }
    fn expand(
        state: &AmazonsState,
        policy: &[f32],
        moves: &mut Vec<AmazonsMove>,
        priors: &mut Vec<f32>,
    ) {
        game::expand_into(state, policy, moves, priors);
    }
}

// ── Concrete PUCT MCTS (eval-match) ───────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Params {
    pub node_budget: u32,
    pub c_puct: f32,
    pub dir_alpha: f32,
    pub dir_eps: f32,
    pub temp_moves: u32,
    pub max_ply: u32,
}
