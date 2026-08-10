//! Built-in (non-NN) move chooser.
//!
//! Lets the engine answer `RequestMove` without a model server — used for
//! strong baselines like the perfect Connect 4 solver. This is deliberately
//! separate from the MCTS+model path: a `Solver` is whatever can pick a move
//! for a position on its own.

use crate::Game;

/// Result of a built-in search.
#[derive(Clone, Debug)]
pub struct SearchOutcome<M> {
    /// The move to play. The caller guarantees `state` was non-terminal.
    pub best_move: M,
    /// Value for the side to move, in [-1, 1]: +1 forced win, 0 draw,
    /// -1 forced loss. Magnitude may encode distance-to-result.
    pub value: f64,
    /// Positions examined, for telemetry / UI.
    pub nodes: u64,
}

/// Per-move budget for a solver, as asked for by `RequestMove`. Built-in
/// solvers may ignore it; external engines usually map it onto their own
/// time/playout controls.
#[derive(Clone, Copy, Debug, Default)]
pub struct SolverBudget {
    pub max_millis: Option<u32>,
    pub max_visits: Option<u32>,
}

/// A game that can choose a move without a neural-network model.
pub trait Solver: Game {
    /// Choose a move for a non-terminal `state`.
    fn choose_move(state: &Self::State) -> SearchOutcome<Self::Move>;

    /// Choose a move using an *external* reference engine the game crate drives
    /// itself — MoHex over GTP for Hex, and the obvious home for GNU Go in go9.
    /// `None` (the default) means no such engine is configured, and the engine
    /// server turns that into a failed-precondition rather than silently
    /// falling back to [`Solver::choose_move`]: a caller that asked to play the
    /// reference engine must not be answered by a different one.
    fn choose_move_external(
        _state: &Self::State,
        _budget: SolverBudget,
    ) -> Option<anyhow::Result<SearchOutcome<Self::Move>>> {
        None
    }
}
