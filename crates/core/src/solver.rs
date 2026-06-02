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

/// A game that can choose a move without a neural-network model.
pub trait Solver: Game {
    /// Choose a move for a non-terminal `state`.
    fn choose_move(state: &Self::State) -> SearchOutcome<Self::Move>;
}
