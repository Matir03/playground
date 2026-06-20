//! Pure algorithm core. No I/O, no tokio runtime, no tonic.
//!
//! The engine-server and model-client crates wrap this with transport.
//! See `docs/architecture.md` for the boundary rationale.

pub mod batch_search;
pub mod game;
pub mod latent_state;
pub mod model;
pub mod mcts;
pub mod latent_mcts;
pub mod recording;
pub mod solver;

pub use game::{Game, Outcome, Player};
pub use latent_state::LatentGameState;
pub use model::{LatentModel, MixturePolicy, ModelClient, PolicyCandidate};
pub use solver::{SearchOutcome, Solver};
