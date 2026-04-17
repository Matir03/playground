//! Pure algorithm core. No I/O, no tokio runtime, no tonic.
//!
//! The engine-server and model-client crates wrap this with transport.
//! See `docs/architecture.md` for the boundary rationale.

pub mod game;
pub mod gs;
pub mod model;
pub mod mcts;
pub mod recording;

pub use game::{Game, Outcome, Player};
pub use gs::Gs;
pub use model::{ModelClient, PolicyCandidate};
