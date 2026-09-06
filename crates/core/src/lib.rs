//! Pure algorithm core. No I/O, no tokio runtime, no tonic.
//!
//! The engine-server and model-client crates wrap this with transport.
//! See `docs/architecture.md` for the boundary rationale.

pub mod batch_search;
pub mod game;
pub mod latent_state;
pub mod model;
pub mod mcts;
pub mod sampled_mcts;
pub mod latent_mcts;
pub mod puct;
pub mod recording;
pub mod trt_cache;
pub mod solver;

pub use game::{Game, Outcome, Player};
pub use latent_state::LatentGameState;
pub use model::{LatentModel, MixturePolicy, ModelClient, MovePolicy, PolicyCandidate};
pub use puct::fpu;
pub use sampled_mcts::{SampledGame, SampledModel, SampledParams, SampledSearchResult, SearchStats};
pub use solver::{
    Analysis, AnalysisAux, AnalysisEdge, AnalysisNode, Analyzable, SearchOutcome, Solver,
    SolverBudget,
};
