//! MCTS in GS-space with KR-UCT + progressive widening and on-demand
//! reification. Stub — see docs/algorithm.md for the intended semantics.
//!
//! Filled in during M4. The shape of this module should not leak tonic or
//! tokio runtime concerns — all NN calls go through `ModelClient`.

use crate::game::Game;
use crate::gs::Gs;
use crate::model::ModelClient;

/// Search configuration. Field list will grow as the algorithm solidifies.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// B — number of (action, next_gs) candidates produced by the policy head
    /// per expansion.
    pub policy_top_b: u32,
    /// K — number of visits between reifications at the same node.
    /// Progressive widening will grow this with visit count.
    pub reification_interval: u32,
    /// Multiplier for exploration noise on reified siblings.
    pub sibling_noise_base: f64,
    /// UCT/KR-UCT exploration constant. Specific kernel params TBD.
    pub exploration_c: f64,
    /// Optional RNG seed for deterministic tests.
    pub seed: Option<u64>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            policy_top_b: 8,
            reification_interval: 8,
            sibling_noise_base: 0.25,
            exploration_c: 1.4,
            seed: None,
        }
    }
}

/// Tree node — only stubbed here.
#[derive(Debug, Default)]
pub struct Node {
    pub gs: Option<Gs>,
    pub visits: u32,
    pub value_sum: f64,
    pub prior: f64,
    pub children: Vec<u32>,
}

/// Run a search from `root_state`, returning the chosen move and a tree
/// suitable for recording.
///
/// TODO(M4): implement. For now this is a stub so the rest of the crate
/// compiles.
pub async fn search<G: Game, M: ModelClient<G>>(
    _game: std::marker::PhantomData<G>,
    _model: &M,
    _root_state: &G::State,
    _config: &SearchConfig,
) -> anyhow::Result<SearchResult<G>> {
    anyhow::bail!("mcts::search is not implemented yet (see docs/algorithm.md)")
}

#[derive(Debug)]
pub struct SearchResult<G: Game> {
    pub chosen: G::Move,
    pub nodes: Vec<Node>,
}
