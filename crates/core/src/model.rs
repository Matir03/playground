//! ModelClient trait — the algorithm's view of the neural network.
//!
//! The only implementation that matters in production lives in the
//! `model-client` crate and calls the Python ModelService over gRPC.
//! Tests use a deterministic mock that returns fixed outputs.

use crate::game::Game;
use crate::gs::Gs;
use async_trait::async_trait;
use bytes::Bytes;

#[derive(Clone, Debug)]
pub struct PolicyCandidate {
    pub prior: f64,
    pub action_vector: Bytes,
    pub next_gs: Gs,
}

#[async_trait]
pub trait ModelClient<G: Game>: Send + Sync {
    async fn encode(&self, state: &G::State) -> anyhow::Result<Gs>;
    async fn eval(&self, gs: &Gs) -> anyhow::Result<f64>;
    async fn policy(&self, gs: &Gs, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>>;

    /// Given `from_state` (whose encoding is `from_gs`) and a target GS,
    /// produce a concrete move that approximately realizes the transition.
    /// `action_seed` is the policy action-head output for this edge, optionally
    /// perturbed with `noise_scale`.
    async fn reify(
        &self,
        from_state: &G::State,
        from_gs: &Gs,
        to_gs: &Gs,
        action_seed: Option<&[u8]>,
        noise_scale: Option<f64>,
    ) -> anyhow::Result<G::Move>;

    /// Current weights version reported by the server. Engine stamps this
    /// onto each GameRecord.
    async fn model_version(&self) -> anyhow::Result<String>;
}
