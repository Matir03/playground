//! ModelClient trait — the algorithm's view of the neural network.
//!
//! The only implementation that matters in production lives in the
//! `model-client` crate and calls the Python ModelService over gRPC.
//! Tests use a deterministic mock that returns fixed outputs.

use crate::game::Game;
use crate::latent_state::LatentGameState;
use async_trait::async_trait;
use bytes::Bytes;

#[derive(Clone, Debug)]
pub struct PolicyCandidate {
    pub prior: f64,
    pub action_vector: Bytes,
    pub next_lgs: LatentGameState,
}

// ── Reification-based latent-space algorithm (docs/algorithm.md) ───────────────
//
// The new algorithm's policy is a density over next-LGS, modelled as a
// diagonal-covariance Gaussian mixture (MDN). It is served through `LatentModel`
// (kept separate from the discrete `ModelClient` so the discrete games — go9,
// amazons — are unaffected). KR-PUCT samples candidate next-LGS from the
// mixture and uses its density as the PUCT prior.

/// A diagonal-covariance Gaussian mixture over next-LGS space — the MDN policy
/// head's output. `means`/`stds` are row-major `[k][dim]`.
#[derive(Clone, Debug)]
pub struct MixturePolicy {
    pub dim: usize,
    pub weights: Vec<f32>, // length k, sums to 1
    pub means: Vec<f32>,   // k*dim, row-major, in (-1, 1)
    pub stds: Vec<f32>,    // k*dim, row-major, > 0
}

impl MixturePolicy {
    pub fn k(&self) -> usize {
        self.weights.len()
    }

    /// log p(x) under the mixture (stable log-sum-exp over components). `x` has
    /// length `dim`.
    pub fn log_density(&self, x: &[f32]) -> f64 {
        let d = self.dim;
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();
        let mut logs: Vec<f64> = Vec::with_capacity(self.k());
        for c in 0..self.k() {
            let w = self.weights[c] as f64;
            if w <= 0.0 {
                logs.push(f64::NEG_INFINITY);
                continue;
            }
            let (mut quad, mut logdet) = (0.0f64, 0.0f64);
            for i in 0..d {
                let mu = self.means[c * d + i] as f64;
                let sd = (self.stds[c * d + i] as f64).max(1e-6);
                let z = (x[i] as f64 - mu) / sd;
                quad += z * z;
                logdet += sd.ln();
            }
            logs.push(w.ln() - 0.5 * (quad + d as f64 * ln_2pi) - logdet);
        }
        let m = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if m == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        m + logs.iter().map(|&l| (l - m).exp()).sum::<f64>().ln()
    }

    /// p(x) under the mixture.
    pub fn density(&self, x: &[f32]) -> f64 {
        self.log_density(x).exp()
    }

    /// Sample one next-LGS point from the mixture.
    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec<f32> {
        use rand_distr::StandardNormal;
        let d = self.dim;
        let r: f32 = rng.gen();
        let mut acc = 0.0f32;
        let mut comp = self.k().saturating_sub(1);
        for (j, &w) in self.weights.iter().enumerate() {
            acc += w;
            if r <= acc {
                comp = j;
                break;
            }
        }
        let mut out = vec![0f32; d];
        for i in 0..d {
            let n: f32 = rng.sample(StandardNormal);
            out[i] = self.means[comp * d + i] + self.stds[comp * d + i] * n;
        }
        out
    }
}

/// The reification-based algorithm's view of the network (docs/algorithm.md).
/// Distinct from `ModelClient`: the policy is an MDN over next-LGS and the
/// reifier decodes an LGS→LGS edge directly (no action-vector seed).
#[async_trait]
pub trait LatentModel<G: Game>: Send + Sync {
    async fn encode(&self, state: &G::State) -> anyhow::Result<LatentGameState>;
    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64>;
    /// MDN policy: a Gaussian mixture over next-LGS space.
    async fn policy(&self, lgs: &LatentGameState) -> anyhow::Result<MixturePolicy>;
    /// Decode the edge `from_lgs → to_lgs` into a concrete move legal from
    /// `from_state` (the engine remains authoritative on legality).
    async fn reify(
        &self,
        from_state: &G::State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
    ) -> anyhow::Result<G::Move>;
    async fn model_version(&self) -> anyhow::Result<String>;
}

#[async_trait]
pub trait ModelClient<G: Game>: Send + Sync {
    async fn encode(&self, state: &G::State) -> anyhow::Result<LatentGameState>;
    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64>;
    async fn policy(&self, lgs: &LatentGameState, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>>;

    /// Given `from_state` (whose encoding is `from_lgs`) and a target LGS,
    /// produce a concrete move that approximately realizes the transition.
    /// `action_seed` is the policy action-head output for this edge, optionally
    /// perturbed with `noise_scale`.
    async fn reify(
        &self,
        from_state: &G::State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
        action_seed: Option<&[u8]>,
        noise_scale: Option<f64>,
    ) -> anyhow::Result<G::Move>;

    /// Current weights version reported by the server. Engine stamps this
    /// onto each GameRecord.
    async fn model_version(&self) -> anyhow::Result<String>;
}
