//! In-process Hex AZ net via candle — plain AlphaZero: a ResNet trunk, one
//! spatial policy head (169 raw logits, one per cell) and a scalar value head.
//! No latent state, no reifier, no encoding space: planes in, policy/value out.
//!
//! Mirrors `python/src/playground/model_server/hex_net.py`; the parameter names
//! here and there must stay in sync, since the checkpoints are the same
//! `.safetensors` files.
//!
//! Both heads live in the **canonical frame** (see `game::fill_planes_into`): the
//! net always sees itself as the player connecting top↔bottom, so policy logit
//! `k` means cell `canon_index(state, k)` on the real board.

use std::path::Path;

use candle_core::{DType, Device, Module, ModuleT, Tensor};
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear,
    VarBuilder,
};

use async_trait::async_trait;
use bytes::Bytes;
use playground_core::batch_search::MoveGame;
use playground_core::{Game, LatentGameState, ModelClient, PolicyCandidate};

use crate::game::{Hex, HexMove, HexState, N, NN, PLANES as IN_PLANES};
use crate::mcts::{Evaluator, POLICY_DIM};

struct ResBlock {
    c1: Conv2d,
    b1: BatchNorm,
    c2: Conv2d,
    b2: BatchNorm,
}

impl ResBlock {
    fn load(c: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            c1: conv2d(c, c, 3, cfg, vb.pp("c1"))?,
            b1: batch_norm(c, BatchNormConfig::default(), vb.pp("b1"))?,
            c2: conv2d(c, c, 3, cfg, vb.pp("c2"))?,
            b2: batch_norm(c, BatchNormConfig::default(), vb.pp("b2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.b1.forward_t(&self.c1.forward(x)?, false)?.relu()?;
        let y = self.b2.forward_t(&self.c2.forward(&y)?, false)?;
        x.add(&y)?.relu()
    }
}

struct AzNet {
    stem: Conv2d,
    stem_bn: BatchNorm,
    blocks: Vec<ResBlock>,
    pol_conv: Conv2d,
    val_conv: Conv2d,
    val_bn: BatchNorm,
    val_fc1: Linear,
    val_fc2: Linear,
}

impl AzNet {
    fn load(channels: usize, blocks_n: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let cfg1 = Conv2dConfig::default(); // 1×1
        let mut blocks = Vec::with_capacity(blocks_n);
        let blocks_vb = vb.pp("blocks");
        for i in 0..blocks_n {
            blocks.push(ResBlock::load(channels, blocks_vb.pp(i.to_string()))?);
        }
        Ok(Self {
            stem: conv2d(IN_PLANES, channels, 3, cfg, vb.pp("stem"))?,
            stem_bn: batch_norm(channels, BatchNormConfig::default(), vb.pp("stem_bn"))?,
            blocks,
            // One logit per cell: (B,1,13,13) → (B,169), index r*13+c.
            pol_conv: conv2d(channels, 1, 3, cfg, vb.pp("pol_conv"))?,
            val_conv: conv2d(channels, 1, 1, cfg1, vb.pp("val_conv"))?,
            val_bn: batch_norm(1, BatchNormConfig::default(), vb.pp("val_bn"))?,
            val_fc1: linear(NN, 64, vb.pp("val_fc1"))?,
            val_fc2: linear(64, 1, vb.pp("val_fc2"))?,
        })
    }

    fn trunk(&self, planes: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.stem_bn.forward_t(&self.stem.forward(planes)?, false)?.relu()?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        Ok(x)
    }

    /// (B,2,13,13) → (values[B], policy_logits[B*169]).
    fn value_policy_batch(&self, planes: &Tensor) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
        let b = planes.dim(0)?;
        let x = self.trunk(planes)?;
        let policy = self.pol_conv.forward(&x)?.reshape((b, POLICY_DIM))?;
        let policy = policy.flatten_all()?.to_vec1::<f32>()?;
        let v = self.val_bn.forward_t(&self.val_conv.forward(&x)?, false)?.relu()?.flatten_from(1)?;
        let v = self.val_fc1.forward(&v)?.relu()?;
        let vals = self.val_fc2.forward(&v)?.tanh()?.flatten_all()?.to_vec1::<f32>()?;
        Ok((vals, policy))
    }
}

/// Trunk size, read from the checkpoint's sibling `config.json` (written by the
/// learner). Falls back to the defaults the learner also uses.
fn read_arch(checkpoint: &Path) -> (usize, usize) {
    let cfg = checkpoint.with_file_name("config.json");
    if let Ok(text) = std::fs::read_to_string(&cfg) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let arch = &v["arch"];
            let channels = arch["channels"].as_u64().unwrap_or(64) as usize;
            let blocks = arch["blocks"].as_u64().unwrap_or(6) as usize;
            return (channels, blocks);
        }
    }
    (64, 6)
}

/// A compute device for the in-process net. `cuda` needs the `cuda` feature.
pub fn pick_device(cuda: bool) -> anyhow::Result<Device> {
    if cuda {
        #[cfg(feature = "cuda")]
        {
            return Ok(Device::new_cuda(0)?);
        }
        #[cfg(not(feature = "cuda"))]
        {
            anyhow::bail!("cuda requested but this binary was built without the `cuda` feature");
        }
    }
    Ok(Device::Cpu)
}

/// The Hex AZ net running in-process via candle.
pub struct LocalModel {
    net: AzNet,
    device: Device,
    model_version: String,
}

impl LocalModel {
    pub fn load(checkpoint: &str) -> anyhow::Result<Self> {
        Self::load_on(checkpoint, Device::Cpu)
    }

    pub fn load_on(checkpoint: &str, device: Device) -> anyhow::Result<Self> {
        let path = Path::new(checkpoint);
        let (channels, blocks) = read_arch(path);
        // SAFETY: the mmap outlives the load; tensors are copied out of it.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)? };
        let net = AzNet::load(channels, blocks, vb)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("hex");
        Ok(Self { net, device, model_version: format!("hex-{stem}") })
    }

    /// Batched forward over positions: (values, raw policy logits[169] each).
    pub fn value_policy_batch(&self, states: &[&HexState]) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let b = states.len();
        let stride = IN_PLANES * NN;
        let mut data = vec![0f32; b * stride];
        for (i, s) in states.iter().enumerate() {
            crate::game::fill_planes_into(s, &mut data[i * stride..(i + 1) * stride]);
        }
        let planes = Tensor::from_vec(data, (b, IN_PLANES, N, N), &self.device)?;
        Ok(self.net.value_policy_batch(&planes)?)
    }
}

/// The batched engine sees the net as a pure forward over pre-encoded planes; the
/// per-move priors are the game's masked softmax (`MoveGame::expand`).
impl playground_core::batch_search::BatchModel for LocalModel {
    fn forward(&self, planes: &[f32], b: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let t = Tensor::from_slice(planes, (b, IN_PLANES, N, N), &self.device)?;
        Ok(self.net.value_policy_batch(&t)?)
    }
}

/// Batch-1 forward for the sequential search (MoHex matches, interactive play).
impl Evaluator for LocalModel {
    fn evaluate(&self, state: &HexState) -> anyhow::Result<(f32, Vec<f32>)> {
        let (vals, logits) = self.value_policy_batch(&[state])?;
        Ok((vals[0], logits))
    }
}

/// Adapter for the engine server's interactive path, which is written against the
/// latent-space `ModelClient`. Hex is plain AZ, so there is nothing latent here:
/// the "latent" payload *is* the encoded state, `policy` enumerates real legal
/// moves with the net's priors, and `reify` just decodes the move — every edge
/// reifies trivially and the search degenerates to a plain AZ tree over real
/// positions. Same shim as amazons.
#[async_trait]
impl ModelClient<Hex> for LocalModel {
    async fn encode(&self, state: &HexState) -> anyhow::Result<LatentGameState> {
        Ok(LatentGameState::new(Hex::encode_state(state), vec![], self.model_version.clone()))
    }

    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        let state = Hex::decode_state(&lgs.payload)?;
        let (vals, _) = self.value_policy_batch(&[&state])?;
        Ok(vals[0] as f64)
    }

    async fn policy(&self, lgs: &LatentGameState, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>> {
        let state = Hex::decode_state(&lgs.payload)?;
        let (_vals, logits) = self.value_policy_batch(&[&state])?;
        let (mut moves, mut priors) = (Vec::new(), Vec::new());
        <Hex as MoveGame>::expand(&state, &logits, &mut moves, &mut priors);
        let mut order: Vec<usize> = (0..moves.len()).collect();
        order.sort_by(|&a, &b| priors[b].total_cmp(&priors[a]));
        let mut out = Vec::with_capacity(top_b as usize);
        for &i in order.iter().take(top_b as usize) {
            let child = <Hex as Game>::apply(&state, &moves[i]);
            out.push(PolicyCandidate {
                prior: priors[i] as f64,
                action_vector: Bytes::from(Hex::encode_move(&moves[i])),
                next_lgs: LatentGameState::new(
                    Hex::encode_state(&child),
                    vec![],
                    self.model_version.clone(),
                ),
            });
        }
        Ok(out)
    }

    async fn reify(
        &self,
        from_state: &HexState,
        _from_lgs: &LatentGameState,
        _to_lgs: &LatentGameState,
        action_seed: Option<&[u8]>,
        _noise_scale: Option<f64>,
    ) -> anyhow::Result<HexMove> {
        if let Some(seed) = action_seed {
            if let Ok(mv) = Hex::decode_move(seed) {
                if let HexMove::Play { r, c } = mv {
                    if from_state.cell_at(r as usize * N + c as usize) == 0 {
                        return Ok(mv);
                    }
                }
            }
        }
        Hex::legal_moves(from_state)
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no legal move from a non-terminal state"))
    }

    async fn model_version(&self) -> anyhow::Result<String> {
        Ok(self.model_version.clone())
    }
}
