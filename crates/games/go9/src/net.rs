//! In-process 9×9 Go net — runs inference inside the engine via candle (no gRPC).
//!
//! The same network as `python/.../model_server/go9_net.py`, reimplemented with
//! `candle` so the reification MCTS in `crates/core` calls `encode`/`eval`/
//! `policy`/`reify` as local function calls. Checkpoints are the exact
//! `.safetensors` Python writes; the architecture is read from the sibling
//! `config.json` (`{"arch": {"kind": "go9-resnet", "channels", "blocks"}}`).
//!
//! Contract (so GS bytes are interchangeable with Python):
//!   * planes are 3×9×9 (side-to-move stones, opponent stones, side-to-move
//!     constant), matching `Go9::to_features`;
//!   * GS is `GS_DIM` little-endian f32, shape `[GS_DIM]`;
//!   * an action vector is 16 little-endian f32 with slot 0 = the action index
//!     (0..80 = play r*9+c, 81 = pass);
//!   * reify is a structural action-index decode clamped to legal moves.

use std::path::Path;

use async_trait::async_trait;
use bytes::Bytes;
use candle_core::{DType, Device, Module, ModuleT, Tensor};
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear,
    VarBuilder,
};
use playground_core::{Game, LatentGameState, ModelClient, PolicyCandidate};

use crate::game::{Go9, Go9Move, Go9State, N};

pub const GS_DIM: usize = 128;
const CELLS: usize = N * N; // 81
pub const ACTIONS: usize = CELLS + 1; // 82: board points + pass
const PASS: usize = CELLS; // action index of pass
const ACTION_DIM: usize = 16;
const IN_PLANES: usize = 3;

/// One residual block (conv-bn-relu-conv-bn + skip), matching `_ResBlock`.
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

/// The full net: ResNet trunk + shared encoder/eval/policy heads.
struct Net {
    stem: Conv2d,
    stem_bn: BatchNorm,
    blocks: Vec<ResBlock>,
    enc_fc: Linear,
    val_fc1: Linear,
    val_fc2: Linear,
    pol_prior: Linear,
    pol_next: Linear,
}

impl Net {
    fn load(channels: usize, blocks_n: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let mut blocks = Vec::with_capacity(blocks_n);
        let blocks_vb = vb.pp("blocks");
        for i in 0..blocks_n {
            blocks.push(ResBlock::load(channels, blocks_vb.pp(i.to_string()))?);
        }
        Ok(Self {
            stem: conv2d(IN_PLANES, channels, 3, cfg, vb.pp("stem"))?,
            stem_bn: batch_norm(channels, BatchNormConfig::default(), vb.pp("stem_bn"))?,
            blocks,
            enc_fc: linear(channels * CELLS, GS_DIM, vb.pp("enc_fc"))?,
            val_fc1: linear(GS_DIM, 64, vb.pp("val_fc1"))?,
            val_fc2: linear(64, 1, vb.pp("val_fc2"))?,
            pol_prior: linear(GS_DIM, ACTIONS, vb.pp("pol_prior"))?,
            pol_next: linear(GS_DIM, ACTIONS * GS_DIM, vb.pp("pol_next"))?,
        })
    }

    /// planes (B,3,9,9) -> GS (B,GS_DIM), tanh-bounded.
    fn encode(&self, planes: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.stem_bn.forward_t(&self.stem.forward(planes)?, false)?.relu()?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let x = x.flatten_from(1)?;
        self.enc_fc.forward(&x)?.tanh()
    }

    /// GS (1,GS_DIM) -> scalar value in [-1, 1] for the side to move.
    fn value(&self, gs: &Tensor) -> candle_core::Result<f32> {
        let x = self.val_fc1.forward(gs)?.relu()?;
        let v = self.val_fc2.forward(&x)?.tanh()?;
        Ok(v.flatten_all()?.to_vec1::<f32>()?[0])
    }

    /// GS (1,GS_DIM) -> (value, full softmax priors over the 82 actions). The
    /// self-play forward; skips the pol_next matmul.
    fn value_policy(&self, gs: &Tensor) -> candle_core::Result<(f32, Vec<f32>)> {
        let x = self.val_fc1.forward(gs)?.relu()?;
        let v = self.val_fc2.forward(&x)?.tanh()?.flatten_all()?.to_vec1::<f32>()?[0];
        let pri = candle_nn::ops::softmax(&self.pol_prior.forward(gs)?, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok((v, pri))
    }

    /// Batched self-play forward: planes (B,3,9,9) -> (values[B], probs[B*82]).
    fn value_policy_batch(&self, planes: &Tensor) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
        let gs = self.encode(planes)?;
        let x = self.val_fc1.forward(&gs)?.relu()?;
        let v = self.val_fc2.forward(&x)?.tanh()?;
        let vals = v.flatten_all()?.to_vec1::<f32>()?;
        let pri = candle_nn::ops::softmax(&self.pol_prior.forward(&gs)?, 1)?;
        let probs = pri.flatten_all()?.to_vec1::<f32>()?;
        Ok((vals, probs))
    }

    /// GS (1,GS_DIM) -> (priors over 82, predicted next-GS per action (82,GS_DIM)).
    fn policy_heads(&self, gs: &Tensor) -> candle_core::Result<(Vec<f32>, Tensor)> {
        let priors = candle_nn::ops::softmax(&self.pol_prior.forward(gs)?, 1)?;
        let nexts = self.pol_next.forward(gs)?.tanh()?.reshape((ACTIONS, GS_DIM))?;
        Ok((priors.flatten_all()?.to_vec1::<f32>()?, nexts))
    }
}

fn read_arch(checkpoint: &Path) -> (usize, usize) {
    let cfg = checkpoint.with_file_name("config.json");
    if let Ok(text) = std::fs::read_to_string(&cfg) {
        // Python's json writes bare NaN/Infinity (invalid JSON); sanitize first.
        let text = text.replace("NaN", "null").replace("Infinity", "null");
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let arch = &v["arch"];
            if arch["kind"] == "go9-resnet" {
                return (
                    arch["channels"].as_u64().unwrap_or(96) as usize,
                    arch["blocks"].as_u64().unwrap_or(10) as usize,
                );
            }
        }
    }
    (96, 10)
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

/// A `ModelClient` that runs the Go 9×9 net in-process via candle.
pub struct LocalModel {
    net: Net,
    device: Device,
    model_version: String,
}

impl LocalModel {
    /// Load a `.safetensors` checkpoint on CPU (architecture from config.json).
    pub fn load(checkpoint: &str) -> anyhow::Result<Self> {
        Self::load_on(checkpoint, Device::Cpu)
    }

    /// Load a checkpoint onto a chosen device (see `pick_device`).
    pub fn load_on(checkpoint: &str, device: Device) -> anyhow::Result<Self> {
        let path = Path::new(checkpoint);
        let (channels, blocks) = read_arch(path);
        // SAFETY: the mmap lives for the VarBuilder's load; tensors are copied.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)? };
        let net = Net::load(channels, blocks, vb)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("local");
        Ok(Self { net, device, model_version: format!("go9-{stem}") })
    }

    /// Fill `out` (length `b*3*CELLS`) with the 3 side-to-move planes of `states`.
    fn fill_planes(states: &[&Go9State], out: &mut [f32]) {
        for (i, s) in states.iter().enumerate() {
            let own = s.to_move;
            let base = i * IN_PLANES * CELLS;
            let side: f32 = if s.to_move == 1 { 1.0 } else { 0.0 };
            for r in 0..N {
                for c in 0..N {
                    let cell = s.board[r][c];
                    let idx = r * N + c;
                    if cell == own {
                        out[base + idx] = 1.0;
                    } else if cell != 0 {
                        out[base + CELLS + idx] = 1.0;
                    }
                    out[base + 2 * CELLS + idx] = side;
                }
            }
        }
    }

    fn planes(&self, state: &Go9State) -> candle_core::Result<Tensor> {
        let mut data = vec![0f32; IN_PLANES * CELLS];
        Self::fill_planes(&[state], &mut data);
        Tensor::from_vec(data, (1, IN_PLANES, N, N), &self.device)
    }

    fn gs_tensor(&self, gs: &LatentGameState) -> candle_core::Result<Tensor> {
        Tensor::from_vec(f32_le(&gs.payload), (1, GS_DIM), &self.device)
    }

    fn gs_state(&self, payload: Vec<f32>) -> LatentGameState {
        LatentGameState::new(f32_to_bytes(&payload), vec![GS_DIM as i32], self.model_version.clone())
    }

    /// AlphaZero-style self-play forward: state -> (value, softmax over 82).
    /// `&self`-only so the self-play MCTS can call it from many native threads.
    pub fn policy_value(&self, state: &Go9State) -> anyhow::Result<(f32, [f32; ACTIONS])> {
        let gs = self.net.encode(&self.planes(state)?)?;
        let (v, pri) = self.net.value_policy(&gs)?;
        let mut arr = [0f32; ACTIONS];
        arr.copy_from_slice(&pri[..ACTIONS]);
        Ok((v, arr))
    }

    /// Batched self-play forward over concrete states.
    pub fn policy_value_batch(
        &self,
        states: &[&Go9State],
    ) -> anyhow::Result<Vec<(f32, [f32; ACTIONS])>> {
        let b = states.len();
        let mut data = vec![0f32; b * IN_PLANES * CELLS];
        Self::fill_planes(states, &mut data);
        let planes = Tensor::from_vec(data, (b, IN_PLANES, N, N), &self.device)?;
        let (vals, probs) = self.net.value_policy_batch(&planes)?;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let mut arr = [0f32; ACTIONS];
            arr.copy_from_slice(&probs[i * ACTIONS..(i + 1) * ACTIONS]);
            out.push((vals[i], arr));
        }
        Ok(out)
    }
}

#[async_trait]
impl ModelClient<Go9> for LocalModel {
    async fn encode(&self, state: &Go9State) -> anyhow::Result<LatentGameState> {
        let gs = self.net.encode(&self.planes(state)?)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(self.gs_state(gs))
    }

    async fn eval(&self, gs: &LatentGameState) -> anyhow::Result<f64> {
        Ok(self.net.value(&self.gs_tensor(gs)?)? as f64)
    }

    async fn policy(&self, gs: &LatentGameState, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>> {
        let t = self.gs_tensor(gs)?;
        let (priors, nexts) = self.net.policy_heads(&t)?;
        let mut order: Vec<usize> = (0..ACTIONS).collect();
        order.sort_by(|&a, &b| priors[b].total_cmp(&priors[a]));
        let mut out = Vec::with_capacity(top_b as usize);
        for &a in order.iter().take(top_b as usize) {
            let next_gs = nexts.get(a)?.to_vec1::<f32>()?;
            out.push(PolicyCandidate {
                prior: priors[a] as f64,
                action_vector: Bytes::from(action_vector_for(a)),
                next_gs: self.gs_state(next_gs),
            });
        }
        Ok(out)
    }

    async fn reify(
        &self,
        from_state: &Go9State,
        _from_gs: &LatentGameState,
        _to_gs: &LatentGameState,
        action_seed: Option<&[u8]>,
        _noise_scale: Option<f64>,
    ) -> anyhow::Result<Go9Move> {
        let legal = Go9::legal_moves(from_state);
        let decoded = action_seed.and_then(decode_action).map(action_to_move);
        // Clamp to a legal move: the decoded action if legal, else the first legal.
        if let Some(mv) = decoded {
            if legal.contains(&mv) {
                return Ok(mv);
            }
        }
        Ok(legal.into_iter().next().unwrap_or(Go9Move::Pass))
    }

    async fn model_version(&self) -> anyhow::Result<String> {
        Ok(self.model_version.clone())
    }
}

/// Map an action index to a Go move (0..80 = play r*9+c, 81 = pass).
pub fn action_to_move(a: usize) -> Go9Move {
    if a >= PASS {
        Go9Move::Pass
    } else {
        Go9Move::Play { r: (a / N) as u8, c: (a % N) as u8 }
    }
}

/// Map a Go move to its action index.
pub fn move_to_action(mv: &Go9Move) -> usize {
    match mv {
        Go9Move::Pass => PASS,
        Go9Move::Play { r, c } => *r as usize * N + *c as usize,
    }
}

fn action_vector_for(a: usize) -> Vec<u8> {
    let mut v = [0f32; ACTION_DIM];
    v[0] = a as f32;
    f32_to_bytes(&v)
}

fn decode_action(seed: &[u8]) -> Option<usize> {
    if seed.len() < 4 {
        return None;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&seed[..4]);
    let a = f32::from_le_bytes(b).round();
    (a >= 0.0 && (a as usize) < ACTIONS).then_some(a as usize)
}

fn f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Parity check against the Python net (run with `-- --nocapture`). Gated on
    // the checkpoint existing so CI without the data registry still passes.
    #[tokio::test]
    async fn parity_with_python_net() {
        let ckpt = "../../../data/models/go9/random/checkpoint.safetensors";
        if !Path::new(ckpt).exists() {
            eprintln!("skip: no checkpoint");
            return;
        }
        let m = LocalModel::load(ckpt).unwrap();
        let mut s = Go9::initial_state();
        for mv in [Go9Move::Play { r: 2, c: 2 }, Go9Move::Play { r: 6, c: 6 },
                   Go9Move::Play { r: 4, c: 4 }] {
            s = Go9::apply(&s, &mv);
        }
        let (v, probs) = m.policy_value(&s).unwrap();
        let psum: f32 = probs.iter().sum();
        eprintln!("value {v:.5}  policy_sum {psum:.5}  pass_prior {:.5}", probs[PASS]);
        assert!((psum - 1.0).abs() < 1e-3, "policy must be a distribution");
    }
}
