//! In-process Connect 4 net — runs inference inside the engine (no gRPC).
//!
//! This is the same network as `python/.../model_server/net.py` reimplemented
//! with `candle` so the reification MCTS in `crates/core` can call
//! `encode` / `eval` / `policy` / `reify` as local function calls instead of a
//! gRPC round-trip to the Python model server per node. The checkpoints are the
//! exact `.safetensors` files Python writes; we load them directly and read the
//! architecture from the sibling `config.json` (`{"arch": {...}}`), so the
//! `random`, `bc-perfect`, and larger `selfplay` ResNet checkpoints all load.
//!
//! Contract parity with the Python server (so LGS bytes are interchangeable):
//!   * planes are 2×6×7 (side-to-move's stones, opponent's), row 0 = bottom;
//!   * LGS is 64 little-endian f32 (`LGS_DIM`), shape `[64]`;
//!   * an action vector is 16 little-endian f32 with slot 0 = the column;
//!   * reify is a structural column decode clamped to legal moves.

use std::path::Path;

use async_trait::async_trait;
use bytes::Bytes;
use candle_core::{DType, Device, Module, ModuleT, Tensor};
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear,
    VarBuilder,
};
use playground_core::{LatentGameState, ModelClient, PolicyCandidate};

use crate::game::{Connect4, Connect4Move, Connect4State, C4_COLS, C4_ROWS};

const LGS_DIM: usize = 64;
const ACTION_DIM: usize = 16;
const CELLS: usize = C4_ROWS * C4_COLS;

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

/// The trunk: either the small 2-conv net or the AlphaZero-style ResNet.
enum Trunk {
    Small { c1: Conv2d, c2: Conv2d },
    Res { stem: Conv2d, stem_bn: BatchNorm, blocks: Vec<ResBlock> },
}

impl Trunk {
    fn forward(&self, planes: &Tensor) -> candle_core::Result<Tensor> {
        let x = match self {
            Trunk::Small { c1, c2 } => {
                let x = c1.forward(planes)?.relu()?;
                c2.forward(&x)?.relu()?
            }
            Trunk::Res { stem, stem_bn, blocks } => {
                let mut x = stem_bn.forward_t(&stem.forward(planes)?, false)?.relu()?;
                for blk in blocks {
                    x = blk.forward(&x)?;
                }
                x
            }
        };
        x.flatten_from(1) // (B, channels*cells)
    }
}

/// The full net: trunk + shared encoder/eval/policy heads (`_C4Base`).
struct Net {
    trunk: Trunk,
    enc_fc: Linear,
    val_fc1: Linear,
    val_fc2: Linear,
    pol_prior: Linear,
    pol_next: Linear,
}

impl Net {
    fn load(arch: &Arch, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let (trunk, trunk_features) = match arch {
            Arch::Small => {
                let trunk = Trunk::Small {
                    c1: conv2d(2, 32, 3, cfg, vb.pp("enc_conv1"))?,
                    c2: conv2d(32, 32, 3, cfg, vb.pp("enc_conv2"))?,
                };
                (trunk, 32 * CELLS)
            }
            Arch::Res { channels, blocks } => {
                let mut bs = Vec::with_capacity(*blocks);
                let blocks_vb = vb.pp("blocks");
                for i in 0..*blocks {
                    bs.push(ResBlock::load(*channels, blocks_vb.pp(i.to_string()))?);
                }
                let trunk = Trunk::Res {
                    stem: conv2d(2, *channels, 3, cfg, vb.pp("stem"))?,
                    stem_bn: batch_norm(*channels, BatchNormConfig::default(), vb.pp("stem_bn"))?,
                    blocks: bs,
                };
                (trunk, channels * CELLS)
            }
        };
        Ok(Self {
            trunk,
            enc_fc: linear(trunk_features, LGS_DIM, vb.pp("enc_fc"))?,
            val_fc1: linear(LGS_DIM, 64, vb.pp("val_fc1"))?,
            val_fc2: linear(64, 1, vb.pp("val_fc2"))?,
            pol_prior: linear(LGS_DIM, C4_COLS, vb.pp("pol_prior"))?,
            pol_next: linear(LGS_DIM, C4_COLS * LGS_DIM, vb.pp("pol_next"))?,
        })
    }

    /// planes (1,2,6,7) -> LGS (1,64), tanh-bounded.
    fn encode(&self, planes: &Tensor) -> candle_core::Result<Tensor> {
        self.enc_fc.forward(&self.trunk.forward(planes)?)?.tanh()
    }

    /// LGS (1,64) -> scalar value in [-1, 1] for the side to move.
    fn value(&self, lgs: &Tensor) -> candle_core::Result<f32> {
        let x = self.val_fc1.forward(lgs)?.relu()?;
        let v = self.val_fc2.forward(&x)?.tanh()?;
        v.flatten_all()?.to_vec1::<f32>().map(|v| v[0])
    }

    /// LGS (1,64) -> (value, full column priors over 7). The self-play forward:
    /// matches Python's `policy_value` (softmax over all columns, renormalised
    /// over legal moves by the caller). Skips the `pol_next` matmul, which only
    /// the reification policy head needs.
    fn value_policy(&self, lgs: &Tensor) -> candle_core::Result<(f32, Vec<f32>)> {
        let x = self.val_fc1.forward(lgs)?.relu()?;
        let v = self.val_fc2.forward(&x)?.tanh()?.flatten_all()?.to_vec1::<f32>()?[0];
        let pri = candle_nn::ops::softmax(&self.pol_prior.forward(lgs)?, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok((v, pri))
    }

    /// Batched self-play forward: planes (B,2,6,7) -> (values[B], probs[B*7]),
    /// the latter row-major softmax over the 7 columns. This is what the GPU
    /// inference server calls — coalescing many search threads' leaves into one
    /// forward is the whole point of the batched path.
    fn value_policy_batch(&self, planes: &Tensor) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
        let lgs = self.encode(planes)?; // (B,64)
        let x = self.val_fc1.forward(&lgs)?.relu()?;
        let v = self.val_fc2.forward(&x)?.tanh()?; // (B,1)
        let vals = v.flatten_all()?.to_vec1::<f32>()?;
        let pri = candle_nn::ops::softmax(&self.pol_prior.forward(&lgs)?, 1)?; // (B,7)
        let probs = pri.flatten_all()?.to_vec1::<f32>()?;
        Ok((vals, probs))
    }

    /// LGS (1,64) -> (column priors over 7, predicted next-LGS per column (7,64)).
    fn policy_heads(&self, lgs: &Tensor) -> candle_core::Result<(Vec<f32>, Tensor)> {
        let priors = candle_nn::ops::softmax(&self.pol_prior.forward(lgs)?, 1)?;
        let nexts = self.pol_next.forward(lgs)?.tanh()?.reshape((C4_COLS, LGS_DIM))?;
        Ok((priors.flatten_all()?.to_vec1::<f32>()?, nexts))
    }
}

/// Architecture descriptor read from the checkpoint's sibling `config.json`.
enum Arch {
    Small,
    Res { channels: usize, blocks: usize },
}

fn read_arch(checkpoint: &Path) -> Arch {
    let cfg = checkpoint.with_file_name("config.json");
    if let Ok(text) = std::fs::read_to_string(&cfg) {
        // Python's json writes bare NaN/Infinity (e.g. an unset winrate), which
        // is invalid JSON; serde rejects it. Sanitize so we can still read the
        // arch rather than silently falling back to the wrong architecture.
        let text = text.replace("NaN", "null").replace("Infinity", "null");
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let arch = &v["arch"];
            if arch["kind"] == "resnet" {
                return Arch::Res {
                    channels: arch["channels"].as_u64().unwrap_or(64) as usize,
                    blocks: arch["blocks"].as_u64().unwrap_or(5) as usize,
                };
            }
        }
    }
    Arch::Small
}

/// A `ModelClient` that runs the Connect 4 net in-process via candle.
pub struct LocalModel {
    net: Net,
    device: Device,
    model_version: String,
}

/// A compute device for the in-process net. `cuda` is only available when the
/// crate is built with the `cuda` feature (it pulls candle's CUDA kernels).
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

impl LocalModel {
    /// Load a `.safetensors` checkpoint on CPU (architecture from config.json).
    pub fn load(checkpoint: &str) -> anyhow::Result<Self> {
        Self::load_on(checkpoint, Device::Cpu)
    }

    /// Load a checkpoint onto a chosen device (see `pick_device`).
    pub fn load_on(checkpoint: &str, device: Device) -> anyhow::Result<Self> {
        let path = Path::new(checkpoint);
        let arch = read_arch(path);
        // SAFETY: the mmap lives for the VarBuilder's load; tensors are copied.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)?
        };
        let net = Net::load(&arch, vb)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("local");
        Ok(Self { net, device, model_version: format!("connect4-{stem}") })
    }

    /// Fill `out` (length `b*2*CELLS`) with the side-to-move planes of `states`.
    fn fill_planes(states: &[&Connect4State], out: &mut [f32]) {
        for (i, state) in states.iter().enumerate() {
            let mover = if state.ply % 2 == 0 { 1u8 } else { 2u8 };
            let base = i * 2 * CELLS;
            for r in 0..C4_ROWS {
                for c in 0..C4_COLS {
                    let cell = state.cells[r][c];
                    let idx = r * C4_COLS + c;
                    if cell == mover {
                        out[base + idx] = 1.0;
                    } else if cell != 0 {
                        out[base + CELLS + idx] = 1.0;
                    }
                }
            }
        }
    }

    /// Batched self-play forward over concrete states -> per-state
    /// `(value, softmax priors over 7 columns)`. The GPU inference server's hot
    /// path: one host→device copy, one forward, one device→host copy per batch.
    pub fn policy_value_batch(
        &self,
        states: &[&Connect4State],
    ) -> anyhow::Result<Vec<(f32, [f32; C4_COLS])>> {
        let b = states.len();
        let mut data = vec![0f32; b * 2 * CELLS];
        Self::fill_planes(states, &mut data);
        let planes = Tensor::from_vec(data, (b, 2, C4_ROWS, C4_COLS), &self.device)?;
        let (vals, probs) = self.net.value_policy_batch(&planes)?;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let mut arr = [0f32; C4_COLS];
            arr.copy_from_slice(&probs[i * C4_COLS..(i + 1) * C4_COLS]);
            out.push((vals[i], arr));
        }
        Ok(out)
    }

    fn planes(&self, state: &Connect4State) -> candle_core::Result<Tensor> {
        let mover = if state.ply % 2 == 0 { 1u8 } else { 2u8 };
        let mut data = vec![0f32; 2 * CELLS];
        for r in 0..C4_ROWS {
            for c in 0..C4_COLS {
                let cell = state.cells[r][c];
                let idx = r * C4_COLS + c;
                if cell == mover {
                    data[idx] = 1.0;
                } else if cell != 0 {
                    data[CELLS + idx] = 1.0;
                }
            }
        }
        Tensor::from_vec(data, (1, 2, C4_ROWS, C4_COLS), &self.device)
    }

    fn lgs_tensor(&self, lgs: &LatentGameState) -> candle_core::Result<Tensor> {
        let v = f32_le(&lgs.payload);
        Tensor::from_vec(v, (1, LGS_DIM), &self.device)
    }

    fn gs_state(&self, payload: Vec<f32>) -> LatentGameState {
        LatentGameState::new(f32_to_bytes(&payload), vec![LGS_DIM as i32], self.model_version.clone())
    }

    /// AlphaZero-style self-play forward: a concrete state -> (value for the
    /// side to move, full softmax priors over the 7 columns). Synchronous and
    /// `&self`-only so the self-play MCTS can call it from many native threads
    /// (`LocalModel` is `Send + Sync`); candle tensors are `Arc`-backed and
    /// inference allocates fresh outputs, so concurrent forwards are safe.
    pub fn policy_value(&self, state: &Connect4State) -> anyhow::Result<(f32, [f32; C4_COLS])> {
        let planes = self.planes(state)?;
        let lgs = self.net.encode(&planes)?;
        let (v, pri) = self.net.value_policy(&lgs)?;
        let mut arr = [0f32; C4_COLS];
        arr.copy_from_slice(&pri[..C4_COLS]);
        Ok((v, arr))
    }
}

/// The batched-self-play engine sees the net as a pure forward over pre-encoded
/// planes (workers encode — `fill_planes` is off the GPU thread's path).
impl playground_core::batch_search::BatchModel for LocalModel {
    fn forward(&self, planes: &[f32], b: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let t = Tensor::from_vec(planes.to_vec(), (b, 2, C4_ROWS, C4_COLS), &self.device)?;
        Ok(self.net.value_policy_batch(&t)?)
    }
}

#[async_trait]
impl ModelClient<Connect4> for LocalModel {
    async fn encode(&self, state: &Connect4State) -> anyhow::Result<LatentGameState> {
        let planes = self.planes(state)?;
        let lgs = self.net.encode(&planes)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(self.gs_state(lgs))
    }

    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        let t = self.lgs_tensor(lgs)?;
        Ok(self.net.value(&t)? as f64)
    }

    async fn policy(&self, lgs: &LatentGameState, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>> {
        let t = self.lgs_tensor(lgs)?;
        let (priors, nexts) = self.net.policy_heads(&t)?;
        // Highest-prior columns first, like the Python policy head.
        let mut order: Vec<usize> = (0..C4_COLS).collect();
        order.sort_by(|&a, &b| priors[b].total_cmp(&priors[a]));
        let mut out = Vec::with_capacity(top_b as usize);
        for &col in order.iter().take(top_b as usize) {
            let next_lgs = nexts.get(col)?.to_vec1::<f32>()?;
            out.push(PolicyCandidate {
                prior: priors[col] as f64,
                action_vector: Bytes::from(action_vector_for_column(col)),
                next_lgs: self.gs_state(next_lgs),
            });
        }
        Ok(out)
    }

    async fn reify(
        &self,
        from_state: &Connect4State,
        _from_lgs: &LatentGameState,
        _to_lgs: &LatentGameState,
        action_seed: Option<&[u8]>,
        _noise_scale: Option<f64>,
    ) -> anyhow::Result<Connect4Move> {
        use playground_core::Game;
        let legal: Vec<u8> = Connect4::legal_moves(from_state).iter().map(|m| m.col).collect();
        let col = action_seed
            .and_then(decode_column)
            .filter(|c| legal.contains(c))
            .or_else(|| legal.first().copied())
            .unwrap_or(0);
        Ok(Connect4Move { col })
    }

    async fn model_version(&self) -> anyhow::Result<String> {
        Ok(self.model_version.clone())
    }
}

fn action_vector_for_column(col: usize) -> Vec<u8> {
    let mut v = [0f32; ACTION_DIM];
    v[0] = col as f32;
    f32_to_bytes(&v)
}

fn decode_column(seed: &[u8]) -> Option<u8> {
    if seed.len() < 4 {
        return None;
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&seed[..4]);
    let col = f32::from_le_bytes(b).round();
    (col >= 0.0).then_some(col as u8)
}

fn f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use playground_core::Game;

    fn state_after(moves: &[u8]) -> Connect4State {
        let mut s = Connect4::initial_state();
        for &c in moves {
            s = Connect4::apply(&s, &Connect4Move { col: c });
        }
        s
    }

    // Raw forward throughput (run with `--ignored -- --nocapture`).
    #[tokio::test]
    #[ignore]
    async fn bench_forward() {
        for (name, ckpt) in [
            ("random", "../../../data/models/connect4/random/checkpoint.safetensors"),
            ("selfplay", "../../../data/models/connect4/selfplay/checkpoint.safetensors"),
        ] {
            if !Path::new(ckpt).exists() {
                continue;
            }
            let m = LocalModel::load(ckpt).unwrap();
            let s = state_after(&[3, 3, 2, 4]);
            // encode only (the trunk — the expensive head)
            let n = 5000;
            let t = std::time::Instant::now();
            for _ in 0..n {
                let _ = m.encode(&s).await.unwrap();
            }
            let us = t.elapsed().as_micros() as f64 / n as f64;
            eprintln!("{name}: encode {us:.1} us/call  ({:.0}/s)", 1e6 / us);
        }
    }

    // Parity check against the Python net (run with `-- --nocapture`). Gated on
    // the checkpoints existing so CI without the data registry still passes.
    #[tokio::test]
    async fn parity_with_python_net() {
        for (name, ckpt) in [
            ("random", "../../../data/models/connect4/random/checkpoint.safetensors"),
            ("selfplay", "../../../data/models/connect4/selfplay/checkpoint.safetensors"),
        ] {
            if !Path::new(ckpt).exists() {
                eprintln!("skip {name}: no checkpoint");
                continue;
            }
            let m = LocalModel::load(ckpt).unwrap();
            let s = state_after(&[3, 3, 2, 4]);
            let lgs = m.encode(&s).await.unwrap();
            let v = m.eval(&lgs).await.unwrap();
            let cands = m.policy(&lgs, 7).await.unwrap();
            let head: Vec<f32> = f32_le(&lgs.payload)[..5].to_vec();
            let mut priors = vec![0f64; C4_COLS];
            for c in &cands {
                priors[decode_column(&c.action_vector).unwrap() as usize] = c.prior;
            }
            eprintln!("== {name} ==");
            eprintln!("lgs[:5] {head:?}");
            eprintln!("value {v:.5}");
            eprintln!("priors {priors:?}");
        }
    }
}
