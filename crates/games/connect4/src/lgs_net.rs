//! In-process Connect 4 net for the reification-based latent-space algorithm.
//!
//! candle port of `python/.../model_server/c4_lgs_net.py` (the LGS net), kept in
//! exact key/shape parity so the `.safetensors` Python writes load directly.
//! Implements [`playground_core::LatentModel`] so the KR-PUCT search can call
//! `encode`/`eval`/`policy`/`reify` as local calls. Separate from `net.rs` (the
//! discrete-AZ baseline net) — the two coexist.
//!
//! Heads: encoder (ResNet trunk + global pooling) -> tanh LGS; eval; MDN policy
//! over next-LGS (K diagonal Gaussians); reifier `[LGS||next-LGS] -> 7 logits`.

// TODO(stage3): remove once the LGS self-play path constructs `LgsModel`.
#![allow(dead_code)]

use std::path::Path;

use async_trait::async_trait;
use candle_core::{DType, Device, Module, ModuleT, Tensor};
use candle_nn::{batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, VarBuilder};
use playground_core::{Game, LatentGameState, LatentModel, MixturePolicy};

use crate::game::{Connect4, Connect4Move, Connect4State, C4_COLS, C4_ROWS};

const LGS_DIM: usize = 64;
const CELLS: usize = C4_ROWS * C4_COLS;
const SIGMA_MIN: f64 = 0.05;

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

/// Global pooling bias (matches `GlobalPoolBias` in Python): pool mean+max over
/// the board, project, broadcast back as a per-channel bias.
struct GlobalPoolBias {
    fc: Linear,
}

impl GlobalPoolBias {
    fn load(c: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self { fc: linear(2 * c, c, vb.pp("fc"))? })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, c, _h, _w) = x.dims4()?;
        let mean = x.mean(3)?.mean(2)?; // (B,C)
        let mx = x.max(3)?.max(2)?; // (B,C)
        let pooled = Tensor::cat(&[mean, mx], 1)?; // (B,2C)
        let bias = self.fc.forward(&pooled)?.reshape((b, c, 1, 1))?;
        x.broadcast_add(&bias)
    }
}

struct Net {
    stem: Conv2d,
    stem_bn: BatchNorm,
    blocks: Vec<ResBlock>,
    gpool: GlobalPoolBias,
    enc_fc: Linear,
    val_fc1: Linear,
    val_fc2: Linear,
    mix_logits: Linear,
    means: Linear,
    log_stds: Linear,
    reif1: Linear,
    reif2: Linear,
    k: usize,
}

impl Net {
    fn load(arch: &Arch, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let c = arch.channels;
        let mut blocks = Vec::with_capacity(arch.blocks);
        let blocks_vb = vb.pp("blocks");
        for i in 0..arch.blocks {
            blocks.push(ResBlock::load(c, blocks_vb.pp(i.to_string()))?);
        }
        let feat = c * CELLS;
        Ok(Self {
            stem: conv2d(2, c, 3, cfg, vb.pp("stem"))?,
            stem_bn: batch_norm(c, BatchNormConfig::default(), vb.pp("stem_bn"))?,
            blocks,
            gpool: GlobalPoolBias::load(c, vb.pp("gpool"))?,
            enc_fc: linear(feat, LGS_DIM, vb.pp("enc_fc"))?,
            val_fc1: linear(LGS_DIM, 64, vb.pp("val_fc1"))?,
            val_fc2: linear(64, 1, vb.pp("val_fc2"))?,
            mix_logits: linear(LGS_DIM, arch.k, vb.pp("mix_logits"))?,
            means: linear(LGS_DIM, arch.k * LGS_DIM, vb.pp("means"))?,
            log_stds: linear(LGS_DIM, arch.k * LGS_DIM, vb.pp("log_stds"))?,
            reif1: linear(2 * LGS_DIM, 128, vb.pp("reif1"))?,
            reif2: linear(128, C4_COLS, vb.pp("reif2"))?,
            k: arch.k,
        })
    }

    /// planes (B,2,6,7) -> LGS (B,64), tanh-bounded.
    fn encode(&self, planes: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.stem_bn.forward_t(&self.stem.forward(planes)?, false)?.relu()?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        x = self.gpool.forward(&x)?;
        let flat = x.flatten_from(1)?;
        self.enc_fc.forward(&flat)?.tanh()
    }

    fn value(&self, lgs: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.val_fc1.forward(lgs)?.relu()?;
        self.val_fc2.forward(&x)?.tanh() // (B,1)
    }

    /// LGS (B,64) -> (weights (B,k), means (B,k,d), stds (B,k,d)).
    fn policy_mdn(&self, lgs: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let b = lgs.dim(0)?;
        let w = candle_nn::ops::softmax(&self.mix_logits.forward(lgs)?, 1)?;
        let mu = self.means.forward(lgs)?.tanh()?.reshape((b, self.k, LGS_DIM))?;
        let raw = self.log_stds.forward(lgs)?;
        // softplus(z) = ln(1+exp(z)); + SIGMA_MIN floor.
        let std = ((raw.exp()? + 1.0)?.log()? + SIGMA_MIN)?.reshape((b, self.k, LGS_DIM))?;
        Ok((w, mu, std))
    }

    /// (LGS (1,64), next-LGS (1,64)) -> column logits (7,).
    fn reify_logits(&self, lgs: &Tensor, next_lgs: &Tensor) -> candle_core::Result<Vec<f32>> {
        let cat = Tensor::cat(&[lgs, next_lgs], 1)?;
        let h = self.reif1.forward(&cat)?.relu()?;
        self.reif2.forward(&h)?.flatten_all()?.to_vec1::<f32>()
    }
}

struct Arch {
    channels: usize,
    blocks: usize,
    k: usize,
}

fn read_arch(checkpoint: &Path) -> Arch {
    let mut a = Arch { channels: 64, blocks: 5, k: 8 };
    let cfg = checkpoint.with_file_name("config.json");
    if let Ok(text) = std::fs::read_to_string(&cfg) {
        let text = text.replace("NaN", "null").replace("Infinity", "null");
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let arch = &v["arch"];
            if arch["kind"] == "lgs" {
                a.channels = arch["channels"].as_u64().unwrap_or(64) as usize;
                a.blocks = arch["blocks"].as_u64().unwrap_or(5) as usize;
                a.k = arch["k"].as_u64().unwrap_or(8) as usize;
            }
        }
    }
    a
}

/// A `LatentModel` running the Connect 4 LGS net in-process via candle.
pub struct LgsModel {
    net: Net,
    device: Device,
    model_version: String,
}

impl LgsModel {
    pub fn load(checkpoint: &str) -> anyhow::Result<Self> {
        Self::load_on(checkpoint, Device::Cpu)
    }

    pub fn load_on(checkpoint: &str, device: Device) -> anyhow::Result<Self> {
        let path = Path::new(checkpoint);
        let arch = read_arch(path);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)? };
        let net = Net::load(&arch, vb)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("local");
        Ok(Self { net, device, model_version: format!("connect4-lgs-{stem}") })
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
        Tensor::from_vec(lgs.to_f32(), (1, LGS_DIM), &self.device)
    }
}

#[async_trait]
impl LatentModel<Connect4> for LgsModel {
    async fn encode(&self, state: &Connect4State) -> anyhow::Result<LatentGameState> {
        let planes = self.planes(state)?;
        let v = self.net.encode(&planes)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(LatentGameState::from_f32(&v, self.model_version.clone()))
    }

    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        let t = self.lgs_tensor(lgs)?;
        Ok(self.net.value(&t)?.flatten_all()?.to_vec1::<f32>()?[0] as f64)
    }

    async fn policy(&self, lgs: &LatentGameState) -> anyhow::Result<MixturePolicy> {
        let t = self.lgs_tensor(lgs)?;
        let (w, mu, std) = self.net.policy_mdn(&t)?;
        Ok(MixturePolicy {
            dim: LGS_DIM,
            weights: w.flatten_all()?.to_vec1::<f32>()?,
            means: mu.flatten_all()?.to_vec1::<f32>()?,
            stds: std.flatten_all()?.to_vec1::<f32>()?,
        })
    }

    async fn reify(
        &self,
        from_state: &Connect4State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
    ) -> anyhow::Result<Connect4Move> {
        let legal: Vec<u8> = Connect4::legal_moves(from_state).iter().map(|m| m.col).collect();
        let logits = self.net.reify_logits(&self.lgs_tensor(from_lgs)?, &self.lgs_tensor(to_lgs)?)?;
        // Pick the highest-logit *legal* column; structural fallback keeps play legal.
        let best = legal
            .iter()
            .copied()
            .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]));
        Ok(Connect4Move { col: best.or_else(|| legal.first().copied()).unwrap_or(0) })
    }

    async fn model_version(&self) -> anyhow::Result<String> {
        Ok(self.model_version.clone())
    }
}

/// Batched LGS net for the central inference server (the throughput path). Owns
/// the candle net on its device; each method runs one batched forward over a
/// stack of requests coalesced from many concurrent searches. See lgs_selfplay.
pub struct LgsServer {
    net: Net,
    device: Device,
    pub model_version: String,
}

impl LgsServer {
    pub fn load_on(checkpoint: &str, device: Device) -> anyhow::Result<Self> {
        let path = Path::new(checkpoint);
        let arch = read_arch(path);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)? };
        let net = Net::load(&arch, vb)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("local");
        Ok(Self { net, device, model_version: format!("connect4-lgs-{stem}") })
    }

    /// `b` × planes (PLANES each) → `b` × LGS (LGS_DIM each), flattened.
    pub fn encode_batch(&self, planes: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let t = Tensor::from_vec(planes.to_vec(), (b, 2, C4_ROWS, C4_COLS), &self.device)?;
        Ok(self.net.encode(&t)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// `b` × LGS → `b` values.
    pub fn eval_batch(&self, lgs: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let t = Tensor::from_vec(lgs.to_vec(), (b, LGS_DIM), &self.device)?;
        Ok(self.net.value(&t)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// `b` × LGS → MDN params: `(k, weights b·k, means b·k·d, stds b·k·d)`.
    pub fn policy_batch(
        &self,
        lgs: &[f32],
        b: usize,
    ) -> anyhow::Result<(usize, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let t = Tensor::from_vec(lgs.to_vec(), (b, LGS_DIM), &self.device)?;
        let (w, mu, std) = self.net.policy_mdn(&t)?;
        Ok((
            self.net.k,
            w.flatten_all()?.to_vec1::<f32>()?,
            mu.flatten_all()?.to_vec1::<f32>()?,
            std.flatten_all()?.to_vec1::<f32>()?,
        ))
    }

    /// `b` × (from-LGS, to-LGS) → `b` × 7 reifier logits, flattened.
    pub fn reify_batch(&self, from: &[f32], to: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let f = Tensor::from_vec(from.to_vec(), (b, LGS_DIM), &self.device)?;
        let g = Tensor::from_vec(to.to_vec(), (b, LGS_DIM), &self.device)?;
        let cat = Tensor::cat(&[&f, &g], 1)?;
        let h = self.net.reif1.forward(&cat)?.relu()?;
        Ok(self.net.reif2.forward(&h)?.flatten_all()?.to_vec1::<f32>()?)
    }
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

    // Parity vs the Python LGS net: same checkpoint -> same LGS/value/MDN. Gated
    // on the parity checkpoint existing (written by c4_lgs_net's smoke test).
    #[tokio::test]
    async fn parity_with_python_lgs_net() {
        let ckpt = "../../../data/models/connect4_lgs/parity/checkpoint.safetensors";
        let refp = "../../../data/models/connect4_lgs/parity/parity_ref.json";
        if !Path::new(ckpt).exists() || !Path::new(refp).exists() {
            eprintln!("skip parity: no parity checkpoint");
            return;
        }
        let m = LgsModel::load(ckpt).unwrap();
        let s = state_after(&[3, 3, 2, 4]);
        let lgs = m.encode(&s).await.unwrap();
        let v = m.eval(&lgs).await.unwrap();
        let pol = m.policy(&lgs).await.unwrap();

        let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refp).unwrap()).unwrap();
        let ref_v = r["value"].as_f64().unwrap();
        let ref_lgs5: Vec<f64> = r["lgs5"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let lgs_f = lgs.to_f32();
        eprintln!("candle value {v:.6} vs torch {ref_v:.6}");
        assert!((v - ref_v).abs() < 1e-4, "value parity: {v} vs {ref_v}");
        for (i, &rv) in ref_lgs5.iter().enumerate() {
            assert!((lgs_f[i] as f64 - rv).abs() < 1e-4, "lgs[{i}] {} vs {rv}", lgs_f[i]);
        }
        let ref_w: Vec<f64> = r["w"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        for (i, &rw) in ref_w.iter().enumerate() {
            assert!((pol.weights[i] as f64 - rw).abs() < 1e-4, "w[{i}] {} vs {rw}", pol.weights[i]);
        }
        assert!((pol.weights.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
}
