//! In-process Amazons AZ net via candle (no gRPC). Pure ResNet trunk + a value
//! head and TWO spatial policy heads (queen-move + arrow), mirroring
//! `python/.../model_server/amazons_net.py` (parameter names must stay in sync).
//!
//! The policy output is **raw logits**: queen head (7200) ‖ arrow head (7200) =
//! `POLICY_DIM`. The per-move prior (softmax over legal of `q_logit + a_logit`) is
//! computed by the search (`MoveGame::move_priors`), not here. Each head is the
//! 72 conv channels reshaped `(B,72,N,N) -> permute (B,N,N,72) -> (B,7200)`, so the
//! flat index is `(r*N+c)*72 + movetype` — matching `moves::queen_index`/`arrow_index`.

use std::path::Path;

use candle_core::{DType, Device, Module, ModuleT, Tensor};
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear,
    VarBuilder,
};

use crate::game::{fill_planes_into, AmazonsState, CELLS, N, PLANES};
use crate::moves::{MOVE_TYPES, PER_HEAD};

const IN_PLANES: usize = PLANES; // 4

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
    q_conv: Conv2d,
    a_conv: Conv2d,
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
            q_conv: conv2d(channels, MOVE_TYPES, 3, cfg, vb.pp("q_conv"))?,
            a_conv: conv2d(channels, MOVE_TYPES, 3, cfg, vb.pp("a_conv"))?,
            val_conv: conv2d(channels, 1, 1, cfg1, vb.pp("val_conv"))?,
            val_bn: batch_norm(1, BatchNormConfig::default(), vb.pp("val_bn"))?,
            val_fc1: linear(CELLS, 64, vb.pp("val_fc1"))?,
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

    /// (B,4,N,N) -> (values[B], policy_logits[B*POLICY_DIM]); policy = queen ‖ arrow.
    fn value_policy_batch(&self, planes: &Tensor) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
        let b = planes.dim(0)?;
        let x = self.trunk(planes)?;
        let head = |conv: &Conv2d| -> candle_core::Result<Tensor> {
            // (B,72,N,N) -> (B,N,N,72) -> (B,7200), so idx = (r*N+c)*72 + mt.
            conv.forward(&x)?.permute((0, 2, 3, 1))?.contiguous()?.reshape((b, PER_HEAD))
        };
        let q = head(&self.q_conv)?;
        let a = head(&self.a_conv)?;
        let policy = Tensor::cat(&[&q, &a], 1)?.flatten_all()?.to_vec1::<f32>()?;
        let v = self.val_bn.forward_t(&self.val_conv.forward(&x)?, false)?.relu()?.flatten_from(1)?;
        let v = self.val_fc1.forward(&v)?.relu()?;
        let vals = self.val_fc2.forward(&v)?.tanh()?.flatten_all()?.to_vec1::<f32>()?;
        Ok((vals, policy))
    }
}

fn read_arch(checkpoint: &Path) -> (usize, usize) {
    let cfg = checkpoint.with_file_name("config.json");
    if let Ok(text) = std::fs::read_to_string(&cfg) {
        let text = text.replace("NaN", "null").replace("Infinity", "null");
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let arch = &v["arch"];
            let channels = arch["channels"].as_u64().unwrap_or(96) as usize;
            let blocks = arch["blocks"].as_u64().unwrap_or(10) as usize;
            return (channels, blocks);
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

/// Amazons AZ net running in-process via candle.
pub struct LocalModel {
    net: AzNet,
    device: Device,
}

impl LocalModel {
    #[allow(dead_code)]
    pub fn load(checkpoint: &str) -> anyhow::Result<Self> {
        Self::load_on(checkpoint, Device::Cpu)
    }

    pub fn load_on(checkpoint: &str, device: Device) -> anyhow::Result<Self> {
        let path = Path::new(checkpoint);
        let (channels, blocks) = read_arch(path);
        // SAFETY: mmap lives for the load; tensors are copied.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)? };
        let net = AzNet::load(channels, blocks, vb)?;
        Ok(Self { net, device })
    }

    #[allow(dead_code)]
    fn fill_planes(states: &[&AmazonsState], out: &mut [f32]) {
        let stride = IN_PLANES * CELLS;
        for (i, s) in states.iter().enumerate() {
            fill_planes_into(s, &mut out[i * stride..(i + 1) * stride]);
        }
    }

    /// Batched forward over concrete states: (value, raw policy logits[POLICY_DIM]).
    #[allow(dead_code)]
    pub fn value_policy_batch(
        &self,
        states: &[&AmazonsState],
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let b = states.len();
        let mut data = vec![0f32; b * IN_PLANES * CELLS];
        Self::fill_planes(states, &mut data);
        let planes = Tensor::from_vec(data, (b, IN_PLANES, N, N), &self.device)?;
        Ok(self.net.value_policy_batch(&planes)?)
    }
}

/// The batched move-search engine sees the net as a pure forward over pre-encoded
/// planes; the policy returned here is raw logits (combined per-move by the game).
impl playground_core::batch_search::BatchModel for LocalModel {
    fn forward(&self, planes: &[f32], b: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let t = Tensor::from_slice(planes, (b, IN_PLANES, N, N), &self.device)?;
        Ok(self.net.value_policy_batch(&t)?)
    }
}
