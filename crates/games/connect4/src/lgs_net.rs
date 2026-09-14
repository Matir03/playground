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
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear,
    VarBuilder,
};
use playground_core::{Game, LatentGameState, LatentModel, MixturePolicy, MovePolicy};

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
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
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
        Ok(Self {
            fc: linear(2 * c, c, vb.pp("fc"))?,
        })
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

/// Value head over the (shared) LGS. `Legacy` is the original single 64-wide
/// hidden layer (val_fc1/val_fc2); `Mlp` is a deeper/wider stack
/// (val_hidden_layers.{i} + val_out) produced by c4_lgs_grow_value.py. The
/// config.json's optional `val_hidden` array selects which one to load.
enum ValueHead {
    Legacy { val_fc1: Linear, val_fc2: Linear },
    Mlp { layers: Vec<Linear>, out: Linear },
}

impl ValueHead {
    fn load(arch: &Arch, vb: &VarBuilder) -> candle_core::Result<Self> {
        match &arch.val_hidden {
            None => Ok(ValueHead::Legacy {
                val_fc1: linear(LGS_DIM, 64, vb.pp("val_fc1"))?,
                val_fc2: linear(64, 1, vb.pp("val_fc2"))?,
            }),
            Some(hidden) => {
                let lvb = vb.pp("val_hidden_layers");
                let mut layers = Vec::with_capacity(hidden.len());
                let mut in_dim = LGS_DIM;
                for (i, &h) in hidden.iter().enumerate() {
                    layers.push(linear(in_dim, h, lvb.pp(i.to_string()))?);
                    in_dim = h;
                }
                Ok(ValueHead::Mlp {
                    layers,
                    out: linear(in_dim, 1, vb.pp("val_out"))?,
                })
            }
        }
    }

    fn forward(&self, lgs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            ValueHead::Legacy { val_fc1, val_fc2 } => {
                let x = val_fc1.forward(lgs)?.relu()?;
                val_fc2.forward(&x)?.tanh()
            }
            ValueHead::Mlp { layers, out } => {
                let mut x = lgs.clone();
                for layer in layers {
                    x = layer.forward(&x)?.relu()?;
                }
                out.forward(&x)?.tanh()
            }
        }
    }
}

/// Edge head: turns a latent edge / next-LGS into a concrete action.
/// `Reifier` is the original ((LGS, next-LGS) -> 7 column logits). `Decoder` is a
/// position decoder (next-LGS -> board-plane logits); the search decodes a sampled
/// next-LGS to a target board and infers the move (see `move_from_decoded`).
/// config.json's optional `decoder` bool selects which one to load, so a gens/
/// dir can hold both kinds for ladder matches.
enum EdgeHead {
    Reifier {
        reif1: Linear,
        reif2: Linear,
    },
    Decoder {
        dec_fc1: Linear,
        dec_fc2: Linear,
        dec_out: Linear,
    },
}

impl EdgeHead {
    fn load(arch: &Arch, vb: &VarBuilder) -> candle_core::Result<Self> {
        if arch.decoder {
            Ok(EdgeHead::Decoder {
                dec_fc1: linear(LGS_DIM, 256, vb.pp("dec_fc1"))?,
                dec_fc2: linear(256, 256, vb.pp("dec_fc2"))?,
                dec_out: linear(256, 2 * CELLS, vb.pp("dec_out"))?,
            })
        } else {
            Ok(EdgeHead::Reifier {
                reif1: linear(2 * LGS_DIM, 128, vb.pp("reif1"))?,
                reif2: linear(128, C4_COLS, vb.pp("reif2"))?,
            })
        }
    }
}

/// Policy head over next-LGS. `Mdn` is the continuous mixture (general games).
/// `Move` is the discrete per-move head for C4: `pol_prior` (A move-prior logits)
/// + `pol_next` (A predicted next-encodings). Selected by config `move_policy`.
enum PolicyHead {
    Mdn {
        mix_logits: Linear,
        means: Linear,
        log_stds: Linear,
    },
    Move {
        pol_prior: Linear,
        pol_next: Linear,
    },
}

impl PolicyHead {
    fn load(arch: &Arch, vb: &VarBuilder) -> candle_core::Result<Self> {
        if arch.move_policy {
            Ok(PolicyHead::Move {
                pol_prior: linear(LGS_DIM, C4_COLS, vb.pp("pol_prior"))?,
                pol_next: linear(LGS_DIM, C4_COLS * LGS_DIM, vb.pp("pol_next"))?,
            })
        } else {
            Ok(PolicyHead::Mdn {
                mix_logits: linear(LGS_DIM, arch.k, vb.pp("mix_logits"))?,
                means: linear(LGS_DIM, arch.k * LGS_DIM, vb.pp("means"))?,
                log_stds: linear(LGS_DIM, arch.k * LGS_DIM, vb.pp("log_stds"))?,
            })
        }
    }
}

struct Net {
    stem: Conv2d,
    stem_bn: BatchNorm,
    blocks: Vec<ResBlock>,
    gpool: GlobalPoolBias,
    enc_fc: Linear,
    value_head: ValueHead,
    policy: PolicyHead,
    edge: EdgeHead,
    k: usize,
}

impl Net {
    fn load(arch: &Arch, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
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
            value_head: ValueHead::load(arch, &vb)?,
            policy: PolicyHead::load(arch, &vb)?,
            edge: EdgeHead::load(arch, &vb)?,
            k: arch.k,
        })
    }

    /// planes (B,2,6,7) -> LGS (B,64), tanh-bounded.
    fn encode(&self, planes: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self
            .stem_bn
            .forward_t(&self.stem.forward(planes)?, false)?
            .relu()?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        x = self.gpool.forward(&x)?;
        let flat = x.flatten_from(1)?;
        self.enc_fc.forward(&flat)?.tanh()
    }

    fn value(&self, lgs: &Tensor) -> candle_core::Result<Tensor> {
        self.value_head.forward(lgs) // (B,1)
    }

    /// LGS (B,64) -> (weights (B,k), means (B,k,d), stds (B,k,d)). MDN nets only.
    fn policy_mdn(&self, lgs: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let b = lgs.dim(0)?;
        match &self.policy {
            PolicyHead::Mdn {
                mix_logits,
                means,
                log_stds,
            } => {
                let w = candle_nn::ops::softmax(&mix_logits.forward(lgs)?, 1)?;
                let mu = means.forward(lgs)?.tanh()?.reshape((b, self.k, LGS_DIM))?;
                let raw = log_stds.forward(lgs)?;
                // softplus(z) = ln(1+exp(z)); + SIGMA_MIN floor.
                let std =
                    ((raw.exp()? + 1.0)?.log()? + SIGMA_MIN)?.reshape((b, self.k, LGS_DIM))?;
                Ok((w, mu, std))
            }
            PolicyHead::Move { .. } => candle_core::bail!("policy_mdn called on a move-policy net"),
        }
    }

    /// LGS (B,64) -> (prior softmax (B,A), next-latents (B,A,d) tanh). Move nets only.
    fn move_policy(&self, lgs: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let b = lgs.dim(0)?;
        match &self.policy {
            PolicyHead::Move {
                pol_prior,
                pol_next,
            } => {
                let prior = candle_nn::ops::softmax(&pol_prior.forward(lgs)?, 1)?; // (B,A)
                let nxt = pol_next
                    .forward(lgs)?
                    .tanh()?
                    .reshape((b, C4_COLS, LGS_DIM))?; // (B,A,d)
                Ok((prior, nxt))
            }
            PolicyHead::Mdn { .. } => candle_core::bail!("move_policy called on an MDN net"),
        }
    }

    fn is_move_policy(&self) -> bool {
        matches!(self.policy, PolicyHead::Move { .. })
    }

    /// (LGS (1,64), next-LGS (1,64)) -> column logits (7,). Reifier nets only.
    fn reify_logits(&self, lgs: &Tensor, next_lgs: &Tensor) -> candle_core::Result<Vec<f32>> {
        match &self.edge {
            EdgeHead::Reifier { reif1, reif2 } => {
                let cat = Tensor::cat(&[lgs, next_lgs], 1)?;
                let h = reif1.forward(&cat)?.relu()?;
                reif2.forward(&h)?.flatten_all()?.to_vec1::<f32>()
            }
            EdgeHead::Decoder { .. } => candle_core::bail!("reify_logits called on a decoder net"),
        }
    }

    /// next-LGS (B,64) -> board-plane logits (B, 2*CELLS). Decoder nets only.
    fn decode(&self, lgs: &Tensor) -> candle_core::Result<Tensor> {
        match &self.edge {
            EdgeHead::Decoder {
                dec_fc1,
                dec_fc2,
                dec_out,
            } => {
                let h = dec_fc1.forward(lgs)?.relu()?;
                let h = dec_fc2.forward(&h)?.relu()?;
                dec_out.forward(&h)
            }
            EdgeHead::Reifier { .. } => candle_core::bail!("decode called on a reifier net"),
        }
    }

    fn is_decoder(&self) -> bool {
        matches!(self.edge, EdgeHead::Decoder { .. })
    }
}

/// Infer the move that maps `from_state` to the decoder's target board. `decoded`
/// is the 84-value plane-logit output (2×6×7). We always play the legal move
/// whose true successor board is *closest* to the decoder output (nearest legal
/// target — no random fallback), comparing σ(decoded) to each successor's planes.
/// Returns `(move, valid)` where `valid` marks an *exact* decode (the closest
/// successor's occupancy matches the decode: exactly one new stone at its drop
/// cell, none removed) — that flag drives the valid-decode-rate metric.
pub fn move_from_decoded(from_state: &Connect4State, decoded: &[f32]) -> (Connect4Move, bool) {
    let legal: Vec<Connect4Move> = Connect4::legal_moves(from_state);
    if legal.is_empty() {
        return (Connect4Move { col: 0 }, false);
    }
    let sig = |x: f32| 1.0f32 / (1.0 + (-x).exp());
    // Nearest legal successor by L2(σ(decoded), successor planes).
    let mut best = &legal[0];
    let mut best_d = f32::INFINITY;
    for mv in &legal {
        let child = Connect4::apply(from_state, mv);
        let planes = crate::mcts::planes_of(&child);
        let mut d = 0.0f32;
        for i in 0..2 * CELLS {
            let diff = sig(decoded[i]) - planes[i];
            d += diff * diff;
        }
        if d < best_d {
            best_d = d;
            best = mv;
        }
    }
    // Exact-decode flag: decoded occupancy = parent + one stone at a legal drop cell.
    let mut new_cell: Option<(usize, usize)> = None;
    let (mut new_count, mut removed) = (0usize, 0usize);
    for r in 0..C4_ROWS {
        for c in 0..C4_COLS {
            let idx = r * C4_COLS + c;
            let dec_filled = decoded[idx] > 0.0 || decoded[CELLS + idx] > 0.0;
            let par_filled = from_state.cells[r][c] != 0;
            if dec_filled && !par_filled {
                new_count += 1;
                new_cell = Some((r, c));
            } else if !dec_filled && par_filled {
                removed += 1;
            }
        }
    }
    let exact = new_count == 1
        && removed == 0
        && new_cell
            .is_some_and(|(r, c)| (0..C4_ROWS).find(|&rr| from_state.cells[rr][c] == 0) == Some(r));
    (best.clone(), exact)
}

struct Arch {
    channels: usize,
    blocks: usize,
    k: usize,
    /// Hidden widths of an enlarged value-head MLP (None = legacy 64-wide head).
    val_hidden: Option<Vec<usize>>,
    /// Edge head is a position decoder (true) vs the reifier (false, default).
    decoder: bool,
    /// Policy is the discrete per-move head (true) vs the MDN (false, default).
    move_policy: bool,
}

fn read_arch(checkpoint: &Path) -> Arch {
    let mut a = Arch {
        channels: 64,
        blocks: 5,
        k: 8,
        val_hidden: None,
        decoder: false,
        move_policy: false,
    };
    // Prefer a per-checkpoint sidecar (e.g. gen_000153.config.json) so a single
    // gens/ dir can hold mixed archs (legacy + grown value heads); fall back to
    // config.json in the same dir (live/ uses that).
    let sidecar = checkpoint.with_extension("config.json");
    let cfg = if sidecar.exists() {
        sidecar
    } else {
        checkpoint.with_file_name("config.json")
    };
    if let Ok(text) = std::fs::read_to_string(&cfg) {
        let text = text.replace("NaN", "null").replace("Infinity", "null");
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let arch = &v["arch"];
            if arch["kind"] == "lgs" {
                a.channels = arch["channels"].as_u64().unwrap_or(64) as usize;
                a.blocks = arch["blocks"].as_u64().unwrap_or(5) as usize;
                a.k = arch["k"].as_u64().unwrap_or(8) as usize;
                if let Some(vh) = arch["val_hidden"].as_array() {
                    a.val_hidden = Some(
                        vh.iter()
                            .map(|x| x.as_u64().unwrap_or(64) as usize)
                            .collect(),
                    );
                }
                a.decoder = arch["decoder"].as_bool().unwrap_or(false);
                a.move_policy = arch["move_policy"].as_bool().unwrap_or(false);
            }
        }
    }
    a
}

/// Whether a checkpoint's arch is the discrete per-move policy (vs the MDN), read
/// from its sidecar/config.json. Self-play/eval set `SearchConfig.move_policy`
/// from this so the search path matches the net without a separate CLI flag.
pub fn checkpoint_is_move_policy(checkpoint: &str) -> bool {
    read_arch(Path::new(checkpoint)).move_policy
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
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)? };
        let net = Net::load(&arch, vb)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("local");
        Ok(Self {
            net,
            device,
            model_version: format!("connect4-lgs-{stem}"),
        })
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

    async fn move_policy(&self, lgs: &LatentGameState) -> anyhow::Result<MovePolicy> {
        let t = self.lgs_tensor(lgs)?;
        let (prior, nxt) = self.net.move_policy(&t)?;
        Ok(MovePolicy {
            dim: LGS_DIM,
            priors: prior.flatten_all()?.to_vec1::<f32>()?,
            next: nxt.flatten_all()?.to_vec1::<f32>()?,
        })
    }

    async fn reify(
        &self,
        from_state: &Connect4State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
    ) -> anyhow::Result<Connect4Move> {
        if self.net.is_decoder() {
            // Decode the target next-LGS to a board and infer the move.
            let decoded = self
                .net
                .decode(&self.lgs_tensor(to_lgs)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            return Ok(move_from_decoded(from_state, &decoded).0);
        }
        let legal: Vec<u8> = Connect4::legal_moves(from_state)
            .iter()
            .map(|m| m.col)
            .collect();
        let logits = self
            .net
            .reify_logits(&self.lgs_tensor(from_lgs)?, &self.lgs_tensor(to_lgs)?)?;
        // Pick the highest-logit *legal* column; structural fallback keeps play legal.
        let best = legal
            .iter()
            .copied()
            .max_by(|&a, &b| logits[a as usize].total_cmp(&logits[b as usize]));
        Ok(Connect4Move {
            col: best.or_else(|| legal.first().copied()).unwrap_or(0),
        })
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
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)? };
        let net = Net::load(&arch, vb)?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("local");
        Ok(Self {
            net,
            device,
            model_version: format!("connect4-lgs-{stem}"),
        })
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

    /// Fused encode+value: `b` × planes → (`b` × LGS, `b` values). One forward
    /// (shared trunk) so the reify-leaf value rides along with the child encode
    /// instead of a separate eval round-trip.
    pub fn encode_eval_batch(
        &self,
        planes: &[f32],
        b: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let t = Tensor::from_vec(planes.to_vec(), (b, 2, C4_ROWS, C4_COLS), &self.device)?;
        let lgs = self.net.encode(&t)?;
        let val = self.net.value(&lgs)?;
        Ok((
            lgs.flatten_all()?.to_vec1::<f32>()?,
            val.flatten_all()?.to_vec1::<f32>()?,
        ))
    }

    /// Fused move-policy+value: `b` × LGS → (priors `b·A`, next-latents `b·A·d`,
    /// values `b`). One round-trip for discrete-policy node expansion.
    pub fn move_policy_eval_batch(
        &self,
        lgs: &[f32],
        b: usize,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let t = Tensor::from_vec(lgs.to_vec(), (b, LGS_DIM), &self.device)?;
        let (prior, nxt) = self.net.move_policy(&t)?;
        let val = self.net.value(&t)?;
        Ok((
            prior.flatten_all()?.to_vec1::<f32>()?,
            nxt.flatten_all()?.to_vec1::<f32>()?,
            val.flatten_all()?.to_vec1::<f32>()?,
        ))
    }

    /// Fused policy+value: `b` × LGS → (MDN params, `b` values). One round-trip
    /// for node expansion (which always needs both the policy and the value).
    pub fn policy_eval_batch(
        &self,
        lgs: &[f32],
        b: usize,
    ) -> anyhow::Result<(usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let t = Tensor::from_vec(lgs.to_vec(), (b, LGS_DIM), &self.device)?;
        let (w, mu, std) = self.net.policy_mdn(&t)?;
        let val = self.net.value(&t)?;
        Ok((
            self.net.k,
            w.flatten_all()?.to_vec1::<f32>()?,
            mu.flatten_all()?.to_vec1::<f32>()?,
            std.flatten_all()?.to_vec1::<f32>()?,
            val.flatten_all()?.to_vec1::<f32>()?,
        ))
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

    /// `b` × next-LGS → `b` × (2*CELLS) decoder plane-logits, flattened.
    /// Self-play runs the decoder net; the move is inferred CPU-side from the
    /// decoded board via `move_from_decoded`.
    pub fn decode_batch(&self, lgs: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let t = Tensor::from_vec(lgs.to_vec(), (b, LGS_DIM), &self.device)?;
        Ok(self.net.decode(&t)?.flatten_all()?.to_vec1::<f32>()?)
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

        let r: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(refp).unwrap()).unwrap();
        let ref_v = r["value"].as_f64().unwrap();
        let ref_lgs5: Vec<f64> = r["lgs5"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let lgs_f = lgs.to_f32();
        eprintln!("candle value {v:.6} vs torch {ref_v:.6}");
        assert!((v - ref_v).abs() < 1e-4, "value parity: {v} vs {ref_v}");
        for (i, &rv) in ref_lgs5.iter().enumerate() {
            assert!(
                (lgs_f[i] as f64 - rv).abs() < 1e-4,
                "lgs[{i}] {} vs {rv}",
                lgs_f[i]
            );
        }
        let ref_w: Vec<f64> = r["w"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        for (i, &rw) in ref_w.iter().enumerate() {
            assert!(
                (pol.weights[i] as f64 - rw).abs() < 1e-4,
                "w[{i}] {} vs {rw}",
                pol.weights[i]
            );
        }
        assert!((pol.weights.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
}
