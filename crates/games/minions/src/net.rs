//! In-process Minions net via candle.
//!
//! Mirrors `python/src/playground/model_server/minions_net.py` — the parameter
//! names here and there are the same `.safetensors` keys, so a rename on either
//! side silently breaks the other. `tests::parity_with_python_net` pins it
//! against a fixture written by the Python side.
//!
//! Four heads on one trunk: value, the masked-diffusion declaration
//! head that proposes turns, and the energy head that scores a whole
//! declaration. See the Python module for what each is for.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear,
    VarBuilder,
};

use crate::game::PLANES as IN_PLANES;
use crate::map::{N, NUM_CELLS};
use crate::turn::{DECL_PLANES, ENERGY_PLANES, N_OFFSETS, N_SPAWNS, N_TARGETS};
use half::f16;
use half::slice::HalfFloatSliceExt;
use half::vec::HalfFloatVecExt;

/// Declaration logits emitted per cell: 37 move offsets + 3 targets + 6 spawns.
pub const DECL_LOGITS: usize = N_OFFSETS + N_TARGETS + N_SPAWNS;

/// A head's private two-conv neck, matching the Python `nn.Sequential`.
///
/// Torch names `Sequential` children by position, so the conv/BN pairs land at
/// keys "0", "1", "3", "4" — indices 2 and 5 are the ReLUs, which hold no
/// parameters. Getting these names wrong is a silent load failure, so the
/// safetensors-key parity test is what actually guards this.
struct ConvNeck {
    c1: Conv2d,
    b1: BatchNorm,
    c2: Conv2d,
    b2: BatchNorm,
}

impl ConvNeck {
    fn load(
        c_in: usize,
        c_out: usize,
        bn: BatchNormConfig,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            c1: conv2d(c_in, c_out, 3, cfg, vb.pp("0"))?,
            b1: batch_norm(c_out, bn, vb.pp("1"))?,
            c2: conv2d(c_out, c_out, 3, cfg, vb.pp("3"))?,
            b2: batch_norm(c_out, bn, vb.pp("4"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.b1.forward_t(&self.c1.forward(x)?, false)?.relu()?;
        self.b2.forward_t(&self.c2.forward(&x)?, false)?.relu()
    }
}

/// Width of the shared head neck. Mirrors `head_width()` in minions_net.py.
pub fn head_width(channels: usize) -> usize {
    std::cmp::max(32, channels / 2)
}

/// A single conv+BN+ReLU, matching a 3-element `nn.Sequential` (keys "0", "1").
struct ConvNeck1 {
    c: Conv2d,
    b: BatchNorm,
}

impl ConvNeck1 {
    fn load(
        c_in: usize,
        c_out: usize,
        bn: BatchNormConfig,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            c: conv2d(c_in, c_out, 3, cfg, vb.pp("0"))?,
            b: batch_norm(c_out, bn, vb.pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.b.forward_t(&self.c.forward(x)?, false)?.relu()
    }
}

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

/// KataGo global pooling: mean+max across the board, broadcast back as a
/// per-channel bias. Minions needs it more than a placement game does — gold and
/// graveyard counts are global, and the adjudication rule turns on them.
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
        let (b, c, h, w) = x.dims4()?;
        let flat = x.reshape((b, c, h * w))?;
        let pooled = Tensor::cat(&[flat.mean(D::Minus1)?, flat.max(D::Minus1)?], 1)?;
        let bias = self.fc.forward(&pooled)?.reshape((b, c, 1, 1))?;
        x.broadcast_add(&bias)
    }
}

/// What the sampler needs from a net, independent of backend.
///
/// Flat `f32` slices rather than device tensors, so candle and ORT present one
/// surface. That costs candle-CUDA a host round-trip between heads; acceptable
/// because candle is the parity and development path, while ORT + TensorRT is
/// the production one — on Hex that gap measured 4.6k vs 158k positions/s.
///
/// Three entry points rather than one forward, because the sampler computes the
/// trunk *once* per position and then runs the diffusion head `T` times and the
/// energy head once against it. Fusing them would recompute the trunk every
/// round and throw away the only thing that makes T-step sampling affordable.
/// What one trunk pass yields. `h` feeds both heads.
///
/// `h` is fp16 on the host. The host never computes with it -- it only
/// carries it from the trunk to the cache and on to the heads -- so fp16 costs
/// nothing in accuracy (TensorRT runs fp16 inside) and halves the cache and
/// every transfer. `value` stays f32.
pub struct TrunkOut {
    pub h: Vec<f16>,
    pub value: Vec<f32>,
}

/// One diffusion pass over a batch of rows.
///
/// `h` is fixed for the whole pass — only `decl` changes between steps — so a
/// backend can upload the trunk features to the device **once** instead of on
/// every one of the `T` steps. Measured at +15.7% on the diffusion head
/// (`bench-bind`, ABBA, 14 reps) at the production shape.
pub trait DiffusionPass {
    /// Run one step against the pass's fixed `h`.
    fn step(&mut self, decl: &[f32]) -> anyhow::Result<Vec<f32>>;
}

/// Fallback pass for backends with nothing to cache: re-sends `h` every step,
/// i.e. exactly the old behaviour.
pub struct NaivePass<'a> {
    pub net: &'a dyn Backend,
    pub h: &'a [f16],
    pub b: usize,
}

impl DiffusionPass for NaivePass<'_> {
    fn step(&mut self, decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.net.diffusion(self.h, decl, self.b)
    }
}

pub trait Backend: Send + Sync {
    /// Trunk width, needed to size the projection buffers.
    fn channels(&self) -> usize;
    /// `planes (b, IN_PLANES, N, N)` → value and the head projections.
    fn trunk(&self, planes: &[f32], b: usize) -> anyhow::Result<TrunkOut>;
    /// `(h, decl)` → declaration logits `(b, NUM_CELLS, DECL_LOGITS)`.
    fn diffusion(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>>;
    /// `(h, decl)` → `f(s, a)` per row.
    fn energy(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>>;

    /// [`Self::diffusion`] with row `i`'s features taken from row `rows[i]` of
    /// `h` (one row per position, `channels() * NUM_CELLS` wide), so a caller
    /// batching several positions' chains never builds the tiled copy itself.
    /// The default gathers and calls `diffusion`; `OrtNet` copies the rows
    /// straight into its pinned input.
    fn diffusion_rows(&self, h: &[f16], rows: &[u32], decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.diffusion(
            &gather_rows(h, rows, self.channels() * crate::map::NUM_CELLS),
            decl,
            rows.len(),
        )
    }
    /// [`Self::diffusion_rows`] returning only the logit rows `cells` names
    /// (`row * NUM_CELLS + cell`, `row` indexing `rows`):
    /// `(cells.len(), DECL_LOGITS)`.
    fn diffusion_cells(
        &self,
        h: &[f16],
        rows: &[u32],
        decl: &[f32],
        cells: &[u32],
    ) -> anyhow::Result<Vec<f32>> {
        Ok(gather_rows(
            &self.diffusion_rows(h, rows, decl)?,
            cells,
            DECL_LOGITS,
        ))
    }
    /// [`Self::energy`], gathered as [`Self::diffusion_rows`].
    fn energy_rows(&self, h: &[f16], rows: &[u32], decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        self.energy(
            &gather_rows(h, rows, self.channels() * crate::map::NUM_CELLS),
            decl,
            rows.len(),
        )
    }

    /// Whether this net proposes turns with a best-move head (`best.onnx`)
    /// instead of the diffusion sampler.
    fn has_best(&self) -> bool {
        false
    }
    /// `h` → the best-move head's outputs `(b, NUM_CELLS, DECL_LOGITS)`: a
    /// regression toward the one-hot of every declaration field of the
    /// search's favourite turn, read back by argmax.
    fn best(&self, _h: &[f16], _b: usize) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("this net has no best-move head")
    }

    /// Begin a `T`-step diffusion pass with `h` held fixed.
    ///
    /// The default re-sends `h` every step. `OrtNet` overrides it to bind `h`
    /// once — but only when it has a session pool, because binding requires
    /// holding one session for the whole pass, which with a single shared
    /// session would serialize the self-play workers.
    fn diffusion_pass<'a>(
        &'a self,
        h: &'a [f16],
        b: usize,
    ) -> anyhow::Result<Box<dyn DiffusionPass + 'a>>;
    /// Upload `h` (whole rows) to a device-resident store for the rest of a
    /// wave's head calls, when the backend has one; the lease holds the rows
    /// until dropped. `None`: use the per-call methods.
    fn h_lease<'a>(&'a self, _h: &[f16]) -> anyhow::Result<Option<StoreLease<'a>>> {
        Ok(None)
    }
    /// [`Self::diffusion`] with row `i`'s features at store row `rows[i]`,
    /// returning only the logit rows `cells` names (`row * NUM_CELLS + cell`,
    /// `row` indexing `rows`): `(cells.len(), DECL_LOGITS)`.
    fn diffusion_store(
        &self,
        _rows: &[u32],
        _decl: &[f32],
        _cells: &[u32],
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("this backend has no h store")
    }
    /// [`Self::energy`] with row `i`'s features at store row `rows[i]`.
    fn energy_store(&self, _rows: &[u32], _decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("this backend has no h store")
    }

    /// Build anything a backend makes lazily on first use and that would
    /// otherwise stall a measured window (the ORT combiners' engine rebuild).
    fn warm(&self) -> anyhow::Result<()> {
        Ok(())
    }
    /// Flush any profiler traces (ORT's `MINIONS_ORT_PROFILE`); their paths.
    fn end_profiling(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Rows of trunk features uploaded for a wave (`Backend::h_lease`).
pub struct StoreLease<'a> {
    /// The store row holding each uploaded row, in upload order.
    pub rows: Vec<u32>,
    /// Whatever keeps those rows reserved; released on drop.
    pub hold: Box<dyn Send + 'a>,
}

/// Rows `rows` of `h` (each `stride` wide), in that order.
pub fn gather_rows<T: Copy>(h: &[T], rows: &[u32], stride: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(rows.len() * stride);
    for &r in rows {
        let r = r as usize;
        out.extend_from_slice(&h[r * stride..(r + 1) * stride]);
    }
    out
}

/// Repeat one position's trunk features across `n` rows, so a batch of `n`
/// declarations for the *same* state reuses a single trunk pass.
pub fn repeat_trunk<T: Copy>(h: &[T], n: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(h.len() * n);
    for _ in 0..n {
        out.extend_from_slice(h);
    }
    out
}

pub struct MinionsNet {
    pub channels: usize,
    stem: Conv2d,
    stem_bn: BatchNorm,
    blocks: Vec<ResBlock>,
    gpool: GlobalPoolBias,

    head_neck: ConvNeck1,
    head_block: ResBlock,
    val_neck: ConvNeck,
    val_conv: Conv2d,
    val_bn: BatchNorm,
    val_fc1: Linear,
    val_fc2: Linear,

    dif_in: Conv2d,
    dif_bn: BatchNorm,
    dif_block: ResBlock,
    dif_gpool: GlobalPoolBias,
    dif_out: Conv2d,

    eng_in: Conv2d,
    eng_bn: BatchNorm,
    eng_blocks: Vec<ResBlock>,
    eng_gpool: GlobalPoolBias,
    eng_fc1: Linear,
    eng_fc2: Linear,
}

impl MinionsNet {
    pub fn load(
        channels: usize,
        blocks_n: usize,
        energy_blocks_n: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let cfg1 = Conv2dConfig::default();
        let bn = BatchNormConfig::default();

        let mut blocks = Vec::with_capacity(blocks_n);
        let bvb = vb.pp("blocks");
        for i in 0..blocks_n {
            blocks.push(ResBlock::load(channels, bvb.pp(i.to_string()))?);
        }
        let mut eng_blocks = Vec::with_capacity(energy_blocks_n);
        let evb = vb.pp("eng_blocks");
        for i in 0..energy_blocks_n {
            eng_blocks.push(ResBlock::load(channels, evb.pp(i.to_string()))?);
        }

        // Must mirror `MinionsNet.neck` in minions_net.py.
        let neck = std::cmp::max(16, channels / 3);
        // Must mirror `head_width()` in minions_net.py.
        let head_ch = head_width(channels);

        Ok(Self {
            channels,
            stem: conv2d(IN_PLANES, channels, 3, cfg, vb.pp("stem"))?,
            stem_bn: batch_norm(channels, bn, vb.pp("stem_bn"))?,
            blocks,
            gpool: GlobalPoolBias::load(channels, vb.pp("gpool"))?,

            val_neck: ConvNeck::load(channels, neck, bn, vb.pp("val_neck"))?,
            val_conv: conv2d(neck, 1, 1, cfg1, vb.pp("val_conv"))?,
            val_bn: batch_norm(1, bn, vb.pp("val_bn"))?,
            val_fc1: linear(NUM_CELLS, 64, vb.pp("val_fc1"))?,
            val_fc2: linear(64, 1, vb.pp("val_fc2"))?,

            head_neck: ConvNeck1::load(channels, head_ch, bn, vb.pp("head_neck"))?,
            head_block: ResBlock::load(head_ch, vb.pp("head_block"))?,
            dif_in: conv2d(head_ch + DECL_PLANES, channels, 3, cfg, vb.pp("dif_in"))?,
            dif_bn: batch_norm(channels, bn, vb.pp("dif_bn"))?,
            dif_block: ResBlock::load(channels, vb.pp("dif_block"))?,
            dif_gpool: GlobalPoolBias::load(channels, vb.pp("dif_gpool"))?,
            dif_out: conv2d(channels, DECL_LOGITS, 1, cfg1, vb.pp("dif_out"))?,

            eng_in: conv2d(head_ch + ENERGY_PLANES, channels, 3, cfg, vb.pp("eng_in"))?,
            eng_bn: batch_norm(channels, bn, vb.pp("eng_bn"))?,
            eng_blocks,
            eng_gpool: GlobalPoolBias::load(channels, vb.pp("eng_gpool"))?,
            eng_fc1: linear(2 * channels, 64, vb.pp("eng_fc1"))?,
            eng_fc2: linear(64, 1, vb.pp("eng_fc2"))?,
        })
    }

    /// `(B, IN_PLANES, N, N)` → `(B, C, N, N)`. Computed once per position
    /// and reused across every diffusion step and every energy evaluation, which
    /// is what keeps the T-step sampling loop affordable.
    pub fn trunk(&self, planes: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self
            .stem_bn
            .forward_t(&self.stem.forward(planes)?, false)?
            .relu()?;
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        self.gpool.forward(&x)
    }

    /// Trunk → the features both declaration heads condition on.
    ///
    /// Computed once per position and reused across all `T` diffusion rounds
    /// and the energy pass — mirrors `MinionsNet.head_features`.
    pub fn head_features(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        self.head_block.forward(&self.head_neck.forward(h)?)
    }

    /// → `values[B]`.
    pub fn value(&self, h: &Tensor) -> candle_core::Result<Vec<f32>> {
        let vh = self.val_neck.forward(h)?;
        let v = self
            .val_bn
            .forward_t(&self.val_conv.forward(&vh)?, false)?
            .relu()?
            .flatten_from(1)?;
        let v = self.val_fc1.forward(&v)?.relu()?;
        self.val_fc2
            .forward(&v)?
            .tanh()?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    /// Declaration logits for a partially committed turn.
    /// `decl` is `(B, DECL_PLANES, N, N)`; result is `(B, NUM_CELLS, DECL_LOGITS)`.
    pub fn declaration_logits(&self, h: &Tensor, decl: &Tensor) -> candle_core::Result<Tensor> {
        let b = h.dim(0)?;
        let x = Tensor::cat(&[h, decl], 1)?;
        let x = self
            .dif_bn
            .forward_t(&self.dif_in.forward(&x)?, false)?
            .relu()?;
        let x = self.dif_gpool.forward(&self.dif_block.forward(&x)?)?;
        self.dif_out
            .forward(&x)?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((b, NUM_CELLS, DECL_LOGITS))
    }

    /// `f(s, a)` for a whole declaration → one score per batch row.
    pub fn energy(&self, h: &Tensor, decl: &Tensor) -> candle_core::Result<Vec<f32>> {
        let x = Tensor::cat(&[h, decl], 1)?;
        let mut x = self
            .eng_bn
            .forward_t(&self.eng_in.forward(&x)?, false)?
            .relu()?;
        for blk in &self.eng_blocks {
            x = blk.forward(&x)?;
        }
        let x = self.eng_gpool.forward(&x)?;
        let (b, c, hh, ww) = x.dims4()?;
        let flat = x.reshape((b, c, hh * ww))?;
        let pooled = Tensor::cat(&[flat.mean(D::Minus1)?, flat.max(D::Minus1)?], 1)?;
        let y = self.eng_fc1.forward(&pooled)?.relu()?;
        self.eng_fc2.forward(&y)?.flatten_all()?.to_vec1()
    }
}

/// Trunk size, read from the checkpoint's sibling `config.json`.
pub fn read_arch(checkpoint: &Path) -> (usize, usize, usize) {
    let (mut c, mut b, mut e) = (96usize, 8usize, 2usize);
    if let Some(dir) = checkpoint.parent() {
        if let Ok(txt) = std::fs::read_to_string(dir.join("config.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                let a = &v["arch"];
                c = a["channels"].as_u64().unwrap_or(c as u64) as usize;
                b = a["blocks"].as_u64().unwrap_or(b as u64) as usize;
                e = a["energy_blocks"].as_u64().unwrap_or(e as u64) as usize;
            }
        }
    }
    (c, b, e)
}

/// Candle builds both declaration heads `channels` wide with one diffusion
/// block. Distilled nets can size them independently (`dif_ch`, `dif_blocks`,
/// `eng_ch` in `config.json`); ORT reads those shapes from the graphs, but
/// here an extra diffusion block would be skipped without a word, so refuse.
fn check_default_heads(checkpoint: &Path, channels: usize) -> anyhow::Result<()> {
    let Some(txt) = checkpoint
        .parent()
        .and_then(|d| std::fs::read_to_string(d.join("config.json")).ok())
    else {
        return Ok(());
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return Ok(());
    };
    let a = &v["arch"];
    let dif_ch = a["dif_ch"].as_u64().unwrap_or(channels as u64) as usize;
    let eng_ch = a["eng_ch"].as_u64().unwrap_or(channels as u64) as usize;
    let dif_blocks = a["dif_blocks"].as_u64().unwrap_or(1) as usize;
    anyhow::ensure!(
        dif_ch == channels && eng_ch == channels && dif_blocks == 1,
        "candle backend supports only default head shapes \
         (dif_ch {dif_ch}, eng_ch {eng_ch}, dif_blocks {dif_blocks} vs channels {channels}); \
         use --backend ort"
    );
    Ok(())
}

/// Which inference backend to run.
///
/// Same split as Hex: candle is the development and parity path, ORT with the
/// TensorRT EP is the production one. On Hex that gap was 4.6k vs 158k
/// positions/s — roughly a generation per day versus one per half-hour — so
/// anything but a smoke test wants `Ort`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Candle { cuda: bool },
    Ort { ep: String, fp16: bool },
}

impl BackendKind {
    pub fn parse(name: &str, ep: &str, fp16: bool, cuda: bool) -> anyhow::Result<Self> {
        match name {
            "candle" => Ok(Self::Candle { cuda }),
            "ort" => Ok(Self::Ort {
                ep: ep.to_string(),
                fp16,
            }),
            other => anyhow::bail!("unknown --backend {other:?} (use candle|ort)"),
        }
    }

    /// The artefact whose mtime the actor watches to decide a hot-reload.
    pub fn watch_path(&self, live: &Path) -> PathBuf {
        match self {
            // Prefer the champion where one exists. That indirection *is* the
            // promotion gate: self-play data never comes from a net that lost
            // its match.
            Self::Candle { .. } => {
                let best = live.join("best.safetensors");
                if best.exists() {
                    best
                } else {
                    live.join("latest.safetensors")
                }
            }
            Self::Ort { .. } => {
                let best = live.join("best").join("trunk.onnx");
                if best.exists() {
                    best
                } else {
                    live.join("trunk.onnx")
                }
            }
        }
    }
}

/// Load whichever backend `kind` names, resolving champion-vs-latest as above.
pub fn load_backend(
    live: &Path,
    kind: &BackendKind,
    cache_dir: &str,
) -> anyhow::Result<Box<dyn Backend>> {
    let path = kind.watch_path(live);
    match kind {
        BackendKind::Candle { cuda } => Ok(Box::new(LocalNet::load(&path, *cuda)?)),
        BackendKind::Ort { ep, fp16 } => {
            let dir = path.parent().unwrap_or(live);
            Ok(Box::new(crate::ort_net::OrtNet::new(
                dir, ep, *fp16, cache_dir,
            )?))
        }
    }
}

/// The net plus its device, loaded from a checkpoint.
pub struct LocalNet {
    pub net: MinionsNet,
    pub device: Device,
}

/// So callers can reach the net's heads directly off the loaded handle.
impl std::ops::Deref for LocalNet {
    type Target = MinionsNet;
    fn deref(&self) -> &MinionsNet {
        &self.net
    }
}

impl LocalNet {
    pub fn load(checkpoint: &Path, cuda: bool) -> anyhow::Result<Self> {
        let device = if cuda {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        let (c, b, e) = read_arch(checkpoint);
        check_default_heads(checkpoint, c)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[checkpoint], DType::F32, &device)? };
        Ok(Self {
            net: MinionsNet::load(c, b, e, vb)?,
            device,
        })
    }

    pub fn planes_tensor(&self, planes: &[f32], b: usize) -> anyhow::Result<Tensor> {
        Ok(Tensor::from_slice(
            planes,
            (b, IN_PLANES, N, N),
            &self.device,
        )?)
    }

    pub fn decl_tensor(&self, decl: &[f32], b: usize) -> anyhow::Result<Tensor> {
        Ok(Tensor::from_slice(
            decl,
            (b, DECL_PLANES, N, N),
            &self.device,
        )?)
    }

    pub fn h_tensor(&self, h: &[f32], b: usize) -> anyhow::Result<Tensor> {
        let c = head_width(self.net.channels);
        Ok(Tensor::from_slice(h, (b, c, N, N), &self.device)?)
    }

    pub fn energy_decl_tensor(&self, decl: &[f32], b: usize) -> anyhow::Result<Tensor> {
        Ok(Tensor::from_slice(
            decl,
            (b, ENERGY_PLANES, N, N),
            &self.device,
        )?)
    }
}

impl Backend for LocalNet {
    fn channels(&self) -> usize {
        // The head-conditioning width, not the trunk width: this sizes the `h`
        // buffers the sampler repeats across candidate chains.
        head_width(self.net.channels)
    }

    fn trunk(&self, planes: &[f32], b: usize) -> anyhow::Result<TrunkOut> {
        let t = self.planes_tensor(planes, b)?;
        let h = self.net.trunk(&t)?;
        let value = self.net.value(&h)?;
        // Emit the head features, not the raw trunk: they are what the sampler
        // repeats across chains and reuses for every diffusion round.
        let hh: Vec<f32> = self.net.head_features(&h)?.flatten_all()?.to_vec1()?;
        Ok(TrunkOut {
            h: Vec::from_f32_slice(&hh),
            value,
        })
    }

    fn diffusion(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let ht = self.h_tensor(&h.to_f32_vec(), b)?;
        let dt = self.decl_tensor(decl, b)?;
        Ok(self
            .net
            .declaration_logits(&ht, &dt)?
            .flatten_all()?
            .to_vec1()?)
    }

    fn diffusion_pass<'a>(
        &'a self,
        h: &'a [f16],
        b: usize,
    ) -> anyhow::Result<Box<dyn DiffusionPass + 'a>> {
        Ok(Box::new(NaivePass { net: self, h, b }))
    }

    fn energy(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let ht = self.h_tensor(&h.to_f32_vec(), b)?;
        let dt = self.energy_decl_tensor(decl, b)?;
        Ok(self.net.energy(&ht, &dt)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-checks the candle net against PyTorch on a fixed checkpoint.
    ///
    /// Skipped unless `MINIONS_PARITY_DIR` points at a fixture directory holding
    /// `checkpoint.safetensors`, `config.json`, `input.bin` (planes then decl,
    /// f32 LE) and `expected.bin` (value, declaration logits, energy).
    /// `python/minions_parity_fixture.py` writes one.
    #[test]
    fn parity_with_python_net() {
        let Ok(dir) = std::env::var("MINIONS_PARITY_DIR") else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping parity check");
            return;
        };
        let dir = Path::new(&dir);
        let local = LocalNet::load(&dir.join("checkpoint.safetensors"), false)
            .expect("load fixture checkpoint");

        let read = |name: &str| -> Vec<f32> {
            let raw = std::fs::read(dir.join(name)).expect("fixture file");
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let input = read("input.bin");
        let expected = read("expected.bin");

        let b = 2usize;
        let np = b * IN_PLANES * NUM_CELLS;
        let nd = b * DECL_PLANES * NUM_CELLS;
        let planes = local.planes_tensor(&input[..np], b).unwrap();
        let decl = local.decl_tensor(&input[np..np + nd], b).unwrap();

        let h = local.net.trunk(&planes).unwrap();
        let vals = local.net.value(&h).unwrap();
        // Both declaration heads condition on the cached neck output, and the
        // energy head takes the narrower encoding (no move-commit plane).
        let hh = local.net.head_features(&h).unwrap();
        let logits = local.net.declaration_logits(&hh, &decl).unwrap();
        let logits: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let edecl_host: Vec<f32> = {
            let d = &input[np..np + nd];
            let mut out = Vec::with_capacity(b * ENERGY_PLANES * NUM_CELLS);
            for bi in 0..b {
                let base = bi * DECL_PLANES * NUM_CELLS;
                out.extend_from_slice(&d[base..base + 2 * NUM_CELLS]);
                out.extend_from_slice(&d[base + 3 * NUM_CELLS..base + DECL_PLANES * NUM_CELLS]);
            }
            out
        };
        let edecl = local.energy_decl_tensor(&edecl_host, b).unwrap();
        let energy = local.net.energy(&hh, &edecl).unwrap();

        let mut got = Vec::new();
        got.extend_from_slice(&vals);
        got.extend_from_slice(&logits);
        got.extend_from_slice(&energy);

        assert_eq!(got.len(), expected.len(), "output width mismatch");
        let mut worst = 0f32;
        for (i, (a, e)) in got.iter().zip(&expected).enumerate() {
            let d = (a - e).abs();
            if d > worst {
                worst = d;
            }
            assert!(d < 1e-4, "output {i} differs: candle {a} vs torch {e}");
        }
        eprintln!("parity ok, worst abs diff {worst:e}");
    }
}
