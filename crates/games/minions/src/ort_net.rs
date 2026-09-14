//! ONNX Runtime backend (CUDA / TensorRT EP, fp16) — the fast path for
//! self-play, using the GPU's tensor cores. Mirrors `hex/src/ort_net.rs`.
//!
//! Unlike Hex, which is one graph with two outputs, this loads **three**:
//! `trunk.onnx`, `diffusion.onnx` and `energy.onnx`. The sampler runs the trunk
//! once per position and then the diffusion head `T` times and the energy head
//! once against the cached features, so fusing them into one graph would
//! recompute the trunk on every round.
//!
//! The learner exports all three next to `latest.safetensors`; the actor picks
//! them up with `--backend ort`. The onnxruntime shared library is located at
//! runtime via `ORT_DYLIB_PATH`, and TensorRT caches built engines under
//! `<live>/trt_cache` — the first run pays the build, later ones do not.

use std::path::Path;
use std::sync::Mutex;

use ort::ep::cuda::ConvAlgorithmSearch;
use ort::execution_providers::{TensorRT, CUDA};
use ort::session::Session;
use ort::value::TensorRef;

use crate::game::PLANES as IN_PLANES;
use crate::map::{N, NUM_CELLS};
use crate::net::{Backend, DiffusionPass, NaivePass, TrunkOut, DECL_LOGITS};
use crate::turn::{DECL_PLANES, ENERGY_PLANES};

use half::f16;
use half::slice::HalfFloatSliceExt;
use half::vec::HalfFloatVecExt;

/// Whether this process runs the fp16-edge graphs, or -- when an `fp32/`
/// sibling export of the same net is loaded -- the fp32 ones. Always true
/// outside an A/B: `run_loop` flips it per window while `graph_f16_ab` exists,
/// only between windows, when no worker is mid-wave.
pub static GRAPH_F16_USE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether a graph takes and returns fp16 at the edges `h` crosses.
///
/// Current exports are fp16 at `planes`/`h`/`decl`/logits (`onnx_f16io.py`);
/// older snapshots, which the ladder still plays, are fp32 there. Read from
/// the session at load, so both kinds run with no flag, converting only at the
/// edges of an fp32 graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Edges {
    in16: bool,
    out16: bool,
}

fn edges(sess: &Session, input: &str, output: &str) -> Edges {
    let is16 = |outlets: &[ort::value::Outlet], name: &str| {
        outlets
            .iter()
            .find(|o| o.name() == name)
            .and_then(|o| o.dtype().tensor_type())
            == Some(ort::value::TensorElementType::Float16)
    };
    Edges {
        in16: is16(sess.inputs(), input),
        out16: is16(sess.outputs(), output),
    }
}

/// One output as f32, widened when the graph emits fp16.
macro_rules! take_f32 {
    ($outputs:expr, $name:expr, $out16:expr) => {
        if $out16 {
            let (_, v) = $outputs[$name].try_extract_tensor::<f16>().map_err(e2a)?;
            v.to_f32_vec()
        } else {
            let (_, v) = $outputs[$name].try_extract_tensor::<f32>().map_err(e2a)?;
            v.to_vec()
        }
    };
}

/// One output as fp16, narrowed when the graph emits f32.
macro_rules! take_f16 {
    ($outputs:expr, $name:expr, $out16:expr) => {
        if $out16 {
            let (_, v) = $outputs[$name].try_extract_tensor::<f16>().map_err(e2a)?;
            v.to_vec()
        } else {
            let (_, v) = $outputs[$name].try_extract_tensor::<f32>().map_err(e2a)?;
            Vec::<f16>::from_f32_slice(v)
        }
    };
}

fn e2a<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("ort: {e}")
}

pub struct OrtNet {
    /// One session per pool slot. The four self-play workers each run the whole
    /// pipeline (`selfplay::drive_pool`), so a single session per graph makes
    /// them queue on one mutex for every trunk/diffusion/energy call — which is
    /// why 4 workers only bought +17.5% over 1. A pool lets them run concurrently.
    ///
    /// `Option` only so [`Drop`] can take them out — see there.
    trunk: Option<Vec<Mutex<Session>>>,
    diffusion: Option<Vec<Mutex<Session>>>,
    energy: Option<Vec<Mutex<Session>>>,
    /// Edge dtypes of each graph, read at load (see [`Edges`]).
    trunk_io: Edges,
    diffusion_io: Edges,
    energy_io: Edges,
    /// The same net exported with fp32 edges, from `<dir>/fp32/`, when that
    /// exists: the other arm of a `graph_f16_ab` A/B. `None` otherwise.
    fp32_alt: Option<Box<OrtNet>>,
    /// Rotates the starting slot so callers do not all probe slot 0 first.
    rr: std::sync::atomic::AtomicUsize,
    channels: usize,
}

/// How many sessions to build per graph. `1` reproduces the old single-session
/// behaviour exactly, so this is A/B-able without a rebuild.
fn pool_size() -> usize {
    std::env::var("MINIONS_ORT_POOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
}

impl Drop for OrtNet {
    /// Deliberately leak the sessions.
    ///
    /// Dropping one built on the **TensorRT** EP aborts the process with heap
    /// corruption roughly one time in three, always at teardown — a race inside
    /// ORT/TRT's own shutdown, documented the same way in `hex/src/ort_net.rs`.
    /// That would be fatal for the actor, which drops its model every time the
    /// learner publishes new weights. The cost is three leaked sessions per
    /// hot-reload, i.e. per generation.
    fn drop(&mut self) {
        std::mem::forget(self.trunk.take());
        std::mem::forget(self.diffusion.take());
        std::mem::forget(self.energy.take());
    }
}

/// Read a boolean tuning knob from the environment, defaulting to `def`.
///
/// These are experiment knobs rather than user-facing settings — they exist so a
/// paired A/B can flip one provider option without rebuilding, which matters
/// because measuring anything here needs two binaries otherwise.
fn env_flag(name: &str, def: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(v.as_str(), "" | "0" | "false" | "no"),
        Err(_) => def,
    }
}

pub(crate) fn build(path: &Path, ep: &str, fp16: bool, cache_dir: &str) -> anyhow::Result<Session> {
    let cuda = || {
        CUDA::default()
            .with_device_id(0)
            .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
            // Tensor cores, and **off on purpose** — ORT already defaults
            // `use_tf32` off, and turning it on measured *slower* here: −7.0%,
            // −6.7%, −0.9% over three ABBA pairs. It does engage (the
            // candle/ORT parity error moves 3.3e-7 → 1.25e-4, the ~400x a
            // 10-bit mantissa predicts), it just does not pay: at 10x10 with
            // this channel count the convs are occupancy- and launch-limited
            // rather than FLOP-bound, so switching cuDNN to tensor-core
            // algorithms buys nothing and costs their tiling overhead. Same
            // reason the split-conv experiment lost despite removing 28% of the
            // arithmetic. Flip with MINIONS_TF32=1 if the net ever gets wide
            // enough to change that.
            .with_tf32(env_flag("MINIONS_TF32", false))
            // Tensor cores want NHWC, and this model is nothing but small convs.
            // Off by default until measured: the layout transposes it inserts at
            // the graph edges can cost more than the convs save at 10x10.
            .with_prefer_nhwc(env_flag("MINIONS_NHWC", false))
            .with_fuse_conv_bias(env_flag("MINIONS_FUSE_CONV_BIAS", true))
    };
    let providers = match ep {
        "trt" => {
            std::fs::create_dir_all(cache_dir).ok();
            vec![
                TensorRT::default()
                    .with_device_id(0)
                    .with_fp16(fp16)
                    .with_engine_cache(true)
                    .with_engine_cache_path(cache_dir.to_string())
                    .build(),
                cuda().build(), // per-node fallback where TRT cannot build
            ]
        }
        "cuda" => {
            // `fp16` reaches TensorRT only. The CUDA EP has no global
            // reduced-precision switch — it runs whatever dtype the graph
            // declares, and the exports are fp32 — so silently accepting
            // `--fp16` here would let a run believe it was using tensor cores
            // when it was not.
            if fp16 {
                eprintln!(
                    "WARNING: --fp16 has no effect with --ep cuda (the CUDA EP runs the \n\
                     graph's own dtype, and the ONNX exports are fp32). Tensor cores are \n\
                     still used via TF32; for true fp16 use --ep trt --fp16."
                );
            }
            vec![cuda().build()]
        }
        other => anyhow::bail!("unknown --ep {other:?} (use cuda|trt)"),
    };
    Session::builder()
        .map_err(e2a)?
        .with_execution_providers(providers)
        .map_err(e2a)?
        .commit_from_file(path)
        .map_err(e2a)
}

impl OrtNet {
    /// `dir` holds `trunk.onnx`, `diffusion.onnx`, `energy.onnx`.
    pub fn new(dir: &Path, ep: &str, fp16: bool, cache_dir: &str) -> anyhow::Result<Self> {
        Self::with_pool(dir, ep, fp16, cache_dir, pool_size())
    }

    /// As [`Self::new`] but with an explicit pool size, so tests can exercise
    /// the bound-pass path without racing on a process-wide env var.
    pub fn with_pool(
        dir: &Path,
        ep: &str,
        fp16: bool,
        cache_dir: &str,
        n_pool: usize,
    ) -> anyhow::Result<Self> {
        let trunk_path = dir.join("trunk.onnx");
        // Engines collide across checkpoints unless the cache is keyed by model
        // content — see `playground_core::trt_cache` and the 15-hour Hex run it
        // was written after. Keyed on the trunk, which changes every generation.
        let cache = playground_core::trt_cache::content_keyed(cache_dir, &trunk_path)?;
        let n_pool = n_pool.max(1);
        let pool = |path: std::path::PathBuf| -> anyhow::Result<Vec<Mutex<Session>>> {
            (0..n_pool)
                .map(|_| build(&path, ep, fp16, &cache).map(Mutex::new))
                .collect()
        };
        let trunk = pool(trunk_path.clone())?;

        // Trunk width from the sidecar `config.json`, the same source candle's
        // loader uses, so the two backends cannot disagree about the shape. The
        // first `trunk` call asserts the graph really matches.
        // `read_arch` looks for `config.json` beside the path it is given. When
        // the graphs live in `<live>/best/`, the config is one level up.
        let mut probe = dir.join("checkpoint.safetensors");
        if !dir.join("config.json").exists() {
            if let Some(parent) = dir.parent() {
                probe = parent.join("checkpoint.safetensors");
            }
        }
        let (channels, _, _) = crate::net::read_arch(&probe);

        let diffusion = pool(dir.join("diffusion.onnx"))?;
        let energy = pool(dir.join("energy.onnx"))?;
        let io = |p: &[Mutex<Session>], i: &str, o: &str| edges(&p[0].lock().unwrap(), i, o);
        let (trunk_io, diffusion_io, energy_io) = (
            io(&trunk, "planes", "h"),
            io(&diffusion, "h", "logits"),
            io(&energy, "h", "energy"),
        );
        let alt = dir.join("fp32");
        let fp32_alt = if alt.join("trunk.onnx").exists() {
            eprintln!("loaded the fp32 graph set in {} (A/B arm)", alt.display());
            Some(Box::new(Self::with_pool(
                &alt, ep, fp16, cache_dir, n_pool,
            )?))
        } else {
            None
        };
        Ok(Self {
            trunk: Some(trunk),
            diffusion: Some(diffusion),
            energy: Some(energy),
            trunk_io,
            diffusion_io,
            energy_io,
            fp32_alt,
            rr: std::sync::atomic::AtomicUsize::new(0),
            channels,
        })
    }

    /// Take the first free session in the pool, else block on a rotating slot.
    /// `try_lock` first means an idle session is always preferred over waiting.
    fn acquire<'a>(&self, pool: &'a [Mutex<Session>]) -> std::sync::MutexGuard<'a, Session> {
        let start = self.rr.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for i in 0..pool.len() {
            if let Ok(g) = pool[(start + i) % pool.len()].try_lock() {
                return g;
            }
        }
        pool[start % pool.len()].lock().unwrap()
    }

    /// One head call. `h` is fp16 on the host; it and `decl` are widened, and
    /// the output converted, only for a graph whose edges are fp32. Borrowed
    /// views, not copies: at `max_batch x K` rows these buffers run to ~10-19 MB
    /// per call, `T` times per expansion.
    #[allow(clippy::too_many_arguments)]
    fn run_head(
        &self,
        which: &Option<Vec<Mutex<Session>>>,
        io: Edges,
        h: &[f16],
        decl: &[f32],
        b: usize,
        output: &str,
        decl_planes: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let c = crate::net::head_width(self.channels) as i64;
        let hshape = vec![b as i64, c, N as i64, N as i64];
        let dshape = vec![b as i64, decl_planes as i64, N as i64, N as i64];
        let mut sess = self.acquire(which.as_ref().expect("session outlives run"));
        if io.in16 {
            let d16: Vec<f16> = Vec::from_f32_slice(decl);
            let ht = TensorRef::from_array_view((hshape, h)).map_err(e2a)?;
            let dt = TensorRef::from_array_view((dshape, &d16[..])).map_err(e2a)?;
            let outputs = sess
                .run(ort::inputs!["h" => ht, "decl" => dt])
                .map_err(e2a)?;
            Ok(take_f32!(outputs, output, io.out16))
        } else {
            let h32 = h.to_f32_vec();
            let ht = TensorRef::from_array_view((hshape, &h32[..])).map_err(e2a)?;
            let dt = TensorRef::from_array_view((dshape, decl)).map_err(e2a)?;
            let outputs = sess
                .run(ort::inputs!["h" => ht, "decl" => dt])
                .map_err(e2a)?;
            Ok(take_f32!(outputs, output, io.out16))
        }
    }

    /// The graph set this window runs: the fp32 sibling while an A/B has the
    /// fp16 set switched off, otherwise this one.
    fn active(&self) -> &OrtNet {
        match &self.fp32_alt {
            Some(alt) if !GRAPH_F16_USE.load(std::sync::atomic::Ordering::Relaxed) => alt,
            _ => self,
        }
    }
}

impl Backend for OrtNet {
    fn channels(&self) -> usize {
        // Head-conditioning width — `trunk.onnx` emits `head_features(h)`.
        crate::net::head_width(self.channels)
    }

    fn trunk(&self, planes: &[f32], b: usize) -> anyhow::Result<TrunkOut> {
        let net = self.active();
        let shape = vec![b as i64, IN_PLANES as i64, N as i64, N as i64];
        let mut sess = net.acquire(net.trunk.as_ref().expect("session outlives forward"));
        // Features are small integers, exact in fp16.
        let (h, value) = if net.trunk_io.in16 {
            let p16: Vec<f16> = Vec::from_f32_slice(planes);
            let input = TensorRef::from_array_view((shape, &p16[..])).map_err(e2a)?;
            let outputs = sess.run(ort::inputs!["planes" => input]).map_err(e2a)?;
            (
                take_f16!(outputs, "h", net.trunk_io.out16),
                take_f32!(outputs, "value", false),
            )
        } else {
            let input = TensorRef::from_array_view((shape, planes)).map_err(e2a)?;
            let outputs = sess.run(ort::inputs!["planes" => input]).map_err(e2a)?;
            (
                take_f16!(outputs, "h", net.trunk_io.out16),
                take_f32!(outputs, "value", false),
            )
        };
        // Cheap guard against a config.json that has drifted from the graph:
        // silently wrong channel counts would reshape every later head call.
        // `trunk.onnx` emits `head_features(h)`, so the expected width is the
        // head width derived from the trunk width, not the trunk width itself.
        let want = crate::net::head_width(net.channels);
        if h.len() != b * want * NUM_CELLS {
            anyhow::bail!(
                "trunk.onnx emitted {} values for batch {b}, but config.json says {} trunk \
                 channels -> {want} head channels — regenerate the ONNX exports",
                h.len(),
                net.channels
            );
        }
        Ok(TrunkOut { h, value })
    }

    fn diffusion(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let net = self.active();
        let out = net.run_head(
            &net.diffusion,
            net.diffusion_io,
            h,
            decl,
            b,
            "logits",
            DECL_PLANES,
        )?;
        debug_assert_eq!(out.len(), b * NUM_CELLS * DECL_LOGITS);
        Ok(out)
    }

    fn energy(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let net = self.active();
        net.run_head(
            &net.energy,
            net.energy_io,
            h,
            decl,
            b,
            "energy",
            ENERGY_PLANES,
        )
    }

    /// Per-step by default: with fp16 edges `h` is already in the graph's
    /// dtype, so a pass has nothing to convert or cache.
    ///
    /// `BoundPass` would bind `h` on the device once, but it is broken (see
    /// there), fp32-only, and needs a session pool, since it holds one session
    /// for the whole pass -- so it stays behind `MINIONS_ORT_BIND`.
    fn diffusion_pass<'a>(
        &'a self,
        h: &'a [f16],
        b: usize,
    ) -> anyhow::Result<Box<dyn DiffusionPass + 'a>> {
        let net = self.active();
        let pool = net.diffusion.as_ref().expect("session outlives run");
        if !net.diffusion_io.in16 && pool.len() >= 2 && env_flag("MINIONS_ORT_BIND", false) {
            return Ok(Box::new(BoundPass::new(
                net.acquire(pool),
                &h.to_f32_vec(),
                b,
                net.channels,
            )?));
        }
        Ok(Box::new(NaivePass { net, h, b }))
    }
}

/// A diffusion pass holding one pooled session with `h` already on the device.
///
/// **DEAD CODE — `IoBinding` is broken in ort 2.0.0-rc.12 on the CUDA EP.**
/// Gated off behind `MINIONS_ORT_BIND`; do not enable it.
///
/// The prize was real: binding `h` once instead of re-sending it on all `T`
/// diffusion steps measured +15.7% on the head (`bench-bind`, ABBA, 14 reps),
/// and the head is ~71% of self-play at steady state. But `run_binding` only
/// works *once* per session. The first call returns the right answer
/// (reproducibly -2.213553 on the fixture); every later call on that session
/// returns garbage, drifting run to run with byte-identical inputs.
///
/// Ruled out, each by a direct experiment (`bound_h_is_not_mutated_across_runs`):
///   - output buffer reuse — a fresh output `Tensor` per step behaves identically
///     to one reusable bound output, so the output path is not involved;
///   - ORT mutating the bound `h` in place — `max |h_now - h_orig|` is exactly 0;
///   - the queued host->device copy racing the run — `synchronize_inputs` before
///     every run changes nothing;
///   - binding *reuse* — a fresh `IoBinding` per run fails the same way, which
///     also means there is no version of this that keeps `h` resident.
///
/// So the corruption is session-level and survives rebuilding the binding. The
/// route back is an ort upgrade (rc.12 is a release candidate) or a different
/// mechanism for device-resident inputs; re-run
/// `bound_pass_matches_per_step_path` against any new ort before trusting it.
struct BoundPass<'a> {
    sess: std::sync::MutexGuard<'a, Session>,
    binding: ort::session::IoBinding,
    b: usize,
}

impl<'a> BoundPass<'a> {
    fn new(
        sess: std::sync::MutexGuard<'a, Session>,
        h: &[f32],
        b: usize,
        channels: usize,
    ) -> anyhow::Result<Self> {
        use ort::value::Tensor;
        let shape = vec![b as i64, channels as i64, N as i64, N as i64];
        // `bind_input` copies at bind time and the session reuses that copy on
        // every later `run_binding`, so this is the one and only upload of `h`.
        let ht = Tensor::from_array((shape, h.to_vec())).map_err(e2a)?;
        let mut binding = sess.create_binding().map_err(e2a)?;
        binding.bind_input("h", &ht).map_err(e2a)?;
        Ok(Self { sess, binding, b })
    }
}

impl DiffusionPass for BoundPass<'_> {
    fn step(&mut self, decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        use ort::value::Tensor;
        let shape = vec![self.b as i64, DECL_PLANES as i64, N as i64, N as i64];
        let dt = Tensor::from_array((shape, decl.to_vec())).map_err(e2a)?;
        self.binding.bind_input("decl", &dt).map_err(e2a)?;
        // A *fresh* output tensor every step. Binding one reusable output buffer
        // (`bind_output_to_device`) is what the parity test caught: with the
        // same `decl` repeated, successive steps returned -2.87, 30.49, 60.34 —
        // the buffer was being accumulated into rather than overwritten. The
        // upload this optimisation exists to remove is `h`, not the output, so
        // allocating here costs nothing that matters.
        let n_out = self.b * NUM_CELLS * DECL_LOGITS;
        let ot = Tensor::from_array((
            vec![self.b as i64, NUM_CELLS as i64, DECL_LOGITS as i64],
            vec![0f32; n_out],
        ))
        .map_err(e2a)?;
        self.binding.bind_output("logits", ot).map_err(e2a)?;
        let out = self.sess.run_binding(&self.binding).map_err(e2a)?;
        self.binding.synchronize_outputs().map_err(e2a)?;
        let (_, v) = out["logits"].try_extract_tensor::<f32>().map_err(e2a)?;
        Ok(v.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Minions, PLANES};
    use crate::net::LocalNet;
    use playground_core::Game;

    /// Does ORT mutate the bound `h` buffer across runs?
    ///
    /// Both output strategies (one reusable bound output, and a fresh output per
    /// step) produced byte-identical drift, which rules out the output path and
    /// points at the input. ORT may run in-place ops on a bound input buffer.
    #[test]
    fn bound_h_is_not_mutated_across_runs() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            return;
        };
        let dir = std::path::Path::new(&dir);
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        let Ok(ort) = OrtNet::with_pool(dir, "cuda", false, &cache, 2) else {
            return;
        };
        let b = 4usize;
        let c = ort.channels();
        let h0: Vec<f32> = (0..b * c * NUM_CELLS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let decl: Vec<f32> = vec![0.25f32; b * DECL_PLANES * NUM_CELLS];

        use ort::value::Tensor;
        let pool = ort.diffusion.as_ref().unwrap();
        let mut sess = pool[0].lock().unwrap();
        let ht =
            Tensor::from_array((vec![b as i64, c as i64, N as i64, N as i64], h0.clone())).unwrap();
        // Fresh binding per run: if this is deterministic while a reused
        // binding is not, the bug is binding *reuse* — which is the whole
        // point of the optimisation.
        for run in 0..3 {
            let mut binding = sess.create_binding().unwrap();
            binding.bind_input("h", &ht).unwrap();
            let dt = Tensor::from_array((
                vec![b as i64, DECL_PLANES as i64, N as i64, N as i64],
                decl.clone(),
            ))
            .unwrap();
            binding.bind_input("decl", &dt).unwrap();
            let ot = Tensor::from_array((
                vec![b as i64, NUM_CELLS as i64, DECL_LOGITS as i64],
                vec![0f32; b * NUM_CELLS * DECL_LOGITS],
            ))
            .unwrap();
            binding.bind_output("logits", ot).unwrap();
            // `bind_input` only *queues* the host->device copy.
            binding.synchronize_inputs().unwrap();
            let out = sess.run_binding(&binding).unwrap();
            let (_, v) = out["logits"].try_extract_tensor::<f32>().unwrap();
            let first = v[0];
            drop(out);
            // Read the h tensor back: has ORT written into our buffer?
            let (_, hnow) = ht.try_extract_tensor::<f32>().unwrap();
            let drift: f32 = hnow
                .iter()
                .zip(h0.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            println!("run {run}: logits[0]={first:.6}  max |h_now - h_orig| = {drift:.6}");
        }
    }

    /// Diagnostic: is the mismatch from rebinding `decl`, or from `h`?
    #[test]
    fn bound_pass_diagnosis() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            return;
        };
        let dir = std::path::Path::new(&dir);
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        let Ok(ort) = OrtNet::with_pool(dir, "cuda", false, &cache, 2) else {
            return;
        };
        let b = 4usize;
        let c = ort.channels();
        let h: Vec<f32> = (0..b * c * NUM_CELLS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let h: Vec<f16> = Vec::from_f32_slice(&h);
        let d0: Vec<f32> = (0..b * DECL_PLANES * NUM_CELLS)
            .map(|i| ((i % 7) as f32 - 3.0) / 3.0)
            .collect();
        let d1: Vec<f32> = (0..b * DECL_PLANES * NUM_CELLS)
            .map(|i| (((i + 13) % 7) as f32 - 3.0) / 3.0)
            .collect();
        let direct0 = ort.diffusion(&h, &d0, b).unwrap();
        let direct1 = ort.diffusion(&h, &d1, b).unwrap();
        println!("direct0[0]={:.6}  direct1[0]={:.6}", direct0[0], direct1[0]);

        // (a) same decl three times through one bound pass
        let mut pass = ort.diffusion_pass(&h, b).unwrap();
        for i in 0..3 {
            let got = pass.step(&d0).unwrap();
            println!(
                "same-decl step {i}: {:.6}  (direct0 {:.6})",
                got[0], direct0[0]
            );
        }
        drop(pass);
        // (b) d0 then d1
        let mut pass = ort.diffusion_pass(&h, b).unwrap();
        let a = pass.step(&d0).unwrap();
        let bb = pass.step(&d1).unwrap();
        println!("d0 then d1: {:.6} then {:.6}", a[0], bb[0]);
        println!(
            "  matches direct0? {}   matches direct1? {}",
            (bb[0] - direct0[0]).abs() < 1e-4,
            (bb[0] - direct1[0]).abs() < 1e-4
        );
    }

    /// The bound pass must return exactly what the per-step path returns.
    ///
    /// `BoundPass` uploads `h` once and reuses it for every later step, so a
    /// stale or wrongly-shaped binding would silently feed the sampler features
    /// from the wrong rows — producing plausible-looking logits and quietly
    /// corrupting the training data. Nothing else covers it: the Rust suite runs
    /// on candle, which takes `NaivePass`. Needs `MINIONS_ORT_DIR` and a GPU.
    #[test]
    fn bound_pass_matches_per_step_path() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping bound-pass parity");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        // Pool of 2 so `diffusion_pass` takes the binding path, not the fallback.
        let ort = match OrtNet::with_pool(dir, "cuda", false, &cache, 2) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ORT unavailable ({e}) — skipping");
                return;
            }
        };
        assert!(
            OrtNet::with_pool(dir, "cuda", false, &cache, 1).is_ok(),
            "single-session construction must still work"
        );

        let b = 4usize;
        let c = ort.channels();
        // Distinct per-row values: a binding that reused row 0 for every row, or
        // transposed the batch, would still look sane against uniform input.
        let h: Vec<f32> = (0..b * c * NUM_CELLS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let h: Vec<f16> = Vec::from_f32_slice(&h);
        let steps = 4usize;
        let decls: Vec<Vec<f32>> = (0..steps)
            .map(|t| {
                (0..b * DECL_PLANES * NUM_CELLS)
                    .map(|i| (((i + t * 13) % 7) as f32 - 3.0) / 3.0)
                    .collect()
            })
            .collect();

        let mut pass = ort.diffusion_pass(&h, b).expect("open bound pass");
        for (t, decl) in decls.iter().enumerate() {
            let bound = pass.step(decl).expect("bound step");
            let direct = ort.diffusion(&h, decl, b).expect("direct step");
            assert_eq!(bound.len(), direct.len(), "step {t}: length");
            for (i, (x, y)) in bound.iter().zip(direct.iter()).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-4 * (1.0 + y.abs()),
                    "step {t}, element {i}: bound {x} vs per-step {y}"
                );
            }
        }
    }

    /// Candle and ORT must agree on all three graphs.
    ///
    /// They load different artefacts — safetensors versus exported ONNX — so a
    /// bad export, a transposed reshape, or a head wired to the wrong output
    /// name would show up here and nowhere else. Skipped unless
    /// `MINIONS_ORT_DIR` points at a directory holding both.
    #[test]
    fn ort_agrees_with_candle() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping ORT parity");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let candle =
            LocalNet::load(&dir.join("latest.safetensors"), false).expect("load safetensors");
        let ort = match OrtNet::new(dir, "cuda", false, &dir.join("trt_cache").to_string_lossy()) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ORT unavailable ({e}) — skipping");
                return;
            }
        };
        assert_eq!(ort.channels(), candle.channels(), "trunk width disagrees");

        // A real position, not zeros: an all-zero input can hide a wrong
        // reshape, since every channel looks alike.
        let mut s = Minions::initial_state();
        s = Minions::apply(&s, &crate::game::MinionsMove::EndPhase);
        let planes: Vec<f32> = Minions::to_features(&s).iter().map(|&b| b as f32).collect();
        assert_eq!(planes.len(), PLANES * NUM_CELLS);

        let tc = candle.trunk(&planes, 1).unwrap();
        let to = ort.trunk(&planes, 1).unwrap();
        // Relative tolerance, not absolute. The two backends run different
        // convolution kernels on different devices, so fp32 accumulation order
        // differs and the error compounds through the residual blocks —
        // agreement is to a few 1e-4 on activations of order 10, which is
        // numerical noise rather than a wiring bug.
        let worst = |a: &[f32], b: &[f32]| {
            let abs = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let scale = a.iter().fold(1.0f32, |m, v| m.max(v.abs()));
            abs / scale
        };
        assert!(
            worst(&tc.h.to_f32_vec(), &to.h.to_f32_vec()) < 1e-3,
            "trunk differs by {}",
            worst(&tc.h.to_f32_vec(), &to.h.to_f32_vec())
        );
        assert!(
            worst(&tc.value, &to.value) < 1e-3,
            "value differs by {}",
            worst(&tc.value, &to.value)
        );

        // Heads, batched over two distinct declarations so a batch-axis bug
        // cannot hide behind a single row.
        let live = crate::turn::LiveMask::of(&s);
        let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(4);
        let mut decl = vec![0f32; 2 * DECL_PLANES * NUM_CELLS];
        for i in 0..2 {
            let spec = crate::turn::TurnSpec::random_plausible(&s, &mut rng);
            let p = crate::turn::Partial::fully_committed(&spec, &live);
            p.write_planes(
                &mut decl[i * DECL_PLANES * NUM_CELLS..(i + 1) * DECL_PLANES * NUM_CELLS],
            );
        }
        let h2 = crate::net::repeat_trunk(&tc.h, 2);
        let dc = candle.diffusion(&h2, &decl, 2).unwrap();
        let do_ = ort.diffusion(&h2, &decl, 2).unwrap();
        assert_eq!(dc.len(), do_.len());
        assert!(
            worst(&dc, &do_) < 1e-3,
            "diffusion differs by {}",
            worst(&dc, &do_)
        );

        let ec = candle.energy(&h2, &decl, 2).unwrap();
        let eo = ort.energy(&h2, &decl, 2).unwrap();
        assert!(
            worst(&ec, &eo) < 1e-3,
            "energy differs by {}",
            worst(&ec, &eo)
        );
        eprintln!(
            "ort parity ok — trunk {:e}, diffusion {:e}, energy {:e}",
            worst(&tc.h.to_f32_vec(), &to.h.to_f32_vec()),
            worst(&dc, &do_),
            worst(&ec, &eo)
        );
    }
}

/// Microbenchmark: is re-sending the trunk features on every diffusion step
/// actually costing us? Drives the raw session with f32 tensors, so it needs
/// an fp32-edge export (`export_onnx_atomic(..., f16_io=False)`). Arm A rebinds `h` each run (what the sampler does
/// today); arm B binds `h` once and rebinds only `decl`. Interleaved ABBA,
/// because sequential arms on this thermally-capped GPU drift ~19%.
pub fn bench_binding(
    dir: &Path,
    ep: &str,
    fp16: bool,
    cache_dir: &str,
    rows: usize,
    steps: usize,
    reps: usize,
) -> anyhow::Result<()> {
    use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
    use ort::value::Tensor;

    let trunk_path = dir.join("trunk.onnx");
    let cache = playground_core::trt_cache::content_keyed(cache_dir, &trunk_path)?;
    let mut probe = dir.join("checkpoint.safetensors");
    if !dir.join("config.json").exists() {
        if let Some(parent) = dir.parent() {
            probe = parent.join("checkpoint.safetensors");
        }
    }
    let (channels, _, _) = crate::net::read_arch(&probe);
    let mut sess = build(&dir.join("diffusion.onnx"), ep, fp16, &cache)?;

    let c = channels as i64;
    let b = rows as i64;
    let h: Vec<f32> = (0..(rows * channels * NUM_CELLS))
        .map(|i| (i % 17) as f32 * 0.01)
        .collect();
    let mut decl: Vec<f32> = vec![0.0; rows * DECL_PLANES * NUM_CELLS];

    // Arm A: exactly today's path — a fresh TensorRef for both inputs, every step.
    let arm_a = |sess: &mut Session, decl: &mut Vec<f32>| -> anyhow::Result<f64> {
        let t0 = std::time::Instant::now();
        for step in 0..steps {
            let n = decl.len();
            decl[step % n] = step as f32;
            let ht = TensorRef::from_array_view((vec![b, c, N as i64, N as i64], h.as_slice()))
                .map_err(e2a)?;
            let dt = TensorRef::from_array_view((
                vec![b, DECL_PLANES as i64, N as i64, N as i64],
                decl.as_slice(),
            ))
            .map_err(e2a)?;
            let out = sess
                .run(ort::inputs!["h" => ht, "decl" => dt])
                .map_err(e2a)?;
            let (_, v) = out["logits"].try_extract_tensor::<f32>().map_err(e2a)?;
            std::hint::black_box(v.len());
        }
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    };

    // Arm B: bind `h` once per batch, rebind only `decl` per step.
    let arm_b = |sess: &mut Session, decl: &mut Vec<f32>| -> anyhow::Result<f64> {
        let t0 = std::time::Instant::now();
        let mut binding = sess.create_binding().map_err(e2a)?;
        let ht = Tensor::from_array((vec![b, c, N as i64, N as i64], h.clone())).map_err(e2a)?;
        binding.bind_input("h", &ht).map_err(e2a)?;
        let cuda_out = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .map_err(e2a)?;
        binding
            .bind_output_to_device("logits", &cuda_out)
            .map_err(e2a)?;
        for step in 0..steps {
            let n = decl.len();
            decl[step % n] = step as f32;
            let dt = Tensor::from_array((
                vec![b, DECL_PLANES as i64, N as i64, N as i64],
                decl.clone(),
            ))
            .map_err(e2a)?;
            binding.bind_input("decl", &dt).map_err(e2a)?;
            let out = sess.run_binding(&binding).map_err(e2a)?;
            let (_, v) = out["logits"].try_extract_tensor::<f32>().map_err(e2a)?;
            std::hint::black_box(v.len());
        }
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    };

    // Warm up both paths (TRT engine build, allocator growth) before timing.
    arm_a(&mut sess, &mut decl)?;
    arm_b(&mut sess, &mut decl)?;

    let (mut ta, mut tb) = (0.0, 0.0);
    println!("rows {rows}  steps {steps}  reps {reps}  (ABBA interleaved)");
    for rep in 0..reps {
        let a1 = arm_a(&mut sess, &mut decl)?;
        let b1 = arm_b(&mut sess, &mut decl)?;
        let b2 = arm_b(&mut sess, &mut decl)?;
        let a2 = arm_a(&mut sess, &mut decl)?;
        let (a, bb) = ((a1 + a2) / 2.0, (b1 + b2) / 2.0);
        ta += a;
        tb += bb;
        println!(
            "  rep {rep}: rebind-h {a:7.2} ms   bind-once {bb:7.2} ms   {:+.1}%",
            (a - bb) / a * 100.0
        );
    }
    println!(
        "mean: rebind-h {:.2} ms   bind-once {:.2} ms   delta {:+.1}%",
        ta / reps as f64,
        tb / reps as f64,
        (ta - tb) / ta * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod native_f16_tests {
    use super::*;
    use crate::game::Minions;
    use playground_core::Game;

    /// Graphs exported with fp16 edges compute what the fp32 export of the same
    /// weights computes, up to fp16 rounding at the edges. A mis-wired Cast, a
    /// swapped input or a stale head is O(1) off, not O(1e-3). Heads are fed the
    /// fp32 graph's `h`, so their error is not compounded with the trunk's.
    /// Needs the GPU and both exports:
    ///
    ///   MINIONS_ORT_DIR16=<dir> MINIONS_ORT_DIR32=<dir> \
    ///     cargo test --release -p minions-engine -- --ignored native_f16
    #[test]
    #[ignore = "needs a GPU and two exports of one net"]
    fn native_f16_graphs_match_fp32_graphs() {
        let (Ok(d16), Ok(d32)) = (
            std::env::var("MINIONS_ORT_DIR16"),
            std::env::var("MINIONS_ORT_DIR32"),
        ) else {
            return;
        };
        let cache = std::env::temp_dir().join("minions-native-f16-test-trt");
        let cache = cache.to_str().unwrap();
        let n16 = OrtNet::with_pool(Path::new(&d16), "trt", true, cache, 1).unwrap();
        let n32 = OrtNet::with_pool(Path::new(&d32), "trt", true, cache, 1).unwrap();
        let all16 = Edges {
            in16: true,
            out16: true,
        };
        assert!(
            n16.trunk_io == all16 && n16.diffusion_io == all16 && n16.energy_io == all16,
            "{d16} is not an fp16-edge export: {:?} {:?} {:?}",
            n16.trunk_io,
            n16.diffusion_io,
            n16.energy_io
        );
        assert!(
            !n32.trunk_io.in16 && !n32.diffusion_io.in16 && !n32.energy_io.in16,
            "{d32} is not fp32"
        );

        let mut st = Minions::initial_state();
        let mut planes = Vec::new();
        for _ in 0..8 {
            planes.extend(Minions::to_features(&st).iter().map(|&x| x as f32));
            let legal = Minions::legal_moves(&st);
            if legal.is_empty() {
                break;
            }
            st = Minions::apply(&st, &legal[legal.len() / 2]);
        }
        let b = planes.len() / (IN_PLANES * NUM_CELLS);
        let (t16, t32) = (
            n16.trunk(&planes, b).unwrap(),
            n32.trunk(&planes, b).unwrap(),
        );
        let rel = |a: &[f32], z: &[f32]| {
            let scale = a.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            a.iter().zip(z).fold(0f32, |m, (p, q)| m.max((p - q).abs())) / scale
        };
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        };
        let decl: Vec<f32> = (0..b * DECL_PLANES * NUM_CELLS)
            .map(|_| (rnd() < 0.2) as u8 as f32)
            .collect();
        let edecl: Vec<f32> = (0..b * ENERGY_PLANES * NUM_CELLS)
            .map(|_| (rnd() < 0.2) as u8 as f32)
            .collect();
        let d32 = n32.diffusion(&t32.h, &decl, b).unwrap();
        let rh = rel(&t32.h.to_f32_vec(), &t16.h.to_f32_vec());
        let rd = rel(&d32, &n16.diffusion(&t32.h, &decl, b).unwrap());
        let re = rel(
            &n32.energy(&t32.h, &edecl, b).unwrap(),
            &n16.energy(&t32.h, &edecl, b).unwrap(),
        );
        let rp = rel(
            &d32,
            &n16.diffusion_pass(&t32.h, b).unwrap().step(&decl).unwrap(),
        );
        // `value` is a tanh near zero on these positions, so an error relative
        // to its largest magnitude overstates it. Judge it absolutely, and --
        // given the checkpoint -- against candle's fp32 truth: the question is
        // whether fp16 edges make TensorRT's value *worse*, not whether two fp16
        // engines agree to the last bit.
        let abs =
            |a: &[f32], z: &[f32]| a.iter().zip(z).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
        let va = abs(&t32.value, &t16.value);
        eprintln!(
            "value, fp32 edges {:?}\nvalue, fp16 edges {:?}",
            t32.value, t16.value
        );
        if let Ok(ck) = std::env::var("MINIONS_CKPT") {
            let truth = crate::net::LocalNet::load(Path::new(&ck), false)
                .unwrap()
                .trunk(&planes, b)
                .unwrap()
                .value;
            let (e32, e16) = (abs(&truth, &t32.value), abs(&truth, &t16.value));
            eprintln!("value, candle fp32 {truth:?}\nvalue error vs fp32 truth: fp32 edges {e32:.2e}  fp16 edges {e16:.2e}");
            assert!(
                e16 <= 2.0 * e32 + 1e-2,
                "fp16 edges make value worse: {e16} vs {e32}"
            );
        }
        eprintln!("fp16 vs fp32 edges (b={b}): trunk h {rh:.2e}  value |d| {va:.2e}  diffusion {rd:.2e}  energy {re:.2e}  pass {rp:.2e}");
        assert!(va < 5e-2, "value differs by {va}");
        for (name, r) in [
            ("trunk h", rh),
            ("diffusion", rd),
            ("energy", re),
            ("pass", rp),
        ] {
            assert!(r < 2e-2, "{name} disagrees by {r}");
        }
    }
}
