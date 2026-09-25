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

use crate::combine::{Axis, Combiner, HStore, RunFn};
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

/// One output copied out raw, so the session can be released *before* the
/// fp16 -> f32 widening: that conversion took ~115 us of a ~1.4 ms diffusion
/// hold at ~160 rows, while callers queued ~1.4 ms for the session.
enum Raw {
    F16(Vec<f16>),
    F32(Vec<f32>),
}

impl Raw {
    fn into_f32(self) -> Vec<f32> {
        match self {
            Raw::F16(v) => v.to_f32_vec(),
            Raw::F32(v) => v,
        }
    }
}

macro_rules! take_raw {
    ($outputs:expr, $name:expr, $out16:expr) => {
        if $out16 {
            let (_, v) = $outputs[$name].try_extract_tensor::<f16>().map_err(e2a)?;
            Raw::F16(v.to_vec())
        } else {
            let (_, v) = $outputs[$name].try_extract_tensor::<f32>().map_err(e2a)?;
            Raw::F32(v.to_vec())
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

/// Per-graph call accounting, so a window can say *where* a head call's time
/// goes: waiting for the pooled session, host-side conversion, `Session::run`
/// (upload + kernels + download + sync), and extracting the output.
///
/// Added 2026-09-24 because a head 2.3x narrower self-played at the same rate
/// as the full one -- so the head's time is not in its arithmetic, and the
/// phase counters (which time the whole call) cannot say what it is in.
pub mod call_stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub const TRUNK: usize = 0;
    pub const DIFFUSION: usize = 1;
    pub const ENERGY: usize = 2;
    const NAMES: [&str; 3] = ["trunk", "diffusion", "energy"];
    /// Row-count buckets: <=32, <=128, <=512, <=2048, larger.
    const EDGES: [u64; 4] = [32, 128, 512, 2048];

    #[derive(Default)]
    pub struct Graph {
        calls: AtomicU64,
        rows: AtomicU64,
        wait_ns: AtomicU64,
        prep_ns: AtomicU64,
        run_ns: AtomicU64,
        post_ns: AtomicU64,
        hist: [AtomicU64; 5],
    }

    pub static GRAPHS: [Graph; 3] = [Graph::new(), Graph::new(), Graph::new()];

    impl Graph {
        const fn new() -> Self {
            Self {
                calls: AtomicU64::new(0),
                rows: AtomicU64::new(0),
                wait_ns: AtomicU64::new(0),
                prep_ns: AtomicU64::new(0),
                run_ns: AtomicU64::new(0),
                post_ns: AtomicU64::new(0),
                hist: [
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                ],
            }
        }
    }

    /// One call's split, in nanoseconds.
    pub fn record(g: usize, rows: usize, wait: u64, prep: u64, run: u64, post: u64) {
        let s = &GRAPHS[g];
        s.calls.fetch_add(1, Relaxed);
        s.rows.fetch_add(rows as u64, Relaxed);
        s.wait_ns.fetch_add(wait, Relaxed);
        s.prep_ns.fetch_add(prep, Relaxed);
        s.run_ns.fetch_add(run, Relaxed);
        s.post_ns.fetch_add(post, Relaxed);
        let b = EDGES.iter().position(|&e| rows as u64 <= e).unwrap_or(4);
        s.hist[b].fetch_add(1, Relaxed);
    }

    /// Combined calls (`Combiner`) per graph: count, rows, GPU-call time.
    static BATCHES: [[AtomicU64; 3]; 3] = [
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
    ];

    pub fn record_batch(g: usize, rows: usize, ns: u64) {
        BATCHES[g][0].fetch_add(1, Relaxed);
        BATCHES[g][1].fetch_add(rows as u64, Relaxed);
        BATCHES[g][2].fetch_add(ns, Relaxed);
    }

    static UPLOADS: AtomicU64 = AtomicU64::new(0);
    static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);

    /// A `DevicePass` uploading its `h` once.
    pub fn record_upload(_rows: usize, ns: u64) {
        UPLOADS.fetch_add(1, Relaxed);
        UPLOAD_NS.fetch_add(ns, Relaxed);
    }

    /// The window's numbers since the last call, then reset. `None` when no
    /// ORT call was made (candle backend).
    pub fn take_report(secs: f64) -> Option<String> {
        let mut parts = Vec::new();
        for (g, name) in NAMES.iter().enumerate() {
            let s = &GRAPHS[g];
            let calls = s.calls.swap(0, Relaxed);
            let rows = s.rows.swap(0, Relaxed);
            let ms = |a: &AtomicU64| a.swap(0, Relaxed) as f64 / 1e6;
            let (wait, prep, run, post) = (
                ms(&s.wait_ns),
                ms(&s.prep_ns),
                ms(&s.run_ns),
                ms(&s.post_ns),
            );
            let hist: Vec<u64> = s.hist.iter().map(|h| h.swap(0, Relaxed)).collect();
            if calls == 0 {
                continue;
            }
            let per = |x: f64| 1000.0 * x / calls as f64;
            parts.push(format!(
                "{name} {:.0}/s x {:.0} rows: wait {:.0} prep {:.0} run {:.0} post {:.0} us/call (hist <=32/128/512/2k/more {:?})",
                calls as f64 / secs,
                rows as f64 / calls as f64,
                per(wait),
                per(prep),
                per(run),
                per(post),
                hist
            ));
        }
        for (g, name) in NAMES.iter().enumerate() {
            let n = BATCHES[g][0].swap(0, Relaxed);
            let rows = BATCHES[g][1].swap(0, Relaxed);
            let ns = BATCHES[g][2].swap(0, Relaxed);
            if n > 0 {
                parts.push(format!(
                    "{name} combined {:.0}/s x {:.0} rows, {:.0} us each ({:.0}% busy)",
                    n as f64 / secs,
                    rows as f64 / n as f64,
                    ns as f64 / 1e3 / n as f64,
                    ns as f64 / 1e7 / secs
                ));
            }
        }
        let ups = UPLOADS.swap(0, Relaxed);
        let up_ns = UPLOAD_NS.swap(0, Relaxed);
        if ups > 0 {
            parts.push(format!(
                "h uploads {:.0}/s x {:.0} us",
                ups as f64 / secs,
                up_ns as f64 / 1e3 / ups as f64
            ));
        }
        (!parts.is_empty()).then(|| format!("ort calls: {}", parts.join("  |  ")))
    }
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
    /// The proposer: exactly one of `diffusion` and `best` is loaded, from
    /// whichever of `diffusion.onnx` and `best.onnx` the export wrote.
    diffusion: Option<Vec<Mutex<Session>>>,
    best: Option<Vec<Mutex<Session>>>,
    energy: Option<Vec<Mutex<Session>>>,
    /// Edge dtypes of each graph, read at load (see [`Edges`]).
    trunk_io: Edges,
    diffusion_io: Edges,
    best_io: Edges,
    energy_io: Edges,
    /// The same net exported with fp32 edges, from `<dir>/fp32/`, when that
    /// exists: the other arm of a `graph_f16_ab` A/B. `None` otherwise.
    fp32_alt: Option<Box<OrtNet>>,
    /// Rotates the starting slot so callers do not all probe slot 0 first.
    rr: std::sync::atomic::AtomicUsize,
    channels: usize,
    /// Device buffers for `DevicePass`'s `h`, by row count, reused across
    /// passes: allocating one per pass (ort's `Tensor::to`) cost 4.3 ms, six
    /// times the pass it was saving on. Row counts come from a small fixed set
    /// (the sampler pads to TensorRT's shapes), so this stays small.
    dev_h: Mutex<std::collections::HashMap<usize, Vec<ort::value::Tensor<f16>>>>,
    /// The diffusion session's CUDA allocator, kept alive with the buffers.
    dev_alloc: Mutex<Option<ort::memory::Allocator>>,
    /// Pinned host buffers for head calls, by `(graph, rows)` -- see [`Pinned`].
    pinned: Mutex<std::collections::HashMap<(usize, usize), Vec<Pinned>>>,
    /// `CUDA_PINNED` allocators for inputs and outputs, made on first use.
    pin_alloc: Mutex<Option<(ort::memory::Allocator, ort::memory::Allocator)>>,
    /// Cross-worker batching per graph (`call_stats` index), made on first use.
    /// Cross-worker batching per graph, by `COMB_*` slot, made on first use.
    combiners: [std::sync::OnceLock<Combiner>; 5],
    combine_init: Mutex<()>,
    /// The store-indexed head graphs (`diffusion_g.onnx`, `energy_g.onnx`),
    /// when the export wrote them; see [`HStore`].
    diffusion_g: Option<Vec<Mutex<Session>>>,
    energy_g: Option<Vec<Mutex<Session>>>,
    store: std::sync::OnceLock<HStore>,
}

fn store_on() -> bool {
    STORE_USE.load(std::sync::atomic::Ordering::Relaxed) || FORCE_COMBINE.with(|f| f.get())
}

/// Combiner slots: one per graph a combined call can run.
const COMB_TRUNK: usize = call_stats::TRUNK;
const COMB_DIFFUSION_G: usize = 3;
const COMB_ENERGY_G: usize = 4;

/// Whether waves may use the device `h` store and the store-indexed graphs.
/// Self-play flips it per window for an in-process A/B (`store_ab` in the live
/// dir). Opt-in with `MINIONS_ORT_STORE=1`: once head calls were combined,
/// the bytes it saves stopped mattering (production A/B 2026-09-25, -0.5%).
pub static STORE_USE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether head calls are batched across workers ([`Combiner`]). Self-play
/// flips it per window for an in-process A/B (`combine_ab` in the live dir);
/// `MINIONS_ORT_COMBINE=0` turns it off.
pub static COMBINE_USE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

thread_local! {
    /// Set by `OrtNet::warm` to create the combiners whatever `COMBINE_USE`
    /// says this window.
    static FORCE_COMBINE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn combine_on() -> bool {
    env_flag("MINIONS_ORT_COMBINE", true)
        && (COMBINE_USE.load(std::sync::atomic::Ordering::Relaxed)
            || FORCE_COMBINE.with(|f| f.get()))
}

/// Rows one combined call may hold (`MINIONS_ORT_COMBINE_ROWS`, default 2048).
fn combine_cap() -> usize {
    std::env::var("MINIONS_ORT_COMBINE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 16)
        .unwrap_or(2048)
}

/// A head call's `h`: one contiguous block, or rows gathered from a per-position
/// block (`Backend::diffusion_rows`) -- copied straight into the pinned input,
/// so the gathered copy is never built on its own.
#[derive(Clone, Copy)]
pub(crate) enum HSrc<'a> {
    Contig(&'a [f16]),
    /// `(h, rows, stride)`: row `i` of the call is row `rows[i]` of `h`.
    Rows(&'a [f16], &'a [u32], usize),
}

impl HSrc<'_> {
    fn fill(&self, out: &mut [f16]) {
        match *self {
            HSrc::Contig(h) => out.copy_from_slice(h),
            HSrc::Rows(h, rows, stride) => {
                for (o, &r) in out.chunks_exact_mut(stride).zip(rows) {
                    let r = r as usize;
                    o.copy_from_slice(&h[r * stride..(r + 1) * stride]);
                }
            }
        }
    }

    fn contiguous(&self) -> std::borrow::Cow<'_, [f16]> {
        match *self {
            HSrc::Contig(h) => std::borrow::Cow::Borrowed(h),
            HSrc::Rows(h, rows, stride) => {
                std::borrow::Cow::Owned(crate::net::gather_rows(h, rows, stride))
            }
        }
    }
}

/// One head call's inputs and output in page-locked host memory, reused.
///
/// From pageable memory every call's copies are staged through a driver
/// bounce buffer and synchronous: the ORT trace put ~200 us of a 536 us
/// diffusion `model_run` outside the TensorRT kernel, which is what ~1.66 MB
/// at pageable bandwidth costs. Pinned buffers let those copies go straight
/// to DMA, and filling them *before* taking the session means the lock --
/// the diffusion session is the actor's one saturated resource -- is held
/// only for the run itself. Measured 2026-09-24; `MINIONS_ORT_PINNED=0`
/// turns it off.
struct Pinned {
    /// The graph's fp16 inputs, in the order the caller named them.
    ins: Vec<ort::value::Tensor<f16>>,
    /// Holds the preallocated outputs: ORT writes each run's results into them.
    run: ort::session::RunOptions<ort::session::HasSelectedOutputs>,
    /// Each output's host data (owned by `run`, which outlives every read) and
    /// element count, in the order the caller named them.
    outs: Vec<(*const u8, usize)>,
}

// SAFETY: `outs` points into tensors `run` owns; a `Pinned` is only ever used by
// the one thread that popped it from the pool.
unsafe impl Send for Pinned {}

impl Pinned {
    /// Output `i` as f32, read after the run and outside the session lock.
    fn out_f32(&self, i: usize, is_f16: bool) -> Vec<f32> {
        let (p, n) = self.outs[i];
        // SAFETY: see the type -- the buffer is live, `n` elements of this type.
        unsafe {
            if is_f16 {
                std::slice::from_raw_parts(p as *const f16, n).to_f32_vec()
            } else {
                std::slice::from_raw_parts(p as *const f32, n).to_vec()
            }
        }
    }

    fn out_f16(&self, i: usize) -> Vec<f16> {
        let (p, n) = self.outs[i];
        // SAFETY: as `out_f32`; the caller asked for this output as fp16.
        unsafe { std::slice::from_raw_parts(p as *const f16, n).to_vec() }
    }
}

/// Whether calls may use every session in a graph's pool, or only the first.
/// Self-play flips it per window for an in-process A/B (`pool_ab` in the live
/// dir); the pool itself is sized at load by `MINIONS_ORT_POOL`.
pub static POOL_USE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether head and trunk calls take the [`Pinned`] path. Self-play flips it per
/// window for an in-process A/B (`pinned_ab` in the live dir).
pub static PINNED_USE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn pinned_on() -> bool {
    PINNED_USE.load(std::sync::atomic::Ordering::Relaxed) && env_flag("MINIONS_ORT_PINNED", true)
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
        std::mem::forget(self.best.take());
        std::mem::forget(self.energy.take());
        // Device buffers belong to the leaked sessions' allocator.
        std::mem::forget(std::mem::take(&mut *self.dev_h.lock().unwrap()));
        std::mem::forget(self.dev_alloc.lock().unwrap().take());
        std::mem::forget(std::mem::take(&mut *self.pinned.lock().unwrap()));
        std::mem::forget(self.pin_alloc.lock().unwrap().take());
        for c in std::mem::take(&mut self.combiners) {
            std::mem::forget(c);
        }
        std::mem::forget(self.diffusion_g.take());
        std::mem::forget(self.energy_g.take());
        std::mem::forget(std::mem::take(&mut self.store));
    }
}

/// Read a boolean tuning knob from the environment, defaulting to `def`.
///
/// These are experiment knobs rather than user-facing settings — they exist so a
/// paired A/B can flip one provider option without rebuilding, which matters
/// because measuring anything here needs two binaries otherwise.
/// `out`'s logit rows `cells` (`row * NUM_CELLS + cell`), or all of it.
fn pick_cells(out: Vec<f32>, cells: Option<&[u32]>) -> Vec<f32> {
    match cells {
        Some(cells) => crate::net::gather_rows(&out, cells, DECL_LOGITS),
        None => out,
    }
}

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
                    // CUDA graph capture: replay one captured launch sequence
                    // instead of re-issuing every kernel. Self-play is
                    // launch-bound -- ~500 launches/s, >=8 per wave, with the
                    // device only ~35% busy -- so this is the targeted fix.
                    //
                    // OFF by default because it is NOT known to work here yet.
                    // ORT's CUDA-EP documents three requirements: fixed input
                    // shapes, fixed input/output *addresses* (i.e. `IoBinding`),
                    // and a session that is not shared between threads. All
                    // three are violated today: `width_for` feeds ~48 distinct
                    // row counts per graph, `IoBinding` is broken on
                    // ort 2.0.0-rc.12 (see `BoundPass`), and `acquire()` hands
                    // any free pooled session to any worker.
                    //
                    // MEASURED 2026-09-18, and it settles the open question:
                    // the TensorRT flag does NOT sidestep `IoBinding`. With
                    // capture on, the very first inference panics inside ort --
                    // `value/mod.rs:353: expected typeinfo_ptr to not be null`
                    // -- before a single trunk call completes, while the same
                    // binary with capture off ran 9205 trunk and 13019
                    // diffusion calls. A null output typeinfo is what a session
                    // that requires pre-bound, pre-allocated outputs produces.
                    // So this is blocked on the same broken `IoBinding` as
                    // `BoundPass`, and needs an ort upgrade before it is worth
                    // any further work. Leave it off.
                    .with_cuda_graph(env_flag("MINIONS_TRT_CUDA_GRAPH", false))
                    .with_engine_cache(true)
                    .with_engine_cache_path(cache_dir.to_string())
                    // Experiment knobs (unset = TensorRT/ORT defaults).
                    .with_builder_optimization_level(
                        std::env::var("MINIONS_TRT_OPT_LEVEL")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(3),
                    )
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
    let mut b = Session::builder()
        .map_err(e2a)?
        .with_execution_providers(providers)
        .map_err(e2a)?;
    // Per-node timings (including host<->device copies) as Chrome-trace JSON,
    // one file per session: `<prefix>_<graph>_<timestamp>.json`.
    if let Ok(prefix) = std::env::var("MINIONS_ORT_PROFILE") {
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        b = b.with_profiling(format!("{prefix}_{stem}")).map_err(e2a)?;
    }
    b.commit_from_file(path).map_err(e2a)
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

        let (dif_path, best_path) = (dir.join("diffusion.onnx"), dir.join("best.onnx"));
        anyhow::ensure!(
            dif_path.exists() != best_path.exists(),
            "{} must hold exactly one proposer graph, diffusion.onnx or best.onnx",
            dir.display()
        );
        let diffusion = if dif_path.exists() {
            Some(pool(dif_path)?)
        } else {
            None
        };
        let best = if best_path.exists() {
            Some(pool(best_path)?)
        } else {
            None
        };
        let energy = pool(dir.join("energy.onnx"))?;
        let io = |p: &[Mutex<Session>], i: &str, o: &str| edges(&p[0].lock().unwrap(), i, o);
        let absent = Edges {
            in16: false,
            out16: false,
        };
        let (trunk_io, energy_io) = (io(&trunk, "planes", "h"), io(&energy, "h", "energy"));
        let diffusion_io = diffusion
            .as_deref()
            .map_or(absent, |p| io(p, "h", "logits"));
        let best_io = best.as_deref().map_or(absent, |p| io(p, "h", "logits"));
        let (dg_path, eg_path) = (dir.join("diffusion_g.onnx"), dir.join("energy_g.onnx"));
        let (diffusion_g, energy_g) = if env_flag("MINIONS_ORT_STORE", false)
            && diffusion_io.in16
            && diffusion_io.out16
            && dg_path.exists()
            && eg_path.exists()
        {
            (Some(pool(dg_path)?), Some(pool(eg_path)?))
        } else {
            (None, None)
        };
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
            diffusion,
            best,
            energy: Some(energy),
            trunk_io,
            diffusion_io,
            best_io,
            energy_io,
            fp32_alt,
            rr: std::sync::atomic::AtomicUsize::new(0),
            channels,
            dev_h: Mutex::new(Default::default()),
            dev_alloc: Mutex::new(None),
            pinned: Mutex::new(Default::default()),
            pin_alloc: Mutex::new(None),
            combiners: Default::default(),
            combine_init: Mutex::new(()),
            diffusion_g,
            energy_g,
            store: Default::default(),
        })
    }

    /// Flush `MINIONS_ORT_PROFILE` traces: ORT writes them only when profiling
    /// ends, and these sessions are never dropped (see [`Drop`]).
    pub fn end_profiling_all(&self) -> Vec<String> {
        let mut out = Vec::new();
        for pool in [&self.trunk, &self.diffusion, &self.best, &self.energy]
            .into_iter()
            .flatten()
        {
            for s in pool {
                if let Ok(p) = s.lock().unwrap().end_profiling() {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Take the first free session in the pool, else block on a rotating slot.
    /// `try_lock` first means an idle session is always preferred over waiting.
    fn acquire<'a>(&self, pool: &'a [Mutex<Session>]) -> std::sync::MutexGuard<'a, Session> {
        let pool = if POOL_USE.load(std::sync::atomic::Ordering::Relaxed) {
            pool
        } else {
            &pool[..1]
        };
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
        h: HSrc<'_>,
        decl: &[f32],
        b: usize,
        output: &str,
        decl_planes: usize,
        graph: usize,
        cells: Option<&[u32]>,
    ) -> anyhow::Result<Vec<f32>> {
        use std::time::Instant;
        let c = crate::net::head_width(self.channels) as i64;
        let hshape = vec![b as i64, c, N as i64, N as i64];
        let dshape = vec![b as i64, decl_planes as i64, N as i64, N as i64];
        let ns = |a: Instant, b: Instant| (b - a).as_nanos() as u64;
        let pool = which.as_ref().expect("session outlives run");
        if io.in16 && io.out16 && pinned_on() && combine_on() && b <= combine_cap() {
            let run = |ins: Vec<ort::value::DynValue>,
                       mut outs: Vec<ort::value::DynValue>,
                       rows: usize,
                       _: usize,
                       slot: usize|
             -> anyhow::Result<()> {
                use ort::session::{OutputSelector, RunOptions};
                let tr = Instant::now();
                let opts = RunOptions::new().map_err(e2a)?.with_outputs(
                    OutputSelector::no_default()
                        .with(output)
                        .preallocate(output, outs.remove(0)),
                );
                let mut sess = pool[slot % pool.len()].lock().unwrap();
                drop(
                    sess.run_with_options(ort::inputs!["h" => &ins[0], "decl" => &ins[1]], &opts)
                        .map_err(e2a)?,
                );
                drop(sess);
                call_stats::record_batch(graph, rows, tr.elapsed().as_nanos() as u64);
                Ok(())
            };
            let comb = self.combiner(
                graph,
                pool,
                &[("h", Axis::Rows), ("decl", Axis::Rows)],
                &[(output, Axis::Rows)],
                0,
                &run,
            )?;
            let t0 = Instant::now();
            let mut out = Vec::new();
            comb.submit(
                b,
                0,
                &mut |_, _, sl| {
                    h.fill(sl[0].f16());
                    sl[1].f16().convert_from_f32_slice(decl);
                    Ok(())
                },
                &mut |o| {
                    let o = o[0].f16();
                    out = match cells {
                        // Only the named logit rows, converted straight out of
                        // the arena: ~8 of a position's 49 cells are live.
                        Some(cells) => {
                            let w = DECL_LOGITS;
                            let mut v = vec![0f32; cells.len() * w];
                            for (d, &c) in v.chunks_exact_mut(w).zip(cells) {
                                let c = c as usize;
                                o[c * w..(c + 1) * w].convert_to_f32_slice(d);
                            }
                            v
                        }
                        None => o.to_f32_vec(),
                    };
                    Ok(())
                },
                &run,
            )?;
            call_stats::record(graph, b, 0, 0, ns(t0, Instant::now()), 0);
            return Ok(out);
        }
        if io.in16 && io.out16 && pinned_on() {
            let tp = Instant::now();
            let mut pin = self.take_pinned(
                pool,
                graph,
                b,
                &[("h", hshape), ("decl", dshape)],
                &[(output, true)],
            )?;
            h.fill(pin.ins[0].try_extract_tensor_mut::<f16>().map_err(e2a)?.1);
            pin.ins[1]
                .try_extract_tensor_mut::<f16>()
                .map_err(e2a)?
                .1
                .convert_from_f32_slice(decl);
            let t0 = Instant::now();
            let mut sess = self.acquire(pool);
            let t1 = Instant::now();
            drop(
                sess.run_with_options(
                    ort::inputs!["h" => &pin.ins[0], "decl" => &pin.ins[1]],
                    &pin.run,
                )
                .map_err(e2a)?,
            );
            drop(sess);
            let t2 = Instant::now();
            let out = pin.out_f32(0, true);
            self.pinned
                .lock()
                .unwrap()
                .entry((graph, b))
                .or_default()
                .push(pin);
            call_stats::record(
                graph,
                b,
                ns(t0, t1),
                ns(tp, t0),
                ns(t1, t2),
                ns(t2, Instant::now()),
            );
            return Ok(pick_cells(out, cells));
        }
        let hv = h.contiguous();
        let h: &[f16] = &hv;
        let t0 = Instant::now();
        let mut sess = self.acquire(pool);
        let t1 = Instant::now();
        if io.in16 {
            let d16: Vec<f16> = Vec::from_f32_slice(decl);
            let ht = TensorRef::from_array_view((hshape, h)).map_err(e2a)?;
            let dt = TensorRef::from_array_view((dshape, &d16[..])).map_err(e2a)?;
            let t2 = Instant::now();
            let raw = {
                let outputs = sess
                    .run(ort::inputs!["h" => ht, "decl" => dt])
                    .map_err(e2a)?;
                take_raw!(outputs, output, io.out16)
            };
            drop(sess);
            let t3 = Instant::now();
            let out = raw.into_f32();
            call_stats::record(
                graph,
                b,
                ns(t0, t1),
                ns(t1, t2),
                ns(t2, t3),
                ns(t3, Instant::now()),
            );
            Ok(pick_cells(out, cells))
        } else {
            let h32 = h.to_f32_vec();
            let ht = TensorRef::from_array_view((hshape, &h32[..])).map_err(e2a)?;
            let dt = TensorRef::from_array_view((dshape, decl)).map_err(e2a)?;
            let t2 = Instant::now();
            let outputs = sess
                .run(ort::inputs!["h" => ht, "decl" => dt])
                .map_err(e2a)?;
            let t3 = Instant::now();
            let out = take_f32!(outputs, output, io.out16);
            call_stats::record(
                graph,
                b,
                ns(t0, t1),
                ns(t1, t2),
                ns(t2, t3),
                ns(t3, Instant::now()),
            );
            Ok(pick_cells(out, cells))
        }
    }

    /// The combiner in slot `slot` (see `COMB_*`), made and warmed on first
    /// use: under the init lock, so exactly one worker pays for the engine's
    /// range rebuild.
    fn combiner(
        &self,
        slot: usize,
        pool: &[Mutex<Session>],
        ins: &[(&str, Axis)],
        outs: &[(&str, Axis)],
        cap_cells: usize,
        run: &RunFn<'_>,
    ) -> anyhow::Result<&Combiner> {
        if let Some(c) = self.combiners[slot].get() {
            return Ok(c);
        }
        let _g = self.combine_init.lock().unwrap();
        if let Some(c) = self.combiners[slot].get() {
            return Ok(c);
        }
        let made = {
            let sess = self.acquire(pool);
            Combiner::new(&sess, ins, outs, combine_cap(), cap_cells, pool.len())?
        };
        made.warm(run)?;
        let _ = self.combiners[slot].set(made);
        Ok(self.combiners[slot].get().expect("just set"))
    }

    /// The device `h` store, made on first use.
    fn store(&self) -> anyhow::Result<&HStore> {
        if let Some(s) = self.store.get() {
            return Ok(s);
        }
        let _g = self.combine_init.lock().unwrap();
        if let Some(s) = self.store.get() {
            return Ok(s);
        }
        let rows: usize = std::env::var("MINIONS_ORT_STORE_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let pool = self
            .diffusion_g
            .as_ref()
            .expect("store needs the store graphs");
        let made = HStore::new(
            &pool[0].lock().unwrap(),
            crate::net::head_width(self.channels),
            rows,
        )?;
        let _ = self.store.set(made);
        Ok(self.store.get().expect("just set"))
    }

    /// A [`Pinned`] set for `graph` at `b` rows, from the pool or made now.
    /// Made at most a few times per row count: each worker holds one only for
    /// the length of a call. `outputs` pairs each name with whether it is fp16
    /// (else f32).
    fn take_pinned(
        &self,
        pool: &[Mutex<Session>],
        graph: usize,
        b: usize,
        inputs: &[(&str, Vec<i64>)],
        outputs: &[(&str, bool)],
    ) -> anyhow::Result<Pinned> {
        use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
        use ort::session::{OutputSelector, RunOptions};
        if let Some(p) = self
            .pinned
            .lock()
            .unwrap()
            .get_mut(&(graph, b))
            .and_then(|v| v.pop())
        {
            return Ok(p);
        }
        let sess = self.acquire(pool);
        let mut oshapes: Vec<Vec<i64>> = Vec::new();
        for (name, _) in outputs {
            oshapes.push(
                sess.outputs()
                    .iter()
                    .find(|o| o.name() == *name)
                    .and_then(|o| o.dtype().tensor_shape())
                    .ok_or_else(|| anyhow::anyhow!("no tensor output {name}"))?
                    .iter()
                    .map(|&d| if d < 0 { b as i64 } else { d })
                    .collect(),
            );
        }
        let mut alloc = self.pin_alloc.lock().unwrap();
        if alloc.is_none() {
            let mi =
                |t| MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, t);
            *alloc = Some((
                Allocator::new(&sess, mi(MemoryType::CPUInput).map_err(e2a)?).map_err(e2a)?,
                Allocator::new(&sess, mi(MemoryType::CPUOutput).map_err(e2a)?).map_err(e2a)?,
            ));
        }
        drop(sess);
        let (ain, aout) = alloc.as_ref().unwrap();
        let mut ins = Vec::new();
        for (_, shape) in inputs {
            ins.push(ort::value::Tensor::<f16>::new(ain, shape.clone()).map_err(e2a)?);
        }
        let mut sel = OutputSelector::no_default();
        let mut outs = Vec::new();
        for (&(name, is_f16), shape) in outputs.iter().zip(oshapes) {
            sel = sel.with(name);
            sel = if is_f16 {
                let o = ort::value::Tensor::<f16>::new(aout, shape).map_err(e2a)?;
                let (_, v) = o.try_extract_tensor::<f16>().map_err(e2a)?;
                outs.push((v.as_ptr() as *const u8, v.len()));
                sel.preallocate(name, o)
            } else {
                let o = ort::value::Tensor::<f32>::new(aout, shape).map_err(e2a)?;
                let (_, v) = o.try_extract_tensor::<f32>().map_err(e2a)?;
                outs.push((v.as_ptr() as *const u8, v.len()));
                sel.preallocate(name, o)
            };
        }
        let run = RunOptions::new().map_err(e2a)?.with_outputs(sel);
        Ok(Pinned { ins, run, outs })
    }

    /// The best-move head: `h` in, one row of outputs per position.
    fn run_best(&self, h: &[f16], b: usize) -> anyhow::Result<Vec<f32>> {
        let pool = self
            .best
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("this net has no best-move head (no best.onnx)"))?;
        let c = crate::net::head_width(self.channels) as i64;
        let hshape = vec![b as i64, c, N as i64, N as i64];
        let io = self.best_io;
        let mut sess = self.acquire(pool);
        let out = if io.in16 {
            let ht = TensorRef::from_array_view((hshape, h)).map_err(e2a)?;
            let outputs = sess.run(ort::inputs!["h" => ht]).map_err(e2a)?;
            take_f32!(outputs, "logits", io.out16)
        } else {
            let h32 = h.to_f32_vec();
            let ht = TensorRef::from_array_view((hshape, &h32[..])).map_err(e2a)?;
            let outputs = sess.run(ort::inputs!["h" => ht]).map_err(e2a)?;
            take_f32!(outputs, "logits", io.out16)
        };
        Ok(out)
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
    fn end_profiling(&self) -> Vec<String> {
        self.end_profiling_all()
    }

    /// One tiny call through each head's combiner, which creates it and runs
    /// its range-widening batch now rather than in the first measured window.
    fn warm(&self) -> anyhow::Result<()> {
        if !(env_flag("MINIONS_ORT_COMBINE", true) && env_flag("MINIONS_ORT_PINNED", true)) {
            return Ok(());
        }
        FORCE_COMBINE.with(|f| f.set(true));
        let res = (|| -> anyhow::Result<()> {
            let b = 16usize;
            let h = vec![f16::ZERO; b * self.channels() * NUM_CELLS];
            self.trunk(&vec![0.0; b * IN_PLANES * NUM_CELLS], b)?;
            if self.diffusion.is_some() {
                self.diffusion(&h, &vec![0.0; b * DECL_PLANES * NUM_CELLS], b)?;
            }
            self.energy(&h, &vec![0.0; b * ENERGY_PLANES * NUM_CELLS], b)?;
            if let Some(lease) = self.h_lease(&h)? {
                self.diffusion_store(&lease.rows, &vec![0.0; b * DECL_PLANES * NUM_CELLS], &[0])?;
                self.energy_store(&lease.rows, &vec![0.0; b * ENERGY_PLANES * NUM_CELLS])?;
            }
            Ok(())
        })();
        FORCE_COMBINE.with(|f| f.set(false));
        res
    }

    fn channels(&self) -> usize {
        // Head-conditioning width — `trunk.onnx` emits `head_features(h)`.
        crate::net::head_width(self.channels)
    }

    fn trunk(&self, planes: &[f32], b: usize) -> anyhow::Result<TrunkOut> {
        let net = self.active();
        let shape = vec![b as i64, IN_PLANES as i64, N as i64, N as i64];
        let pool = net.trunk.as_ref().expect("session outlives forward");
        let ns = |a: std::time::Instant, b: std::time::Instant| (b - a).as_nanos() as u64;
        // Features are small integers, exact in fp16.
        let (h, value) = if net.trunk_io.in16
            && net.trunk_io.out16
            && pinned_on()
            && combine_on()
            && b <= combine_cap()
        {
            use std::time::Instant;
            let run = |ins: Vec<ort::value::DynValue>,
                       mut outs: Vec<ort::value::DynValue>,
                       rows: usize,
                       _: usize,
                       slot: usize|
             -> anyhow::Result<()> {
                use ort::session::{OutputSelector, RunOptions};
                let tr = Instant::now();
                let value = outs.remove(1);
                let h = outs.remove(0);
                let opts = RunOptions::new().map_err(e2a)?.with_outputs(
                    OutputSelector::no_default()
                        .with("h")
                        .with("value")
                        .preallocate("h", h)
                        .preallocate("value", value),
                );
                let mut sess = pool[slot % pool.len()].lock().unwrap();
                drop(
                    sess.run_with_options(ort::inputs!["planes" => &ins[0]], &opts)
                        .map_err(e2a)?,
                );
                drop(sess);
                call_stats::record_batch(call_stats::TRUNK, rows, tr.elapsed().as_nanos() as u64);
                Ok(())
            };
            let comb = net.combiner(
                COMB_TRUNK,
                pool,
                &[("planes", Axis::Rows)],
                &[("h", Axis::Rows), ("value", Axis::Rows)],
                0,
                &run,
            )?;
            let t0 = Instant::now();
            let mut got = (Vec::new(), Vec::new());
            comb.submit(
                b,
                0,
                &mut |_, _, sl| {
                    sl[0].f16().convert_from_f32_slice(planes);
                    Ok(())
                },
                &mut |o| {
                    got = (o[0].f16().to_vec(), o[1].f32().to_vec());
                    Ok(())
                },
                &run,
            )?;
            call_stats::record(call_stats::TRUNK, b, 0, 0, ns(t0, Instant::now()), 0);
            got
        } else if net.trunk_io.in16 && net.trunk_io.out16 && pinned_on() {
            use std::time::Instant;
            let tp = Instant::now();
            let mut pin = net.take_pinned(
                pool,
                call_stats::TRUNK,
                b,
                &[("planes", shape)],
                &[("h", true), ("value", false)],
            )?;
            pin.ins[0]
                .try_extract_tensor_mut::<f16>()
                .map_err(e2a)?
                .1
                .convert_from_f32_slice(planes);
            let t0 = Instant::now();
            let mut sess = net.acquire(pool);
            let t1 = Instant::now();
            drop(
                sess.run_with_options(ort::inputs!["planes" => &pin.ins[0]], &pin.run)
                    .map_err(e2a)?,
            );
            drop(sess);
            let t2 = Instant::now();
            let r = (pin.out_f16(0), pin.out_f32(1, false));
            net.pinned
                .lock()
                .unwrap()
                .entry((call_stats::TRUNK, b))
                .or_default()
                .push(pin);
            call_stats::record(
                call_stats::TRUNK,
                b,
                ns(t0, t1),
                ns(tp, t0),
                ns(t1, t2),
                ns(t2, Instant::now()),
            );
            r
        } else if net.trunk_io.in16 {
            let t0 = std::time::Instant::now();
            let mut sess = net.acquire(pool);
            let t1 = std::time::Instant::now();
            let p16: Vec<f16> = Vec::from_f32_slice(planes);
            let input = TensorRef::from_array_view((shape, &p16[..])).map_err(e2a)?;
            let t2 = std::time::Instant::now();
            let outputs = sess.run(ort::inputs!["planes" => input]).map_err(e2a)?;
            let t3 = std::time::Instant::now();
            let r = (
                take_f16!(outputs, "h", net.trunk_io.out16),
                take_f32!(outputs, "value", false),
            );
            call_stats::record(
                call_stats::TRUNK,
                b,
                ns(t0, t1),
                ns(t1, t2),
                ns(t2, t3),
                ns(t3, std::time::Instant::now()),
            );
            r
        } else {
            let mut sess = net.acquire(pool);
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

    fn has_best(&self) -> bool {
        self.active().best.is_some()
    }

    fn best(&self, h: &[f16], b: usize) -> anyhow::Result<Vec<f32>> {
        let out = self.active().run_best(h, b)?;
        debug_assert_eq!(out.len(), b * NUM_CELLS * DECL_LOGITS);
        Ok(out)
    }

    fn diffusion(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let net = self.active();
        anyhow::ensure!(
            net.diffusion.is_some(),
            "this net has no diffusion head (it proposes with best.onnx)"
        );
        let out = net.run_head(
            &net.diffusion,
            net.diffusion_io,
            HSrc::Contig(h),
            decl,
            b,
            "logits",
            DECL_PLANES,
            call_stats::DIFFUSION,
            None,
        )?;
        debug_assert_eq!(out.len(), b * NUM_CELLS * DECL_LOGITS);
        Ok(out)
    }

    fn h_lease<'a>(&'a self, h: &[f16]) -> anyhow::Result<Option<crate::net::StoreLease<'a>>> {
        // The store belongs to the primary graph set; an A/B window on the
        // fp32 arm takes the per-call path.
        if !store_on() || self.diffusion_g.is_none() || !std::ptr::eq(self.active(), self) {
            return Ok(None);
        }
        let store = self.store()?;
        let n = h.len() / (crate::net::head_width(self.channels) * NUM_CELLS);
        Ok(store.lease(h)?.map(|l| crate::net::StoreLease {
            rows: (0..n).map(|i| l.row(i)).collect(),
            hold: Box::new(l),
        }))
    }

    fn diffusion_store(
        &self,
        rows: &[u32],
        decl: &[f32],
        cells: &[u32],
    ) -> anyhow::Result<Vec<f32>> {
        use std::time::Instant;
        let pool = self
            .diffusion_g
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no diffusion_g.onnx"))?;
        let store = self.store()?;
        let run = |ins: Vec<ort::value::DynValue>,
                   mut outs: Vec<ort::value::DynValue>,
                   n: usize,
                   _: usize,
                   slot: usize|
         -> anyhow::Result<()> {
            use ort::session::{OutputSelector, RunOptions};
            let tr = Instant::now();
            let opts = RunOptions::new().map_err(e2a)?.with_outputs(
                OutputSelector::no_default()
                    .with("logits_g")
                    .preallocate("logits_g", outs.remove(0)),
            );
            let mut sess = pool[slot % pool.len()].lock().unwrap();
            drop(
                sess.run_with_options(
                    ort::inputs![
                        "h_store" => store.tensor(),
                        "h_idx" => &ins[0],
                        "decl" => &ins[1],
                        "cell_flat" => &ins[2]
                    ],
                    &opts,
                )
                .map_err(e2a)?,
            );
            drop(sess);
            call_stats::record_batch(call_stats::DIFFUSION, n, tr.elapsed().as_nanos() as u64);
            Ok(())
        };
        let comb = self.combiner(
            COMB_DIFFUSION_G,
            pool,
            &[
                ("h_idx", Axis::Rows),
                ("decl", Axis::Rows),
                ("cell_flat", Axis::Cells),
            ],
            &[("logits_g", Axis::Cells)],
            combine_cap() * NUM_CELLS,
            &run,
        )?;
        let t0 = Instant::now();
        let mut out = Vec::new();
        comb.submit(
            rows.len(),
            cells.len(),
            &mut |roff, _, sl| {
                for (o, &r) in sl[0].i64().iter_mut().zip(rows) {
                    *o = r as i64;
                }
                sl[1].f16().convert_from_f32_slice(decl);
                // Cell indices name batch rows, and this job's start at `roff`.
                let base = (roff * NUM_CELLS) as i64;
                for (o, &c) in sl[2].i64().iter_mut().zip(cells) {
                    *o = base + c as i64;
                }
                Ok(())
            },
            &mut |o| {
                out = o[0].f16().to_f32_vec();
                Ok(())
            },
            &run,
        )?;
        call_stats::record(
            call_stats::DIFFUSION,
            rows.len(),
            0,
            0,
            t0.elapsed().as_nanos() as u64,
            0,
        );
        Ok(out)
    }

    fn energy_store(&self, rows: &[u32], decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        use std::time::Instant;
        let pool = self
            .energy_g
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no energy_g.onnx"))?;
        let store = self.store()?;
        let run = |ins: Vec<ort::value::DynValue>,
                   mut outs: Vec<ort::value::DynValue>,
                   n: usize,
                   _: usize,
                   slot: usize|
         -> anyhow::Result<()> {
            use ort::session::{OutputSelector, RunOptions};
            let tr = Instant::now();
            let opts = RunOptions::new().map_err(e2a)?.with_outputs(
                OutputSelector::no_default()
                    .with("energy")
                    .preallocate("energy", outs.remove(0)),
            );
            let mut sess = pool[slot % pool.len()].lock().unwrap();
            drop(
                sess.run_with_options(
                    ort::inputs!["h_store" => store.tensor(), "h_idx" => &ins[0], "decl" => &ins[1]],
                    &opts,
                )
                .map_err(e2a)?,
            );
            drop(sess);
            call_stats::record_batch(call_stats::ENERGY, n, tr.elapsed().as_nanos() as u64);
            Ok(())
        };
        let comb = self.combiner(
            COMB_ENERGY_G,
            pool,
            &[("h_idx", Axis::Rows), ("decl", Axis::Rows)],
            &[("energy", Axis::Rows)],
            0,
            &run,
        )?;
        let t0 = Instant::now();
        let mut out = Vec::new();
        comb.submit(
            rows.len(),
            0,
            &mut |_, _, sl| {
                for (o, &r) in sl[0].i64().iter_mut().zip(rows) {
                    *o = r as i64;
                }
                sl[1].f16().convert_from_f32_slice(decl);
                Ok(())
            },
            &mut |o| {
                out = o[0].f16().to_f32_vec();
                Ok(())
            },
            &run,
        )?;
        call_stats::record(
            call_stats::ENERGY,
            rows.len(),
            0,
            0,
            t0.elapsed().as_nanos() as u64,
            0,
        );
        Ok(out)
    }

    fn diffusion_rows(&self, h: &[f16], rows: &[u32], decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        let net = self.active();
        anyhow::ensure!(
            net.diffusion.is_some(),
            "this net has no diffusion head (it proposes with best.onnx)"
        );
        let stride = crate::net::head_width(net.channels) * NUM_CELLS;
        net.run_head(
            &net.diffusion,
            net.diffusion_io,
            HSrc::Rows(h, rows, stride),
            decl,
            rows.len(),
            "logits",
            DECL_PLANES,
            call_stats::DIFFUSION,
            None,
        )
    }

    fn diffusion_cells(
        &self,
        h: &[f16],
        rows: &[u32],
        decl: &[f32],
        cells: &[u32],
    ) -> anyhow::Result<Vec<f32>> {
        let net = self.active();
        anyhow::ensure!(
            net.diffusion.is_some(),
            "this net has no diffusion head (it proposes with best.onnx)"
        );
        let stride = crate::net::head_width(net.channels) * NUM_CELLS;
        let io = net.diffusion_io;
        if io.in16 && io.out16 && pinned_on() && combine_on() && rows.len() <= combine_cap() {
            return net.run_head(
                &net.diffusion,
                io,
                HSrc::Rows(h, rows, stride),
                decl,
                rows.len(),
                "logits",
                DECL_PLANES,
                call_stats::DIFFUSION,
                Some(cells),
            );
        }
        // Per-call paths keep a few batch shapes: pad to a multiple of 16 by
        // repeating the last row.
        let width = rows.len().div_ceil(16) * 16;
        let mut rows = rows.to_vec();
        rows.resize(width, *rows.last().expect("at least one row"));
        let mut decl = decl.to_vec();
        decl.resize(width * DECL_PLANES * NUM_CELLS, 0.0);
        net.run_head(
            &net.diffusion,
            io,
            HSrc::Rows(h, &rows, stride),
            &decl,
            width,
            "logits",
            DECL_PLANES,
            call_stats::DIFFUSION,
            Some(cells),
        )
    }

    fn energy_rows(&self, h: &[f16], rows: &[u32], decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        let net = self.active();
        let stride = crate::net::head_width(net.channels) * NUM_CELLS;
        net.run_head(
            &net.energy,
            net.energy_io,
            HSrc::Rows(h, rows, stride),
            decl,
            rows.len(),
            "energy",
            ENERGY_PLANES,
            call_stats::ENERGY,
            None,
        )
    }

    fn energy(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
        let net = self.active();
        net.run_head(
            &net.energy,
            net.energy_io,
            HSrc::Contig(h),
            decl,
            b,
            "energy",
            ENERGY_PLANES,
            call_stats::ENERGY,
            None,
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
        let pool = net.diffusion.as_ref().ok_or_else(|| {
            anyhow::anyhow!("this net has no diffusion head (it proposes with best.onnx)")
        })?;
        if !net.diffusion_io.in16 && pool.len() >= 2 && env_flag("MINIONS_ORT_BIND", false) {
            return Ok(Box::new(BoundPass::new(
                net.acquire(pool),
                &h.to_f32_vec(),
                b,
                // `h` is the *head-conditioning* width, not the trunk width:
                // `trunk.onnx` emits `head_features(h)`, which is what
                // `Backend::channels()` reports. Passing the raw `self.channels`
                // field described a 48-wide buffer with a 96-wide shape and made
                // the bound pass fail before it bound anything -- so the guard
                // test could never reach the IoBinding behaviour it exists to
                // check. Latent since the per-head neck refactor narrowed the
                // head inputs below the trunk.
                crate::net::head_width(net.channels),
            )?));
        }
        if net.diffusion_io.in16 && env_flag("MINIONS_ORT_DEVICE_H", false) {
            return Ok(Box::new(DevicePass::new(net, h, b)?));
        }
        Ok(Box::new(NaivePass { net, h, b }))
    }
}

/// A diffusion pass with `h` uploaded to the GPU **once**, then fed to every
/// step as a device-resident input of an ordinary `Session::run`.
///
/// `h` never changes within a pass, yet the per-step path re-sent it on all
/// `T` steps: at the production shape (~160 rows) that is 752 KB and ~120 us of
/// PCIe per step, out of a ~0.75 ms call of which only ~0.15 ms is the network
/// -- measured 2026-09-24, when a 16-wide head ran at 0.60 ms against the
/// 96-wide head's 0.75.
///
/// No `IoBinding` on the model session (that is what `BoundPass` broke on):
/// the upload goes through ort's own copy helper (`Tensor::to`), and the model
/// session sees a plain input that happens to live on the device. The session
/// is taken per step, not held for the pass, so other workers interleave.
struct DevicePass<'a> {
    net: &'a OrtNet,
    /// `Some` for the pass's life; taken back into the pool on drop.
    h: Option<ort::value::Tensor<f16>>,
    b: usize,
}

impl<'a> DevicePass<'a> {
    fn new(net: &'a OrtNet, h: &[f16], b: usize) -> anyhow::Result<Self> {
        use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
        let c = crate::net::head_width(net.channels);
        let shape = vec![b as i64, c as i64, N as i64, N as i64];
        let t0 = std::time::Instant::now();
        let reused = net.dev_h.lock().unwrap().get_mut(&b).and_then(|v| v.pop());
        let mut dev = match reused {
            Some(t) => t,
            None => {
                let mut alloc = net.dev_alloc.lock().unwrap();
                if alloc.is_none() {
                    let sess = net.acquire(net.diffusion.as_ref().expect("diffusion session"));
                    let mi = MemoryInfo::new(
                        AllocationDevice::CUDA,
                        0,
                        AllocatorType::Device,
                        MemoryType::Default,
                    )
                    .map_err(e2a)?;
                    *alloc = Some(Allocator::new(&sess, mi).map_err(e2a)?);
                }
                ort::value::Tensor::<f16>::new(alloc.as_ref().unwrap(), shape.clone())
                    .map_err(e2a)?
            }
        };
        let host = TensorRef::from_array_view((shape, h)).map_err(e2a)?;
        host.copy_into(&mut dev).map_err(e2a)?;
        call_stats::record_upload(b, t0.elapsed().as_nanos() as u64);
        Ok(Self {
            net,
            h: Some(dev),
            b,
        })
    }
}

impl Drop for DevicePass<'_> {
    fn drop(&mut self) {
        if let Some(t) = self.h.take() {
            self.net
                .dev_h
                .lock()
                .unwrap()
                .entry(self.b)
                .or_default()
                .push(t);
        }
    }
}

impl DiffusionPass for DevicePass<'_> {
    fn step(&mut self, decl: &[f32]) -> anyhow::Result<Vec<f32>> {
        use std::time::Instant;
        let net = self.net;
        let dshape = vec![self.b as i64, DECL_PLANES as i64, N as i64, N as i64];
        let t0 = Instant::now();
        let mut sess = net.acquire(net.diffusion.as_ref().expect("session outlives pass"));
        let t1 = Instant::now();
        let d16: Vec<f16> = Vec::from_f32_slice(decl);
        let dt = TensorRef::from_array_view((dshape, &d16[..])).map_err(e2a)?;
        let t2 = Instant::now();
        let raw = {
            let outputs = sess
                .run(ort::inputs!["h" => self.h.as_ref().expect("live pass"), "decl" => dt])
                .map_err(e2a)?;
            take_raw!(outputs, "logits", net.diffusion_io.out16)
        };
        drop(sess);
        let t3 = Instant::now();
        let out = raw.into_f32();
        let ns = |a: Instant, b: Instant| (b - a).as_nanos() as u64;
        call_stats::record(
            call_stats::DIFFUSION,
            self.b,
            ns(t0, t1),
            ns(t1, t2),
            ns(t2, t3),
            ns(t3, Instant::now()),
        );
        Ok(out)
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
    fn device_pass_matches_per_step_path() {
        // The production configuration: TensorRT, fp16 edges, a realistic row
        // count, many steps, two passes alive at once (as with several
        // workers), and passes opened and dropped repeatedly. `BoundPass` was
        // right on its first step and garbage after, so every step is checked.
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping device-pass parity");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        let ort = match OrtNet::with_pool(dir, "trt", true, &cache, 1) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ORT unavailable ({e}) — skipping");
                return;
            }
        };
        assert!(ort.diffusion_io.in16, "needs an fp16-edge export");
        let b = 160usize;
        let c = ort.channels();
        let mk_h = |seed: usize| -> Vec<f16> {
            (0..b * c * NUM_CELLS)
                .map(|i| f16::from_f32((((i * 37 + seed * 11) % 101) as f32 - 50.0) / 25.0))
                .collect()
        };
        let mk_decl = |t: usize| -> Vec<f32> {
            (0..b * DECL_PLANES * NUM_CELLS)
                .map(|i| ((i * 7 + t * 13) % 5 == 0) as u8 as f32)
                .collect()
        };
        let mut worst = 0f32;
        for round in 0..3 {
            let (h1, h2) = (mk_h(2 * round), mk_h(2 * round + 1));
            let mut p1 = DevicePass::new(&ort, &h1, b).expect("pass 1");
            let mut p2 = DevicePass::new(&ort, &h2, b).expect("pass 2");
            for t in 0..10 {
                let d = mk_decl(t + round);
                for (pass, h) in [(&mut p1, &h1), (&mut p2, &h2)] {
                    let dev = pass.step(&d).expect("device step");
                    let direct = ort.diffusion(h, &d, b).expect("direct step");
                    assert_eq!(dev.len(), direct.len());
                    for (i, (x, y)) in dev.iter().zip(&direct).enumerate() {
                        let err = (x - y).abs() / (1.0 + y.abs());
                        worst = worst.max(err);
                        assert!(
                            err <= 1e-3,
                            "round {round} step {t} elem {i}: device {x} vs direct {y}"
                        );
                    }
                }
            }
        }
        eprintln!("device pass parity: worst relative error {worst:e}");
    }

    /// The pinned path reuses one output buffer per set across calls, so a
    /// missed write or a read before the copy lands would hand back the
    /// *previous* call's logits -- plausible numbers, wrong rows. Every call
    /// here changes its inputs, at two row counts, and is checked against the
    /// pageable path -- both heads and the trunk. Env-toggled, so run it alone
    /// (`--test-threads 1`).
    #[test]
    fn pinned_path_matches_pageable_path() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping pinned parity");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        let ort = match OrtNet::with_pool(dir, "trt", true, &cache, 1) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ORT unavailable ({e}) — skipping");
                return;
            }
        };
        assert!(
            ort.diffusion_io.in16 && ort.diffusion_io.out16,
            "needs an fp16-edge export"
        );
        let c = ort.channels();
        let mut worst = 0f32;
        for b in [160usize, 144] {
            for t in 0..12 {
                let h: Vec<f16> = (0..b * c * NUM_CELLS)
                    .map(|i| f16::from_f32((((i * 37 + t * 11) % 101) as f32 - 50.0) / 25.0))
                    .collect();
                let d: Vec<f32> = (0..b * DECL_PLANES * NUM_CELLS)
                    .map(|i| ((i * 7 + t * 13) % 5 == 0) as u8 as f32)
                    .collect();
                let e: Vec<f32> = (0..b * ENERGY_PLANES * NUM_CELLS)
                    .map(|i| ((i * 3 + t * 5) % 7 == 0) as u8 as f32)
                    .collect();
                std::env::set_var("MINIONS_ORT_PINNED", "1");
                let (pd, pe) = (
                    ort.diffusion(&h, &d, b).unwrap(),
                    ort.energy(&h, &e, b).unwrap(),
                );
                std::env::set_var("MINIONS_ORT_PINNED", "0");
                let (qd, qe) = (
                    ort.diffusion(&h, &d, b).unwrap(),
                    ort.energy(&h, &e, b).unwrap(),
                );
                let planes: Vec<f32> = (0..b * IN_PLANES * NUM_CELLS)
                    .map(|i| ((i * 5 + t * 3) % 4) as f32)
                    .collect();
                std::env::set_var("MINIONS_ORT_PINNED", "1");
                let pt = ort.trunk(&planes, b).unwrap();
                std::env::set_var("MINIONS_ORT_PINNED", "0");
                let qt = ort.trunk(&planes, b).unwrap();
                let (ph, qh) = (pt.h.to_f32_vec(), qt.h.to_f32_vec());
                for (p, q) in [(&pd, &qd), (&pe, &qe), (&ph, &qh), (&pt.value, &qt.value)] {
                    assert_eq!(p.len(), q.len());
                    for (i, (x, y)) in p.iter().zip(q.iter()).enumerate() {
                        let err = (x - y).abs() / (1.0 + y.abs());
                        worst = worst.max(err);
                        assert!(
                            err <= 1e-3,
                            "rows {b} call {t} elem {i}: pinned {x} vs pageable {y}"
                        );
                    }
                }
            }
        }
        // Gathered rows, pinned and pageable, against gathering by hand.
        let b = 48usize;
        let h: Vec<f16> = (0..b * c * NUM_CELLS)
            .map(|i| f16::from_f32((((i * 29) % 97) as f32 - 48.0) / 24.0))
            .collect();
        let rows: Vec<u32> = (0..b as u32).rev().map(|r| (r * 7) % b as u32).collect();
        let d: Vec<f32> = (0..b * DECL_PLANES * NUM_CELLS)
            .map(|i| (i % 3 == 0) as u8 as f32)
            .collect();
        let e: Vec<f32> = (0..b * ENERGY_PLANES * NUM_CELLS)
            .map(|i| (i % 5 == 0) as u8 as f32)
            .collect();
        let hg = crate::net::gather_rows(&h, &rows, c * NUM_CELLS);
        for pinned in ["1", "0"] {
            std::env::set_var("MINIONS_ORT_PINNED", pinned);
            assert_eq!(
                ort.diffusion_rows(&h, &rows, &d).unwrap(),
                ort.diffusion(&hg, &d, b).unwrap()
            );
            assert_eq!(
                ort.energy_rows(&h, &rows, &e).unwrap(),
                ort.energy(&hg, &e, b).unwrap()
            );
        }
        std::env::remove_var("MINIONS_ORT_PINNED");
        eprintln!("pinned parity: worst relative error {worst:e}");
    }

    /// The combiner under real concurrency: six threads, row counts that vary
    /// call to call, every input distinct, each result checked against the
    /// same call made alone. A job reading another's slice, a batch run before
    /// its rows were filled, or an arena reused while still being read would
    /// all show up here as wrong rows. Needs `MINIONS_ORT_DIR` and a GPU.
    #[test]
    fn combined_calls_match_single_calls() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping combiner parity");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        let ort = match OrtNet::with_pool(dir, "trt", true, &cache, pool_size()) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ORT unavailable ({e}) — skipping");
                return;
            }
        };
        let c = ort.channels();
        let hs = c * NUM_CELLS;
        // (thread, call) -> (h, rows, decl, energy decl)
        let job = |t: usize, i: usize| {
            let b = 16 * (1 + (t * 7 + i * 3) % 10);
            let npos = b / 2 + 1;
            let h: Vec<f16> = (0..npos * hs)
                .map(|x| f16::from_f32((((x * 31 + t * 17 + i * 5) % 89) as f32 - 44.0) / 22.0))
                .collect();
            let rows: Vec<u32> = (0..b).map(|r| ((r * 5 + t + i) % npos) as u32).collect();
            let d: Vec<f32> = (0..b * DECL_PLANES * NUM_CELLS)
                .map(|x| ((x * 7 + t * 3 + i) % 6 == 0) as u8 as f32)
                .collect();
            let e: Vec<f32> = (0..b * ENERGY_PLANES * NUM_CELLS)
                .map(|x| ((x * 11 + t + i * 2) % 5 == 0) as u8 as f32)
                .collect();
            (h, rows, d, e)
        };
        let (threads, calls) = (6usize, 25usize);
        COMBINE_USE.store(false, std::sync::atomic::Ordering::Relaxed);
        let want: Vec<Vec<(Vec<f32>, Vec<f32>)>> = (0..threads)
            .map(|t| {
                (0..calls)
                    .map(|i| {
                        let (h, rows, d, e) = job(t, i);
                        (
                            ort.diffusion_rows(&h, &rows, &d).unwrap(),
                            ort.energy_rows(&h, &rows, &e).unwrap(),
                        )
                    })
                    .collect()
            })
            .collect();
        COMBINE_USE.store(true, std::sync::atomic::Ordering::Relaxed);
        std::thread::scope(|sc| {
            for t in 0..threads {
                let (ort, want, job) = (&ort, &want, &job);
                sc.spawn(move || {
                    for i in 0..calls {
                        let (h, rows, d, e) = job(t, i);
                        let got = ort.diffusion_rows(&h, &rows, &d).unwrap();
                        assert!(
                            got == want[t][i].0,
                            "thread {t} call {i}: diffusion differs"
                        );
                        let got = ort.energy_rows(&h, &rows, &e).unwrap();
                        assert!(got == want[t][i].1, "thread {t} call {i}: energy differs");
                    }
                });
            }
        });
        eprintln!("{}", call_stats::take_report(1.0).unwrap_or_default());
    }

    /// The store-indexed graphs against the plain ones, on the GPU and under
    /// six threads: each thread leases its own rows, and every gathered logit
    /// row and energy must match the plain call on the same inputs. A wrong
    /// lease row, a cell offset that ignored the job's place in the batch, or
    /// a lease released while in use would all show up as mismatches. The two
    /// are separate TensorRT engines, so fp16-level differences are allowed.
    /// Needs `MINIONS_ORT_DIR` with `*_g.onnx` beside the graphs.
    #[test]
    fn store_calls_match_plain_calls() {
        let Ok(dir) = std::env::var("MINIONS_ORT_DIR") else {
            eprintln!("MINIONS_ORT_DIR unset — skipping store parity");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        let ort = match OrtNet::with_pool(dir, "trt", true, &cache, 1) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("ORT unavailable ({e}) — skipping");
                return;
            }
        };
        if ort.diffusion_g.is_none() {
            eprintln!("store graphs not loaded (MINIONS_ORT_STORE=1 and *_g.onnx) — skipping");
            return;
        }
        let c = ort.channels();
        let hs = c * NUM_CELLS;
        let worst = std::sync::Mutex::new(0f32);
        std::thread::scope(|sc| {
            for t in 0..6usize {
                let (ort, worst) = (&ort, &worst);
                sc.spawn(move || {
                    for i in 0..15usize {
                        let npos = 5 + (t * 3 + i * 7) % 40;
                        let h: Vec<f16> = (0..npos * hs)
                            .map(|x| {
                                f16::from_f32(
                                    (((x * 13 + t * 7 + i * 3) % 83) as f32 - 41.0) / 20.0,
                                )
                            })
                            .collect();
                        let lease = ort.h_lease(&h).unwrap().expect("store has room");
                        let b = 1 + (t + i * 5) % 70;
                        let local: Vec<u32> = (0..b).map(|r| ((r * 7 + t) % npos) as u32).collect();
                        let srows: Vec<u32> =
                            local.iter().map(|&r| lease.rows[r as usize]).collect();
                        let d: Vec<f32> = (0..b * DECL_PLANES * NUM_CELLS)
                            .map(|x| ((x * 5 + t + i) % 7 == 0) as u8 as f32)
                            .collect();
                        let cells: Vec<u32> = (0..b)
                            .flat_map(|r| {
                                (0..NUM_CELLS)
                                    .filter(move |cc| (cc * 3 + r + i) % 5 == 0)
                                    .map(move |cc| (r * NUM_CELLS + cc) as u32)
                            })
                            .collect();
                        let got = ort.diffusion_store(&srows, &d, &cells).unwrap();
                        let full = ort.diffusion_rows(&h, &local, &d).unwrap();
                        let want = crate::net::gather_rows(&full, &cells, DECL_LOGITS);
                        let e: Vec<f32> = (0..b * ENERGY_PLANES * NUM_CELLS)
                            .map(|x| ((x * 3 + t * 2 + i) % 6 == 0) as u8 as f32)
                            .collect();
                        let ge = ort.energy_store(&srows, &e).unwrap();
                        let we = ort.energy_rows(&h, &local, &e).unwrap();
                        let mut w = worst.lock().unwrap();
                        for (x, y) in got.iter().zip(&want).chain(ge.iter().zip(&we)) {
                            let err = (x - y).abs() / (1.0 + y.abs());
                            *w = w.max(err);
                            assert!(err <= 2e-2, "thread {t} call {i}: store {x} vs plain {y}");
                        }
                        assert_eq!(got.len(), want.len());
                        assert_eq!(ge.len(), we.len());
                        drop(lease);
                    }
                });
            }
        });
        eprintln!(
            "store parity: worst relative error {:e}",
            *worst.lock().unwrap()
        );
        eprintln!("{}", call_stats::take_report(1.0).unwrap_or_default());
    }

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
