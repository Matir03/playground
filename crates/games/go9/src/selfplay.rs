//! Multi-threaded 9×9 Go self-play generation — the Rust port of the Connect 4
//! self-play driver (`connect4/src/selfplay.rs`). Native threads share an
//! `Arc<LocalModel>`; two evaluation paths: **direct** (each thread runs its own
//! batch-1 forward, the CPU path) and **batched** (a central inference-server
//! thread coalesces leaf evals into one GPU forward, the like-for-like analogue
//! of the Python shared-memory server but with no IPC/GIL).

use std::io::Write;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use playground_core::batch_search::{self, SearchParams};
use playground_core::{Game, Outcome};

use crate::game::{area_outcome, Go9, Go9State};
use crate::mcts::{self_play_game, planes_of, Evaluator, Example, Params, PLANES};
use crate::net::{LocalModel, ACTIONS};

type EvalOut = (f32, [f32; ACTIONS]);

// ── Direct (batch-1 per thread) ──────────────────────────────────────────────

struct DirectEval {
    model: Arc<LocalModel>,
}

impl Evaluator for DirectEval {
    fn eval(&self, state: &Go9State) -> anyhow::Result<EvalOut> {
        self.model.policy_value(state)
    }
}

// ── Batched (central inference server) ───────────────────────────────────────

struct Req {
    state: Go9State,
    resp: SyncSender<EvalOut>,
}

thread_local! {
    static RESP: (SyncSender<EvalOut>, Receiver<EvalOut>) = sync_channel(1);
}

struct BatchedEval {
    req_tx: Sender<Req>,
}

impl Evaluator for BatchedEval {
    fn eval(&self, state: &Go9State) -> anyhow::Result<EvalOut> {
        RESP.with(|(tx, rx)| {
            self.req_tx
                .send(Req { state: state.clone(), resp: tx.clone() })
                .map_err(|_| anyhow::anyhow!("inference server gone"))?;
            rx.recv().map_err(|_| anyhow::anyhow!("inference response dropped"))
        })
    }
}

/// The inference server: block for a request, drain whatever else is queued (up
/// to `max_batch`), run one batched forward, scatter results. Returns when every
/// `Sender` is dropped (all search threads finished).
fn server_loop(rx: Receiver<Req>, model: &LocalModel, max_batch: usize, counters: &Counters) {
    loop {
        let first = match rx.recv() {
            Ok(r) => r,
            Err(_) => return,
        };
        let mut batch = vec![first];
        while batch.len() < max_batch {
            match rx.try_recv() {
                Ok(r) => batch.push(r),
                Err(_) => break,
            }
        }
        counters.batches.fetch_add(1, Ordering::Relaxed);
        counters.batch_samples.fetch_add(batch.len() as u64, Ordering::Relaxed);
        let states: Vec<&Go9State> = batch.iter().map(|r| &r.state).collect();
        let out = model.policy_value_batch(&states).expect("batched forward failed");
        for (req, res) in batch.into_iter().zip(out) {
            let _ = req.resp.send(res);
        }
    }
}

// ── Driver ───────────────────────────────────────────────────────────────────

struct Counters {
    next: AtomicUsize,
    sims: AtomicU64,
    nodes: AtomicU64,
    examples: AtomicU64,
    batches: AtomicU64,
    batch_samples: AtomicU64,
}

fn worker_loop(
    eval: &impl Evaluator,
    counters: &Counters,
    n_games: usize,
    base_seed: u64,
    params: &Params,
    sink: Option<&Mutex<Vec<Example>>>,
) {
    loop {
        let idx = counters.next.fetch_add(1, Ordering::Relaxed);
        if idx >= n_games {
            break;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(idx as u64));
        let (examples, sims, nodes) =
            self_play_game(eval, params, &mut rng).expect("self-play game failed");
        counters.sims.fetch_add(sims, Ordering::Relaxed);
        counters.nodes.fetch_add(nodes, Ordering::Relaxed);
        counters.examples.fetch_add(examples.len() as u64, Ordering::Relaxed);
        if let Some(sink) = sink {
            sink.lock().unwrap().extend(examples);
        }
    }
}

pub struct GenStats {
    pub sims: u64,
    pub nodes: u64,
    pub examples: u64,
    pub elapsed: f64,
    pub avg_batch: f64,
    /// Network forwards actually run (nodes minus eval-cache hits). Engine path only.
    pub evals: u64,
    pub cache_hits: u64,
    /// GPU-forward duty cycle [0,1] (≈1 = GPU-bound) and the CPU phase split.
    pub gpu_duty: f64,
    pub gather_secs: f64,
    pub wait_secs: f64,
    pub apply_secs: f64,
}

/// Convert per-game search knobs into the engine's `SearchParams`.
fn to_search_params(p: &Params) -> SearchParams {
    SearchParams {
        sims: p.sims,
        node_budget: p.node_budget,
        c_puct: p.c_puct,
        dir_alpha: p.dir_alpha,
        dir_eps: p.dir_eps,
        temp_moves: p.temp_moves,
        explore_eps: p.explore_eps,
        max_ply: p.max_ply,
    }
}

/// Turn the engine's game records into training examples: value = final outcome z
/// (area-adjudicated if the game hit the ply cap), plus the played action and the
/// next position's planes for the reifier consistency loss.
fn examples_from_records(records: Vec<batch_search::GameRecord<Go9>>) -> Vec<Example> {
    let mut out = Vec::new();
    for rec in records {
        let final_state = rec.states.last().unwrap();
        let outcome = match Go9::outcome(final_state) {
            Outcome::InProgress => area_outcome(final_state),
            o => o,
        };
        let winner: Option<u8> = match outcome {
            Outcome::FirstWins => Some(1),
            Outcome::SecondWins => Some(2),
            _ => None,
        };
        for (i, mv) in rec.moves.iter().enumerate() {
            let z = match winner {
                Some(w) => if w == mv.to_move { 1.0 } else { -1.0 },
                None => 0.0,
            };
            let mut policy = [0f32; ACTIONS];
            policy.copy_from_slice(&mv.policy);
            out.push(Example {
                planes: planes_of(&rec.states[i]),
                policy,
                value: z,
                played_action: mv.action as u8,
                has_next: true,
                next_planes: planes_of(&rec.states[i + 1]),
            });
        }
    }
    out
}

/// Generate `n_games` across `threads` native threads. With `batched`, a central
/// inference server coalesces leaf evals (the GPU path). When `collect`, the
/// produced training examples are returned.
pub fn generate(
    model: Arc<LocalModel>,
    n_games: usize,
    threads: usize,
    params: Params,
    base_seed: u64,
    batched: bool,
    max_batch: usize,
    collect: bool,
) -> (GenStats, Vec<Example>) {
    // GPU path: the generic batched engine (cross-game lockstep batching, leaf-side
    // encoding, eval cache). The direct/CPU path keeps the simple per-thread search.
    // `GO9_OLD=1` routes the batched path through the original independent-thread
    // server instead, for A/B comparing the two architectures.
    if batched && std::env::var_os("GO9_OLD").is_none() {
        let (records, cs) = batch_search::run_chunk::<Go9>(
            model.as_ref(),
            n_games,
            to_search_params(&params),
            threads,
            max_batch,
            base_seed,
            false, // Go transposes little (~20% hits); the cache's per-leaf cost isn't worth it
        );
        let examples_count: u64 = records.iter().map(|r| r.moves.len() as u64).sum();
        let stats = GenStats {
            sims: cs.nodes + cs.terminal_sims,
            nodes: cs.nodes,
            examples: examples_count,
            elapsed: cs.elapsed,
            avg_batch: cs.avg_batch(),
            evals: cs.evals,
            cache_hits: cs.cache_hits,
            gpu_duty: cs.gpu_duty(),
            gather_secs: cs.gather_secs,
            wait_secs: cs.wait_secs,
            apply_secs: cs.apply_secs,
        };
        let examples = if collect { examples_from_records(records) } else { Vec::new() };
        return (stats, examples);
    }

    let counters = Counters {
        next: AtomicUsize::new(0),
        sims: AtomicU64::new(0),
        nodes: AtomicU64::new(0),
        examples: AtomicU64::new(0),
        batches: AtomicU64::new(0),
        batch_samples: AtomicU64::new(0),
    };
    let sink: Option<Mutex<Vec<Example>>> = collect.then(|| Mutex::new(Vec::new()));

    let t0 = Instant::now();
    if batched {
        let (req_tx, req_rx) = channel::<Req>();
        std::thread::scope(|scope| {
            let model_ref = &model;
            let counters_ref = &counters;
            scope.spawn(move || server_loop(req_rx, model_ref, max_batch, counters_ref));
            for _ in 0..threads {
                let eval = BatchedEval { req_tx: req_tx.clone() };
                let counters = &counters;
                let params = &params;
                let sink = sink.as_ref();
                scope.spawn(move || worker_loop(&eval, counters, n_games, base_seed, params, sink));
            }
            drop(req_tx);
        });
    } else {
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let eval = DirectEval { model: Arc::clone(&model) };
                let counters = &counters;
                let params = &params;
                let sink = sink.as_ref();
                scope.spawn(move || worker_loop(&eval, counters, n_games, base_seed, params, sink));
            }
        });
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let nbatch = counters.batches.load(Ordering::Relaxed);
    let avg_batch = if nbatch > 0 {
        counters.batch_samples.load(Ordering::Relaxed) as f64 / nbatch as f64
    } else {
        0.0
    };
    let stats = GenStats {
        sims: counters.sims.load(Ordering::Relaxed),
        nodes: counters.nodes.load(Ordering::Relaxed),
        examples: counters.examples.load(Ordering::Relaxed),
        elapsed,
        avg_batch,
        evals: 0,
        cache_hits: 0,
        gpu_duty: 0.0,
        gather_secs: 0.0,
        wait_secs: 0.0,
        apply_secs: 0.0,
    };
    let examples = sink.map(|m| m.into_inner().unwrap()).unwrap_or_default();
    (stats, examples)
}

/// Throughput benchmark (no example collection).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn run_benchmark(
    checkpoint: &str,
    cuda: bool,
    n_games: usize,
    threads: usize,
    params: Params,
    batched: bool,
    max_batch: usize,
    base_seed: u64,
    backend: &str,
    onnx: &str,
    ep: &str,
    fp16: bool,
) -> anyhow::Result<()> {
    // ORT/TensorRT backend: drive the same batched engine via an ONNX Runtime
    // session instead of candle, to compare inference throughput.
    if backend == "ort" {
        let model = crate::ort_net::OrtModel::new(onnx, ep, fp16, "/tmp/go9_trt_cache")?;
        eprintln!(
            "go9 self-play: {n_games} games, {threads} threads, {} nodes/move, \
             backend ort/{ep} fp16={fp16}, onnx {onnx}",
            params.node_budget,
        );
        let (_records, cs) = batch_search::run_chunk::<Go9>(
            &model, n_games, to_search_params(&params), threads, max_batch, base_seed, false,
        );
        let nps = cs.nodes as f64 / cs.elapsed;
        let gps = n_games as f64 / cs.elapsed;
        println!(
            "games {n_games}  nodes {}  total_sims {}  time {:.2}s  nodes/s {nps:.0}  \
             games/s {gps:.2}  avg_batch {:.0}  evals/s {:.0}",
            cs.nodes, cs.nodes + cs.terminal_sims, cs.elapsed, cs.avg_batch(),
            cs.evals as f64 / cs.elapsed,
        );
        let cpu = (cs.gather_secs + cs.wait_secs + cs.apply_secs).max(1e-9);
        println!(
            "  profile: gpu_duty {:.0}%  | gather {:.0}% wait {:.0}% apply {:.0}%",
            100.0 * cs.gpu_duty(),
            100.0 * cs.gather_secs / cpu, 100.0 * cs.wait_secs / cpu, 100.0 * cs.apply_secs / cpu,
        );
        return Ok(());
    }

    let device = crate::net::pick_device(cuda)?;
    let model = Arc::new(LocalModel::load_on(checkpoint, device)?);
    let mode = if batched { format!("batched (max_batch {max_batch})") } else { "direct".into() };
    eprintln!(
        "go9 self-play: {n_games} games, {threads} threads, {} sims/move, device {}, {mode}",
        params.sims,
        if cuda { "cuda" } else { "cpu" },
    );
    let (s, _) = generate(model, n_games, threads, params, base_seed, batched, max_batch, false);
    let nodes_per_s = s.nodes as f64 / s.elapsed;
    let evals_per_s = s.evals as f64 / s.elapsed;
    let games_per_s = n_games as f64 / s.elapsed;
    let hit_rate = if s.nodes > 0 { s.cache_hits as f64 / s.nodes as f64 } else { 0.0 };
    let batch_note = if batched {
        format!("  avg_batch {:.0}  evals/s {evals_per_s:.0}  cache_hit {:.1}%",
                s.avg_batch, 100.0 * hit_rate)
    } else {
        String::new()
    };
    println!(
        "games {n_games}  examples {}  nodes {}  total_sims {}  time {:.2}s  \
         nodes/s {nodes_per_s:.0}  games/s {games_per_s:.2}{batch_note}",
        s.examples, s.nodes, s.sims, s.elapsed,
    );
    if batched {
        // CPU-vs-GPU profile: gpu_duty ≈1 means GPU-bound; the worker phase split
        // (summed over threads) shows where the CPU time goes.
        let cpu = s.gather_secs + s.wait_secs + s.apply_secs;
        println!(
            "  profile: gpu_duty {:.0}%  | worker gather {:.0}% wait {:.0}% apply {:.0}%  \
             (gather={:.1}s wait={:.1}s apply={:.1}s)",
            100.0 * s.gpu_duty,
            100.0 * s.gather_secs / cpu.max(1e-9),
            100.0 * s.wait_secs / cpu.max(1e-9),
            100.0 * s.apply_secs / cpu.max(1e-9),
            s.gather_secs, s.wait_secs, s.apply_secs,
        );
    }
    Ok(())
}

/// Per-record float layout of the dump (must match rust_go9_selfplay_train.py):
/// planes(243) + policy(82) + value(1) + played_action(1) + has_next(1) +
/// next_planes(243).
const RECORD_F32: usize = PLANES + ACTIONS + 1 + 1 + 1 + PLANES;

/// Generate one generation of self-play and write the training examples to
/// `out_path` (u64 count, then `count` records of `RECORD_F32` little-endian f32).
#[allow(clippy::too_many_arguments)]
pub fn run_generate(
    checkpoint: &str,
    cuda: bool,
    out_path: &str,
    n_games: usize,
    threads: usize,
    params: Params,
    batched: bool,
    max_batch: usize,
    base_seed: u64,
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let model = Arc::new(LocalModel::load_on(checkpoint, device)?);
    let budget = if params.node_budget > 0 {
        format!("{} nodes/move", params.node_budget)
    } else if params.move_ms > 0.0 {
        format!("{}ms/move", params.move_ms)
    } else {
        format!("{} sims/move", params.sims)
    };
    eprintln!(
        "go9 self-play gen: {n_games} games, {threads} threads, {budget}, device {}, \
         model {checkpoint} -> {out_path}",
        if cuda { "cuda" } else { "cpu" },
    );
    let (s, examples) =
        generate(model, n_games, threads, params, base_seed, batched, max_batch, true);
    write_dump(out_path, &examples)?;

    let moves = s.examples.max(1) as f64;
    eprintln!(
        "  wrote {} examples ({} games, {:.0} nodes/move, {:.0} sims/move, avg_batch {:.0}) \
         in {:.1}s",
        examples.len(), n_games, s.nodes as f64 / moves, s.sims as f64 / moves,
        s.avg_batch, s.elapsed,
    );
    Ok(())
}

/// Write a replay dump: u64 example count, then `count` records of `RECORD_F32`
/// little-endian f32 (planes, policy, value z, played_action, has_next, next_planes).
pub fn write_dump(out_path: &str, examples: &[Example]) -> anyhow::Result<()> {
    let mut buf = Vec::with_capacity(8 + examples.len() * RECORD_F32 * 4);
    buf.extend_from_slice(&(examples.len() as u64).to_le_bytes());
    for ex in examples {
        for &x in &ex.planes {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &ex.policy {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&ex.value.to_le_bytes());
        buf.extend_from_slice(&(ex.played_action as f32).to_le_bytes());
        buf.extend_from_slice(&(if ex.has_next { 1.0f32 } else { 0.0 }).to_le_bytes());
        for &x in &ex.next_planes {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

/// Highest existing `chunk_NNNNNN.bin` index in `dir`, or None.
fn max_chunk_seq(dir: &std::path::Path) -> Option<u64> {
    std::fs::read_dir(dir).ok()?.filter_map(|e| {
        let name = e.ok()?.file_name().into_string().ok()?;
        name.strip_prefix("chunk_")?.strip_suffix(".bin")?.parse::<u64>().ok()
    }).max()
}

/// Delete oldest `chunk_*.bin` so at most `keep` remain (FIFO replay window).
fn prune_buffer(dir: &std::path::Path, keep: usize) {
    let mut chunks: Vec<(u64, std::path::PathBuf)> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| {
            let e = e.ok()?;
            let n = e.file_name().into_string().ok()?;
            let seq = n.strip_prefix("chunk_")?.strip_suffix(".bin")?.parse::<u64>().ok()?;
            Some((seq, e.path()))
        }).collect(),
        Err(_) => return,
    };
    if chunks.len() <= keep {
        return;
    }
    chunks.sort_by_key(|(seq, _)| *seq);
    for (_, path) in &chunks[..chunks.len() - keep] {
        let _ = std::fs::remove_file(path);
    }
}

/// Actor loop: forever load the latest checkpoint (hot-swap), generate a chunk of
/// `games_per_chunk` games, append it to the buffer, and prune to `max_chunks`.
/// Reloading the checkpoint each chunk is the hot-swap mechanism — the learner
/// publishes `latest.safetensors` atomically and actors pick it up next chunk.
#[allow(clippy::too_many_arguments)]
pub fn run_loop(
    live_dir: &str,
    cuda: bool,
    params: Params,
    games_per_chunk: usize,
    max_chunks: usize,
    threads: usize,
    max_batch: usize,
    base_seed: u64,
    backend: &str,
    ep: &str,
    fp16: bool,
) -> anyhow::Result<()> {
    let latest = format!("{live_dir}/latest.safetensors");
    let onnx = format!("{live_dir}/model.onnx");
    let trt_cache = format!("{live_dir}/trt_cache");
    let buffer = std::path::Path::new(live_dir).join("buffer");
    std::fs::create_dir_all(&buffer)?;
    let ort = backend == "ort";
    // candle needs a device; ORT loads weights from model.onnx (hot-swapped by the
    // learner). For ORT use the CUDA EP — it reloads the ONNX cheaply each chunk,
    // unlike TensorRT which would rebuild its engine on every weight change.
    let device = if ort { None } else { Some(crate::net::pick_device(cuda)?) };
    let mut seq = max_chunk_seq(&buffer).map(|s| s + 1).unwrap_or(0);
    eprintln!(
        "go9 actor loop: backend {backend}{}, {games_per_chunk} games/chunk, {threads} threads, \
         buffer {} (keep {max_chunks})",
        if ort { format!("/{ep} fp16={fp16}") } else { String::new() },
        buffer.display(),
    );
    let sp = to_search_params(&params);
    loop {
        let t0 = Instant::now();
        let chunk_seed = base_seed.wrapping_add(seq.wrapping_mul(games_per_chunk as u64));
        let (examples, nodes, avg_batch) = if ort {
            while !std::path::Path::new(&onnx).exists() {
                std::thread::sleep(std::time::Duration::from_secs(1)); // wait for learner's first export
            }
            let model = crate::ort_net::OrtModel::new(&onnx, ep, fp16, &trt_cache)?;
            let (records, cs) =
                batch_search::run_chunk::<Go9>(&model, games_per_chunk, sp, threads, max_batch, chunk_seed, false);
            (examples_from_records(records), cs.nodes, cs.avg_batch())
        } else {
            let model = Arc::new(LocalModel::load_on(&latest, device.clone().unwrap())?);
            let (s, ex) =
                generate(model, games_per_chunk, threads, params, chunk_seed, cuda, max_batch, true);
            (ex, s.nodes, s.avg_batch)
        };
        let path = buffer.join(format!("chunk_{seq:06}.bin"));
        let tmp = buffer.join(format!("chunk_{seq:06}.bin.tmp"));
        write_dump(tmp.to_str().unwrap(), &examples)?;
        std::fs::rename(&tmp, &path)?;
        prune_buffer(&buffer, max_chunks);
        let moves = examples.len().max(1) as f64;
        eprintln!(
            "  chunk {seq}: {} ex  {:.0} nodes/move  avg_batch {avg_batch:.0}  {:.1}s",
            examples.len(), nodes as f64 / moves, t0.elapsed().as_secs_f64(),
        );
        seq += 1;
    }
}
