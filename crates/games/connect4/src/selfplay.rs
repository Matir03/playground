//! Multi-threaded self-play game generation — the Rust port of the Python
//! multiprocessing self-play (`python/.../training/selfplay.py`).
//!
//! Where Python needs worker *processes* and a shared-memory inference server to
//! dodge the GIL (one process owns the GPU net; workers ship leaf planes over
//! shared memory and block on per-slot events), Rust runs the games on native
//! threads that share an `Arc<LocalModel>`. Two evaluation paths:
//!
//!   * **direct** — each thread runs its own batch-1 forward. Best on CPU, where
//!     batch-1 on the tiny 6×7 board is cheap and parallelism comes from cores.
//!   * **batched** — a central inference-server thread coalesces the search
//!     threads' leaf evaluations into one big forward. This is the GPU path, and
//!     the like-for-like analogue of the Python server: same batching, but the
//!     workers are threads, so there's no IPC, no pickling, and no GIL.
//!
//! Either way it's a throughput benchmark: generate games at a fixed simulation
//! count and report sims/sec.

use std::io::Write;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use playground_core::batch_search::{self, SearchParams};
use playground_core::{Game, Outcome};

use crate::game::{Connect4, Connect4State, C4_COLS};
use crate::mcts::{self_play_game, planes_of, Evaluator, Example, Params};
use crate::net::LocalModel;

type EvalOut = (f32, [f32; C4_COLS]);

// ── Direct (batch-1 per thread) ──────────────────────────────────────────────

/// Evaluator that runs a batch-1 forward on the shared net from the calling
/// thread. Parallelism comes from running many of these on different cores.
struct DirectEval {
    model: Arc<LocalModel>,
}

impl Evaluator for DirectEval {
    fn eval(&self, state: &Connect4State) -> anyhow::Result<EvalOut> {
        self.model.policy_value(state)
    }
}

// ── Batched (central inference server) ───────────────────────────────────────

/// One leaf-evaluation request: the state to evaluate and where to send the
/// result back. `resp` is the requesting thread's reusable response channel.
pub(crate) struct Req {
    state: Connect4State,
    resp: SyncSender<EvalOut>,
}

thread_local! {
    /// Per-thread reusable response channel, so a leaf eval doesn't allocate a
    /// fresh channel each time (one capacity-1 rendezvous per search thread).
    static RESP: (SyncSender<EvalOut>, Receiver<EvalOut>) = sync_channel(1);
}

/// Evaluator that hands its leaf to the central inference server and blocks for
/// the batched result. Each thread holds its own `Sender` clone.
pub(crate) struct BatchedEval {
    pub(crate) req_tx: Sender<Req>,
}

impl Evaluator for BatchedEval {
    fn eval(&self, state: &Connect4State) -> anyhow::Result<EvalOut> {
        RESP.with(|(tx, rx)| {
            self.req_tx
                .send(Req { state: state.clone(), resp: tx.clone() })
                .map_err(|_| anyhow::anyhow!("inference server gone"))?;
            rx.recv().map_err(|_| anyhow::anyhow!("inference response dropped"))
        })
    }
}

/// The inference server: block for a request, drain whatever else is already
/// queued (up to `max_batch`), run one batched forward, scatter the results.
/// Returns when every `Sender` has been dropped (all search threads finished).
/// `counters` is optional so the same server backs both self-play (which wants
/// the avg-batch stats) and the evaluator (which does not).
fn server_loop(rx: Receiver<Req>, model: &LocalModel, max_batch: usize, counters: Option<&Counters>) {
    loop {
        let first = match rx.recv() {
            Ok(r) => r,
            Err(_) => return, // all senders dropped: shut down
        };
        let mut batch = vec![first];
        while batch.len() < max_batch {
            match rx.try_recv() {
                Ok(r) => batch.push(r),
                Err(_) => break,
            }
        }
        if let Some(c) = counters {
            c.batches.fetch_add(1, Ordering::Relaxed);
            c.batch_samples.fetch_add(batch.len() as u64, Ordering::Relaxed);
        }
        let states: Vec<&Connect4State> = batch.iter().map(|r| &r.state).collect();
        let out = model
            .policy_value_batch(&states)
            .expect("batched forward failed");
        for (req, res) in batch.into_iter().zip(out) {
            let _ = req.resp.send(res); // requester may have gone away; ignore
        }
    }
}

/// Counters-free batched inference server, for callers (the evaluator) that want
/// the GPU coalescing of `server_loop` but not the self-play batch stats.
pub(crate) fn run_inference_server(rx: Receiver<Req>, model: &LocalModel, max_batch: usize) {
    server_loop(rx, model, max_batch, None)
}

// ── Driver ───────────────────────────────────────────────────────────────────

struct Counters {
    next: AtomicUsize,    // next game index to claim
    sims: AtomicU64,      // total simulations run (root-to-leaf traversals)
    nodes: AtomicU64,     // total tree nodes built (distinct positions searched)
    examples: AtomicU64,  // total training examples produced
    batches: AtomicU64,   // batched forwards executed (batched mode)
    batch_samples: AtomicU64, // leaves summed over those forwards
}

/// Play games claimed from the shared counter until they run out. When `sink` is
/// given, each finished game's training examples are appended to it (one lock
/// per game — negligible against a game's cost); otherwise they're just counted.
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

/// Aggregate stats from a generation run.
pub struct GenStats {
    pub sims: u64,
    pub nodes: u64,
    pub examples: u64,
    pub elapsed: f64,
    pub avg_batch: f64,
    /// Network forwards actually run (nodes minus eval-cache hits). Engine path only.
    pub evals: u64,
    pub cache_hits: u64,
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
        max_ply: 0, // Connect 4 always plays to a natural terminal
        // Connect 4 has no area-score / resignation concept; play to the end.
        settle_plies: 0,
        resign_threshold: 0.0,
        resign_consecutive: 0,
    }
}

/// Turn the engine's game records into training examples: value = final outcome z
/// from each position's side-to-move POV.
fn examples_from_records(records: Vec<batch_search::GameRecord<Connect4>>) -> Vec<Example> {
    let mut out = Vec::new();
    for rec in records {
        let winner: Option<u8> = match Connect4::outcome(rec.states.last().unwrap()) {
            Outcome::FirstWins => Some(1),
            Outcome::SecondWins => Some(2),
            _ => None,
        };
        for (i, mv) in rec.moves.iter().enumerate() {
            let z = match winner {
                Some(w) => if w == mv.to_move { 1.0 } else { -1.0 },
                None => 0.0,
            };
            let mut policy = [0f32; C4_COLS];
            policy.copy_from_slice(&mv.policy);
            out.push(Example { planes: planes_of(&rec.states[i]), policy, value: z });
        }
    }
    out
}

/// Generate `n_games` across `threads` native threads. With `batched`, a central
/// inference server coalesces leaf evals (the GPU path); otherwise each thread
/// runs its own batch-1 forward. When `collect`, the produced training examples
/// are returned (else an empty Vec — the benchmark path).
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
    if batched {
        let (records, cs) = batch_search::run_chunk::<Connect4>(
            model.as_ref(),
            n_games,
            to_search_params(&params),
            threads,
            max_batch,
            base_seed,
            true, // Connect 4 transposes heavily: ~91% eval-cache hit rate
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
            scope.spawn(move || server_loop(req_rx, model_ref, max_batch, Some(counters_ref)));
            for _ in 0..threads {
                let eval = BatchedEval { req_tx: req_tx.clone() };
                let counters = &counters;
                let params = &params;
                let sink = sink.as_ref();
                scope.spawn(move || worker_loop(&eval, counters, n_games, base_seed, params, sink));
            }
            drop(req_tx); // drop the spare so the server can detect shutdown
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
    };
    let examples = sink.map(|m| m.into_inner().unwrap()).unwrap_or_default();
    (stats, examples)
}

/// Run a self-play throughput benchmark and print the results.
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
) -> anyhow::Result<()> {
    let device = crate::net::pick_device(cuda)?;
    let model = Arc::new(LocalModel::load_on(checkpoint, device)?);
    let mode = if batched { format!("batched (max_batch {max_batch})") } else { "direct".into() };
    eprintln!(
        "rust self-play: {n_games} games, {threads} threads, {} sims/move, \
         device {}, {mode}, model {checkpoint}",
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
        "games {n_games}  examples {}  total_sims {}  nodes {}  \
         time {:.2}s  nodes/s {nodes_per_s:.0}  games/s {games_per_s:.2}{batch_note}",
        s.examples, s.sims, s.nodes, s.elapsed,
    );
    Ok(())
}

/// Self-play training-data record on disk: a little-endian `u64` example count,
/// followed by that many records of 84 plane f32 + 7 policy f32 + 1 value f32
/// (`RECORD_F32` floats each). `python/rust_selfplay_train.py` reads this back.
const RECORD_F32: usize = 2 * 6 * 7 + C4_COLS + 1;

/// Generate one generation of self-play and write the training examples to
/// `out_path`. This is the data source for the Rust-self-play training loop.
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
        "rust self-play gen: {n_games} games, {threads} threads, {budget}, \
         device {}, model {checkpoint} -> {out_path}",
        if cuda { "cuda" } else { "cpu" },
    );
    let (s, examples) =
        generate(model, n_games, threads, params, base_seed, batched, max_batch, true);
    write_dump(out_path, &examples)?;

    let moves = s.examples.max(1) as f64;
    eprintln!(
        "  wrote {} examples ({} games, {:.0} nodes/move, {:.0} sims/move, \
         avg_batch {:.0}) in {:.1}s",
        examples.len(), n_games, s.nodes as f64 / moves, s.sims as f64 / moves,
        s.avg_batch, s.elapsed,
    );
    Ok(())
}

/// Write a replay dump: u64 example count, then `count` records of `RECORD_F32`
/// little-endian f32 (planes, policy, value z).
pub fn write_dump(out_path: &str, examples: &[Example]) -> anyhow::Result<()> {
    let mut buf = Vec::with_capacity(8 + examples.len() * RECORD_F32 * 4);
    buf.extend_from_slice(&(examples.len() as u64).to_le_bytes());
    for ex in examples {
        for &x in ex.planes.iter().chain(ex.policy.iter()).chain(std::iter::once(&ex.value)) {
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

/// Cooperative pause: while the evaluator holds `live/PAUSE` (so it can run its
/// match with the full GPU), park here and advertise `live/actor.paused` so the
/// evaluator knows self-play has actually stopped. Checked between chunks, so the
/// actor yields the GPU at chunk granularity (the evaluator runs infrequently).
fn pause_gate(live_dir: &str) {
    let pause = std::path::Path::new(live_dir).join("PAUSE");
    if !pause.exists() {
        return;
    }
    let marker = std::path::Path::new(live_dir).join("actor.paused");
    let _ = std::fs::File::create(&marker);
    eprintln!("  [actor] paused for evaluation");
    while pause.exists() {
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    let _ = std::fs::remove_file(&marker);
    eprintln!("  [actor] resumed");
}

/// Actor loop: forever load the latest checkpoint (hot-swap), generate a chunk of
/// `games_per_chunk` games, append it to the buffer, and prune to `max_chunks`.
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
) -> anyhow::Result<()> {
    let latest = format!("{live_dir}/latest.safetensors");
    let buffer = std::path::Path::new(live_dir).join("buffer");
    std::fs::create_dir_all(&buffer)?;
    let device = crate::net::pick_device(cuda)?;
    let mut seq = max_chunk_seq(&buffer).map(|s| s + 1).unwrap_or(0);
    eprintln!(
        "c4 actor loop: {games_per_chunk} games/chunk, {threads} threads, device {}, \
         buffer {} (keep {max_chunks}), eps {}, model {latest}",
        if cuda { "cuda" } else { "cpu" }, buffer.display(), params.explore_eps,
    );
    loop {
        pause_gate(live_dir);
        let t0 = Instant::now();
        let model = Arc::new(LocalModel::load_on(&latest, device.clone())?);
        let chunk_seed = base_seed.wrapping_add(seq.wrapping_mul(games_per_chunk as u64));
        let (s, examples) =
            generate(model, games_per_chunk, threads, params, chunk_seed, cuda, max_batch, true);
        let path = buffer.join(format!("chunk_{seq:06}.bin"));
        let tmp = buffer.join(format!("chunk_{seq:06}.bin.tmp"));
        write_dump(tmp.to_str().unwrap(), &examples)?;
        std::fs::rename(&tmp, &path)?;
        prune_buffer(&buffer, max_chunks);
        let moves = s.examples.max(1) as f64;
        eprintln!(
            "  chunk {seq}: {} ex  {:.0} nodes/move  avg_batch {:.0}  {:.1}s",
            examples.len(), s.nodes as f64 / moves, s.avg_batch, t0.elapsed().as_secs_f64(),
        );
        seq += 1;
    }
}
