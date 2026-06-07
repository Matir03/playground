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

use crate::game::{Connect4State, C4_COLS};
use crate::mcts::{self_play_game, Evaluator, Example, Params};
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
struct Req {
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
struct BatchedEval {
    req_tx: Sender<Req>,
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
fn server_loop(rx: Receiver<Req>, model: &LocalModel, max_batch: usize, counters: &Counters) {
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
        counters.batches.fetch_add(1, Ordering::Relaxed);
        counters.batch_samples.fetch_add(batch.len() as u64, Ordering::Relaxed);
        let states: Vec<&Connect4State> = batch.iter().map(|r| &r.state).collect();
        let out = model
            .policy_value_batch(&states)
            .expect("batched forward failed");
        for (req, res) in batch.into_iter().zip(out) {
            let _ = req.resp.send(res); // requester may have gone away; ignore
        }
    }
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
    let sims_per_s = s.sims as f64 / s.elapsed;
    let games_per_s = n_games as f64 / s.elapsed;
    let batch_note = if batched { format!("  avg_batch {:.0}", s.avg_batch) } else { String::new() };
    println!(
        "games {n_games}  examples {}  total_sims {}  \
         time {:.2}s  sims/s {sims_per_s:.0}  games/s {games_per_s:.2}{batch_note}",
        s.examples, s.sims, s.elapsed,
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
    let budget = if params.move_ms > 0.0 {
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

    let mut buf = Vec::with_capacity(8 + examples.len() * RECORD_F32 * 4);
    buf.extend_from_slice(&(examples.len() as u64).to_le_bytes());
    for ex in &examples {
        for &x in ex.planes.iter().chain(ex.policy.iter()).chain(std::iter::once(&ex.value)) {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    f.write_all(&buf)?;
    f.flush()?;

    let moves = s.examples.max(1) as f64;
    eprintln!(
        "  wrote {} examples ({} games, {:.0} nodes/move, {:.0} sims/move, \
         avg_batch {:.0}) in {:.1}s",
        examples.len(), n_games, s.nodes as f64 / moves, s.sims as f64 / moves,
        s.avg_batch, s.elapsed,
    );
    Ok(())
}
