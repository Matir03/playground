//! Connect 4 engine binary.
//!
//! See docs/protocols.md for subcommand semantics.

mod book;
mod eval;
mod game;
mod mcts;
mod net;
mod selfplay;
mod solver;

use std::collections::HashMap;
use std::net::SocketAddr;

use async_trait::async_trait;
use clap::{Parser, Subcommand};
use game::Connect4;
use net::LocalModel;
use playground_core::{LatentGameState, ModelClient, PolicyCandidate};
use playground_model_client::RemoteModelClient;

/// A model the engine can serve: either a remote Python model server (gRPC) or
/// the net embedded in-process via candle (no per-node round-trips). One engine
/// can mix both — `--model name=uri` is remote, `--model-local name=ckpt` is
/// embedded — so the `EngineServer`'s single model type is this enum.
enum C4Model {
    Remote(RemoteModelClient<Connect4>),
    Local(LocalModel),
}

#[async_trait]
impl ModelClient<Connect4> for C4Model {
    async fn encode(&self, state: &game::Connect4State) -> anyhow::Result<LatentGameState> {
        match self {
            C4Model::Remote(m) => m.encode(state).await,
            C4Model::Local(m) => m.encode(state).await,
        }
    }
    async fn eval(&self, gs: &LatentGameState) -> anyhow::Result<f64> {
        match self {
            C4Model::Remote(m) => m.eval(gs).await,
            C4Model::Local(m) => m.eval(gs).await,
        }
    }
    async fn policy(&self, gs: &LatentGameState, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>> {
        match self {
            C4Model::Remote(m) => m.policy(gs, top_b).await,
            C4Model::Local(m) => m.policy(gs, top_b).await,
        }
    }
    async fn reify(
        &self,
        from_state: &game::Connect4State,
        from_gs: &LatentGameState,
        to_gs: &LatentGameState,
        action_seed: Option<&[u8]>,
        noise_scale: Option<f64>,
    ) -> anyhow::Result<game::Connect4Move> {
        match self {
            C4Model::Remote(m) => m.reify(from_state, from_gs, to_gs, action_seed, noise_scale).await,
            C4Model::Local(m) => m.reify(from_state, from_gs, to_gs, action_seed, noise_scale).await,
        }
    }
    async fn model_version(&self) -> anyhow::Result<String> {
        match self {
            C4Model::Remote(m) => m.model_version().await,
            C4Model::Local(m) => m.model_version().await,
        }
    }
}

#[derive(Parser)]
#[command(name = "connect4-engine", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Long-lived gRPC server. Interactive play + matches.
    Serve {
        #[arg(long, default_value = "127.0.0.1:7100")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7200")]
        model_uri: String,
        /// Named NN model server(s), as `name=uri`. Repeat for several (e.g.
        /// `--model random=http://127.0.0.1:7200 --model trained=...:7201`).
        /// If none given (and no `--model-local`), `model_uri` is registered
        /// under the name "nn".
        #[arg(long = "model", value_name = "NAME=URI")]
        models: Vec<String>,
        /// Named in-process NN(s), as `name=checkpoint.safetensors`. Repeat for
        /// several. These run inference inside the engine via candle — no Python
        /// model server and no per-node gRPC round-trips — so the MCTS reaches
        /// far more visits per time budget. The architecture is read from the
        /// checkpoint's sibling `config.json`.
        #[arg(long = "model-local", value_name = "NAME=CHECKPOINT")]
        models_local: Vec<String>,
        /// Path to the Connect 4 opening book. Defaults to the bundled
        /// `data/7x6.book`; the perfect engine still works without it but is
        /// slow in the opening.
        #[arg(long)]
        book: Option<String>,
    },
    /// As Serve, but also accepts StartSelfPlay requests.
    SelfPlay {
        #[arg(long, default_value = "127.0.0.1:7100")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7200")]
        model_uri: String,
        #[arg(long = "model", value_name = "NAME=URI")]
        models: Vec<String>,
        #[arg(long = "model-local", value_name = "NAME=CHECKPOINT")]
        models_local: Vec<String>,
        #[arg(long, default_value_t = 16)]
        workers: u32,
        #[arg(long)]
        book: Option<String>,
    },
    /// Client-side match runner.
    Match {
        #[arg(long)]
        first_player_uri: String,
        #[arg(long)]
        second_player_uri: String,
        #[arg(long, default_value_t = 1)]
        num_games: u32,
    },
    /// Newline-delimited stdio protocol for debugging / tools.
    Stdio,
    /// Self-play generation throughput benchmark (the Rust port of the Python
    /// multiprocessing self-play). Generates games on native threads sharing the
    /// in-process net and reports sims/sec — for comparing against Python.
    SelfplayBench {
        /// Checkpoint to play with (architecture from its sibling config.json).
        #[arg(long)]
        model: String,
        /// Run inference on the GPU (requires a `cuda`-feature build).
        #[arg(long)]
        cuda: bool,
        /// Coalesce leaf evals from all threads into batched forwards via a
        /// central inference server. Required for a sane GPU run; on CPU the
        /// default batch-1-per-thread path is usually faster.
        #[arg(long)]
        batched: bool,
        /// Max leaves coalesced into one batched forward (with --batched).
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 64)]
        games: usize,
        /// Worker threads (true-parallel tree work; defaults to all cores). With
        /// --batched, oversubscribing cores helps fill batches (threads waiting
        /// on the server don't burn a core).
        #[arg(long)]
        threads: Option<usize>,
        /// Fixed simulations per move (the throughput unit; avoids the
        /// time-budget confound a comparison would otherwise have).
        #[arg(long, default_value_t = 200)]
        sims: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 1.0)]
        dir_alpha: f32,
        #[arg(long, default_value_t = 0.25)]
        dir_eps: f32,
        #[arg(long, default_value_t = 10)]
        temp_moves: u32,
        /// Per-node ε-greedy forced exploration (random visits excluded from the
        /// policy target). 0 = off.
        #[arg(long, default_value_t = 0.25)]
        explore_eps: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Generate one generation of self-play games and write the training
    /// examples to `--out` (consumed by python/rust_selfplay_train.py). Same
    /// engine as `selfplay-bench` but collects + dumps examples, and supports a
    /// per-move time budget via `--move-ms`.
    SelfplayGen {
        #[arg(long)]
        model: String,
        /// Output file for the training examples (binary; see selfplay.rs).
        #[arg(long)]
        out: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        batched: bool,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 64)]
        games: usize,
        #[arg(long)]
        threads: Option<usize>,
        /// Per-move node budget: search until the tree holds this many distinct
        /// nodes (overrides --move-ms and --sims when > 0). Consistent depth
        /// regardless of saturation.
        #[arg(long, default_value_t = 0)]
        node_budget: u32,
        /// Per-move time budget in ms (overrides --sims when > 0).
        #[arg(long, default_value_t = 0.0)]
        move_ms: f64,
        #[arg(long, default_value_t = 200)]
        sims: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 1.0)]
        dir_alpha: f32,
        #[arg(long, default_value_t = 0.25)]
        dir_eps: f32,
        #[arg(long, default_value_t = 10)]
        temp_moves: u32,
        /// Per-node ε-greedy forced exploration (random visits excluded from the
        /// policy target). 0 = off.
        #[arg(long, default_value_t = 0.25)]
        explore_eps: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Continuous self-play actor: forever load live/latest.safetensors (hot-swap),
    /// generate a chunk of games, append to live/buffer, prune to --max-chunks.
    SelfplayLoop {
        #[arg(long, default_value = "data/models/connect4/live")]
        live: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 64)]
        games_per_chunk: usize,
        #[arg(long, default_value_t = 60)]
        max_chunks: usize,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 0)]
        node_budget: u32,
        #[arg(long, default_value_t = 0.0)]
        move_ms: f64,
        #[arg(long, default_value_t = 200)]
        sims: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 1.0)]
        dir_alpha: f32,
        #[arg(long, default_value_t = 0.25)]
        dir_eps: f32,
        #[arg(long, default_value_t = 10)]
        temp_moves: u32,
        #[arg(long, default_value_t = 0.25)]
        explore_eps: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Absolute-strength eval: play a checkpoint's MCTS vs the perfect solver and
    /// print the net's score (the distance-to-perfect signal).
    EvalSolver {
        #[arg(long)]
        model: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 40)]
        games: u32,
        #[arg(long, default_value_t = 200)]
        node_budget: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        /// Random opening plies for variety (each game).
        #[arg(long, default_value_t = 4)]
        opening: u32,
        /// Worker threads playing games in parallel (leaf evals batched).
        #[arg(long)]
        threads: Option<usize>,
        /// Max leaves coalesced into one batched forward.
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

/// Default worker-thread count: all available cores.
fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // candle's CPU matmul/conv parallelizes over rayon, but for batch-1
    // inference on Connect 4's tiny 6×7 board the threading overhead dominates
    // and *slows* the in-process net (measured ~4× slower at 11 threads than 1).
    // Default to a single thread; users can still override via the env var.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve { addr, model_uri, models, models_local, book }
        | Cmd::SelfPlay { addr, model_uri, models, models_local, book, .. } => {
            if let Some(path) = book {
                // The solver reads this on first use (see solver::book()).
                std::env::set_var("CONNECT4_BOOK", path);
            }
            let mut clients: HashMap<String, C4Model> = HashMap::new();
            // In-process nets first (`--model-local name=checkpoint`).
            for spec in &models_local {
                let (name, ckpt) = spec
                    .split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("--model-local must be NAME=CHECKPOINT, got {spec:?}"))?;
                clients.insert(name.to_string(), C4Model::Local(LocalModel::load(ckpt)?));
            }
            // Remote model servers (`--model name=uri`). With neither flavor of
            // model given, register `--model-uri` as "nn" for back-compat.
            let remote_specs: Vec<(String, String)> = if models.is_empty() && clients.is_empty() {
                vec![("nn".to_string(), model_uri)]
            } else {
                models
                    .iter()
                    .map(|s| {
                        s.split_once('=')
                            .map(|(n, u)| (n.to_string(), u.to_string()))
                            .ok_or_else(|| anyhow::anyhow!("--model must be NAME=URI, got {s:?}"))
                    })
                    .collect::<anyhow::Result<_>>()?
            };
            for (name, uri) in remote_specs {
                clients.insert(name, C4Model::Remote(RemoteModelClient::<Connect4>::connect(&uri).await?));
            }
            playground_engine_server::run::<Connect4, _>(addr, clients).await?;
        }
        Cmd::Match { .. } => {
            anyhow::bail!("match subcommand: implement in M2");
        }
        Cmd::Stdio => {
            anyhow::bail!("stdio subcommand: implement when a second tool needs it");
        }
        Cmd::SelfplayBench {
            model, cuda, batched, max_batch, games, threads,
            sims, c_puct, dir_alpha, dir_eps, temp_moves, explore_eps, seed,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims, move_ms: 0.0, node_budget: 0, c_puct, dir_alpha, dir_eps, temp_moves, explore_eps,
            };
            // Run off the async runtime: this is CPU-bound, synchronous work.
            tokio::task::spawn_blocking(move || {
                selfplay::run_benchmark(
                    &model, cuda, games, threads, params, batched, max_batch, seed,
                )
            })
            .await??;
        }
        Cmd::SelfplayGen {
            model, out, cuda, batched, max_batch, games, threads, node_budget, move_ms,
            sims, c_puct, dir_alpha, dir_eps, temp_moves, explore_eps, seed,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims, move_ms, node_budget, c_puct, dir_alpha, dir_eps, temp_moves, explore_eps,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_generate(
                    &model, cuda, &out, games, threads, params, batched, max_batch, seed,
                )
            })
            .await??;
        }
        Cmd::SelfplayLoop {
            live, cuda, max_batch, games_per_chunk, max_chunks, threads, node_budget, move_ms,
            sims, c_puct, dir_alpha, dir_eps, temp_moves, explore_eps, seed,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims, move_ms, node_budget, c_puct, dir_alpha, dir_eps, temp_moves, explore_eps,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_loop(&live, cuda, params, games_per_chunk, max_chunks, threads, max_batch, seed)
            })
            .await??;
        }
        Cmd::EvalSolver { model, cuda, games, node_budget, c_puct, opening, threads, max_batch, seed } => {
            // Deterministic-ish net play (argmax, no root noise) vs the solver.
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims: 0, move_ms: 0.0, node_budget, c_puct,
                dir_alpha: 1.0, dir_eps: 0.0, temp_moves: 0, explore_eps: 0.0,
            };
            tokio::task::spawn_blocking(move || {
                eval::run_eval_vs_solver(&model, cuda, games, params, opening, seed, threads, max_batch)
                    .map(|_| ())
            })
            .await??;
        }
    }
    Ok(())
}
