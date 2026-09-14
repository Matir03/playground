//! 9×9 Go engine binary. See connect4-engine for subcommand semantics.

mod eval;
mod game;
mod gnugo;
mod mcts;
mod net;
mod ort_net;
mod selfplay;

use std::collections::HashMap;
use std::net::SocketAddr;

use async_trait::async_trait;
use clap::{Parser, Subcommand};
use game::Go9;
use net::LocalModel;
use playground_core::{LatentGameState, ModelClient, PolicyCandidate};
use playground_model_client::RemoteModelClient;

/// A model the engine can serve: a remote Python model server (gRPC) or the net
/// embedded in-process via candle. `--model name=uri` is remote,
/// `--model-local name=ckpt` is embedded; one engine can mix both.
enum Go9Model {
    Remote(RemoteModelClient<Go9>),
    Local(LocalModel),
}

#[async_trait]
impl ModelClient<Go9> for Go9Model {
    async fn encode(&self, state: &game::Go9State) -> anyhow::Result<LatentGameState> {
        match self {
            Go9Model::Remote(m) => m.encode(state).await,
            Go9Model::Local(m) => m.encode(state).await,
        }
    }
    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        match self {
            Go9Model::Remote(m) => m.eval(lgs).await,
            Go9Model::Local(m) => m.eval(lgs).await,
        }
    }
    async fn policy(
        &self,
        lgs: &LatentGameState,
        top_b: u32,
    ) -> anyhow::Result<Vec<PolicyCandidate>> {
        match self {
            Go9Model::Remote(m) => m.policy(lgs, top_b).await,
            Go9Model::Local(m) => m.policy(lgs, top_b).await,
        }
    }
    async fn reify(
        &self,
        from_state: &game::Go9State,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
        action_seed: Option<&[u8]>,
        noise_scale: Option<f64>,
    ) -> anyhow::Result<game::Go9Move> {
        match self {
            Go9Model::Remote(m) => {
                m.reify(from_state, from_lgs, to_lgs, action_seed, noise_scale)
                    .await
            }
            Go9Model::Local(m) => {
                m.reify(from_state, from_lgs, to_lgs, action_seed, noise_scale)
                    .await
            }
        }
    }
    async fn model_version(&self) -> anyhow::Result<String> {
        match self {
            Go9Model::Remote(m) => m.model_version().await,
            Go9Model::Local(m) => m.model_version().await,
        }
    }
}

#[derive(Parser)]
#[command(name = "go9-engine", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Long-lived gRPC server. Interactive play + matches.
    Serve {
        #[arg(long, default_value = "127.0.0.1:7101")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7201")]
        model_uri: String,
        /// Named remote NN model server(s), as `name=uri`.
        #[arg(long = "model", value_name = "NAME=URI")]
        models: Vec<String>,
        /// Named in-process NN(s), as `name=checkpoint.safetensors` (candle).
        #[arg(long = "model-local", value_name = "NAME=CHECKPOINT")]
        models_local: Vec<String>,
        /// Run in-process NN inference on the GPU (needs a `cuda`-feature build).
        /// Much faster per move for the larger Go net — use it for play.
        #[arg(long)]
        cuda: bool,
    },
    /// As Serve, but also accepts StartSelfPlay requests.
    SelfPlay {
        #[arg(long, default_value = "127.0.0.1:7101")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7201")]
        model_uri: String,
        #[arg(long = "model", value_name = "NAME=URI")]
        models: Vec<String>,
        #[arg(long = "model-local", value_name = "NAME=CHECKPOINT")]
        models_local: Vec<String>,
        #[arg(long)]
        cuda: bool,
    },
    /// Self-play generation throughput benchmark.
    SelfplayBench {
        #[arg(long)]
        model: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        batched: bool,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 32)]
        games: usize,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 200)]
        sims: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 0.5)]
        dir_alpha: f32,
        #[arg(long, default_value_t = 0.25)]
        dir_eps: f32,
        #[arg(long, default_value_t = 16)]
        temp_moves: u32,
        /// Per-node uniform exploration floor mixed into priors (0 = off). Keeps
        /// self-play from collapsing into a narrow action subspace.
        #[arg(long, default_value_t = 0.25)]
        explore_eps: f32,
        #[arg(long, default_value_t = 80)]
        max_ply: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Inference backend: "candle" (default) or "ort" (ONNX Runtime).
        #[arg(long, default_value = "candle")]
        backend: String,
        /// ONNX model path (required for --backend ort).
        #[arg(long, default_value = "")]
        onnx: String,
        /// ORT execution provider: "cuda" or "trt" (TensorRT).
        #[arg(long, default_value = "trt")]
        ep: String,
        /// Use fp16 (TensorRT EP). fp32 if false. Takes a value: --fp16 true|false.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Generate one generation of self-play games to `--out` (training data).
    SelfplayGen {
        #[arg(long)]
        model: String,
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
        /// Per-move node budget (overrides --move-ms and --sims when > 0).
        #[arg(long, default_value_t = 0)]
        node_budget: u32,
        #[arg(long, default_value_t = 0.0)]
        move_ms: f64,
        #[arg(long, default_value_t = 200)]
        sims: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 0.5)]
        dir_alpha: f32,
        #[arg(long, default_value_t = 0.25)]
        dir_eps: f32,
        #[arg(long, default_value_t = 16)]
        temp_moves: u32,
        /// Per-node uniform exploration floor mixed into priors (0 = off).
        #[arg(long, default_value_t = 0.25)]
        explore_eps: f32,
        /// End a self-play game at this ply and adjudicate by area score.
        #[arg(long, default_value_t = 80)]
        max_ply: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Continuous self-play actor: forever load live/latest.safetensors (hot-swap),
    /// generate a chunk of games, append to live/buffer, prune to --max-chunks.
    SelfplayLoop {
        /// The live workspace dir (holds latest.safetensors, config.json, buffer/).
        #[arg(long, default_value = "data/models/go9/live")]
        live: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 32)]
        games_per_chunk: usize,
        #[arg(long, default_value_t = 60)]
        max_chunks: usize,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 800)]
        node_budget: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 0.5)]
        dir_alpha: f32,
        /// Root Dirichlet exploration weight. This noise is baked into the visit
        /// counts and can't be subtracted from the policy target, so keep it low.
        #[arg(long, default_value_t = 0.1)]
        dir_eps: f32,
        #[arg(long, default_value_t = 16)]
        temp_moves: u32,
        /// Root-only ε-greedy forced exploration (excluded from the policy target).
        /// Default off: rely on Dirichlet so we don't burn search budget on random
        /// root visits that make the (already budget-limited) targets flatter.
        #[arg(long, default_value_t = 0.0)]
        explore_eps: f32,
        /// Safety backstop ply cap (not the primary terminator — score-stability +
        /// resignation end most games first; this only bounds chaotic early/random
        /// nets that never settle, and prevents non-terminating repetition cycles).
        #[arg(long, default_value_t = 120)]
        max_ply: u32,
        /// Score-stability early stop: end a game once the area score is unchanged
        /// for this many consecutive plies (the borders are settled). 0 = off.
        /// Off by default: a weak net can hit "stable" with dead stones still on the
        /// board, so area scoring would mislabel the outcome — play games out for now.
        #[arg(long, default_value_t = 0)]
        settle_plies: u32,
        /// Resignation: a side resigns/loses when its root win-estimate stays at or
        /// below this for `resign-consecutive` of its own moves in a row.
        #[arg(long, default_value_t = -0.95)]
        resign_threshold: f32,
        /// Consecutive own-moves at/under `resign-threshold` before a side resigns
        /// (0 = resignation off).
        #[arg(long, default_value_t = 4)]
        resign_consecutive: u32,
        /// Fraction of games that never resign and play to the end — the resign
        /// calibration safeguard, so a too-aggressive threshold can't corrupt
        /// training with false resignations. 0 = every game may resign.
        #[arg(long, default_value_t = 0.25)]
        resign_playout_frac: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Inference backend: "candle" (default) or "ort" (ONNX Runtime; reads
        /// <live>/model.onnx, which the learner exports each snapshot).
        #[arg(long, default_value = "candle")]
        backend: String,
        /// ORT execution provider: "cuda" (recommended for hot-swap) or "trt".
        #[arg(long, default_value = "cuda")]
        ep: String,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Play a head-to-head match between two checkpoints; print A's score (ELO).
    EvalMatch {
        #[arg(long)]
        model_a: String,
        #[arg(long)]
        model_b: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 16)]
        games: usize,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 128)]
        node_budget: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 80)]
        max_ply: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Absolute-strength eval: the net (concrete value+policy MCTS) vs GNU Go (GTP).
    GnugoMatch {
        #[arg(long)]
        model: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 10)]
        games: u32,
        #[arg(long, default_value_t = 1, help = "GNU Go strength 1..10")]
        level: u32,
        #[arg(long, default_value_t = 800, help = "net node budget per move")]
        visits: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 200)]
        max_ply: u32,
        #[arg(long, default_value = "gnugo")]
        gnugo: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    Match {
        #[arg(long)]
        first_player_uri: String,
        #[arg(long)]
        second_player_uri: String,
        #[arg(long, default_value_t = 1)]
        num_games: u32,
    },
    Stdio,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Batch-1 candle inference on the 9×9 board doesn't benefit from rayon's
    // intra-op threading; default to 1 thread and parallelise across games.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve {
            addr,
            model_uri,
            models,
            models_local,
            cuda,
        }
        | Cmd::SelfPlay {
            addr,
            model_uri,
            models,
            models_local,
            cuda,
        } => {
            let device = net::pick_device(cuda)?;
            let mut clients: HashMap<String, Go9Model> = HashMap::new();
            for spec in &models_local {
                let (name, ckpt) = spec.split_once('=').ok_or_else(|| {
                    anyhow::anyhow!("--model-local must be NAME=CHECKPOINT, got {spec:?}")
                })?;
                clients.insert(
                    name.to_string(),
                    Go9Model::Local(LocalModel::load_on(ckpt, device.clone())?),
                );
            }
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
                clients.insert(
                    name,
                    Go9Model::Remote(RemoteModelClient::<Go9>::connect(&uri).await?),
                );
            }
            playground_engine_server::run::<Go9, _>(addr, clients).await?;
        }
        Cmd::SelfplayBench {
            model,
            cuda,
            batched,
            max_batch,
            games,
            threads,
            sims,
            c_puct,
            dir_alpha,
            dir_eps,
            temp_moves,
            explore_eps,
            max_ply,
            seed,
            backend,
            onnx,
            ep,
            fp16,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims,
                move_ms: 0.0,
                node_budget: 0,
                c_puct,
                dir_alpha,
                dir_eps,
                temp_moves,
                explore_eps,
                max_ply,
                settle_plies: 0,
                resign_threshold: 0.0,
                resign_consecutive: 0,
                resign_playout_frac: 0.0,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_benchmark(
                    &model, cuda, games, threads, params, batched, max_batch, seed, &backend,
                    &onnx, &ep, fp16,
                )
            })
            .await??;
        }
        Cmd::SelfplayGen {
            model,
            out,
            cuda,
            batched,
            max_batch,
            games,
            threads,
            node_budget,
            move_ms,
            sims,
            c_puct,
            dir_alpha,
            dir_eps,
            temp_moves,
            explore_eps,
            max_ply,
            seed,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims,
                move_ms,
                node_budget,
                c_puct,
                dir_alpha,
                dir_eps,
                temp_moves,
                explore_eps,
                max_ply,
                settle_plies: 0,
                resign_threshold: 0.0,
                resign_consecutive: 0,
                resign_playout_frac: 0.0,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_generate(
                    &model, cuda, &out, games, threads, params, batched, max_batch, seed,
                )
            })
            .await??;
        }
        Cmd::SelfplayLoop {
            live,
            cuda,
            max_batch,
            games_per_chunk,
            max_chunks,
            threads,
            node_budget,
            c_puct,
            dir_alpha,
            dir_eps,
            temp_moves,
            explore_eps,
            max_ply,
            seed,
            settle_plies,
            resign_threshold,
            resign_consecutive,
            resign_playout_frac,
            backend,
            ep,
            fp16,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims: 0,
                move_ms: 0.0,
                node_budget,
                c_puct,
                dir_alpha,
                dir_eps,
                temp_moves,
                explore_eps,
                max_ply,
                settle_plies,
                resign_threshold,
                resign_consecutive,
                resign_playout_frac,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_loop(
                    &live,
                    cuda,
                    params,
                    games_per_chunk,
                    max_chunks,
                    threads,
                    max_batch,
                    seed,
                    &backend,
                    &ep,
                    fp16,
                )
            })
            .await??;
        }
        Cmd::EvalMatch {
            model_a,
            model_b,
            cuda,
            games,
            threads,
            max_batch,
            node_budget,
            c_puct,
            max_ply,
            seed,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            // Match play: net's best move (argmax), but sample the first few
            // moves for game diversity (temp_moves), and no root noise.
            let params = mcts::Params {
                sims: 0,
                move_ms: 0.0,
                node_budget,
                c_puct,
                dir_alpha: 0.5,
                dir_eps: 0.0,
                temp_moves: 8,
                explore_eps: 0.0,
                max_ply,
                settle_plies: 0,
                resign_threshold: 0.0,
                resign_consecutive: 0,
                resign_playout_frac: 0.0,
            };
            tokio::task::spawn_blocking(move || {
                eval::run_match(
                    &model_a, &model_b, cuda, games, params, threads, max_batch, seed,
                )
                .map(|_| ())
            })
            .await??;
        }
        Cmd::GnugoMatch {
            model,
            cuda,
            games,
            level,
            visits,
            c_puct,
            max_ply,
            gnugo,
            seed,
        } => {
            tokio::task::spawn_blocking(move || {
                gnugo::run_gnugo_match(
                    &model, cuda, &gnugo, level, games, visits, c_puct, max_ply, seed,
                )
                .map(|_| ())
            })
            .await??;
        }
        Cmd::Match { .. } => anyhow::bail!("match: implement in M2"),
        Cmd::Stdio => anyhow::bail!("stdio: implement when needed"),
    }
    Ok(())
}
