//! Amazons engine binary. AZ self-play training (`selfplay-loop`, `eval-match`,
//! `selfplay-bench`) plus the gRPC serve stub. See go9-engine for the pattern.

mod eval;
mod game;
mod mcts;
mod moves;
mod net;
mod ort_net;
mod selfplay;

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use game::Amazons;
use playground_core::batch_search::SearchParams;
use playground_model_client::RemoteModelClient;

#[derive(Parser)]
#[command(name = "amazons-engine", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// gRPC serve (remote model). Interactive play.
    Serve {
        #[arg(long, default_value = "127.0.0.1:7102")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7202")]
        model_uri: String,
    },
    /// Self-play generation throughput benchmark.
    SelfplayBench {
        #[arg(long, default_value = "")]
        model: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 128)]
        games: usize,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 800)]
        node_budget: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 120)]
        max_ply: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "candle")]
        backend: String,
        #[arg(long, default_value = "")]
        onnx: String,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Continuous self-play actor: hot-load live weights, append chunks to buffer.
    SelfplayLoop {
        #[arg(long, default_value = "data/models/amazons_az/live")]
        live: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 128)]
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
        /// Root Dirichlet weight (baked into visit counts; keep low).
        #[arg(long, default_value_t = 0.1)]
        dir_eps: f32,
        #[arg(long, default_value_t = 16)]
        temp_moves: u32,
        /// Safety backstop ply cap (Amazons ends naturally well under this).
        #[arg(long, default_value_t = 120)]
        max_ply: u32,
        #[arg(long, default_value_t = -0.95)]
        resign_threshold: f32,
        #[arg(long, default_value_t = 4)]
        resign_consecutive: u32,
        /// Fraction of games that never resign (resign-calibration safeguard).
        #[arg(long, default_value_t = 0.25)]
        resign_playout_frac: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Inference backend: "candle" or "ort" (reads <live>/model.onnx).
        #[arg(long, default_value = "candle")]
        backend: String,
        #[arg(long, default_value = "cuda")]
        ep: String,
        #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Head-to-head match between two checkpoints; print A's score (for ELO).
    EvalMatch {
        #[arg(long)]
        model_a: String,
        #[arg(long)]
        model_b: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 32)]
        games: usize,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 400)]
        node_budget: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 120)]
        max_ply: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve { addr, model_uri } => {
            let model = RemoteModelClient::<Amazons>::connect(&model_uri).await?;
            let models = std::collections::HashMap::from([("nn".to_string(), model)]);
            playground_engine_server::run::<Amazons, _>(addr, models).await?;
        }
        Cmd::SelfplayBench {
            model, cuda, games, threads, max_batch, node_budget, c_puct, max_ply, seed,
            backend, onnx, ep, fp16,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = SearchParams {
                sims: 0, node_budget, c_puct, dir_alpha: 0.5, dir_eps: 0.0, temp_moves: 0,
                explore_eps: 0.0, max_ply, settle_plies: 0, resign_threshold: 0.0,
                resign_consecutive: 0, resign_playout_frac: 0.0,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_benchmark(&model, cuda, games, threads, params, max_batch, seed,
                    &backend, &onnx, &ep, fp16)
            })
            .await??;
        }
        Cmd::SelfplayLoop {
            live, cuda, max_batch, games_per_chunk, max_chunks, threads, node_budget, c_puct,
            dir_alpha, dir_eps, temp_moves, max_ply, resign_threshold, resign_consecutive,
            resign_playout_frac, seed, backend, ep, fp16,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = SearchParams {
                sims: 0, node_budget, c_puct, dir_alpha, dir_eps, temp_moves, explore_eps: 0.0,
                max_ply, settle_plies: 0, resign_threshold, resign_consecutive, resign_playout_frac,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_loop(&live, cuda, params, games_per_chunk, max_chunks, threads,
                    max_batch, seed, &backend, &ep, fp16)
            })
            .await??;
        }
        Cmd::EvalMatch {
            model_a, model_b, cuda, games, threads, max_batch, node_budget, c_puct, max_ply, seed,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                node_budget, c_puct, dir_alpha: 0.5, dir_eps: 0.0, temp_moves: 6, max_ply,
            };
            tokio::task::spawn_blocking(move || {
                eval::run_match(&model_a, &model_b, cuda, games, params, threads, max_batch, seed)
                    .map(|_| ())
            })
            .await??;
        }
    }
    Ok(())
}
