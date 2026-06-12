//! 9×9 Go engine binary. See connect4-engine for subcommand semantics.

mod game;
mod mcts;
mod net;
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
    async fn eval(&self, gs: &LatentGameState) -> anyhow::Result<f64> {
        match self {
            Go9Model::Remote(m) => m.eval(gs).await,
            Go9Model::Local(m) => m.eval(gs).await,
        }
    }
    async fn policy(&self, gs: &LatentGameState, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>> {
        match self {
            Go9Model::Remote(m) => m.policy(gs, top_b).await,
            Go9Model::Local(m) => m.policy(gs, top_b).await,
        }
    }
    async fn reify(
        &self,
        from_state: &game::Go9State,
        from_gs: &LatentGameState,
        to_gs: &LatentGameState,
        action_seed: Option<&[u8]>,
        noise_scale: Option<f64>,
    ) -> anyhow::Result<game::Go9Move> {
        match self {
            Go9Model::Remote(m) => m.reify(from_state, from_gs, to_gs, action_seed, noise_scale).await,
            Go9Model::Local(m) => m.reify(from_state, from_gs, to_gs, action_seed, noise_scale).await,
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
        #[arg(long, default_value_t = 80)]
        max_ply: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
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
        /// End a self-play game at this ply and adjudicate by area score.
        #[arg(long, default_value_t = 80)]
        max_ply: u32,
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
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
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
        Cmd::Serve { addr, model_uri, models, models_local }
        | Cmd::SelfPlay { addr, model_uri, models, models_local } => {
            let mut clients: HashMap<String, Go9Model> = HashMap::new();
            for spec in &models_local {
                let (name, ckpt) = spec
                    .split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("--model-local must be NAME=CHECKPOINT, got {spec:?}"))?;
                clients.insert(name.to_string(), Go9Model::Local(LocalModel::load(ckpt)?));
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
                clients.insert(name, Go9Model::Remote(RemoteModelClient::<Go9>::connect(&uri).await?));
            }
            playground_engine_server::run::<Go9, _>(addr, clients).await?;
        }
        Cmd::SelfplayBench {
            model, cuda, batched, max_batch, games, threads,
            sims, c_puct, dir_alpha, dir_eps, temp_moves, max_ply, seed,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims, move_ms: 0.0, node_budget: 0, c_puct, dir_alpha, dir_eps, temp_moves, max_ply,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_benchmark(&model, cuda, games, threads, params, batched, max_batch, seed)
            })
            .await??;
        }
        Cmd::SelfplayGen {
            model, out, cuda, batched, max_batch, games, threads, node_budget, move_ms,
            sims, c_puct, dir_alpha, dir_eps, temp_moves, max_ply, seed,
        } => {
            let threads = threads.unwrap_or_else(default_threads);
            let params = mcts::Params {
                sims, move_ms, node_budget, c_puct, dir_alpha, dir_eps, temp_moves, max_ply,
            };
            tokio::task::spawn_blocking(move || {
                selfplay::run_generate(
                    &model, cuda, &out, games, threads, params, batched, max_batch, seed,
                )
            })
            .await??;
        }
        Cmd::Match { .. } => anyhow::bail!("match: implement in M2"),
        Cmd::Stdio => anyhow::bail!("stdio: implement when needed"),
    }
    Ok(())
}
