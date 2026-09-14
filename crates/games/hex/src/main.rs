//! Hex 13×13 engine binary: AZ self-play training (`selfplay-loop`, `eval-match`,
//! `selfplay-bench`, `vs-mohex`), the gRPC serve path, and the MoHex bridge
//! (`verify` cross-checks the rules, `match` plays MoHex against a stand-in).
//! See amazons-engine for the pattern the training subcommands follow.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Instant;

use async_trait::async_trait;
use clap::{Parser, Subcommand};
use playground_core::batch_search::SearchParams;
use playground_core::{Game, LatentGameState, ModelClient, PolicyCandidate};
use playground_model_client::RemoteModelClient;

use hex_engine::game::{Hex, HexMove, HexState};
use hex_engine::{eval, mohex, net, selfplay};

/// A model the engine can serve: a remote Python model server (gRPC) or the AZ net
/// embedded in-process via candle. `--model-local name=ckpt` is embedded, the
/// default `--model-uri` is remote. Mirrors amazons-engine.
enum HexModel {
    Remote(RemoteModelClient<Hex>),
    Local(net::LocalModel),
}

#[async_trait]
impl ModelClient<Hex> for HexModel {
    async fn encode(&self, state: &HexState) -> anyhow::Result<LatentGameState> {
        match self {
            Self::Remote(m) => m.encode(state).await,
            Self::Local(m) => m.encode(state).await,
        }
    }
    async fn eval(&self, lgs: &LatentGameState) -> anyhow::Result<f64> {
        match self {
            Self::Remote(m) => m.eval(lgs).await,
            Self::Local(m) => m.eval(lgs).await,
        }
    }
    async fn policy(
        &self,
        lgs: &LatentGameState,
        top_b: u32,
    ) -> anyhow::Result<Vec<PolicyCandidate>> {
        match self {
            Self::Remote(m) => m.policy(lgs, top_b).await,
            Self::Local(m) => m.policy(lgs, top_b).await,
        }
    }
    async fn reify(
        &self,
        from_state: &HexState,
        from_lgs: &LatentGameState,
        to_lgs: &LatentGameState,
        action_seed: Option<&[u8]>,
        noise_scale: Option<f64>,
    ) -> anyhow::Result<HexMove> {
        match self {
            Self::Remote(m) => {
                m.reify(from_state, from_lgs, to_lgs, action_seed, noise_scale)
                    .await
            }
            Self::Local(m) => {
                m.reify(from_state, from_lgs, to_lgs, action_seed, noise_scale)
                    .await
            }
        }
    }
    async fn model_version(&self) -> anyhow::Result<String> {
        match self {
            Self::Remote(m) => m.model_version().await,
            Self::Local(m) => m.model_version().await,
        }
    }
}

/// Version string that names the compiled backends, because a `cargo build`
/// without `--features cuda` silently replaces a cuda binary at the same path —
/// and the only symptom is `--cuda` failing at run time, possibly hours later.
#[cfg(feature = "cuda")]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " +cuda");
#[cfg(not(feature = "cuda"))]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (built without cuda)");

#[derive(Parser)]
#[command(name = "hex-engine", version = VERSION)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Serve {
        #[arg(long, default_value = "127.0.0.1:7104")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7204")]
        model_uri: String,
        /// Embedded AZ net(s), as `name=checkpoint.safetensors`. The UI's NN seat
        /// selects by name; with none given only the remote model server is served.
        #[arg(long = "model-local", value_name = "NAME=CHECKPOINT")]
        models_local: Vec<String>,
        /// Run the embedded net on the GPU (candle cuda). Needs the `cuda` feature.
        #[arg(long)]
        cuda: bool,
        /// Play MoHex for `RequestMove(player = EXTERNAL)` seats. Needs
        /// scripts/fetch-mohex.sh; without it those requests are refused.
        #[arg(long)]
        mohex_bin: Option<String>,
        /// Default MoHex playouts per move (a request's budget overrides it).
        #[arg(long, default_value_t = 10_000)]
        mohex_playouts: u32,
        /// Default MoHex seconds per move (0 = budget by playouts only).
        #[arg(long, default_value_t = 0.0)]
        mohex_time: f64,
    },
    /// Continuous AZ self-play actor: hot-load live weights, append chunks to the
    /// learner's replay buffer.
    SelfplayLoop {
        #[arg(long, default_value = "data/models/hex_az/live")]
        live: String,
        #[arg(long)]
        cuda: bool,
        /// Concurrent games kept in flight (overcommit ~3× max-batch).
        #[arg(long, default_value_t = 512)]
        concurrency: usize,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        /// Wall-clock seconds of self-play per buffer chunk.
        #[arg(long, default_value_t = 60.0)]
        chunk_secs: f64,
        /// Replay-buffer window, in chunks (a backstop — see --max-examples).
        #[arg(long, default_value_t = 200)]
        max_chunks: usize,
        /// Replay-buffer window in *examples*, which is what training actually
        /// cares about. A chunk is a time slice, so its size tracks GPU speed:
        /// bounding by chunks alone let the buffer collapse 4x when the GPU
        /// clocked down. 0 = chunk count only.
        #[arg(long, default_value_t = 300_000)]
        max_examples: u64,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 400)]
        node_budget: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        /// Root Dirichlet noise (AZ exploration).
        #[arg(long, default_value_t = 0.3)]
        dir_alpha: f32,
        #[arg(long, default_value_t = 0.25)]
        dir_eps: f32,
        /// Sample from the visit counts for this many opening plies, then argmax.
        #[arg(long, default_value_t = 12)]
        temp_moves: u32,
        /// KataGo forced playouts + policy target pruning (0 = off, 2.0 = KataGo's k):
        /// explore the Dirichlet noise, then subtract it back out of the target.
        #[arg(long, default_value_t = 2.0)]
        forced_playout_k: f32,
        /// Playout cap randomization: fraction of moves that get the full
        /// `--node-budget` and become training examples (0 = every move does).
        #[arg(long, default_value_t = 0.25)]
        full_search_prob: f32,
        /// Visits for the other moves — they advance the game but aren't recorded.
        #[arg(long, default_value_t = 100)]
        fast_node_budget: u32,
        /// Random openings: each game plays N ~ U[0, this] uniformly random legal
        /// moves before the search takes over, and none of them are trained on
        /// (0 = off; 169 = the whole board). Keeps the training distribution
        /// moving even while the champion generating the data is frozen.
        #[arg(long, default_value_t = 0)]
        random_opening_max: u32,
        /// Shape of the opening length: P(N=n) ∝ this^-n, truncated at
        /// --random-opening-max (1.0 = uniform). 1.01 tilts toward short openings
        /// while still reaching deep ones.
        #[arg(long, default_value_t = 1.0)]
        random_opening_decay: f32,
        /// Exit once this many training examples have been written (0 = run until
        /// killed). The generate→train→eval cycle uses this so the phases never
        /// share the GPU.
        #[arg(long, default_value_t = 0)]
        target_examples: u64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Inference backend: `candle` (latest.safetensors) or `ort` (model.onnx).
        #[arg(long, default_value = "candle")]
        backend: String,
        /// ORT execution provider: `trt` or `cuda`.
        #[arg(long, default_value = "trt")]
        ep: String,
        /// fp16 for the TensorRT engine.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Self-play throughput benchmark (runs for `--secs` wall-clock).
    SelfplayBench {
        #[arg(long)]
        model: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 512)]
        concurrency: usize,
        #[arg(long, default_value_t = 20.0)]
        secs: f64,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 256)]
        max_batch: usize,
        #[arg(long, default_value_t = 400)]
        node_budget: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Inference backend: `candle` (a .safetensors) or `ort` (a .onnx).
        #[arg(long, default_value = "candle")]
        backend: String,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Head-to-head between two checkpoints (relative strength / ELO).
    EvalMatch {
        /// Candidate: a .safetensors for `--backend candle`, a .onnx for `ort`.
        #[arg(long)]
        a: String,
        /// Incumbent, same format as `--a`.
        #[arg(long)]
        b: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 32)]
        games: usize,
        #[arg(long, default_value_t = 400)]
        node_budget: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 4)]
        temp_moves: u32,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long, default_value_t = 64)]
        max_batch: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Inference backend for both sides: `candle` (.safetensors) or `ort` (.onnx).
        #[arg(long, default_value = "candle")]
        backend: String,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Absolute strength: the net (with search) against MoHex.
    VsMohex {
        #[arg(long)]
        model: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value = "mohex")]
        mohex_bin: String,
        #[arg(long, default_value_t = 10)]
        games: u32,
        /// Net simulations per move.
        #[arg(long, default_value_t = 400)]
        sims: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        #[arg(long, default_value_t = 1000)]
        mohex_playouts: u32,
        #[arg(long, default_value_t = 0.0)]
        mohex_time: f64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    SelfPlay {
        #[arg(long, default_value = "127.0.0.1:7104")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7204")]
        model_uri: String,
        #[arg(long, default_value_t = 16)]
        workers: u32,
    },
    /// Cross-check our rules against MoHex over random games.
    Verify {
        #[arg(long, default_value = "mohex")]
        mohex_bin: String,
        #[arg(long, default_value_t = 20)]
        games: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Play MoHex against a random-move stand-in (until the AZ net lands, this
    /// is the end-to-end check that the bridge drives a whole game).
    Match {
        #[arg(long, default_value = "mohex")]
        mohex_bin: String,
        #[arg(long, default_value_t = 2)]
        games: u32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// MoHex playouts per move (0 = its default).
        #[arg(long, default_value_t = 1000)]
        mohex_playouts: u32,
        /// MoHex seconds per move (0 = unlimited, budget by playouts).
        #[arg(long, default_value_t = 0.0)]
        mohex_time: f64,
    },
    /// Print the net's raw value and top policy cells for a position (given as the
    /// moves played from the empty board). Debug aid, and the Rust↔Python parity check.
    NetEval {
        #[arg(long)]
        model: String,
        /// Moves played from the empty board, e.g. `g7,d4,h7`.
        #[arg(long, default_value = "")]
        moves: String,
        #[arg(long, default_value_t = 5)]
        top: usize,
        /// Evaluate the transpose+colour-swapped position instead, mapping the
        /// result back to real cells. Hex's symmetry makes this the *same*
        /// position with the roles exchanged, so the output must match the
        /// unmirrored one exactly — the end-to-end check that the canonical
        /// frame (planes and policy indices) is wired up right for a live net.
        #[arg(long)]
        mirror: bool,
        /// Inference backend: `candle` (a .safetensors) or `ort` (a .onnx). Run it
        /// on both to check a backend (notably TensorRT fp16) still agrees.
        #[arg(long, default_value = "candle")]
        backend: String,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Play one self-play game with the net and print the moves — what the net
    /// actually does, as opposed to what it scores.
    PlayGame {
        #[arg(long)]
        model: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 200)]
        sims: u32,
        #[arg(long, default_value_t = 1.5)]
        c_puct: f32,
        /// Sample from the visit counts for this many opening plies, then argmax.
        #[arg(long, default_value_t = 0)]
        temp_moves: u32,
        /// Root Dirichlet noise, as self-play uses it (0 = off, the eval default).
        #[arg(long, default_value_t = 0.0)]
        dir_eps: f32,
        #[arg(long, default_value_t = 0.3)]
        dir_alpha: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Print the final board.
        #[arg(long)]
        board: bool,
        /// Also print the root visit distribution for each move — the policy target
        /// self-play would record, so it can be compared against the actor's.
        #[arg(long)]
        verbose: bool,
    },
    /// Raw forward throughput of the net (positions/s), with no search around it —
    /// the ceiling `selfplay-bench` can approach, and how you tell a GPU-bound
    /// actor from a CPU-bound one.
    NetBench {
        #[arg(long)]
        model: String,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 50)]
        iters: usize,
        /// Inference backend: `candle` (a .safetensors) or `ort` (a .onnx).
        #[arg(long, default_value = "candle")]
        backend: String,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        fp16: bool,
    },
    /// Random-playout throughput of the rules (moves/s), the AZ self-play floor.
    Bench {
        #[arg(long, default_value_t = 2000)]
        games: u32,
    },
    Stdio,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve {
            addr,
            model_uri,
            models_local,
            cuda,
            mohex_bin,
            mohex_playouts,
            mohex_time,
        } => {
            if let Some(bin) = &mohex_bin {
                mohex::install(bin, mohex_playouts, mohex_time)?;
                tracing::info!(bin, mohex_playouts, mohex_time, "mohex opponent ready");
            }
            serve(addr, &model_uri, &models_local, cuda).await?;
        }
        Cmd::SelfPlay {
            addr, model_uri, ..
        } => serve(addr, &model_uri, &[], false).await?,
        Cmd::SelfplayLoop {
            live,
            cuda,
            concurrency,
            max_batch,
            chunk_secs,
            max_chunks,
            max_examples,
            threads,
            node_budget,
            c_puct,
            dir_alpha,
            dir_eps,
            temp_moves,
            forced_playout_k,
            full_search_prob,
            fast_node_budget,
            random_opening_max,
            random_opening_decay,
            target_examples,
            seed,
            backend,
            ep,
            fp16,
        } => {
            let mut params = search_params(node_budget, c_puct, dir_alpha, dir_eps, temp_moves);
            params.forced_playout_k = forced_playout_k;
            params.full_search_prob = full_search_prob;
            params.fast_node_budget = fast_node_budget;
            params.random_opening_max = random_opening_max;
            params.random_opening_decay = random_opening_decay;
            selfplay::run_loop(
                &live,
                cuda,
                params,
                concurrency,
                chunk_secs,
                max_chunks,
                max_examples,
                threads.unwrap_or_else(default_threads),
                max_batch,
                seed,
                parse_backend(&backend, &ep, fp16)?,
                target_examples,
            )?;
        }
        Cmd::SelfplayBench {
            model,
            cuda,
            concurrency,
            secs,
            threads,
            max_batch,
            node_budget,
            c_puct,
            seed,
            backend,
            ep,
            fp16,
        } => {
            let params = search_params(node_budget, c_puct, 0.3, 0.25, 12);
            let backend = parse_backend(&backend, &ep, fp16)?;
            selfplay::run_benchmark(
                &model,
                cuda,
                concurrency,
                threads.unwrap_or_else(default_threads),
                params,
                max_batch,
                seed,
                secs,
                backend,
            )?;
            // ONNX Runtime's TensorRT EP intermittently aborts inside its own
            // teardown (heap corruption, ~1 run in 3, always *after* the results
            // are printed). Everything we own is already written, so leave
            // without unwinding rather than let a shutdown bug fail the run.
            if backend.is_ort() {
                use std::io::Write;
                std::io::stdout().flush().ok();
                std::process::exit(0);
            }
        }
        Cmd::EvalMatch {
            a,
            b,
            cuda,
            games,
            node_budget,
            c_puct,
            temp_moves,
            threads,
            max_batch,
            seed,
            backend,
            ep,
            fp16,
        } => {
            eval::run_match(
                &a,
                &b,
                cuda,
                games,
                node_budget,
                c_puct,
                temp_moves,
                threads.unwrap_or_else(default_threads),
                max_batch,
                seed,
                parse_backend(&backend, &ep, fp16)?,
            )?;
        }
        Cmd::VsMohex {
            model,
            cuda,
            mohex_bin,
            games,
            sims,
            c_puct,
            mohex_playouts,
            mohex_time,
            seed,
        } => {
            eval::run_vs_mohex(
                &model,
                cuda,
                &mohex_bin,
                games,
                sims,
                c_puct,
                mohex_playouts,
                mohex_time,
                seed,
            )?;
        }
        Cmd::Verify {
            mohex_bin,
            games,
            seed,
        } => {
            mohex::verify_rules(&mohex_bin, games, seed)?;
        }
        Cmd::Match {
            mohex_bin,
            games,
            seed,
            mohex_playouts,
            mohex_time,
        } => {
            let mut rng = seed | 1;
            mohex::play_match(&mohex_bin, games, mohex_playouts, mohex_time, move |s| {
                let moves = Hex::legal_moves(s);
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                moves[(rng >> 33) as usize % moves.len()]
            })?;
        }
        Cmd::NetEval {
            model,
            moves,
            top,
            mirror,
            backend,
            ep,
            fp16,
        } => net_eval(
            &model,
            &moves,
            top,
            mirror,
            parse_backend(&backend, &ep, fp16)?,
        )?,
        Cmd::PlayGame {
            model,
            cuda,
            sims,
            c_puct,
            temp_moves,
            dir_eps,
            dir_alpha,
            seed,
            board,
            verbose,
        } => play_game(
            &model, cuda, sims, c_puct, temp_moves, dir_eps, dir_alpha, seed, board, verbose,
        )?,
        Cmd::NetBench {
            model,
            cuda,
            batch,
            iters,
            backend,
            ep,
            fp16,
        } => net_bench(
            &model,
            cuda,
            batch,
            iters,
            parse_backend(&backend, &ep, fp16)?,
        )?,
        Cmd::Bench { games } => bench(games),
        Cmd::Stdio => anyhow::bail!("stdio: implement when needed"),
    }
    Ok(())
}

async fn serve(
    addr: SocketAddr,
    model_uri: &str,
    models_local: &[String],
    cuda: bool,
) -> anyhow::Result<()> {
    let mut models: HashMap<String, HexModel> = HashMap::new();
    if models_local.is_empty() {
        let remote = RemoteModelClient::<Hex>::connect(model_uri).await?;
        models.insert("nn".to_string(), HexModel::Remote(remote));
    } else {
        let device = net::pick_device(cuda)?;
        for spec in models_local {
            let (name, ckpt) = spec.split_once('=').ok_or_else(|| {
                anyhow::anyhow!("--model-local wants NAME=CHECKPOINT, got {spec:?}")
            })?;
            let m = net::LocalModel::load_on(ckpt, device.clone())?;
            tracing::info!(name, ckpt, "loaded embedded AZ net");
            models.insert(name.to_string(), HexModel::Local(m));
        }
    }
    playground_engine_server::run::<Hex, _>(addr, models).await
}

/// Search knobs for self-play. Hex always terminates within 169 plies, so the ply
/// cap is a backstop and neither score-stability nor resignation applies (a
/// resignation threshold would need calibrating against a trained net first).
fn search_params(
    node_budget: u32,
    c_puct: f32,
    dir_alpha: f32,
    dir_eps: f32,
    temp_moves: u32,
) -> SearchParams {
    SearchParams {
        sims: 0,
        node_budget,
        c_puct,
        dir_alpha,
        dir_eps,
        temp_moves,
        explore_eps: 0.0,
        max_ply: 200,
        settle_plies: 0,
        resign_threshold: 0.0,
        resign_consecutive: 0,
        resign_playout_frac: 0.0,
        forced_playout_k: 0.0,
        full_search_prob: 0.0,
        fast_node_budget: 0,
        random_opening_max: 0,
        random_opening_decay: 1.0,
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

/// Value + top policy cells for a position, as JSON. `moves` is a comma-separated
/// list of cells played from the empty board.
fn net_eval(
    model: &str,
    moves: &str,
    top: usize,
    mirror: bool,
    backend: selfplay::Backend,
) -> anyhow::Result<()> {
    use playground_core::batch_search::BatchModel;
    let mut state = <Hex as Game>::initial_state();
    for token in moves.split(',').filter(|t| !t.trim().is_empty()) {
        let (r, c) = hex_engine::game::parse_cell(token)
            .ok_or_else(|| anyhow::anyhow!("bad cell {token:?}"))?;
        state = <Hex as Game>::apply(
            &state,
            &HexMove::Play {
                r: r as u8,
                c: c as u8,
            },
        );
    }
    // In the mirrored frame every cell maps through the long diagonal, so map the
    // reported moves back to make the two outputs directly comparable.
    let unmap = |mv: HexMove| match (mirror, mv) {
        (true, HexMove::Play { r, c }) => HexMove::Play { r: c, c: r },
        (_, mv) => mv,
    };
    if mirror {
        state = hex_engine::game::transpose_swap(&state);
    }
    let m: Box<dyn BatchModel> = match backend {
        selfplay::Backend::Candle => Box::new(net::LocalModel::load(model)?),
        selfplay::Backend::Ort { ep, fp16 } => Box::new(hex_engine::ort_net::OrtModel::new(
            model,
            ep,
            fp16,
            &format!(
                "{}/trt_cache",
                std::path::Path::new(model)
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .display()
            ),
        )?),
    };
    let mut planes = vec![0f32; hex_engine::mcts::PLANES];
    hex_engine::game::fill_planes_into(&state, &mut planes);
    let (vals, logits) = m.forward(&planes, 1)?;
    // Report in board terms: the priors the search would actually use.
    let (mut mvs, mut priors) = (Vec::new(), Vec::new());
    <Hex as playground_core::batch_search::MoveGame>::expand(
        &state,
        &logits,
        &mut mvs,
        &mut priors,
    );
    let mut order: Vec<usize> = (0..mvs.len()).collect();
    order.sort_by(|&a, &b| priors[b].total_cmp(&priors[a]));
    let best: Vec<serde_json::Value> = order
        .iter()
        .take(top)
        .map(|&i| match unmap(mvs[i]) {
            HexMove::Play { r, c } => serde_json::json!({
                "cell": hex_engine::game::cell_name(r as usize, c as usize),
                "prior": priors[i],
            }),
            HexMove::Swap => serde_json::json!({ "cell": "swap", "prior": priors[i] }),
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "to_move": if (state.to_move == 1) != mirror { "black" } else { "white" },
            "mirrored": mirror,
            "value": vals[0],
            "logit_sum": logits.iter().sum::<f32>(),
            "top": best,
        })
    );
    Ok(())
}

/// `--backend`/`--ep`/`--fp16` → the self-play backend. `ep` is leaked because
/// ORT's provider names live for the process anyway and `Backend` is `Copy`.
fn parse_backend(backend: &str, ep: &str, fp16: bool) -> anyhow::Result<selfplay::Backend> {
    match backend {
        "candle" => Ok(selfplay::Backend::Candle),
        "ort" => match ep {
            "trt" => Ok(selfplay::Backend::Ort { ep: "trt", fp16 }),
            "cuda" => Ok(selfplay::Backend::Ort { ep: "cuda", fp16 }),
            other => anyhow::bail!("unknown --ep {other:?} (use cuda|trt)"),
        },
        other => anyhow::bail!("unknown --backend {other:?} (use candle|ort)"),
    }
}

/// Play one game, net against itself, printing every move.
fn play_game(
    model: &str,
    cuda: bool,
    sims: u32,
    c_puct: f32,
    temp_moves: u32,
    dir_eps: f32,
    dir_alpha: f32,
    seed: u64,
    show_board: bool,
    verbose: bool,
) -> anyhow::Result<()> {
    use hex_engine::mcts::{self, Params};
    use rand::SeedableRng;

    let m = net::LocalModel::load_on(model, net::pick_device(cuda)?)?;
    let params = Params {
        sims,
        c_puct,
        dir_alpha,
        dir_eps,
        temp_moves,
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut state = <Hex as Game>::initial_state();
    let mut played: Vec<String> = Vec::new();
    while state.winner == 0 {
        let (moves, visits, value) = mcts::run_search(&m, &state, &params, &mut rng)?;
        let mv = mcts::pick_move(&moves, &visits, state.ply as u32, temp_moves, &mut rng);
        let name = match mv {
            HexMove::Play { r, c } => hex_engine::game::cell_name(r as usize, c as usize),
            HexMove::Swap => "swap".to_string(),
        };
        println!(
            "{:>3}. {} {name:<4} v {:+.3}",
            state.ply + 1,
            if state.to_move == 1 { "B" } else { "W" },
            value,
        );
        if verbose {
            let total: u32 = visits.iter().sum();
            let mut order: Vec<usize> = (0..visits.len()).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(visits[i]));
            let top: Vec<String> = order
                .iter()
                .take(5)
                .filter(|&&i| visits[i] > 0)
                .map(|&i| match moves[i] {
                    HexMove::Play { r, c } => format!(
                        "{}:{}",
                        hex_engine::game::cell_name(r as usize, c as usize),
                        visits[i]
                    ),
                    HexMove::Swap => format!("swap:{}", visits[i]),
                })
                .collect();
            println!(
                "      root visits {total} over {} moves: {}",
                visits.iter().filter(|&&v| v > 0).count(),
                top.join(" ")
            );
        }
        played.push(name);
        state = <Hex as Game>::apply(&state, &mv);
    }
    if show_board {
        print!("{}", hex_engine::game::render(&state));
    }
    println!(
        "result: {} wins in {} plies   moves: {}",
        if state.winner == 1 { "black" } else { "white" },
        state.ply,
        played.join(","),
    );
    Ok(())
}

/// Time `iters` forwards of `batch` positions (after a warm-up).
fn net_bench(
    model: &str,
    cuda: bool,
    batch: usize,
    iters: usize,
    backend: selfplay::Backend,
) -> anyhow::Result<()> {
    use playground_core::batch_search::BatchModel;
    let m: Box<dyn BatchModel> = match backend {
        selfplay::Backend::Candle => {
            Box::new(net::LocalModel::load_on(model, net::pick_device(cuda)?)?)
        }
        selfplay::Backend::Ort { ep, fp16 } => Box::new(hex_engine::ort_net::OrtModel::new(
            model,
            ep,
            fp16,
            &format!(
                "{}/trt_cache",
                std::path::Path::new(model)
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .display()
            ),
        )?),
    };
    let planes = vec![0f32; batch * hex_engine::mcts::PLANES];
    m.forward(&planes, batch)?; // warm-up (kernel load / TRT engine build)
    let t0 = Instant::now();
    for _ in 0..iters {
        m.forward(&planes, batch)?;
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "net-bench [{}]: batch {batch} × {iters} in {secs:.2}s — {:.1} forwards/s, {:.0} positions/s",
        match backend {
            selfplay::Backend::Candle if cuda => "candle/cuda".to_string(),
            selfplay::Backend::Candle => "candle/cpu".to_string(),
            b => b.label(),
        },
        iters as f64 / secs,
        (batch * iters) as f64 / secs,
    );
    Ok(())
}

/// Random playouts to completion, measuring rules throughput (legal-move
/// generation + apply + the connectivity check that decides the winner).
fn bench(games: u32) {
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut moves_played = 0u64;
    let mut black_wins = 0u32;
    let start = Instant::now();
    for _ in 0..games {
        let mut state = Hex::initial_state();
        while state.winner == 0 {
            let moves = Hex::legal_moves(&state);
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let mv = moves[(rng >> 33) as usize % moves.len()];
            state = Hex::apply(&state, &mv);
            moves_played += 1;
        }
        black_wins += u32::from(state.winner == 1);
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{games} random games in {secs:.2}s — {:.0} moves/s, {:.1} plies/game, black won {:.1}%",
        moves_played as f64 / secs,
        moves_played as f64 / f64::from(games),
        100.0 * f64::from(black_wins) / f64::from(games),
    );
}
