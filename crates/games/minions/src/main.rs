//! Simplified-Minions engine binary. Subcommand semantics follow
//! arimaa-engine: `serve` is the interactive gRPC-Web endpoint the web UI
//! talks to, and works without a model server running (the model channel is
//! lazy) — only the NN seat needs one.

mod soak;

use std::net::SocketAddr;

use clap::{Parser, Subcommand};
use minions_engine::game::Minions;
use playground_model_client::RemoteModelClient;

#[derive(Parser)]
#[command(name = "minions-engine", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Serve {
        #[arg(long, default_value = "127.0.0.1:7105")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7205")]
        model_uri: String,
        /// Weights for the interactive net seat. Without this the UI's "net"
        /// player reports that none is configured rather than quietly playing
        /// the greedy heuristic instead.
        #[arg(long)]
        net_dir: Option<String>,
        #[arg(long, default_value_t = String::from("ort"))]
        backend: String,
        #[arg(long, default_value_t = String::from("trt"))]
        ep: String,
        #[arg(long, default_value_t = true)]
        fp16: bool,
        #[arg(long, default_value_t = false)]
        cuda: bool,
        /// Self-play's recorded search (`search.json` in the live dir, written
        /// by `selfplay-loop`). The net seat and the analyser run exactly this
        /// search, so the UI shows what self-play does. Required with
        /// `--net-dir`.
        #[arg(long)]
        search_config: Option<String>,
    },
    SelfPlay {
        #[arg(long, default_value = "127.0.0.1:7105")]
        addr: SocketAddr,
        #[arg(long, default_value = "http://127.0.0.1:7205")]
        model_uri: String,
        #[arg(long, default_value_t = 16)]
        workers: u32,
    },
    /// Continuous self-play actor: play games with the sampled search, append
    /// chunks to the learner's replay buffer, hot-reload weights between flushes.
    SelfplayLoop {
        #[arg(long, default_value = "data/models/minions/live")]
        live: String,
        /// `candle` (development, parity) or `ort` (production — on Hex the ORT
        /// TensorRT path ran ~45x faster end to end).
        #[arg(long, default_value = "candle")]
        backend: String,
        /// ORT execution provider: `trt` (tensor cores) or `cuda`.
        #[arg(long, default_value = "trt")]
        ep: String,
        /// fp16 for the TensorRT EP — this is what puts tensor cores to work.
        #[arg(long)]
        fp16: bool,
        /// Run the candle backend on the GPU. Needs `--features cuda`.
        #[arg(long)]
        cuda: bool,
        /// Games in flight. The pool is deliberately far larger than
        /// `--max-batch` so there is always a full batch of games wanting an
        /// expansion, and a finished game is replaced at once rather than
        /// letting the batch narrow. This is the batch multiplier —
        /// every forward is `concurrency` (or `concurrency x candidates`) wide
        /// instead of 1, and at batch 1 the GPU is ~99% idle on launch latency.
        ///
        /// Diminishing past the point where the heads saturate (measured: the
        /// diffusion head tops out near batch 512, so `concurrency x candidates`
        /// beyond that buys little). Note the whole batch advances together, so
        /// `--chunk-secs` must exceed the time for one batch of games to finish
        /// or nothing is ever flushed.
        #[arg(long, default_value_t = 128)]
        concurrency: usize,
        /// Positions per batched forward — the batch multiplier. At batch 1 the
        /// GPU is ~99% idle on launch latency (measured: 817 vs 115,000
        /// positions/s for the trunk). Batches are padded to this exact width so
        /// TensorRT builds one engine per graph rather than one per shape.
        ///
        /// A single GPU consumer on purpose: concurrent CUDA launches from many
        /// threads serialize badly (the Connect 4 engine measured that collapse
        /// at ~1 game/s), so throughput comes from a wider batch, not threads.
        #[arg(long, default_value_t = 32)]
        max_batch: usize,
        #[arg(long, default_value_t = 200)]
        sims: u32,
        #[arg(long, default_value_t = 50)]
        fast_sims: u32,
        #[arg(long, default_value_t = 0.25)]
        full_search_prob: f64,
        /// Candidate turns proposed per node expansion.
        #[arg(long, default_value_t = 12)]
        candidates: usize,
        /// Candidate turns for *fast* (playout-cap) searches. These are never
        /// recorded, and at `--fast-sims` visits a large candidate set is spread
        /// too thin to form a tree, so this is cheaper *and* better shaped.
        #[arg(long, default_value_t = 6)]
        candidates_fast: usize,
        /// Sampler draws per node: the node is asked for exactly this many
        /// proposals, and a shortfall of distinct ones against `--candidates`
        /// is filled with random playouts. Default: `--candidates`.
        #[arg(long)]
        draws: Option<usize>,
        /// Draws per node for fast searches. Default: `--candidates-fast`.
        #[arg(long)]
        draws_fast: Option<usize>,
        /// Diffusion rounds. 1 = mean-field; set to a large number for the
        /// exact-density diagnostic mode.
        #[arg(long, default_value_t = 4)]
        t_steps: usize,
        /// Diffusion rounds for *fast* (playout-cap) searches, which advance the
        /// game but yield no training data. Diffusion is the largest single term
        /// in expansion cost and scales linearly in this.
        #[arg(long, default_value_t = 2)]
        t_steps_fast: usize,
        /// Share of candidates drawn by forced exploration — a random legal turn
        /// played out, independent of the net.
        #[arg(long, default_value_t = 0.25)]
        random_fraction: f32,
        /// Random legal turns played before the search takes over, per game.
        /// None become training examples; the point is to keep the training
        /// distribution moving when self-play comes from one frozen net.
        #[arg(long, default_value_t = 15)]
        random_opening_max: u32,
        /// Shape of the opening length: `P(n) ∝ decay^-n` on `0..=max`.
        /// 15 / 1.1335 has mean 5, a quarter of the 20-turn cap. 1.0 is uniform.
        #[arg(long, default_value_t = 1.1335)]
        random_opening_decay: f32,
        /// Pool workers. Each owns a disjoint slice of the pool, so one worker's
        /// CPU phases (commit, realize, playout) overlap another's GPU phases.
        /// 1 restores the single-threaded actor.
        #[arg(long, default_value_t = 4)]
        workers: usize,
        /// Visits guaranteed to each forced-exploration candidate at the root,
        /// regardless of its prior. Subtracted again before the policy target is
        /// formed, so they cost search budget but never enter training. 0 falls
        /// back to KataGo's prior-scaled floor alone, which cannot buy a first
        /// visit for a candidate the energy head scores near zero.
        #[arg(long, default_value_t = 2.0)]
        min_explore_visits: f32,
        /// Flat visits guaranteed to each forced-exploration candidate. These
        /// are excluded from PUCT and from the policy target; they buy a value
        /// estimate only.
        #[arg(long, default_value_t = 30)]
        forced_visits: u32,
        /// Non-explore candidates drawn at random to also get a quota, on top
        /// of every explore candidate.
        #[arg(long, default_value_t = 4)]
        forced_extra: u32,
        /// KataGo forced-playout constant, applied at the root: a low-prior
        /// candidate cannot be starved before it is tried. 0 disables root
        /// forced playouts, leaving the root sweep's one seeded visit per
        /// candidate as the only guarantee of a first look.
        #[arg(long, default_value_t = 2.0)]
        k_forced: f32,
        /// KR-PUCT kernel bandwidth over the turn metric (`turn::distance`).
        /// Candidates within roughly this distance share visit counts and
        /// values instead of being scored as unrelated actions.
        ///
        /// **0 (the default) is the flat PUCT search, bit-for-bit** — the
        /// kernel is an alternative mode, not a replacement, and both run
        /// against the same net.
        #[arg(long, default_value_t = 0.0)]
        kernel_sigma: f32,
        /// Which visits back up into a candidate's value: `all` (default),
        /// `true-only`, or `node-gate`.
        ///
        /// Forced visits and refinement hand-overs are bought, not chosen.
        /// `all` folds them into the value as the search has always done;
        /// `true-only` counts none of them; `node-gate` ignores a candidate
        /// whose visits are *all* bought but keeps the whole count once one is
        /// earned. `min-two` ignores provenance and gates on thinness instead:
        /// a candidate with a single visit says nothing to its parent, and from
        /// the second on it contributes its whole count.
        /// **`all` reproduces the previous search bit-for-bit.**
        #[arg(long, default_value = "all")]
        value_visits: playground_core::sampled_mcts::ValueVisits,
        /// Candidates generated per model call. 0 (default) produces a node's
        /// whole set in one call.
        ///
        /// With a chunk, a wave becomes `max_batch` requests of this size
        /// instead of `max_batch` requests of `--candidates` — identical tensor
        /// shape and head rows, spread over more *nodes*. 69% of nodes are
        /// visited once and never look at a full candidate set; a node that
        /// earns visits asks for its next chunk, and that request batches
        /// alongside expansions because it is the same shape.
        #[arg(long, default_value_t = 1)]
        pw_chunk: usize,
        /// Sampler draws in a row that may add nothing at an interior node
        /// before its widen fills with a random playout instead. 0 = keep
        /// drawing until the node's draw budget is spent.
        #[arg(long, default_value_t = 3)]
        draw_retries: u32,
        /// Refine every this many visits to a node (full and fast searches):
        /// redraw part of the picked turn instead of descending. 0 = off.
        #[arg(long, default_value_t = 0.0)]
        resample_every: f32,
        /// Deepest node that refines, root = 0. Default: every depth.
        #[arg(long)]
        resample_depth: Option<u32>,
        /// Diffusion rounds a refinement redraws with.
        #[arg(long, default_value_t = 2)]
        resample_t: usize,
        /// Most of a turn a refinement masks again (share uniform in [0, this]).
        #[arg(long, default_value_t = 0.5)]
        resample_frac: f32,
        /// Write `search.json` for these settings into `--live` and exit
        /// without playing. Every run writes it on start anyway; this records
        /// the search without starting self-play, for eval and the UI.
        #[arg(long)]
        write_search_config: bool,
        /// Score candidates with the energy head (off = uniform priors).
        #[arg(long)]
        no_energy: bool,
        #[arg(long, default_value_t = 60)]
        chunk_secs: u64,
        #[arg(long, default_value_t = 200_000)]
        max_examples: usize,
        #[arg(long, default_value_t = 200)]
        max_chunks: usize,
        /// Exit once this many examples have been produced (0 = run forever).
        #[arg(long, default_value_t = 0)]
        target_examples: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Play a candidate net against a champion (or the greedy heuristic) and
    /// print `A_score`, which the eval loop parses to decide promotion.
    EvalMatch {
        #[arg(long)]
        a: String,
        /// Champion checkpoint; omit to play the greedy heuristic instead.
        #[arg(long)]
        b: Option<String>,
        #[arg(long, default_value_t = 20)]
        games: u32,
        /// Eval's search settings are its own (see `eval::match_search`); the
        /// search that runs them is self-play's.
        #[arg(long, default_value_t = 100)]
        sims: u32,
        #[arg(long, default_value_t = 16)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        t_steps: usize,
        /// Diffusion rounds for seat B; defaults to A's. Set it to play one net
        /// against itself at two sampler depths.
        #[arg(long)]
        t_steps_b: Option<usize>,
        /// Candidates (and draws) per node for seat B; defaults to A's.
        #[arg(long)]
        candidates_b: Option<usize>,
        /// Simulations per turn for seat B; defaults to A's (`--sims`).
        #[arg(long)]
        sims_b: Option<u32>,
        /// Seat A refines every this many visits to a node: the visit redraws
        /// part of the picked turn and visits the result. 0 = off.
        #[arg(long, default_value_t = 0.0)]
        resample_every: f32,
        /// As `--resample-every`, for seat B; defaults to A's.
        #[arg(long)]
        resample_every_b: Option<f32>,
        /// Progressive-widening `beta` for seat A; 0 turns widening off (every
        /// candidate a node earns is selectable at once). Default: eval's.
        #[arg(long)]
        pw_beta: Option<f32>,
        /// As `--pw-beta`, for seat B; defaults to A's.
        #[arg(long)]
        pw_beta_b: Option<f32>,
        /// Diffusion rounds a resample redraws with (both seats).
        #[arg(long, default_value_t = 2)]
        resample_t: usize,
        /// Most of a turn a resample masks again: the share is uniform in
        /// `[0, this]` (both seats).
        #[arg(long, default_value_t = 0.5)]
        resample_frac: f32,
        /// Deepest node that refines, root = 0 (both seats). Default: all.
        #[arg(long)]
        resample_depth: Option<u32>,
        /// Seat B's `--resample-t`, `--resample-frac` and `--resample-depth`;
        /// each defaults to A's.
        #[arg(long)]
        resample_t_b: Option<usize>,
        #[arg(long)]
        resample_frac_b: Option<f32>,
        #[arg(long)]
        resample_depth_b: Option<u32>,
        /// Both seats: no forced visits, no explore quotas and no random
        /// fills -- every candidate is a sampler draw, chosen by plain PUCT.
        #[arg(long)]
        no_explore: bool,
        #[arg(long, default_value = "candle")]
        backend: String,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long)]
        fp16: bool,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// KR-PUCT kernel bandwidth for seat A. 0 = today's flat PUCT search.
        #[arg(long, default_value_t = 0.0)]
        kernel_sigma: f32,
        /// Kernel bandwidth for seat B; defaults to A's, so both seats search
        /// the same way unless you ask for the flat-vs-KR comparison.
        #[arg(long)]
        kernel_sigma_b: Option<f32>,
        /// Value-visit mode for seat A: `all` (default), `true-only`,
        /// `node-gate` or `min-two`. See the selfplay flag of the same name.
        #[arg(long, default_value = "all")]
        value_visits: playground_core::sampled_mcts::ValueVisits,
        /// Value-visit mode for seat B; defaults to A's. Set only this to play
        /// one mode against another on a single checkpoint, which is the whole
        /// point of the knob.
        #[arg(long)]
        value_visits_b: Option<playground_core::sampled_mcts::ValueVisits>,
        /// Write a turn-by-turn transcript of the first N games here. These are
        /// the very games the promotion gate scored, so the sample and the
        /// number always agree.
        #[arg(long)]
        render_file: Option<String>,
        #[arg(long, default_value_t = 5)]
        render_games: u32,
    },
    /// Time the three graphs on a backend, at the batch sizes the sampler
    /// actually uses. Far quicker than a self-play chunk, and it isolates
    /// inference cost from tree work.
    Bench {
        #[arg(long, default_value = "data/models/minions/live")]
        live: String,
        #[arg(long, default_value = "candle")]
        backend: String,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long)]
        fp16: bool,
        #[arg(long)]
        cuda: bool,
        /// Candidates per expansion — the batch the heads run at.
        #[arg(long, default_value_t = 32)]
        candidates: usize,
        /// Diffusion rounds per expansion.
        #[arg(long, default_value_t = 6)]
        t_steps: usize,
        #[arg(long, default_value_t = 3)]
        secs: u64,
        /// Sweep batch sizes instead of timing one configuration, to show where
        /// the GPU stops being latency-bound.
        #[arg(long)]
        sweep: bool,
    },
    /// Write a bundle the Python validator checks itself against: a real replay
    /// chunk plus this side's `Partial::write_planes` output for known
    /// declarations. Anywhere Rust and Python implement the same encoding twice
    /// is a place they can silently diverge.
    DumpValidation {
        #[arg(long, default_value = "data/minions_validation")]
        out: String,
        #[arg(long, default_value_t = 24)]
        cases: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// Hammer the realizer with random declarations, checking that every action
    /// it emits is legal and reporting fidelity, throughput and how much the
    /// greedy attack assignment gives up against an exact one.
    Soak {
        #[arg(long, default_value_t = 8)]
        threads: usize,
        #[arg(long, default_value_t = 600)]
        secs: u64,
        /// Skip the per-action `legal_moves` check (much faster; measures raw
        /// realizer throughput rather than correctness).
        #[arg(long)]
        no_verify: bool,
        /// Skip the exact-assignment oracle, which is exponential.
        #[arg(long)]
        no_oracle: bool,
        #[arg(long, default_value_t = 20260825)]
        seed: u64,
        #[arg(long, default_value = "data/minions_soak.json")]
        out: String,
    },
    /// Play the greedy solver against itself and print the final board.
    /// Play one full game with a trained net against itself, rendering every
    /// turn — the only way to actually see what the policy has learned, as
    /// opposed to inferring it from aggregate statistics.
    NetGame {
        #[arg(long)]
        live: String,
        #[arg(long, default_value_t = 100)]
        sims: u32,
        #[arg(long, default_value_t = 32)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        t_steps: usize,
        #[arg(long, default_value_t = String::from("candle"))]
        backend: String,
        #[arg(long, default_value_t = String::from("trt"))]
        ep: String,
        #[arg(long)]
        fp16: bool,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Opponent for the second seat: `net` (the same checkpoint), `greedy`,
        /// or a path to another checkpoint — which is how you watch the current
        /// candidate play the champion, the pairing the promotion gate scores.
        #[arg(long, default_value_t = String::from("net"))]
        opponent: String,
        /// Play this many games instead of one. A single game is a sample of
        /// one and reads as a story; several show which habits are consistent.
        #[arg(long, default_value_t = 1)]
        games: u32,
    },
    /// Measure whether re-sending trunk features every diffusion step costs us.
    BenchBind {
        #[arg(long)]
        live: String,
        #[arg(long, default_value_t = String::from("trt"))]
        ep: String,
        #[arg(long, default_value_t = false)]
        fp16: bool,
        #[arg(long, default_value_t = 384)]
        rows: usize,
        #[arg(long, default_value_t = 4)]
        steps: usize,
        #[arg(long, default_value_t = 5)]
        reps: usize,
    },
    GreedyGame {
        #[arg(long, default_value_t = 1)]
        games: u32,
        /// Print the board after every completed turn.
        #[arg(long)]
        verbose: bool,
    },
    /// Generate random positions for the small value net's distillation set.
    ///
    /// Same opening law as self-play — a bid, then uniformly random *legal*
    /// turns — but the turn count is drawn from a normal truncated above rather
    /// than self-play's decaying geometric. Planes are written in the replay
    /// buffer's own encoding so the buffer can validate a net trained on this.
    GenPositions {
        #[arg(long, default_value = "data/minions_nnue/positions")]
        out: String,
        #[arg(long, default_value_t = 10_000_000)]
        count: usize,
        #[arg(long, default_value_t = 20.0)]
        mean: f64,
        #[arg(long, default_value_t = 5.0)]
        stdev: f64,
        /// Draws longer than this are rejected, not clamped.
        #[arg(long, default_value_t = 20)]
        max_turns: u32,
        #[arg(long, default_value_t = 250_000)]
        shard_size: usize,
        #[arg(long, default_value_t = 8)]
        threads: usize,
        #[arg(long, default_value_t = 20260918)]
        seed: u64,
    },
    /// Measure the small value net's evaluation rate in the language it would
    /// ship in. Both Python harnesses were overhead-bound — they reported rates
    /// flat across an 8x change in hidden width — so neither could answer
    /// whether the 100k/s design target is reachable.
    NnueBench {
        /// Active features per position. Generated positions measure 27.1 of
        /// 588 (range 14–41), so 27 is the realistic operating point.
        #[arg(long, default_value_t = 27)]
        active: usize,
        #[arg(long, default_value_t = 200_000)]
        reps: usize,
    },
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
            net_dir,
            backend,
            ep,
            fp16,
            cuda,
            search_config,
        } => {
            if let Some(dir) = net_dir {
                let path = search_config.ok_or_else(|| {
                    anyhow::anyhow!(
                        "--net-dir needs --search-config (the live dir's search.json), so the \
                         seat searches exactly as self-play does"
                    )
                })?;
                let (actor, sampler) =
                    minions_engine::search_config::load(std::path::Path::new(&path))?;
                let kind = minions_engine::net::BackendKind::parse(&backend, &ep, fp16, cuda)?;
                eprintln!(
                    "net seat: {dir} ({backend}/{ep}); search from {path}: {} sims, K={}, T={}, \
                     random {:.2}, forced {}",
                    actor.sims,
                    sampler.k_candidates,
                    sampler.t_steps,
                    sampler.random_fraction,
                    actor.forced_visits
                );
                minions_engine::net_seat::configure(minions_engine::net_seat::NetSeatConfig {
                    dir: std::path::PathBuf::from(&dir),
                    kind,
                    actor,
                    sampler,
                });
            } else {
                eprintln!("net seat: not configured (pass --net-dir to enable it)");
            }
            let model = RemoteModelClient::<Minions>::connect(&model_uri).await?;
            let models = std::collections::HashMap::from([("nn".to_string(), model)]);
            playground_engine_server::run::<Minions, _>(addr, models).await?;
        }
        Cmd::SelfPlay {
            addr, model_uri, ..
        } => {
            let model = RemoteModelClient::<Minions>::connect(&model_uri).await?;
            let models = std::collections::HashMap::from([("nn".to_string(), model)]);
            playground_engine_server::run::<Minions, _>(addr, models).await?;
        }
        Cmd::SelfplayLoop {
            live,
            backend,
            ep,
            fp16,
            cuda,
            concurrency,
            max_batch,
            sims,
            fast_sims,
            full_search_prob,
            candidates,
            candidates_fast,
            draws,
            draws_fast,
            t_steps,
            t_steps_fast,
            random_fraction,
            random_opening_max,
            random_opening_decay,
            min_explore_visits,
            forced_visits,
            forced_extra,
            k_forced,
            workers,
            no_energy,
            kernel_sigma,
            value_visits,
            pw_chunk,
            draw_retries,
            resample_every,
            resample_depth,
            resample_t,
            resample_frac,
            write_search_config,
            chunk_secs,
            max_examples,
            max_chunks,
            target_examples,
            seed,
        } => {
            let kind = minions_engine::net::BackendKind::parse(&backend, &ep, fp16, cuda)?;
            let actor = minions_engine::selfplay::ActorConfig {
                sims,
                fast_sims,
                full_search_prob,
                backend: kind,
                min_explore_visits,
                forced_visits,
                forced_extra,
                k_forced,
                workers,
                random_opening_max,
                random_opening_decay,
                kernel_sigma,
                value_visits,
                pw_chunk,
                draw_retries,
                resample_every,
                resample_max_depth: resample_depth.unwrap_or(u32::MAX),
                ..Default::default()
            };
            let sampler = minions_engine::sampler::SamplerConfig {
                t_steps,
                t_steps_fast,
                k_candidates: candidates,
                k_candidates_fast: candidates_fast,
                k_draws: draws.unwrap_or(candidates),
                k_draws_fast: draws_fast.unwrap_or(candidates_fast),
                random_fraction,
                use_energy: !no_energy,
                resample_t,
                resample_frac,
                seed,
                ..Default::default()
            };
            if write_search_config {
                std::fs::create_dir_all(&live)?;
                let path = std::path::Path::new(&live).join(minions_engine::search_config::FILE);
                minions_engine::search_config::save(&path, &actor, &sampler)?;
                println!("wrote {}", path.display());
                return Ok(());
            }
            minions_engine::selfplay::run_loop(
                &live,
                actor,
                sampler,
                concurrency,
                max_batch,
                chunk_secs,
                max_examples,
                max_chunks,
                target_examples,
                seed,
            )?
        }
        Cmd::EvalMatch {
            a,
            b,
            games,
            sims,
            candidates,
            t_steps,
            t_steps_b,
            candidates_b,
            sims_b,
            resample_every,
            resample_every_b,
            resample_t,
            resample_frac,
            resample_depth,
            resample_t_b,
            resample_frac_b,
            resample_depth_b,
            pw_beta,
            pw_beta_b,
            no_explore,
            backend,
            ep,
            fp16,
            cuda,
            seed,
            kernel_sigma,
            kernel_sigma_b,
            value_visits,
            value_visits_b,
            render_file,
            render_games,
        } => {
            use minions_engine::eval::{load_seat, match_search, run_match, Seat};
            // Every game advances together, so the live count falls as games
            // finish. Padding to the full match width keeps one tensor shape
            // for the whole match instead of letting TensorRT rebuild an
            // engine each time a game ends.
            let (mut sampler, params) =
                match_search(sims, candidates, t_steps, games as usize, seed);
            sampler.resample_t = resample_t;
            sampler.resample_frac = resample_frac;
            let kind = minions_engine::net::BackendKind::parse(&backend, &ep, fp16, cuda)?;
            let sa = load_seat(std::path::Path::new(&a), &kind, sampler)?;
            let tb = t_steps_b.unwrap_or(t_steps);
            let cb = candidates_b.unwrap_or(candidates);
            let sampler_b = minions_engine::sampler::SamplerConfig {
                t_steps: tb,
                t_steps_fast: tb,
                k_candidates: cb,
                k_candidates_fast: cb,
                k_draws: cb,
                k_draws_fast: cb,
                resample_t: resample_t_b.unwrap_or(resample_t),
                resample_frac: resample_frac_b.unwrap_or(resample_frac),
                ..sampler
            };
            let sb = match &b {
                Some(p) => load_seat(std::path::Path::new(p), &kind, sampler_b)?,
                None => Seat::Greedy,
            };
            use playground_core::sampled_mcts::SampledParams;
            let params = if no_explore {
                SampledParams {
                    k_forced: 0.0,
                    min_explore_visits: 0.0,
                    forced_visits: 0,
                    forced_extra: 0,
                    fills: false,
                    ..params
                }
            } else {
                params
            };
            let pa = SampledParams {
                kernel_sigma,
                value_visits,
                resample_every,
                resample_max_depth: resample_depth.unwrap_or(u32::MAX),
                pw_beta: pw_beta.unwrap_or(params.pw_beta),
                ..params
            };
            let pb = SampledParams {
                simulations: sims_b.unwrap_or(sims),
                kernel_sigma: kernel_sigma_b.unwrap_or(kernel_sigma),
                value_visits: value_visits_b.unwrap_or(value_visits),
                resample_every: resample_every_b.unwrap_or(resample_every),
                pw_beta: pw_beta_b.or(pw_beta).unwrap_or(params.pw_beta),
                resample_max_depth: resample_depth_b.or(resample_depth).unwrap_or(u32::MAX),
                ..params
            };
            let rf = render_file.as_ref().map(std::path::PathBuf::from);
            let render = rf.as_deref().map(|p| (p, render_games));
            let (score, w, l, d, margin, game_score) =
                run_match(&sa, &sb, games, (&pa, &pb), render)?;
            println!("A_wins {w} B_wins {l} draws {d}");
            println!("A_score {score:.4}");
            // Mean net-worth margin from A's seat. Reported alongside the score
            // because the score alone can be pinned at 0.5 by seat asymmetry.
            println!("A_margin {margin:.3}");
            // Mean terminal score from A's seat: the margin plus a speed term,
            // so a fast win outranks a slow one that ends on the same material.
            //
            // Deliberately *not* called `A_score` -- that name is already the
            // win rate the eval loop gates promotion on, and a second line with
            // the same key would be matched by its regex instead.
            println!("A_gamescore {game_score:.3}");
        }
        Cmd::Bench {
            live,
            backend,
            ep,
            fp16,
            cuda,
            candidates,
            t_steps,
            secs,
            sweep,
        } => bench(
            &live, &backend, &ep, fp16, cuda, candidates, t_steps, secs, sweep,
        )?,
        Cmd::DumpValidation { out, cases, seed } => dump_validation(&out, cases, seed)?,
        Cmd::Soak {
            threads,
            secs,
            no_verify,
            no_oracle,
            seed,
            out,
        } => soak::run(threads, secs, !no_verify, !no_oracle, seed, &out)?,
        Cmd::NetGame {
            live,
            sims,
            candidates,
            t_steps,
            backend,
            ep,
            fp16,
            cuda,
            seed,
            opponent,
            games,
        } => {
            use minions_engine::eval::{load_seat, match_search, Seat};
            use playground_core::sampled_mcts::search;
            use playground_core::Game;
            // Eval's search: the policy as it plays, no random candidates and
            // the most-visited turn.
            let (sampler, params) = match_search(sims, candidates, t_steps, 0, seed);
            let kind = minions_engine::net::BackendKind::parse(&backend, &ep, fp16, cuda)?;
            let seat = load_seat(std::path::Path::new(&live), &kind, sampler)?;
            // Second seat: the net again, or the heuristic. Yellow is the net.
            let foe = match opponent.as_str() {
                "greedy" => Seat::Greedy,
                "net" => load_seat(std::path::Path::new(&live), &kind, sampler)?,
                path => load_seat(std::path::Path::new(path), &kind, sampler)?,
            };
            for g in 0..games.max(1) {
                // Alternate colours between games. Minions is not symmetric — the
                // second player is paid START_MONEY_SECOND — so watching one seat
                // only would show half the policy's habits.
                let a_yellow = g % 2 == 0;
                println!(
                    "\n===== game {} of {games}  ({} is yellow) =====",
                    g + 1,
                    if a_yellow { "A" } else { "B" }
                );
                let mut params = params;
                params.seed = seed ^ ((g as u64) << 24);
                let mut state = Minions::initial_state();
                println!("{}", state.render());
                let mut turn = 0u32;
                while !Minions::outcome(&state).is_terminal() {
                    // Yellow (side 0) is the net; the opponent takes blue.
                    let actor = if (state.side == 0) == a_yellow {
                        &seat
                    } else {
                        &foe
                    };
                    let spec = match actor {
                        Seat::Net(m) => {
                            let r = search::<minions_engine::turn::MinionsTurns, _>(
                                &**m,
                                state.clone(),
                                &params,
                                turn,
                            )?;
                            r.candidates
                                .get(r.chosen)
                                .map(|(s, _)| *s)
                                .unwrap_or_default()
                        }
                        Seat::Greedy => minions_engine::turn::TurnSpec::from_play(&state, |st| {
                            if Minions::outcome(st).is_terminal() {
                                return None;
                            }
                            Some(<Minions as playground_core::Solver>::choose_move(st).best_move)
                        }),
                    };
                    let r = minions_engine::turn::realize(&state, &spec);
                    state = r.next;
                    turn += 1;
                    let nw = [
                        minions_engine::game::net_worth(&state, 0),
                        minions_engine::game::net_worth(&state, 1),
                    ];
                    println!(
                        "turn {turn}  gold {:?}  net worth {nw:?}  actions {}",
                        state.money,
                        r.actions.len()
                    );
                    println!("{}", state.render());
                }
                println!(
                    "outcome: {:?} after {} turns",
                    Minions::outcome(&state),
                    state.turn
                );
            }
        }
        Cmd::BenchBind {
            live,
            ep,
            fp16,
            rows,
            steps,
            reps,
        } => {
            let dir = std::path::Path::new(&live);
            let cache = dir.join("trt_cache");
            minions_engine::ort_net::bench_binding(
                dir,
                &ep,
                fp16,
                cache.to_str().unwrap(),
                rows,
                steps,
                reps,
            )?;
        }
        Cmd::GreedyGame { games, verbose } => greedy_games(games, verbose),
        Cmd::GenPositions {
            out,
            count,
            mean,
            stdev,
            max_turns,
            shard_size,
            threads,
            seed,
        } => {
            minions_engine::genpos::generate(
                &out,
                minions_engine::genpos::GenConfig {
                    count,
                    mean,
                    stdev,
                    max_turns,
                    shard_size,
                    threads,
                    seed,
                },
            )?;
        }
        Cmd::NnueBench { active, reps } => minions_engine::nnue::bench_sweep(active, reps),
    }
    Ok(())
}

fn greedy_games(games: u32, verbose: bool) {
    use playground_core::{Game, Outcome, Solver};

    let mut tally = [0u32; 3]; // first wins, second wins, draws
    for g in 0..games {
        let mut state = Minions::initial_state();
        let mut turn = state.turn;
        while Minions::outcome(&state) == Outcome::InProgress {
            let mv = Minions::choose_move(&state).best_move;
            state = Minions::apply(&state, &mv);
            if verbose && state.turn != turn {
                turn = state.turn;
                println!("{}", state.render());
            }
        }
        let outcome = Minions::outcome(&state);
        match outcome {
            Outcome::FirstWins => tally[0] += 1,
            Outcome::SecondWins => tally[1] += 1,
            _ => tally[2] += 1,
        }
        println!(
            "game {g}: {outcome:?} after {} turns / {} actions",
            state.turn, state.ply
        );
        if !verbose {
            println!("{}", state.render());
        }
    }
    println!(
        "first {} / second {} / draw {}",
        tally[0], tally[1], tally[2]
    );
}

/// Emit the cross-implementation validation bundle. See `python/minions_validate.py`.
fn dump_validation(out: &str, cases: usize, seed: u64) -> anyhow::Result<()> {
    use minions_engine::game::{Minions, PLANES};
    use minions_engine::map::NUM_CELLS;
    use minions_engine::turn::{
        LiveMask, Partial, TurnSpec, C_MOVE, C_SPAWN, C_TARGET, DECL_PLANES,
    };
    use playground_core::Game;
    use rand::SeedableRng;

    std::fs::create_dir_all(out)?;
    let dir = std::path::Path::new(out);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    // Declaration-plane cases: a spec, a committed mask, and the planes Rust
    // produces for them. Partial commitment matters — the sampler feeds the net
    // half-built declarations, so an encoding bug there would only show up
    // mid-diffusion.
    let mut specs = Vec::new();
    let mut masks = Vec::new();
    let mut lives = Vec::new();
    let mut planes = Vec::new();
    let mut state = Minions::initial_state();
    for i in 0..cases {
        if Minions::outcome(&state).is_terminal() {
            state = Minions::initial_state();
        }
        let spec = if i % 2 == 0 {
            TurnSpec::wild(&mut rng)
        } else {
            TurnSpec::random_plausible(&state, &mut rng)
        };
        let live = LiveMask::of(&state);
        let mut p = Partial {
            spec,
            committed: [0; NUM_CELLS],
        };
        // Commit a varying fraction, including none and all.
        let frac = i as f64 / (cases.max(2) - 1) as f64;
        for (c, bit) in [C_MOVE, C_TARGET, C_SPAWN].into_iter().enumerate() {
            let cells = match c {
                0 => live.moves,
                1 => live.targets,
                _ => live.spawns,
            };
            for cell in minions_engine::bitboard::cells(cells) {
                if rand::Rng::gen_bool(&mut rng, frac) {
                    p.committed[cell as usize] |= bit;
                }
            }
        }
        let mut buf = vec![0f32; DECL_PLANES * NUM_CELLS];
        p.write_planes(&mut buf);
        let mut live_bytes = vec![0u8; NUM_CELLS];
        for (cells, bit) in [
            (live.moves, C_MOVE),
            (live.targets, C_TARGET),
            (live.spawns, C_SPAWN),
        ] {
            for cell in minions_engine::bitboard::cells(cells) {
                live_bytes[cell as usize] |= bit;
            }
        }
        specs.extend_from_slice(&p.spec.to_bytes());
        masks.extend_from_slice(&p.committed);
        lives.extend_from_slice(&live_bytes);
        planes.extend(buf.iter().flat_map(|v| v.to_le_bytes()));
        state = minions_engine::turn::realize(&state, &spec).next;
    }
    std::fs::write(dir.join("decl_specs.bin"), &specs)?;
    std::fs::write(dir.join("decl_masks.bin"), &masks)?;
    std::fs::write(dir.join("decl_live.bin"), &lives)?;
    std::fs::write(dir.join("decl_planes.bin"), &planes)?;

    // A features case, so the learner's plane reshape is checked against the
    // encoder that actually feeds the net.
    let s0 = Minions::initial_state();
    std::fs::write(dir.join("features.bin"), Minions::to_features(&s0))?;
    std::fs::write(
        dir.join("meta.json"),
        format!(
            r#"{{"cases": {cases}, "cells": {NUM_CELLS}, "decl_planes": {DECL_PLANES}, "in_planes": {PLANES}}}"#
        ),
    )?;
    println!("wrote validation bundle to {out} ({cases} declaration cases)");
    Ok(())
}

/// Time each graph at the batch the sampler uses, then derive the cost of one
/// node expansion: 1 trunk (batch 1) + T diffusion + 1 energy (batch K).
#[allow(clippy::too_many_arguments)]
fn bench(
    live: &str,
    backend: &str,
    ep: &str,
    fp16: bool,
    cuda: bool,
    k: usize,
    t_steps: usize,
    secs: u64,
    sweep: bool,
) -> anyhow::Result<()> {
    use minions_engine::game::Minions;
    use minions_engine::map::NUM_CELLS;
    use minions_engine::net::{load_backend, repeat_trunk, BackendKind};
    use minions_engine::turn::{LiveMask, Partial, TurnSpec, DECL_PLANES, ENERGY_PLANES};
    use playground_core::Game;
    use std::time::{Duration, Instant};

    let live_path = std::path::Path::new(live);
    let kind = BackendKind::parse(backend, ep, fp16, cuda)?;
    let cache = live_path.join("trt_cache").to_string_lossy().into_owned();
    println!("loading {kind:?} from {live} ...");
    let t0 = Instant::now();
    let net = load_backend(live_path, &kind, &cache)?;
    println!(
        "loaded in {:.1}s (channels {})",
        t0.elapsed().as_secs_f64(),
        net.channels()
    );

    let state = Minions::initial_state();
    let planes: Vec<f32> = Minions::to_features(&state)
        .iter()
        .map(|&b| b as f32)
        .collect();
    let live_mask = LiveMask::of(&state);
    let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(1);
    let stride = DECL_PLANES * NUM_CELLS;
    // The energy head takes the declaration planes *minus* plane 2, the
    // move-commit flag (`ENERGY_PLANES`; see `turn.rs` -- with everything
    // committed that flag degenerates into a liveness mask `h` already
    // carries). So it needs its own buffer, written by `write_energy_planes`,
    // exactly as production does in `sampler.rs`. Feeding it the 12-plane
    // buffer made every `bench` run die at the energy step with
    // "shape [k, 11, 7, 7] ... larger than the length of the data provided".
    let estride = ENERGY_PLANES * NUM_CELLS;
    let mut decl = vec![0f32; k * stride];
    let mut edecl = vec![0f32; k * estride];
    for i in 0..k {
        let spec = TurnSpec::random_plausible(&state, &mut rng);
        let p = Partial::fully_committed(&spec, &live_mask);
        p.write_planes(&mut decl[i * stride..(i + 1) * stride]);
        p.write_energy_planes(&mut edecl[i * estride..(i + 1) * estride]);
    }

    let tr = net.trunk(&planes, 1)?;
    let h_k = repeat_trunk(&tr.h, k);

    let timed = |name: &str, f: &dyn Fn() -> anyhow::Result<()>| -> anyhow::Result<f64> {
        f()?; // warm up: TensorRT builds and caches its engine on first use
        let deadline = Instant::now() + Duration::from_secs(secs);
        let start = Instant::now();
        let mut n = 0u64;
        while Instant::now() < deadline {
            f()?;
            n += 1;
        }
        let per = start.elapsed().as_secs_f64() / n.max(1) as f64;
        println!("  {name:<12} {n:>7} calls   {:>8.3} ms/call", per * 1000.0);
        Ok(per)
    };

    if sweep {
        // At small batches a GPU spends its time on launch latency rather than
        // arithmetic, so throughput-per-item keeps improving long after the
        // batch looks "big enough". This is the measurement that says how much
        // batching across concurrent games is worth.
        println!("\nbatch sweep (throughput per item; higher is better):");
        println!(
            "  {:>6}  {:>12}  {:>12}  {:>12}",
            "batch", "trunk/s", "diffusion/s", "energy/s"
        );
        for &b in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
            let planes_b = repeat_trunk(&planes, b);
            let h_b = repeat_trunk(&tr.h, b);
            let mut decl_b = vec![0f32; b * stride];
            let mut edecl_b = vec![0f32; b * estride];
            for i in 0..b {
                let src = (i % k) * stride;
                decl_b[i * stride..(i + 1) * stride].copy_from_slice(&decl[src..src + stride]);
                let esrc = (i % k) * estride;
                edecl_b[i * estride..(i + 1) * estride]
                    .copy_from_slice(&edecl[esrc..esrc + estride]);
            }
            let rate = |f: &dyn Fn() -> anyhow::Result<()>| -> anyhow::Result<f64> {
                f()?;
                let deadline = Instant::now() + Duration::from_millis(700);
                let start = Instant::now();
                let mut n = 0u64;
                while Instant::now() < deadline {
                    f()?;
                    n += 1;
                }
                Ok(n as f64 * b as f64 / start.elapsed().as_secs_f64())
            };
            let rt = rate(&|| net.trunk(&planes_b, b).map(|_| ()))?;
            let rd = rate(&|| net.diffusion(&h_b, &decl_b, b).map(|_| ()))?;
            let re = rate(&|| net.energy(&h_b, &edecl_b, b).map(|_| ()))?;
            println!("  {b:>6}  {rt:>12.0}  {rd:>12.0}  {re:>12.0}");
        }
        return Ok(());
    }

    println!("timing (trunk at batch 1, heads at batch {k}, {secs}s each):");
    let t_trunk = timed("trunk", &|| net.trunk(&planes, 1).map(|_| ()))?;
    let t_diff = timed("diffusion", &|| net.diffusion(&h_k, &decl, k).map(|_| ()))?;
    let t_energy = timed("energy", &|| net.energy(&h_k, &edecl, k).map(|_| ()))?;

    let per_expansion = t_trunk + t_steps as f64 * t_diff + t_energy;
    println!(
        "\nper expansion (1 trunk + {t_steps} diffusion + 1 energy): {:.2} ms -> {:.0}/s",
        per_expansion * 1000.0,
        1.0 / per_expansion
    );
    for sims in [100u32, 200] {
        let per_game = per_expansion * sims as f64 * 40.0;
        println!(
            "  at {sims} sims: {:.1}s of inference per game, {:.3} games/s/thread",
            per_game,
            1.0 / per_game
        );
    }
    Ok(())
}
