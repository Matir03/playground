//! Simplified-Minions engine binary. Subcommand semantics follow
//! arimaa-engine: `serve` is the interactive gRPC-Web endpoint the web UI
//! talks to, and works without a model server running (the model channel is
//! lazy) — only the NN seat needs one.

mod soak;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
        /// Play the "net" seat with the best-first step search instead
        /// (docs/minions-steps.md): a step net (`.onnx`, `.bin` MLP, or
        /// `old:<dir>`). Each turn searches for the UI's time budget, else
        /// `--steps-secs`.
        #[arg(long)]
        steps_net: Option<String>,
        #[arg(long, default_value_t = 5.0)]
        steps_secs: f64,
        #[arg(long, default_value_t = 32)]
        steps_heads: usize,
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
        #[arg(long, default_value_t = 0.0)]
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
        #[arg(long, default_value_t = 0)]
        forced_visits: u32,
        /// Non-explore candidates drawn at random to also get a quota, on top
        /// of every explore candidate.
        #[arg(long, default_value_t = 0)]
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
        /// How a node's value is formed: `avg` (default) averages every value
        /// backed up through it; `minimax` makes it the best child instead, so
        /// one refutation is not diluted by the siblings sampled beside it.
        /// **`avg` reproduces the previous search bit-for-bit.**
        #[arg(long, default_value = "avg")]
        value_backup: playground_core::sampled_mcts::ValueBackup,
        /// What an interior node's widening budget `ceil(pw_beta * n^pw_alpha)`
        /// counts: `children` (default) the candidates it holds, or `attempts`
        /// the widen requests it has made.
        ///
        /// `children` budgets results -- a widen that brings back a duplicate
        /// costs nothing and the node simply asks again, and candidates arriving
        /// from draws, fills or explore injection count against the budget and
        /// hold growth back. `attempts` budgets requests: each widen spends one
        /// whatever it returns, so the rate follows how often the node was asked
        /// to grow rather than what came of it.
        /// `attempts` is the default since it beat `children` 285-114 over 400
        /// games on gen405; `children` is the pre-2026-09-19 search.
        #[arg(long, default_value = "attempts")]
        widen_by: playground_core::sampled_mcts::WidenBy,
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
        /// Move pruned chunks here instead of deleting them.
        #[arg(long)]
        archive_dir: Option<String>,
        /// Cap on the archive's size; the oldest chunks are deleted past it.
        #[arg(long, default_value_t = 50.0)]
        archive_max_gb: f64,
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
        /// What seat A's widening budget counts: `children` or `attempts`.
        /// Default: eval's.
        #[arg(long)]
        widen_by: Option<playground_core::sampled_mcts::WidenBy>,
        /// As `--widen-by`, for seat B; defaults to A's. Set only this to play
        /// the two widening schedules against each other on one checkpoint.
        #[arg(long)]
        widen_by_b: Option<playground_core::sampled_mcts::WidenBy>,
        /// Completions drawn from the last diffusion step for seat A, the best
        /// by NNUE kept and the rest offered as extra candidates. 1 = the
        /// ordinary sampler. Needs `--nnue-weights`.
        #[arg(long, default_value_t = 1)]
        last_stage_samples: usize,
        /// As `--last-stage-samples`, for seat B; defaults to A's. Set only
        /// this to play best-of-N against the ordinary sampler on one
        /// checkpoint.
        #[arg(long)]
        last_stage_samples_b: Option<usize>,
        /// Distilled value net used to order those completions. Required by
        /// `--last-stage-samples` above 1; never reaches the search's values.
        #[arg(long, default_value = "data/minions_nnue/student_512_long.bin")]
        nnue_weights: String,
        /// Seat A picks resample variables by the head's belief -- each one
        /// unmasked with probability `1 - p(value already there)` -- instead of
        /// a uniform random share. Ignores `--resample-frac`.
        #[arg(long)]
        resample_bias: bool,
        /// As `--resample-bias`, for seat B; defaults to A's. Set only this to
        /// play biased unmasking against the random one on one checkpoint.
        #[arg(long)]
        resample_bias_b: Option<bool>,
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
        /// Value-backup mode for seat A: `avg` (default) or `minimax`.
        #[arg(long, default_value = "avg")]
        value_backup: playground_core::sampled_mcts::ValueBackup,
        /// Value-backup mode for seat B; defaults to A's. Set only this to play
        /// averaging against minimax on a single checkpoint.
        #[arg(long)]
        value_backup_b: Option<playground_core::sampled_mcts::ValueBackup>,
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
        /// Record *every* game to this path as JSON: per game the settled bid,
        /// which seat had yellow, the outcome, and the position after each
        /// turn as a hex-encoded `GameState.payload`. Unlike `--render-file`,
        /// which writes a human transcript of the first few games, this is the
        /// machine-readable record the web viewer replays.
        #[arg(long)]
        record_file: Option<String>,
        /// Also record each turn's search: the root's top candidates with the
        /// visits, priors, energies and values the search actually selected
        /// on, so a recorded game can be read the way the analysis view reads
        /// a live one. Roughly 2 MB per 80 games on top of the positions.
        #[arg(long)]
        record_search: bool,
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
        /// Time every graph at exactly this many rows instead (0 = trunk at 1,
        /// heads at `candidates`) -- the production shape is ~100-170 rows.
        #[arg(long, default_value_t = 0)]
        rows: usize,
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
    /// Greedy match between two step evaluators (each step: evaluate every
    /// option, take the best). Evaluators: `old:<dir>`, `<path>.onnx`, or an
    /// MLP's `<path>.bin` (CPU). `--b` may list several, comma-separated: one
    /// match against each, in turn.
    StepsMatch {
        #[arg(long)]
        a: String,
        #[arg(long)]
        b: String,
        #[arg(long, default_value_t = 100)]
        pairs: usize,
        /// 0 = standard opening with bidding; otherwise random openings.
        #[arg(long, default_value_t = 22)]
        opening_max: u32,
        #[arg(long, default_value_t = 1.0884)]
        opening_decay: f32,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Leaf expansions per turn for each side (0 = greedy on static values).
        #[arg(long, default_value_t = 0)]
        expansions: u64,
        /// B's expansions per turn, if different from A's.
        #[arg(long)]
        expansions_b: Option<u64>,
        /// Backup power-mean exponent (`inf` = max) for A, and for B if different.
        #[arg(long, default_value_t = 2.0)]
        power: f32,
        #[arg(long)]
        power_b: Option<f32>,
        #[arg(long, default_value_t = 32)]
        heads: usize,
        /// Game-pair threads.
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Seat A (and B with --two-tier-b) searches with `two_tier` instead
        /// of best-first, on the same expansion budget.
        #[arg(long)]
        two_tier_a: bool,
        #[arg(long)]
        two_tier_b: bool,
        /// Two-tier: UCT constant of the in-turn search.
        #[arg(long, default_value_t = 0.1)]
        t2_c: f32,
        /// Two-tier: tier-2 descents that may end on known turns before a
        /// node's widening is dropped.
        #[arg(long, default_value_t = 20)]
        t2_retries: u32,
        /// Two-tier widening value: `current` (the newest turn's value now)
        /// or `static` (its value when found).
        #[arg(long, default_value = "current")]
        t2_widen: String,
        /// A or B given as a lattice net directory (tier1.onnx + tier2.onnx)
        /// plays by tier-1 lattice search: visits per turn, and in flight.
        #[arg(long, default_value_t = 200)]
        n1: u32,
        /// B's tier-1 visits, if different from A's.
        #[arg(long)]
        n1_b: Option<u32>,
        #[arg(long, default_value_t = 16)]
        lattice_inflight: usize,
        /// Check every candidate option eagerly in the lattices (self-play,
        /// lattice-gen, defers them: `steps::LAZY_WITNESS`). Eager checks gave
        /// single games a heavy tail of 30+ minutes.
        #[arg(long)]
        no_lazy_witness: bool,
    },
    /// The step search (best-first) against the old sampled-AZ MCTS, standard
    /// openings with bidding. Old side defaults to the old ladder's eval
    /// search: 400 sims, 16 candidates, 4 diffusion steps, no refinement.
    StepsVsMcts {
        /// Step net (`.onnx`).
        #[arg(long)]
        a: String,
        /// Old model directory (trunk/diffusion/energy ONNX + config.json).
        #[arg(long)]
        old: String,
        #[arg(long, default_value_t = 50)]
        pairs: usize,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Step side: leaf expansions per turn (best-first).
        #[arg(long, default_value_t = 2000)]
        expansions: u64,
        #[arg(long, default_value_t = 32)]
        heads: usize,
        #[arg(long, default_value_t = 400)]
        sims: u32,
        #[arg(long, default_value_t = 16)]
        candidates: usize,
        #[arg(long, default_value_t = 4)]
        t_steps: usize,
        #[arg(long, default_value_t = 0.0)]
        resample_every: f32,
        /// Step-player threads.
        #[arg(long, default_value_t = 8)]
        threads: usize,
        /// Rescale the step budget each round to the MCTS's time per move
        /// (`--expansions` is then only the first round's budget).
        #[arg(long)]
        equal_time: bool,
        /// `a` as a lattice net directory: tier-1 visits per turn, in flight.
        #[arg(long, default_value_t = 200)]
        n1: u32,
        #[arg(long, default_value_t = 16)]
        lattice_inflight: usize,
    },
    /// Check a step-trained MLP's CPU evaluator against its ONNX graph on
    /// random step-tree positions.
    StepsCheck {
        /// Evaluator specs as for steps-match (`.bin` CPU MLP, `.onnx`, `old:<dir>`).
        #[arg(long)]
        a: String,
        #[arg(long)]
        b: String,
        #[arg(long, default_value = "cuda")]
        ep: String,
        #[arg(long, default_value_t = 2000)]
        positions: usize,
    },
    /// Self-play with the best-first step search (docs/minions-steps.md):
    /// writes every well-visited search node as a value record.
    /// Replay a dumped exact-search query corpus (`STEPS_EXACT_DUMP`) at a
    /// given budget.
    /// Serve saved self-play trees (`steps-selfplay --save-trees`) to the web
    /// viewer, node by node (`tree_serve`).
    StepsTreeServe {
        /// The directory holding the runs (`data/models/minions/steps/trees`).
        #[arg(long)]
        dir: String,
        #[arg(long, default_value = "127.0.0.1:7106")]
        addr: String,
    },
    StepsExactBench {
        #[arg(long)]
        corpus: String,
        #[arg(long, default_value_t = 1000)]
        budget: u32,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Write each query's outcome at this budget (one byte: 0 found,
        /// 1 impossible, 2 unknown) for `steps-exact-diag`.
        #[arg(long)]
        labels_out: Option<String>,
        /// Instead of replaying: profile the queries against these labels and
        /// score the candidate impossibility checks.
        #[arg(long)]
        diag: Option<String>,
    },
    /// Turn-lattice generation (docs/minions-lattice.md): tier-1 searches
    /// every turn, tier-2 walks for the tier-2 net's records.
    LatticeGen {
        #[arg(long)]
        out: String,
        /// A lattice net directory (tier1.onnx, tier2.onnx, lattice.json), or
        /// a steps-mode value net (`value_sparse.onnx`) behind the bridge.
        #[arg(long)]
        onnx: String,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = 2)]
        batcher: usize,
        #[arg(long, default_value_t = 8)]
        workers: usize,
        /// Games per worker.
        #[arg(long, default_value_t = 32)]
        games: usize,
        #[arg(long, default_value_t = 400)]
        n1: u32,
        /// Tier-1 visits and tier-2 walk descents in flight per game.
        #[arg(long, default_value_t = 8)]
        inflight: usize,
        #[arg(long, default_value_t = 20)]
        n2: u32,
        #[arg(long, default_value_t = 1.5)]
        c1: f32,
        #[arg(long, default_value_t = 1.5)]
        c2: f32,
        #[arg(long, default_value_t = 1.0)]
        pw_c: f32,
        #[arg(long, default_value_t = 0.5)]
        pw_alpha: f32,
        #[arg(long, default_value_t = 20)]
        retries: u32,
        #[arg(long, default_value_t = 1.0)]
        temp1: f64,
        #[arg(long, default_value_t = 0.3)]
        dir1_alpha: f64,
        #[arg(long, default_value_t = 0.25)]
        dir1_frac: f64,
        #[arg(long, default_value_t = 1.0)]
        temp2: f64,
        #[arg(long, default_value_t = 0.3)]
        dir2_alpha: f64,
        #[arg(long, default_value_t = 0.25)]
        dir2_frac: f64,
        /// Games start MAX_TURNS - G turns in, G geometric with this mean.
        #[arg(long, default_value_t = 10.0)]
        from_end_mean: f64,
        /// G's median instead of its mean (if positive): turn MAX_TURNS - m
        /// is the median start.
        #[arg(long, default_value_t = 0.0)]
        from_end_median: f64,
        /// The start turn itself geometric on 0, 1, ... (below the cap) with
        /// this mean, if positive (overrides both of the above).
        #[arg(long, default_value_t = 0.0)]
        start_turn_mean: f64,
        #[arg(long, default_value_t = 60.0)]
        secs: f64,
        #[arg(long, default_value_t = 0)]
        target_t2: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Check every offered option's witness on the spot instead of
        /// deferring the exact search to the child's expansion.
        #[arg(long)]
        no_lazy_witness: bool,
        /// For lattice nets: execution provider and runners (the CUDA EP
        /// computed the one-graph export wrongly for some shapes; use trt).
        #[arg(long, default_value = "trt")]
        lattice_ep: String,
        #[arg(long, default_value_t = 1)]
        lattice_sessions: usize,
        #[arg(long, default_value_t = 20000)]
        chunk_records: usize,
        /// Also record non-root tier-1 turns with at least this many visits
        /// as tier-1 value targets (0 = roots only).
        #[arg(long, default_value_t = 16)]
        t1_min_visits: u32,
        /// Also record tier-2 nodes, and their kids with at least this many
        /// visits, as tier-1 value targets on partial-turn views (0 = none).
        #[arg(long, default_value_t = 0)]
        view_min_visits: u32,
        /// Tier-2 records from the tier-1 search (each kid's targets: the
        /// visits and mean value of the tier-1 turns containing it) instead
        /// of the walk.
        #[arg(long)]
        t2_from_t1: bool,
        /// Distill the evaluator into tier 2: expansions become the tier-2
        /// records (child values as value targets, a per-cell softmax of
        /// them at this temperature as prior targets; 0 = off).
        #[arg(long, default_value_t = 0.0)]
        distill_tau: f64,
        /// The share of expansions recorded when distilling.
        #[arg(long, default_value_t = 0.25)]
        distill_frac: f64,
        /// Self-distillation: the share of the nodes on the path to each
        /// chosen turn recorded from completions (0 = off; then the other
        /// tier-2 record modes apply).
        #[arg(long, default_value_t = 0.0)]
        sd_frac: f64,
        /// Self-distillation: descents per kid.
        #[arg(long, default_value_t = 1)]
        sd_descents: u32,
        /// Self-distillation: temperature of the per-cell softmax priors.
        #[arg(long, default_value_t = 0.02)]
        sd_tau: f64,
        /// The walk's descents select on the lattice's accumulated statistics
        /// (the tier-1 search's included) instead of a fresh per-ply search.
        #[arg(long)]
        walk_accumulated: bool,
        /// Value turn starts (complete turns, tier-1 nodes) with this steps
        /// value net instead of the lattice net's tier 1; tier 2 still
        /// expands from --onnx.
        #[arg(long)]
        sink_values: Option<String>,
        /// The walk's descents in flight per game (default: --inflight). With
        /// n2 descents per ply all in flight at once, their virtual losses
        /// spread them by prior order before any value comes back.
        #[arg(long)]
        walk_inflight: Option<usize>,
        /// Walk records count the complete turns containing each kid (the
        /// marginals) instead of each descent's first step.
        #[arg(long)]
        walk_marginal: bool,
        /// Walk records with completed-Q targets (Gumbel-style policy
        /// improvement over edge statistics at each locked node).
        #[arg(long)]
        walk_cq: bool,
        /// Records in the lattice view (original position plus decisions,
        /// per-kind output slots) and energy records (t1e/).
        #[arg(long)]
        lview: bool,
        /// Interior tier-1 turns with at least this many visits also write
        /// energy groups (0: the root's only).
        #[arg(long, default_value_t = 8)]
        energy_min_visits: u32,
        /// Groups of games per worker taking turns, so a worker advances one
        /// group while another's net call is in flight (1 = no overlap).
        #[arg(long, default_value_t = 2)]
        pipeline: usize,
        /// Openings played up front, all at once (batched), before the
        /// searches start; searches play theirs inline once these run out.
        #[arg(long, default_value_t = 768)]
        openings: usize,
        /// Threads for the opening phase (0 = all hardware threads).
        #[arg(long, default_value_t = 0)]
        open_workers: usize,
        /// A saved opening pool: games start from it (oldest first), the
        /// rest is written back, and `--openings` is ignored.
        #[arg(long)]
        pool: Option<std::path::PathBuf>,
        /// Refill the pool to this many openings and stop (no searches).
        #[arg(long, default_value_t = 0)]
        refill_to: usize,
    },
    StepsSelfplay {
        /// Record output directory.
        #[arg(long)]
        out: String,
        /// Evaluate with the sampled-AZ champion's value head on the projected
        /// board (a dir holding trunk.onnx + config.json).
        #[arg(long)]
        old_net: Option<String>,
        /// Evaluate with a step-trained value net: an ONNX graph on the GPU, or
        /// an MLP's `mlp.bin` on the CPU.
        #[arg(long)]
        onnx: Option<String>,
        #[arg(long, default_value = "trt")]
        ep: String,
        #[arg(long, default_value_t = 8)]
        workers: usize,
        /// ONNX sessions for a GPU step net (0 = one per worker).
        #[arg(long, default_value_t = 0)]
        sessions: usize,
        /// Shared-queue GPU runner threads for a GPU step net (0 = each worker
        /// calls its own session).
        #[arg(long, default_value_t = 0)]
        batcher: usize,
        #[arg(long, default_value_t = 32)]
        heads: usize,
        /// Threads building each wave's leaf expansions, per worker. With one
        /// worker, a large --heads and this at the core count, self-play
        /// searches a single game tree with the whole machine.
        #[arg(long, default_value_t = 1)]
        expand_threads: usize,
        /// Games sharing one search tree (one worker, --expand-threads > 1):
        /// the descents are spread over their positions each round.
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 2.0)]
        power: f32,
        /// Leaf expansions per move.
        #[arg(long, default_value_t = 2000)]
        expansions: u64,
        #[arg(long, default_value_t = 0.0)]
        move_secs: f64,
        #[arg(long, default_value_t = 20)]
        min_visits: u32,
        #[arg(long, default_value_t = 1.0)]
        temp: f32,
        #[arg(long, default_value_t = 0.3)]
        dirichlet_alpha: f32,
        #[arg(long, default_value_t = 0.25)]
        dirichlet_eps: f32,
        /// Random turns before the search takes over (0 = none: the game is
        /// already randomized by the Dirichlet noise and the bid).
        #[arg(long, default_value_t = 0)]
        opening_max: u32,
        #[arg(long, default_value_t = 1.0884)]
        opening_decay: f32,
        /// Search: `descent` (to the end of the game, the self-play search) or
        /// `bfs` (best-first, the eval search).
        #[arg(long, default_value = "descent")]
        search: String,
        /// Turn walk: `ordinal` (ordinal distribution + completed action
        /// selection) or `visits` (descents with --temp and Dirichlet noise).
        #[arg(long, default_value = "ordinal")]
        select: String,
        /// Probability of keeping each eligible node as a record.
        #[arg(long, default_value_t = 1.0)]
        record_frac: f64,
        /// Keep records in proportion to visits^alpha (normalized so the
        /// expected count is unchanged; 0 = uniform).
        #[arg(long, default_value_t = 0.5)]
        record_alpha: f64,
        /// Defer exact witness searches to each child's first expansion
        /// (`steps::LAZY_WITNESS`).
        #[arg(long)]
        lazy_witness: bool,
        #[arg(long, default_value_t = 1_000_000)]
        target_records: u64,
        #[arg(long, default_value_t = 20_000)]
        chunk_records: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Save each shared tree (the whole tree at every round) under this
        /// directory, for the web viewer (`tree_save`). Needs --games > 1.
        #[arg(long)]
        save_trees: Option<String>,
        /// Give every sampled position its own best-first search of this many
        /// expansions at the end of the generation (the eval procedure) and
        /// record its root value instead of the self-play tree's (0 = off).
        #[arg(long, default_value_t = 0)]
        resample_expansions: u64,
        /// Leaves per iteration of those searches.
        #[arg(long, default_value_t = 8)]
        resample_heads: usize,
        /// Threads running those searches (0 = twice --expand-threads).
        #[arg(long, default_value_t = 0)]
        resample_threads: usize,
        /// Those searches: best-first, or two-tier (`two_tier`).
        #[arg(long, default_value = "best-first")]
        resample_search: String,
        /// Two-tier: UCT constant of the in-turn search.
        #[arg(long, default_value_t = 0.1)]
        t2_c: f32,
        /// Two-tier: tier-2 descents that may end on known turns before a
        /// node's widening is dropped.
        #[arg(long, default_value_t = 20)]
        t2_retries: u32,
        /// Two-tier widening value: `current` (the newest turn's value now)
        /// or `static` (its value when found).
        #[arg(long, default_value = "current")]
        t2_widen: String,
    },
}

/// A match spec naming a lattice net directory (tier1.onnx + tier2.onnx):
/// its tier-1 net as a steps value net (for bids) and the lattice player.
/// Anything else is a steps spec, passed through.
fn lattice_play(spec: &str, n1: u32, inflight: usize) -> anyhow::Result<(String, Option<minions_engine::stepsp::LatticePlay>)> {
    // `bridge:<steps net>`: the lattice search on a steps value net (tier-1
    // values, tier-2 child values on partial turns, uniform priors).
    if let Some(net) = spec.strip_prefix("bridge:") {
        let eval: std::sync::Arc<dyn minions_engine::lattice::LatticeEval> =
            std::sync::Arc::new(minions_engine::lattice::StepsBridge { eval: load_step_eval_runners(net, "trt", 1)? });
        let play = minions_engine::stepsp::LatticePlay { eval, n1, inflight, params: Default::default() };
        return Ok((net.to_string(), Some(play)));
    }
    let dir = std::path::Path::new(spec);
    if !dir.join("tier1.onnx").is_file() {
        return Ok((spec.to_string(), None));
    }
    let eval = std::sync::Arc::new(minions_engine::lattice_ort::OrtLatticeEval::load(dir, "trt", true, 1)?);
    let play = minions_engine::stepsp::LatticePlay { eval, n1, inflight, params: Default::default() };
    Ok((dir.join("value_sparse.onnx").to_string_lossy().into_owned(), Some(play)))
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
            steps_net,
            steps_secs,
            steps_heads,
        } => {
            if let Some(spec) = steps_net {
                eprintln!("net seat: step search with {spec}, {steps_secs}s per turn, {steps_heads} heads");
                minions_engine::net_seat::configure_steps(minions_engine::net_seat::StepsSeat {
                    eval: load_step_eval(&spec, &ep)?,
                    params: minions_engine::bfs::Params { heads: steps_heads, ..Default::default() },
                    secs: steps_secs,
                    max_expansions: u64::MAX,
                });
            } else if let Some(dir) = net_dir {
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
            value_backup,
            pw_chunk,
            widen_by,
            draw_retries,
            resample_every,
            resample_depth,
            resample_t,
            resample_frac,
            write_search_config,
            chunk_secs,
            max_examples,
            max_chunks,
            archive_dir,
            archive_max_gb,
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
                value_backup,
                pw_chunk,
                widen_by,
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
                archive_dir.map(|d| minions_engine::selfplay::Archive {
                    dir: d.into(),
                    max_bytes: (archive_max_gb * 1e9) as u64,
                }),
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
            widen_by,
            widen_by_b,
            last_stage_samples,
            last_stage_samples_b,
            nnue_weights,
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
            value_backup,
            value_backup_b,
            render_file,
            render_games,
            record_file,
            record_search,
            resample_bias,
            resample_bias_b,
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
            sampler.resample_bias = resample_bias;
            // Biased unmasking only acts inside a resample, so with refinement
            // off it is a silent no-op: both seats run the identical search and
            // the match measures nothing but seat noise. Caught the hard way --
            // a 40-game run reported 22-18 having never called the code path.
            if (resample_bias || resample_bias_b.unwrap_or(false)) && resample_every == 0.0 {
                anyhow::bail!(
                    "--resample-bias needs --resample-every > 0: refinement is off, \
                     so the flag would change nothing"
                );
            }
            sampler.last_stage_samples = last_stage_samples;
            let kind = minions_engine::net::BackendKind::parse(&backend, &ep, fp16, cuda)?;
            // One evaluator per seat: it owns its scratch, and the whole point
            // of the match is that the seats differ in exactly this.
            let load_nnue = |n: usize| -> anyhow::Result<Option<minions_engine::nnue::Evaluator>> {
                if n > 1 {
                    Ok(Some(minions_engine::nnue::Evaluator::load(
                        std::path::Path::new(&nnue_weights),
                    )?))
                } else {
                    Ok(None)
                }
            };
            let sa = load_seat(
                std::path::Path::new(&a),
                &kind,
                sampler,
                load_nnue(last_stage_samples)?,
            )?;
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
                resample_bias: resample_bias_b.unwrap_or(resample_bias),
                last_stage_samples: last_stage_samples_b.unwrap_or(last_stage_samples),
                ..sampler
            };
            let sb = match &b {
                Some(p) => load_seat(
                    std::path::Path::new(p),
                    &kind,
                    sampler_b,
                    load_nnue(sampler_b.last_stage_samples)?,
                )?,
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
                value_backup,
                resample_every,
                resample_max_depth: resample_depth.unwrap_or(u32::MAX),
                pw_beta: pw_beta.unwrap_or(params.pw_beta),
                widen_by: widen_by.unwrap_or(params.widen_by),
                ..params
            };
            let pb = SampledParams {
                simulations: sims_b.unwrap_or(sims),
                kernel_sigma: kernel_sigma_b.unwrap_or(kernel_sigma),
                value_visits: value_visits_b.unwrap_or(value_visits),
                value_backup: value_backup_b.unwrap_or(value_backup),
                resample_every: resample_every_b.unwrap_or(resample_every),
                pw_beta: pw_beta_b.or(pw_beta).unwrap_or(params.pw_beta),
                widen_by: widen_by_b.or(widen_by).unwrap_or(params.widen_by),
                resample_max_depth: resample_depth_b.or(resample_depth).unwrap_or(u32::MAX),
                ..params
            };
            let rf = render_file.as_ref().map(std::path::PathBuf::from);
            let render = rf.as_deref().map(|p| (p, render_games));
            let cf = record_file.as_ref().map(std::path::PathBuf::from);
            let (score, w, l, d, margin, game_score) = run_match(
                &sa,
                &sb,
                games,
                (&pa, &pb),
                render,
                cf.as_deref(),
                record_search,
            )?;
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
            rows,
        } => bench(
            &live, &backend, &ep, fp16, cuda, candidates, t_steps, secs, sweep, rows,
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
            let seat = load_seat(std::path::Path::new(&live), &kind, sampler, None)?;
            // Second seat: the net again, or the heuristic. Yellow is the net.
            let foe = match opponent.as_str() {
                "greedy" => Seat::Greedy,
                "net" => load_seat(std::path::Path::new(&live), &kind, sampler, None)?,
                path => load_seat(std::path::Path::new(path), &kind, sampler, None)?,
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
        Cmd::StepsMatch { a, b, pairs, opening_max, opening_decay, ep, seed, expansions, expansions_b, power, power_b, heads, threads, two_tier_a, two_tier_b, t2_c, t2_retries, t2_widen, n1, n1_b, lattice_inflight, no_lazy_witness } => {
            minions_engine::steps::LAZY_WITNESS.store(!no_lazy_witness, std::sync::atomic::Ordering::Relaxed);
            // A match evaluates one batch at a time per game thread, so one
            // runner (two sessions per net, not four) is enough: the session
            // builds were most of a short match's time and memory.
            anyhow::ensure!(t2_widen == "current" || t2_widen == "static", "--t2-widen: current or static");
            let tt = minions_engine::two_tier::TwoTierParams { c_uct: t2_c, retries: t2_retries, widen_current: t2_widen == "current" };
            let player = |spec: &str, expansions: u64, power: f32, two: bool, n1: u32| -> anyhow::Result<minions_engine::stepsp::Player> {
                let params = minions_engine::bfs::Params { heads, power, ..Default::default() };
                let two_tier = two.then(|| tt.clone());
                let (spec, lattice) = lattice_play(spec, n1, lattice_inflight)?;
                Ok(minions_engine::stepsp::Player { eval: load_step_eval_runners(&spec, &ep, 1)?, expansions, params, two_tier, lattice, solver: false })
            };
            let pa = player(&a, expansions, power, two_tier_a, n1)?;
            // `--b x,y,...`: a match against each in turn (seed + its index),
            // A loaded once and each B freed after its match.
            for (k, b) in b.split(',').enumerate() {
                let t = std::time::Instant::now();
                // `greedy`: the heuristic solver (A's net only for bids).
                let pb = if b == "greedy" {
                    Ok(minions_engine::stepsp::Player { solver: true, lattice: None, two_tier: None, expansions: 0, ..pa.clone() })
                } else {
                    player(b, expansions_b.unwrap_or(expansions), power_b.unwrap_or(power), two_tier_b, n1_b.unwrap_or(n1))
                };
                let res = pb.and_then(|pb| {
                    minions_engine::stepsp::play_match(&pa, &pb, pairs, opening_max, opening_decay, seed + k as u64, threads)
                });
                let (score, wdl) = match res {
                    Ok(r) => r,
                    Err(e) => {
                        println!("A {a}\nB {b}\nError: {e:#}");
                        continue;
                    }
                };
                let n = (2 * pairs) as f64;
                let se = (score * (1.0 - score) / n).sqrt();
                println!(
                    "A {a}\nB {b}\nA score {score:.3} +- {se:.3} over {} games (A wins {}, B wins {}, draws {}) in {:.0}s",
                    2 * pairs, wdl[0], wdl[1], wdl[2], t.elapsed().as_secs_f64()
                );
            }
        }
        Cmd::StepsVsMcts { a, old, pairs, ep, seed, expansions, heads, sims, candidates, t_steps, resample_every, threads, equal_time, n1, lattice_inflight } => {
            let (a_eval, lattice) = lattice_play(&a, n1, lattice_inflight)?;
            let t = std::time::Instant::now();
            let steps = minions_engine::stepsp::Player {
                eval: load_step_eval(&a_eval, &ep)?,
                expansions,
                params: minions_engine::bfs::Params { heads, ..Default::default() },
                two_tier: None,
                lattice,
                solver: false,
            };
            let (sampler, params) = minions_engine::eval::match_search(sims, candidates, t_steps, 2 * pairs, seed);
            let params = playground_core::sampled_mcts::SampledParams { resample_every, ..params };
            let kind = minions_engine::net::BackendKind::parse("ort", &ep, true, false)?;
            let seat = minions_engine::eval::load_seat(std::path::Path::new(&old), &kind, sampler, None)?;
            let (score, wdl, (tw, tl)) = minions_engine::stepsp::play_vs_mcts(&steps, &seat, &params, pairs, seed, threads, equal_time)?;
            let n = (2 * pairs) as f64;
            let se = (score * (1.0 - score) / n).sqrt();
            println!(
                "steps {a} ({expansions} expansions) vs old MCTS {old} ({sims} sims, {candidates} candidates)\nsteps score {score:.3} +- {se:.3} over {} games (steps wins {}, MCTS wins {}, draws {}); mean turn of steps wins {tw:.1}, losses {tl:.1}; {:.0}s",
                2 * pairs, wdl[0], wdl[1], wdl[2], t.elapsed().as_secs_f64()
            );
        }
        Cmd::StepsCheck { a, b, ep, positions } => {
            use rand::{Rng, SeedableRng};
            let (ea, eb) = (load_step_eval(&a, &ep)?, load_step_eval(&b, &ep)?);
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
            let mut views = Vec::new();
            while views.len() < positions {
                let mut real = minions_engine::game::MinionsState::with_bid(rng.gen_range(-5..=10));
                for _ in 0..rng.gen_range(0..20) {
                    let mut s = minions_engine::steps::StepState::begin(&real, 1);
                    let f0 = s.flipped;
                    while !s.is_terminal() && s.flipped == f0 {
                        views.push(s.view());
                        let ch = s.children();
                        if ch.is_empty() {
                            break;
                        }
                        s.apply(ch[rng.gen_range(0..ch.len())]);
                    }
                    if s.is_terminal() {
                        break;
                    }
                    real = s.real_base();
                }
            }
            views.truncate(positions);
            let a = ea.eval(&views)?;
            let b = eb.eval(&views)?;
            let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let mean = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
            println!("{} positions: max |a - b| {d:.2e}, mean {mean:.2e}; a mean value {:.3}", a.len(), a.iter().sum::<f32>() / a.len() as f32);
        }
        Cmd::StepsTreeServe { dir, addr } => {
            minions_engine::tree_serve::serve(std::path::Path::new(&dir), &addr)?;
        }
        Cmd::StepsExactBench { corpus, budget, limit, labels_out, diag } => {
            let corpus = std::path::Path::new(&corpus);
            match diag {
                Some(labels) => minions_engine::steps::exact_diag(corpus, std::path::Path::new(&labels))?,
                None => minions_engine::steps::exact_bench(corpus, budget, limit, labels_out.as_deref().map(std::path::Path::new))?,
            }
        }
        Cmd::LatticeGen {
            out, onnx, ep, batcher, workers, games, n1, inflight, n2, c1, c2, pw_c, pw_alpha, retries,
            temp1, dir1_alpha, dir1_frac, temp2, dir2_alpha, dir2_frac, from_end_mean, from_end_median, start_turn_mean, secs,
            target_t2, seed, chunk_records, no_lazy_witness, lattice_ep, lattice_sessions, t1_min_visits, view_min_visits, t2_from_t1, distill_tau, distill_frac, sd_frac, sd_descents, sd_tau, walk_accumulated, sink_values, walk_inflight, walk_marginal, walk_cq, lview, energy_min_visits, pipeline, openings, open_workers, pool, refill_to,
        } => {
            minions_engine::steps::LAZY_WITNESS.store(!no_lazy_witness, std::sync::atomic::Ordering::Relaxed);
            // A directory is the lattice nets (tier1/tier2.onnx); a file a steps value net behind the bridge.
            let bridge: std::sync::Arc<dyn minions_engine::lattice::LatticeEval> = if std::path::Path::new(&onnx).is_dir() {
                std::sync::Arc::new(minions_engine::lattice_ort::OrtLatticeEval::load(std::path::Path::new(&onnx), &lattice_ep, true, lattice_sessions)?)
            } else {
                let eval = load_step_eval_runners(&onnx, &ep, batcher)?;
                std::sync::Arc::new(minions_engine::lattice::StepsBridge { eval })
            };
            let bridge: std::sync::Arc<dyn minions_engine::lattice::LatticeEval> = match &sink_values {
                Some(net) => std::sync::Arc::new(minions_engine::lattice::HybridEval {
                    expand: bridge,
                    values: std::sync::Arc::new(minions_engine::lattice::StepsBridge { eval: load_step_eval_runners(net, &ep, batcher)? }),
                }),
                None => bridge,
            };
            let cfg = minions_engine::latgen::GenConfig {
                out: out.into(),
                workers,
                games,
                n1,
                inflight1: inflight,
                t1: minions_engine::tier1::T1Params { c_puct: c1, pw_c, pw_alpha, retries, c2 },
                walk: minions_engine::lattice::WalkParams {
                    n2,
                    c2,
                    temp: temp2,
                    dir_alpha: dir2_alpha,
                    dir_frac: dir2_frac,
                    inflight: walk_inflight.unwrap_or(inflight),
                    sd_descents,
                    sd_tau,
                    fresh: !walk_accumulated,
                    first_step: !walk_marginal,
                    cq: walk_cq,
                    ..minions_engine::lattice::WalkParams::default()
                },
                temp1,
                dir1_alpha,
                dir1_frac,
                from_end_mean,
                from_end_median,
                start_turn_mean,
                secs,
                target_t2,
                seed,
                chunk_records,
                t1_min_visits,
                view_min_visits,
                t2_from_t1,
                distill_tau,
                distill_frac,
                sd_frac,
                lview,
                energy_min_visits,
                pipeline,
                openings,
                pool,
                refill_to,
                open_workers: if open_workers > 0 { open_workers } else { std::thread::available_parallelism().map_or(8, |n| n.get()) },
            };
            minions_engine::latgen::generate(&cfg, bridge)?;
        }
        Cmd::StepsSelfplay {
            out,
            old_net,
            onnx,
            ep,
            workers,
            sessions,
            batcher,
            heads,
            expand_threads,
            games,
            power,
            expansions,
            move_secs,
            min_visits,
            temp,
            dirichlet_alpha,
            dirichlet_eps,
            opening_max,
            opening_decay,
            search,
            select,
            record_frac,
            record_alpha,
            lazy_witness,
            target_records,
            chunk_records,
            seed,
            save_trees,
            resample_expansions,
            resample_heads,
            resample_threads,
            resample_search,
            t2_c,
            t2_retries,
            t2_widen,
        } => {
            use std::sync::Arc;
            anyhow::ensure!(search == "descent" || search == "bfs", "--search descent|bfs");
            anyhow::ensure!(save_trees.is_none() || games > 1, "--save-trees needs --games > 1");
            anyhow::ensure!(select == "ordinal" || select == "visits", "--select ordinal|visits");
            minions_engine::steps::LAZY_WITNESS.store(lazy_witness, std::sync::atomic::Ordering::Relaxed);
            let eval: Arc<dyn minions_engine::bfs::Evaluator> = match (old_net, onnx) {
                (Some(dir), None) => {
                    let dir = std::path::Path::new(&dir);
                    let cache = dir.join("trt_cache").to_string_lossy().into_owned();
                    let net = minions_engine::ort_net::OrtNet::new(dir, &ep, true, &cache)?;
                    Arc::new(minions_engine::bfs::OldNetEval { net: Box::new(net) })
                }
                (None, Some(path)) if path.ends_with(".bin") => {
                    // A step-trained MLP's raw weights: evaluate on the CPU.
                    Arc::new(minions_engine::bfs::CpuMlpEval::load(std::path::Path::new(&path))?)
                }
                (None, Some(path)) => {
                    let p = std::path::Path::new(&path);
                    let cache = p.with_extension("trt_cache").to_string_lossy().into_owned();
                    if batcher > 0 {
                        Arc::new(minions_engine::bfs::BatchedOnnxEval::load(p, &ep, true, &cache, batcher)?)
                    } else {
                        Arc::new(minions_engine::bfs::OnnxEval::load(p, &ep, true, &cache, if sessions == 0 { workers } else { sessions })?)
                    }
                }
                _ => anyhow::bail!("give exactly one of --old-net / --onnx"),
            };
            let cfg = minions_engine::stepsp::Config {
                workers,
                params: minions_engine::bfs::Params { heads, power, threads: expand_threads },
                expansions,
                move_secs,
                min_visits,
                temp,
                dirichlet_alpha,
                dirichlet_eps,
                opening_max,
                opening_decay,
                descent: search == "descent",
                ordinal: select == "ordinal",
                record_frac,
                record_alpha,
                target_records,
                chunk_records,
                seed,
                games,
                save_trees: save_trees.map(std::path::PathBuf::from),
                resample_expansions,
                resample_heads,
                resample_threads: if resample_threads == 0 { 2 * expand_threads } else { resample_threads },
                resample_two_tier: match resample_search.as_str() {
                    "best-first" => None,
                    "two-tier" => { anyhow::ensure!(t2_widen == "current" || t2_widen == "static", "--t2-widen: current or static"); Some(minions_engine::two_tier::TwoTierParams { c_uct: t2_c, retries: t2_retries, widen_current: t2_widen == "current" }) }
                    o => anyhow::bail!("--resample-search: best-first or two-tier, not {o}"),
                },
            };
            anyhow::ensure!(games == 1 || workers == 1 && expand_threads > 1, "--games needs --workers 1 and --expand-threads > 1");
            minions_engine::stepsp::run(cfg, eval, std::path::Path::new(&out))?;
        }
    }
    Ok(())
}

/// A step evaluator by spec: `old:<dir>` (the sampled-AZ champion's value head
/// on projected boards), `<path>.bin` (a step MLP on the CPU), or an ONNX graph.
fn load_step_eval(spec: &str, ep: &str) -> anyhow::Result<std::sync::Arc<dyn minions_engine::bfs::Evaluator>> {
    load_step_eval_runners(spec, ep, 2)
}

/// [`load_step_eval`] with `runners` GPU threads for an ONNX net (each holds
/// one session, i.e. one TensorRT engine load, per batch shape).
fn load_step_eval_runners(spec: &str, ep: &str, runners: usize) -> anyhow::Result<std::sync::Arc<dyn minions_engine::bfs::Evaluator>> {
    use std::sync::Arc;
    if spec == "material" {
        return Ok(Arc::new(minions_engine::bfs::MaterialEval));
    }
    if let Some(dir) = spec.strip_prefix("old:") {
        let dir = std::path::Path::new(dir);
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        let net = minions_engine::ort_net::OrtNet::new(dir, ep, true, &cache)?;
        return Ok(Arc::new(minions_engine::bfs::OldNetEval { net: Box::new(net) }));
    }
    let p = std::path::Path::new(spec);
    if spec.ends_with(".bin") {
        return Ok(Arc::new(minions_engine::bfs::CpuMlpEval::load(p)?));
    }
    let cache = p.with_extension("trt_cache").to_string_lossy().into_owned();
    Ok(Arc::new(minions_engine::bfs::BatchedOnnxEval::load(p, ep, true, &cache, runners)?))
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
    rows: usize,
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

    if rows > 0 {
        // Every graph at the same production-sized batch, one thread, no
        // contention: the floor each call costs before any queueing.
        let planes_r = repeat_trunk(&planes, rows);
        let h_r = repeat_trunk(&tr.h, rows);
        let tile = |src: &[f32], stride: usize| -> Vec<f32> {
            (0..rows)
                .flat_map(|i| src[(i % k) * stride..(i % k + 1) * stride].iter().copied())
                .collect()
        };
        let (decl_r, edecl_r) = (tile(&decl, stride), tile(&edecl, estride));
        // `MINIONS_BENCH_THREADS=n`: n threads share the net, as self-play's
        // workers do, and the aggregate rate is what counts -- whether a
        // contended call is bound by its rows or by its count.
        let threads: usize = std::env::var("MINIONS_BENCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        if threads > 1 {
            let (decl_r, edecl_r) = (tile(&decl, stride), tile(&edecl, estride));
            net.trunk(&planes_r, rows)?;
            net.diffusion(&h_r, &decl_r, rows)?;
            net.energy(&h_r, &edecl_r, rows)?;
            // One wave's worth of calls in production's proportions:
            // 3 trunk : 6 diffusion : 2 energy.
            let mix = |i: u64| -> anyhow::Result<()> {
                match i % 11 {
                    0..=2 => net.trunk(&planes_r, rows).map(|_| ()),
                    3..=8 => net.diffusion(&h_r, &decl_r, rows).map(|_| ()),
                    _ => net.energy(&h_r, &edecl_r, rows).map(|_| ()),
                }
            };
            for (name, only_diff) in [("diffusion", true), ("mix", false)] {
                let deadline = Instant::now() + Duration::from_secs(secs);
                let start = Instant::now();
                let calls: u64 = std::thread::scope(|sc| {
                    let hs: Vec<_> = (0..threads)
                        .map(|t| {
                            let (h_r, decl_r, net) = (&h_r, &decl_r, &*net);
                            let mix = &mix;
                            sc.spawn(move || {
                                let mut n = 0u64;
                                while Instant::now() < deadline {
                                    if only_diff {
                                        net.diffusion(h_r, decl_r, rows).expect("diffusion");
                                    } else {
                                        mix(n + t as u64).expect("mix");
                                    }
                                    n += 1;
                                }
                                n
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).sum()
                });
                let s = start.elapsed().as_secs_f64();
                println!(
                    "  {threads} threads {name:<10} {:>8.0} calls/s  {:>9.0} rows/s",
                    calls as f64 / s,
                    calls as f64 * rows as f64 / s
                );
            }
            if let Some(r) = minions_engine::ort_net::call_stats::take_report(1.0) {
                println!("{r}");
            }
            return Ok(());
        }
        println!("timing at {rows} rows, {secs}s each:");
        timed("trunk", &|| net.trunk(&planes_r, rows).map(|_| ()))?;
        timed("diffusion", &|| {
            net.diffusion(&h_r, &decl_r, rows).map(|_| ())
        })?;
        timed("energy", &|| net.energy(&h_r, &edecl_r, rows).map(|_| ()))?;
        // A whole T-step pass as the sampler runs it: one `diffusion_pass`,
        // then `t_steps` steps against it.
        let per = timed("dif pass", &|| {
            let mut p = net.diffusion_pass(&h_r, rows)?;
            for _ in 0..t_steps {
                p.step(&decl_r)?;
            }
            Ok(())
        })?;
        println!(
            "  = {:.3} ms per diffusion step within a {t_steps}-step pass",
            per * 1e3 / t_steps as f64
        );
        for p in net.end_profiling() {
            println!("profile: {p}");
        }
        if let Some(r) = minions_engine::ort_net::call_stats::take_report(1.0) {
            println!("{r}");
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
