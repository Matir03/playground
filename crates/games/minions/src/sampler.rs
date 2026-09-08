//! The sampler: proposes candidate turns for [`playground_core::sampled_mcts`].
//!
//! This is the piece the whole design turns on. A mean-field proposal — sample
//! each cell's declaration independently — converges under cross-entropy to the
//! *marginals* of the search policy, and with ~25 units the product of marginals
//! puts almost no mass on the joint, because good turns are coordinated
//! configurations. Scoring afterwards cannot repair that: reweighting only
//! redistributes mass over what was already sampled. It is a *support* failure.
//!
//! So candidates come from an any-order masked diffusion over the ~80 live
//! declaration variables. Start fully masked; each round, predict every masked
//! variable, sample, and commit the most confident fraction. Once a round
//! commits "that skeleton dies", later rounds condition on it and move units
//! adjacent — the correlation is learned rather than assumed away.
//!
//! Three properties fall out that matter here:
//!
//! - **`T` is a dial.** `T = 1` is mean-field, `T = n` commits one variable at a
//!   time so the product of its conditionals is a proper normalized density —
//!   exact `q(a)`, and therefore an honest effective-sample-size diagnostic, from
//!   the same weights that run fast in self-play at `T ≈ 6`.
//! - **Constraints are exact, not repaired.** Destinations commit progressively,
//!   so a claimed hex is masked out of later rounds and collisions are impossible
//!   by construction rather than fixed up (and biased) by the realizer.
//! - **Partial resampling is free**, which is what progressive widening wants.
//!
//! Independent of the learned proposal, a slice of every candidate set is drawn
//! uniformly at random ([`SamplerConfig::random_fraction`]). That is support
//! coverage that does not depend on the sampler being any good, and it is the
//! structural answer to self-reinforcing support collapse. Such a candidate is
//! only *kept* by the training target if it earns organic visits — see
//! `SampledSearchResult::policy_target`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use playground_core::sampled_mcts::Expansion;
use playground_core::{Game, SampledModel};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::bitboard::{self, bit, Bb, FULL};
use crate::game::{Minions, MinionsState};
use crate::map::{NUM_CELLS, NO_CELL};
use crate::net::{Backend, DECL_LOGITS};
use crate::turn::{
    realize, LiveMask, MinionsTurns, Partial, TurnSpec, C_MOVE, C_SPAWN, C_TARGET, DECL_PLANES,
    MOVE_DEST, N_OFFSETS, N_SPAWNS, N_TARGETS, TARGET_UNSUMMON,
};
use crate::units::BUYABLE;

// Logit layout within a cell's DECL_LOGITS block.
const OFF_MOVES: usize = 0;
const OFF_TARGETS: usize = N_OFFSETS;
const OFF_SPAWNS: usize = N_OFFSETS + N_TARGETS;

#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    /// Diffusion rounds. 1 = mean-field, `n` = exact autoregressive (and exact
    /// density). ~6 in self-play.
    pub t_steps: usize,
    /// Rounds for *fast* (playout-cap) searches, which advance the game but
    /// produce no training data.
    ///
    /// Diffusion is 61% of expansion time and its cost is linear in `T`, so
    /// spending full sampler quality on moves that are never learned from is
    /// pure waste — the same argument that justifies playout-cap randomization
    /// in the first place.
    pub t_steps_fast: usize,
    /// Candidates proposed per node expansion.
    pub k_candidates: usize,
    /// Candidates for *fast* (playout-cap) searches.
    ///
    /// Diffusion cost is linear in `K` just as it is in `T`, and the two
    /// multiply. Cutting it here is not only cheaper but better shaped: a fast
    /// search spends `fast_sims` visits over `K` candidates, so at K=32 and 50
    /// sims each candidate is visited ~1.6 times and no tree forms at all.
    /// Nothing in the training target depends on this — fast searches are never
    /// recorded — so the only cost is the strength of the moves actually played.
    pub k_candidates_fast: usize,
    /// Share of candidates drawn uniformly instead of from the net.
    pub random_fraction: f32,
    /// Softmax temperature on the declaration logits.
    pub temperature: f32,
    /// Score candidates with the energy head. Off = uniform priors, which is the
    /// P1 configuration before `f` has been trained.
    pub use_energy: bool,
    /// Pad every batched call up to this many positions, then discard the
    /// padding. 0 leaves the batch at whatever the caller passed.
    ///
    /// TensorRT builds a separate engine per input shape, so a caller whose
    /// batch shrinks — eval, where games finish at different plies — would set
    /// it rebuilding mid-match. Self-play pads before it calls; padding here
    /// gives every other caller the same guarantee.
    pub pad_to: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            t_steps: 4,
            t_steps_fast: 2,
            k_candidates: 12,
            k_candidates_fast: 6,
            random_fraction: 0.25,
            temperature: 1.0,
            use_energy: true,
            pad_to: 0,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    /// Candidates to propose for the given search mode.
    ///
    /// Clamped to `k_candidates`: a fast search that proposed *more* candidates
    /// than a full one would be backwards, and is easy to reach by lowering
    /// `--candidates` alone.
    pub fn k_for(&self, full: bool) -> usize {
        if full { self.k_candidates } else { self.k_candidates_fast.min(self.k_candidates) }
    }
}

/// Per-position masks the chain loop needs, computed once per expansion rather
/// than once per chain.
struct GameCtx {
    live: LiveMask,
    move_opts: Vec<u64>,
    target_opts: Vec<u8>,
    spawn_opts: u8,
    n_live: usize,
}

/// Score one still-masked variable and push its Gumbel draw onto `picks`.
///
/// Generic over `allowed` rather than taking `&dyn Fn`: this is called for every
/// live variable of every row on every diffusion step (~1.5k calls per batch at
/// the production shape), and each call probed the predicate once per class in
/// *two* separate passes. Monomorphising lets the bounds check and the option
/// mask fold into the loop, and caching the scaled logits means the second pass
/// no longer redoes the division or the predicate.
#[allow(clippy::too_many_arguments)]
fn consider<F: Fn(usize) -> bool>(
    chain: &Partial,
    cell: u8,
    field: u8,
    off: usize,
    n_cls: usize,
    allowed: F,
    logits: &[f32],
    temp: f32,
    picks: &mut Vec<(u8, u8, u8)>,
    rng: &mut ChaCha8Rng,
) {
    if chain.is_committed(cell, field) {
        return;
    }
    let row = cell as usize * DECL_LOGITS + off;
    let (mut best_v, mut best_g) = (0usize, f32::NEG_INFINITY);
    let mut max_l = f32::NEG_INFINITY;
    for v in 0..n_cls {
        if !allowed(v) {
            continue;
        }
        let l = logits[row + v] / temp;
        if l > max_l {
            max_l = l;
        }
        // Gumbel-max: the argmax of (logit + Gumbel noise) is an exact
        // categorical draw, and taking the top few gives sampling without
        // replacement.
        let u: f32 = rng.gen_range(1e-9f32..1.0);
        let gm = l - (-u.ln()).ln();
        if gm > best_g {
            best_g = gm;
            best_v = v;
        }
    }
    if max_l == f32::NEG_INFINITY {
        return; // nothing legal here
    }
    picks.push((cell, field, best_v as u8));
}

/// One diffusion round for one chain: sample every still-masked variable, then
/// commit the most confident ones up to the schedule's target.
fn commit_round(
    chain: &mut Partial,
    claimed: &mut Bb,
    cx: &GameCtx,
    logits: &[f32],
    want: usize,
    temperature: f32,
    rng: &mut ChaCha8Rng,
    // Scratch, owned by the caller: this runs once per row per step, so a
    // fresh `Vec` here was ~1.5k allocate-grow-free cycles per batch.
    picks: &mut Vec<(u8, u8, u8)>,
) {
    let temp = temperature.max(1e-3);
    picks.clear();

    for cell in bitboard::cells(cx.live.moves) {
        let opts = cx.move_opts[cell as usize];
        let taken = *claimed;
        let allow = |v: usize| {
            if opts & (1 << v) == 0 {
                return false;
            }
            let d = MOVE_DEST[cell as usize][v];
            // Offset 0 is "stay" and never claims a hex.
            v == 0 || (d != NO_CELL && !bitboard::has(taken, d))
        };
        consider(chain, cell, C_MOVE, OFF_MOVES, N_OFFSETS, allow, logits, temp, picks, rng);
    }
    for cell in bitboard::cells(cx.live.targets) {
        let opts = cx.target_opts[cell as usize];
        let allow = |v: usize| opts & (1 << v) != 0;
        consider(chain, cell, C_TARGET, OFF_TARGETS, N_TARGETS, allow, logits, temp, picks, rng);
    }
    for cell in bitboard::cells(cx.live.spawns) {
        let so = cx.spawn_opts;
        let allow = |v: usize| so & (1 << v) != 0;
        consider(chain, cell, C_SPAWN, OFF_SPAWNS, N_SPAWNS, allow, logits, temp, picks, rng);
    }

    let already = chain.committed.iter().map(|c| c.count_ones() as usize).sum::<usize>();
    let take = want.saturating_sub(already).max(1).min(picks.len());
    // Commit a *uniformly random* subset, not the most confident ones.
    //
    // Confidence ordering (MaskGIT's heuristic) makes the commit order depend on
    // the sampled values, which has two costs. It biases samples toward
    // high-probability configurations — mode-seeking, so the tails of the search
    // policy are under-sampled. And it makes the proposal density `q(a)`
    // intractable: recovering it would mean summing over every order that could
    // have produced the sample, so neither an exact ESS nor a GFlowNet
    // `log P_F` is computable. A value-independent random order makes the chain
    // an any-order autoregressive model whose density is a plain product of
    // conditionals.
    //
    // Partial Fisher-Yates: only the first `take` slots need to be settled.
    for i in 0..take {
        let j = i + (rng.gen::<u64>() as usize) % (picks.len() - i);
        picks.swap(i, j);
    }
    for &(cell, field, value) in picks.iter().take(take) {
        if field == C_MOVE && value != 0 {
            // Picks within a round are all chosen before any is applied, so two
            // can name the same hex; the loser is left uncommitted for a later
            // round.
            let d = MOVE_DEST[cell as usize][value as usize];
            if d == NO_CELL || bitboard::has(*claimed, d) {
                continue;
            }
            *claimed |= bit(d);
        }
        chain.commit(cell, field, value);
    }
}

/// Where the time in an expansion actually goes.
///
/// Guessing at this was already wrong once — the isolated forward benchmark said
/// TensorRT was 1.6x faster than the CUDA EP, but end to end it bought 10%,
/// which only makes sense if the GPU is no longer the bottleneck. These counters
/// say which part is.
#[derive(Default, Debug)]
pub struct Phases {
    pub features_ns: AtomicU64,
    pub trunk_ns: AtomicU64,
    pub decl_write_ns: AtomicU64,
    pub diffusion_ns: AtomicU64,
    pub commit_ns: AtomicU64,
    pub realize_ns: AtomicU64,
    pub energy_ns: AtomicU64,
    /// Drawing forced-exploration candidates by playing a random legal turn.
    ///
    /// Timed separately because it was invisible before: it sits outside every
    /// other phase, so the percentages below still summed to 100 while wall time
    /// grew, and the profile silently understated it. It is also strongly
    /// position-dependent — 28µs at mid-game against 83µs late, since both the
    /// `legal_moves` enumeration and the playout length scale with unit count.
    pub playout_ns: AtomicU64,
}

impl Phases {
    fn add(counter: &AtomicU64, t: std::time::Instant) {
        counter.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Phase shares **plus** the fraction of worker time no counter covers.
    ///
    /// The shares below are normalised by the sum of the counters, not by wall
    /// clock — so anything uninstrumented is invisible *and* does not dilute the
    /// rest. The tree itself is uninstrumented: PUCT selection, backup, node
    /// allocation, the `state_key` dedupe in `provide`, and the `G::apply` calls
    /// that dedupe performs all sit outside every phase. `unaccounted` is what
    /// makes that visible; read the phase shares as "of measured work", and this
    /// as "how much of the worker's time we are not measuring at all".
    pub fn report_wall(&self, secs: f64, workers: usize) -> String {
        let body = self.report();
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        let measured_ms = g(&self.features_ns)
            + g(&self.trunk_ns)
            + g(&self.decl_write_ns)
            + g(&self.diffusion_ns)
            + g(&self.commit_ns)
            + g(&self.realize_ns)
            + g(&self.energy_ns)
            + g(&self.playout_ns);
        let worker_ms = secs * 1000.0 * workers.max(1) as f64;
        if worker_ms <= 0.0 {
            return body;
        }
        let unaccounted = (1.0 - measured_ms / worker_ms).max(0.0) * 100.0;
        format!("{body}  [unaccounted {unaccounted:.0}% — search/backup/dedupe]")
    }

    pub fn report(&self) -> String {
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        let (f, t, w, d, c, r, e, p) = (
            g(&self.features_ns), g(&self.trunk_ns), g(&self.decl_write_ns),
            g(&self.diffusion_ns), g(&self.commit_ns), g(&self.realize_ns),
            g(&self.energy_ns), g(&self.playout_ns),
        );
        let total = (f + t + w + d + c + r + e + p).max(1e-9);
        let pct = |x: f64| 100.0 * x / total;
        format!(
            "features {:.0}% trunk {:.0}% decl-write {:.0}% diffusion {:.0}% \
commit {:.0}% realize {:.0}% playout {:.0}% energy {:.0}%  (GPU {:.0}%, CPU {:.0}%)",
            pct(f), pct(t), pct(w), pct(d), pct(c), pct(r), pct(p), pct(e),
            pct(t + d + e), pct(f + w + c + r + p),
        )
    }
}

pub struct MinionsModel {
    /// Candle or ORT — the sampler does not care which; see [`Backend`].
    pub net: Box<dyn Backend>,
    pub cfg: SamplerConfig,
    pub phases: Phases,
    /// Rounds the next `expand_batch` should use. The actor sets this to match
    /// the search mode of the batch it has gathered; there is a single GPU
    /// consumer, so batches never interleave.
    /// Seed for per-batch RNGs. Each expansion derives its own stream from this
    /// plus the caller's counter, so concurrent workers never share a generator
    /// — a shared `Mutex<ChaCha8Rng>` would both serialise them and make each
    /// worker's draws depend on the other's interleaving.
    seed: u64,
    /// Distinguishes successive `expand_batch` calls so each draws its own RNG
    /// stream.
    ///
    /// Self-play passes an explicit counter to [`Self::expand_batch_mode`], but
    /// the `SampledModel` trait method has no such argument and used to hardcode
    /// `0` — which made `seed ^ 0 * K` a *constant*, so every expansion on the
    /// `search()` path drew the identical stream. Given a position the sampler
    /// then proposed exactly the same candidate turns every time, and with
    /// argmax selection the whole game replayed move for move. That is why the
    /// interactive net looked deterministic: the diffusion sampler was random,
    /// but it was handed the same seed on every call.
    expand_seq: AtomicU64,
}

impl MinionsModel {
    pub fn new(net: Box<dyn Backend>, cfg: SamplerConfig) -> Self {
        let seed = cfg.seed;
        Self { net, cfg, phases: Phases::default(), seed, expand_seq: AtomicU64::new(0) }
    }

    /// Legal move offsets per cell, as a 64-bit mask over [`crate::turn::OFFSETS`].
    ///
    /// Masking here rather than letting the realizer drop things is most of why
    /// declarations are realizable at all: the sampler never proposes a
    /// destination the unit cannot reach.
    fn move_options(state: &MinionsState, live: &LiveMask) -> Vec<u64> {
        let enemy = live.targets;
        let ground = FULL & !enemy;
        let mut out = vec![0u64; NUM_CELLS];
        for c in bitboard::cells(live.moves) {
            let Some((_, unit)) = state.piece_at(c) else { continue };
            let st = unit.stats();
            let pass = if st.flying { FULL } else { ground };
            // Empty *or* friendly-occupied: a hex whose occupant is itself
            // declared to move away is a legal destination, and the realizer's
            // fixed point sorts out the ordering. Enemy-held hexes are excluded
            // — a flier may path over one but never land on it.
            let reach = bitboard::reachable(bit(c), st.speed, pass) & !enemy & FULL;
            let mut m = 1u64; // offset 0 (stay) is always available
            for (k, &d) in MOVE_DEST[c as usize].iter().enumerate().skip(1) {
                if d != NO_CELL && bitboard::has(reach, d) {
                    m |= 1 << k;
                }
            }
            out[c as usize] = m;
        }
        out
    }

    /// Target classes available per enemy cell. `unsummon` only removes a
    /// non-persistent unit, so proposing it on a persistent one would be a wasted
    /// declaration.
    fn target_options(state: &MinionsState, live: &LiveMask) -> Vec<u8> {
        let mut out = vec![0u8; NUM_CELLS];
        for c in bitboard::cells(live.targets) {
            let Some((_, unit)) = state.piece_at(c) else { continue };
            let mut m = 0b011; // stay, kill
            if !unit.stats().persistent {
                m |= 1 << TARGET_UNSUMMON;
            }
            out[c as usize] = m;
        }
        out
    }

    /// Spawn classes available per spawn cell: "nothing", plus any unit already
    /// in reinforcements or affordable right now. Affordability is a shared
    /// budget, so this is a per-cell over-approximation and the realizer drops
    /// what the purse cannot actually cover.
    fn spawn_options(state: &MinionsState) -> u8 {
        let me = state.side as usize;
        let mut m = 1u8; // "nothing"
        for u in BUYABLE {
            let slot = u.reinforcement_slot().expect("buyable");
            if state.reinforcements[me][slot] > 0 || u.stats().cost <= state.money[me] {
                m |= 1 << (u as u8);
            }
        }
        m
    }

    /// Number of variables that should be committed after round `t` of `T`,
    /// on MaskGIT's cosine schedule.
    fn committed_after(t: usize, total_steps: usize, n: usize) -> usize {
        if t >= total_steps {
            return n;
        }
        let frac = 1.0 - (std::f64::consts::FRAC_PI_2 * t as f64 / total_steps as f64).cos();
        ((n as f64) * frac).ceil() as usize
    }

    /// Everything about a position the chain loop needs, computed once.
    fn context(state: &MinionsState) -> GameCtx {
        let live = LiveMask::of(state);
        GameCtx {
            move_opts: Self::move_options(state, &live),
            target_opts: Self::target_options(state, &live),
            spawn_opts: Self::spawn_options(state),
            n_live: live.count(),
            live,
        }
    }

    /// Run `k` independent diffusion chains for each of `states`, batching every
    /// round's forward across *all* of them.
    ///
    /// This is the throughput lever. Per round the batch is `games x k` rather
    /// than `k`, and the trunk is shared: measured on this net, the diffusion
    /// head runs 1,285 rows/s at batch 1 and 106,866/s at batch 512.
    fn sample_declarations_batch(
        &self,
        states: &[MinionsState],
        h_all: &[f32],
        k: usize,
        full: bool,
        rng: &mut ChaCha8Rng,
    ) -> anyhow::Result<Vec<Vec<TurnSpec>>> {
        let g = states.len();
        if g == 0 || k == 0 {
            return Ok(vec![Vec::new(); g]);
        }
        let ctx: Vec<GameCtx> = states.iter().map(Self::context).collect();
        let hstride = self.net.channels() * NUM_CELLS;
        let stride = DECL_PLANES * NUM_CELLS;
        let rows = g * k;

        let mut chains = vec![Partial::new(); rows];
        let mut claimed = vec![0 as Bb; rows];
        // One RNG per chain rather than one shared. Chains are independent, so
        // this both makes the commit step parallelisable and keeps it
        // deterministic — a shared RNG would make the result depend on thread
        // interleaving.
        let mut rngs: Vec<ChaCha8Rng> = (0..rows)
            .map(|i| ChaCha8Rng::seed_from_u64(rng.gen::<u64>() ^ i as u64))
            .collect();

        // Each game's trunk features repeated across its k chains.
        let mut h_b = Vec::with_capacity(rows * hstride);
        for gi in 0..g {
            let src = &h_all[gi * hstride..(gi + 1) * hstride];
            for _ in 0..k {
                h_b.extend_from_slice(src);
            }
        }
        let mut decl = vec![0f32; rows * stride];
        let steps = if full { self.cfg.t_steps } else { self.cfg.t_steps_fast }.max(1);

        // Serial within a batch, on purpose.
        //
        // Overlapping a shard's commit with the other shard's forward was tried
        // and measured **catastrophic**: 1605 -> 52 expansions/s. The cause is
        // the invariant the energy pass documents a few lines below — TensorRT
        // builds and caches an engine per input *shape*, so halving the row
        // count introduced new shapes and set it rebuilding. The CPU/GPU overlap
        // this was reaching for is obtained instead by running several pool
        // workers (`ActorConfig::workers`), which keeps every batch at the same
        // fixed width.
        let temp = self.cfg.temperature;
        let width = NUM_CELLS * DECL_LOGITS;
        // `h_b` is identical on every step, so hand it to the backend once and
        // let it keep the features resident on the device for the whole pass.
        let mut pass = self.net.diffusion_pass(&h_b, rows)?;
        // Reused by every `commit_round` call across all rows and steps.
        let mut picks: Vec<(u8, u8, u8)> = Vec::with_capacity(NUM_CELLS);
        for t in 1..=steps {
            let clk = Instant::now();
            for (i, c) in chains.iter().enumerate() {
                c.write_planes(&mut decl[i * stride..(i + 1) * stride]);
            }
            Phases::add(&self.phases.decl_write_ns, clk);

            let clk = Instant::now();
            let logits = pass.step(&decl)?;
            Phases::add(&self.phases.diffusion_ns, clk);

            let clk = Instant::now();
            for row in 0..rows {
                let cx = &ctx[row / k];
                if cx.n_live == 0 {
                    continue;
                }
                let want = Self::committed_after(t, steps, cx.n_live);
                commit_round(
                    &mut chains[row],
                    &mut claimed[row],
                    cx,
                    &logits[row * width..(row + 1) * width],
                    want,
                    temp,
                    &mut rngs[row],
                    &mut picks,
                );
            }
            Phases::add(&self.phases.commit_ns, clk);
        }
        Ok(chains
            .chunks(k)
            .map(|c| c.iter().map(|p| p.spec).collect())
            .collect())
    }

    /// Single-position shim over [`Self::sample_declarations_batch`], used by
    /// the tests. Self-play always goes through the batched path.
    #[cfg(test)]
    fn sample_declarations(
        &self,
        state: &MinionsState,
        h: &[f32],
        k: usize,
        rng: &mut ChaCha8Rng,
    ) -> anyhow::Result<Vec<TurnSpec>> {
        Ok(self
            .sample_declarations_batch(std::slice::from_ref(state), h, k, true, rng)?
            .pop()
            .unwrap_or_default())
    }

}

impl MinionsModel {
    /// Realize a game's declarations, merge by resulting position, and score.
    /// Shared by the single and batched paths so they cannot drift.
    /// Realize each declaration and merge those landing on the same board,
    /// carrying whether the survivor came from forced exploration.
    ///
    /// Provenance has to survive this step. Two declarations can realize to one
    /// position, and if a playout-drawn candidate merges into a model-drawn one
    /// the result still deserves its guaranteed look — so the flag is OR-ed
    /// rather than taken from whichever happened to arrive first.
    fn finish_candidates(
        &self,
        state: &MinionsState,
        declared: Vec<(TurnSpec, bool)>,
    ) -> Vec<(TurnSpec, bool)> {
        // Distinct declarations routinely converge on one board; unmerged they
        // would fragment the visit counts they are supposed to concentrate.
        let mut seen: std::collections::HashMap<Vec<u8>, usize> = Default::default();
        let mut specs: Vec<(TurnSpec, bool)> = Vec::with_capacity(declared.len());
        for (spec, explore) in declared {
            let r = realize(state, &spec);
            let key = Minions::encode_state(&r.next);
            match seen.get(&key) {
                Some(&i) => specs[i].1 |= explore,
                None => {
                    seen.insert(key, specs.len());
                    specs.push((r.spec, explore));
                }
            }
        }
        specs
    }

    /// Candidates carry their **raw energies**; the search turns them into
    /// priors.
    ///
    /// This used to return `exp(f - max)` over the batch it was handed, on the
    /// reasoning that the search renormalises anyway so the constant cancels.
    /// It does not cancel once generation is chunked: with `pw_chunk = 1` this
    /// sees a single candidate, `max` is that candidate's own energy, and the
    /// result is `exp(0) = 1` no matter what the head said. Every candidate
    /// then arrived at the node with the same prior, and normalising over the
    /// grown set left priors that halved with arrival order — 0.5, 0.25,
    /// 0.125 — carrying no policy information at all.
    ///
    /// Kept as a named function rather than inlined so the invariant has
    /// somewhere to live: whatever the head emits reaches the search unscaled.
    fn priors_from(&self, energies: &[f32]) -> Vec<f32> {
        energies.to_vec()
    }
}

impl SampledModel<MinionsTurns> for MinionsModel {
    /// Expand many positions at once: one trunk forward for all of them, one
    /// diffusion forward per round across every chain of every game, and one
    /// energy forward over every candidate of every game.
    fn expand_batch(
        &self,
        states: &[MinionsState],
        k: usize,
    ) -> anyhow::Result<Vec<Expansion<MinionsTurns>>> {
        // A fresh stream per call. Still reproducible for a given `cfg.seed`:
        // the sequence of expansions is what varies, not the starting entropy.
        let seq = self.expand_seq.fetch_add(1, Ordering::Relaxed);
        self.expand_batch_mode(states, k, true, seq)
    }

    fn expand(&self, state: &MinionsState, k: usize) -> anyhow::Result<Expansion<MinionsTurns>> {
        Ok(self
            .expand_batch(std::slice::from_ref(state), k)?
            .pop()
            .expect("one state in, one expansion out"))
    }
}

impl MinionsModel {
    /// Expand a batch, told explicitly which kind of search it serves and which
    /// RNG stream to draw from.
    ///
    /// Both used to be implicit: the diffusion depth and the explore flag lived
    /// in atomics set just before the call, and the RNG behind a shared mutex.
    /// That was sound only while a single thread consumed the GPU. With
    /// concurrent workers the atomics race — one worker can flip the mode out
    /// from under another mid-expansion — and a shared generator both serialises
    /// them and makes each worker's draws depend on the other's interleaving.
    /// Passing both makes an expansion a pure function of its arguments.
    pub fn expand_batch_mode(
        &self,
        states: &[MinionsState],
        k: usize,
        full: bool,
        batch_seq: u64,
    ) -> anyhow::Result<Vec<Expansion<MinionsTurns>>> {
        let real = states.len();
        if real == 0 {
            return Ok(Vec::new());
        }
        // Widen to a fixed batch before touching the GPU, so the tensor shape
        // does not follow the caller's live-game count.
        let padded: Vec<MinionsState>;
        let states: &[MinionsState] = if self.cfg.pad_to > real {
            padded = states
                .iter()
                .cloned()
                .chain(std::iter::repeat_with(|| states[real - 1].clone()))
                .take(self.cfg.pad_to)
                .collect();
            &padded
        } else {
            states
        };
        let g = states.len();
        // Plan from the mover's point of view. Both seats play the same game
        // under a 180° rotation (`MinionsState::canonical`), so canonicalising
        // here lets one set of weights serve both instead of learning each
        // orientation separately. The declarations that come back are in this
        // frame, and are rotated onto the real board before they are returned.
        let flipped: Vec<bool> = states.iter().map(|s| s.side == 1).collect();
        let canon: Vec<MinionsState> = states.iter().map(|s| s.canonical()).collect();
        let states: &[MinionsState] = &canon;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed ^ batch_seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let pstride = crate::game::PLANES * NUM_CELLS;

        // ── one trunk forward for every position ──────────────────────────────
        let t = Instant::now();
        let mut planes = vec![0f32; g * pstride];
        for (i, s) in states.iter().enumerate() {
            for (o, b) in planes[i * pstride..(i + 1) * pstride]
                .iter_mut()
                .zip(Minions::to_features(s))
            {
                *o = b as f32;
            }
        }
        Phases::add(&self.phases.features_ns, t);

        let t = Instant::now();
        let tr = self.net.trunk(&planes, g)?;
        Phases::add(&self.phases.trunk_ns, t);
        let values = tr.value;
        let hstride = self.net.channels() * NUM_CELLS;

        // `k == 0` is a value-only expansion: run the trunk, skip the heads.
        //
        // A node's first visit needs a value and nothing else — it will not
        // choose a move until something descends through it, and 77% of nodes
        // never are. Generating a candidate there was the whole of the residual
        // waste after chunking (43.4% of candidates, one per frontier node).
        if k == 0 {
            let out = (0..real).map(|gi| (values.get(gi).copied().unwrap_or(0.0), Vec::new()));
            return Ok(out.collect());
        }
        let k = k.max(1);
        // Forced exploration keeps its *expected* share even when candidates are
        // generated one at a time. Rounding here is what broke it: chunked
        // generation calls with `k = 1`, and `round(1 * 0.25) = 0`, so every
        // call took the diffusion path and injection silently vanished — the
        // mechanism that took kill fidelity from 3.8% to 99.6%. Taking the
        // fractional part as a coin keeps the long-run fraction exact at any
        // chunk size.
        let k_rand = if full {
            let want = (k as f32) * self.cfg.random_fraction;
            let floor = want.floor();
            let extra = if rng.gen::<f32>() < (want - floor) { 1 } else { 0 };
            floor as usize + extra
        } else {
            0
        };
        let k_diff = k.saturating_sub(k_rand);

        // ── diffusion, batched across every chain of every game ───────────────
        let per_game_specs = self.sample_declarations_batch(states, &tr.h, k_diff, full, &mut rng)?;
        let mut per_game: Vec<Vec<(TurnSpec, bool)>> = per_game_specs
            .into_iter()
            .map(|v| v.into_iter().map(|s| (s, false)).collect())
            .collect();
        // Forced exploration: a slice of every candidate set is drawn without the
        // net, by *playing* a random legal turn rather than declaring one.
        //
        // `random_plausible` declares kill/unsummon per enemy at random, and only
        // ~3.8% of those survive realization (measured by `soak`), so the injected
        // candidates were effectively move-only and left combat — the coordinated
        // play the joint sampler exists for — uncovered. Playouts realize ~99.6%
        // of their kills and, despite declaring 22x fewer, land *more* of them in
        // absolute terms. This is the support-coverage guarantee that does not
        // depend on the sampler being any good.
        let t_play = Instant::now();
        for (gi, decls) in per_game.iter_mut().enumerate() {
            for _ in 0..k_rand {
                decls.push((TurnSpec::random_playout(&states[gi], &mut rng), true));
            }
        }
        Phases::add(&self.phases.playout_ns, t_play);

        // ── realize and merge, per game ───────────────────────────────────────
        let t = Instant::now();
        let specs: Vec<Vec<(TurnSpec, bool)>> = per_game
            .into_iter()
            .enumerate()
            .map(|(gi, d)| self.finish_candidates(&states[gi], d))
            .collect();
        Phases::add(&self.phases.realize_ns, t);

        // ── one energy forward over every candidate of every game ─────────────
        let priors: Vec<Vec<f32>> = if self.cfg.use_energy {
            let total: usize = specs.iter().map(|s| s.len()).sum();
            let dstride = crate::turn::ENERGY_PLANES * NUM_CELLS;
            let mut decl = vec![0f32; total * dstride];
            let mut h_b = Vec::with_capacity(total * hstride);
            let mut at = 0usize;
            for (gi, list) in specs.iter().enumerate() {
                let live = LiveMask::of(&states[gi]);
                let src = &tr.h[gi * hstride..(gi + 1) * hstride];
                for (spec, _) in list {
                    Partial::fully_committed(spec, &live)
                        .write_energy_planes(&mut decl[at * dstride..(at + 1) * dstride]);
                    h_b.extend_from_slice(src);
                    at += 1;
                }
            }
            // Pad to a fixed `g * k` rows. Candidate counts vary after merging,
            // and TensorRT builds (and caches) a separate engine per input
            // shape — a ragged batch would rebuild constantly. The padding is a
            // few percent of wasted work against seconds per rebuild.
            let padded = g * k;
            let energies = if total == 0 {
                Vec::new()
            } else {
                decl.resize(padded * dstride, 0.0);
                h_b.resize(padded * hstride, 0.0);
                let clk = Instant::now();
                let mut e = self.net.energy(&h_b, &decl, padded)?;
                Phases::add(&self.phases.energy_ns, clk);
                e.truncate(total);
                e
            };
            let mut out = Vec::with_capacity(g);
            let mut off = 0usize;
            for list in &specs {
                out.push(self.priors_from(&energies[off..off + list.len()]));
                off += list.len();
            }
            out
        } else {
            specs.iter().map(|l| vec![1.0; l.len()]).collect()
        };

        let mut out: Vec<Expansion<MinionsTurns>> = specs
            .into_iter()
            .zip(priors)
            .enumerate()
            .map(|(gi, (sp, pr))| {
                let cands = sp
                    .into_iter()
                    .zip(pr)
                    .map(|((spec, explore), prior)| (spec, prior, explore))
                    .collect();
                (values.get(gi).copied().unwrap_or(0.0), cands)
            })
            .collect::<Vec<_>>();

        // Back to the real board for the seats we rotated.
        for (gi, (_, cands)) in out.iter_mut().enumerate() {
            if flipped[gi] {
                for (spec, _, _) in cands.iter_mut() {
                    *spec = spec.rotated();
                }
            }
        }
        // Drop the padding: the caller asked about `real` positions and must not
        // see the repeats that kept the tensor shape fixed.
        out.truncate(real);
        Ok(out)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::TARGET_KILL;
    use crate::units::Unit;
    use playground_core::SampledGame;

    /// Forced exploration must keep its share at any chunk size.
    ///
    /// Regression guard: with candidates generated one at a time,
    /// `round(1 * 0.25)` is 0, so injection disappeared entirely and the run
    /// reported "explore: none recorded". The count is now a floor plus a coin
    /// on the fractional part, so the long-run fraction holds for `k = 1` just
    /// as it does for `k = 16`.
    #[test]
    fn explore_injection_keeps_its_share_at_every_chunk_size() {
        let fraction = 0.25f32;
        for k in [1usize, 2, 4, 16] {
            let mut rng = ChaCha8Rng::seed_from_u64(7);
            let trials = 20_000;
            let mut total = 0usize;
            for _ in 0..trials {
                let want = (k as f32) * fraction;
                let floor = want.floor();
                let extra = if rng.gen::<f32>() < (want - floor) { 1 } else { 0 };
                total += floor as usize + extra;
            }
            let observed = total as f32 / (trials * k) as f32;
            assert!(
                (observed - fraction).abs() < 0.01,
                "k={k}: injected {observed:.3} of candidates, wanted {fraction}"
            );
        }
    }

    fn fixture_net() -> Option<crate::net::LocalNet> {
        let dir = std::env::var("MINIONS_PARITY_DIR").ok()?;
        crate::net::LocalNet::load(&std::path::Path::new(&dir).join("checkpoint.safetensors"), false).ok()
    }

    fn model(cfg: SamplerConfig) -> Option<MinionsModel> {
        Some(MinionsModel::new(Box::new(fixture_net()?), cfg))
    }

    #[test]
    fn committed_schedule_is_monotone_and_completes() {
        for n in [1usize, 7, 80] {
            for t_total in [1usize, 4, 6] {
                let mut prev = 0;
                for t in 1..=t_total {
                    let c = MinionsModel::committed_after(t, t_total, n);
                    assert!(c >= prev, "schedule must not un-commit");
                    assert!(c <= n);
                    prev = c;
                }
                assert_eq!(
                    MinionsModel::committed_after(t_total, t_total, n),
                    n,
                    "every variable must be committed by the last round"
                );
            }
        }
    }

    #[test]
    fn move_options_offer_reachable_non_enemy_hexes() {
        let s = Minions::initial_state();
        let me = s.side;
        let live = LiveMask::of(&s);
        let opts = MinionsModel::move_options(&s, &live);
        for c in bitboard::cells(live.moves) {
            let m = opts[c as usize];
            assert!(m & 1 != 0, "staying must always be available");
            let (_, unit) = s.piece_at(c).unwrap();
            for (k, &d) in MOVE_DEST[c as usize].iter().enumerate().skip(1) {
                if m & (1 << k) == 0 {
                    continue;
                }
                assert_ne!(d, NO_CELL);
                // Friendly-occupied is allowed (the occupant may be moving away);
                // enemy-occupied never is.
                assert!(
                    s.piece_at(d).is_none_or(|(sd, _)| sd == me),
                    "offered an enemy-held destination"
                );
                assert!(
                    crate::map::dist(c, d) <= unit.stats().speed as u32,
                    "offered a destination beyond the unit's speed"
                );
            }
        }
    }

    #[test]
    fn unsummon_is_only_offered_against_non_persistent_units() {
        let mut s = Minions::initial_state();
        // Second player's zombies are non-persistent; give them a skeleton too.
        let ghost = crate::map::idx(5, 5);
        s.cells[ghost as usize] = 1 + Unit::Ghost as u8 + 6; // persistent
        let live = LiveMask::of(&s);
        let opts = MinionsModel::target_options(&s, &live);
        assert_eq!(opts[ghost as usize] & (1 << TARGET_UNSUMMON), 0);
        for c in bitboard::cells(live.targets) {
            let (_, u) = s.piece_at(c).unwrap();
            let offered = opts[c as usize] & (1 << TARGET_UNSUMMON) != 0;
            assert_eq!(offered, !u.stats().persistent, "cell {c}");
            assert!(opts[c as usize] & (1 << TARGET_KILL) != 0, "kill is always offered");
        }
    }

    #[test]
    fn spawn_options_track_the_purse() {
        let mut s = Minions::initial_state();
        s.money[0] = 0;
        assert_eq!(MinionsModel::spawn_options(&s), 1, "broke: only 'nothing'");
        s.money[0] = 2; // a zombie costs 2
        let m = MinionsModel::spawn_options(&s);
        assert!(m & (1 << Unit::Zombie as u8) != 0);
        assert_eq!(m & (1 << Unit::Skeleton as u8), 0, "skeleton costs 4");
        s.money[0] = 99;
        let m = MinionsModel::spawn_options(&s);
        for u in BUYABLE {
            assert!(m & (1 << u as u8) != 0, "{} should be affordable", u.stats().name);
        }
    }

    #[test]
    fn sampled_declarations_are_collision_free_and_realizable() {
        let Some(m) = model(SamplerConfig { k_candidates: 8, seed: 3, ..Default::default() })
        else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping");
            return;
        };
        let s = Minions::initial_state();
        let mut planes = vec![0f32; crate::game::PLANES * NUM_CELLS];
        for (o, b) in planes.iter_mut().zip(Minions::to_features(&s)) {
            *o = b as f32;
        }
        let tr = m.net.trunk(&planes, 1).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let specs = m.sample_declarations(&s, &tr.h, 8, &mut rng).unwrap();
        assert_eq!(specs.len(), 8);

        for spec in &specs {
            // No two units declared onto the same hex.
            let mut dests = std::collections::HashSet::new();
            for (c, &k) in spec.moves.iter().enumerate() {
                if k == crate::turn::NO_MOVE || k == 0 {
                    continue;
                }
                let d = MOVE_DEST[c][k as usize];
                if d != NO_CELL {
                    assert!(dests.insert(d), "two units declared onto {d}");
                }
            }
            // And every action it realizes to is legal.
            let r = realize(&s, spec);
            let mut check = s.clone();
            for mv in &r.actions {
                assert!(Minions::legal_moves(&check).contains(mv), "illegal {mv}");
                check = Minions::apply(&check, mv);
            }
            // No threshold on realization *rate* here: these are the fixture's
            // random weights, so the declarations are noise and their fidelity
            // says nothing about the design. What must hold is that reachability
            // masking never proposes an outright impossible move — so anything
            // dropped is a follow-move whose leader was itself blocked, never a
            // destination out of range.
            assert!(
                r.fidelity.done_moves <= r.fidelity.declared_moves,
                "cannot execute more moves than were declared"
            );
            for (c, &kk) in spec.moves.iter().enumerate() {
                if kk == crate::turn::NO_MOVE || kk == 0 {
                    continue;
                }
                let d = MOVE_DEST[c][kk as usize];
                let Some((_, unit)) = s.piece_at(c as u8) else { continue };
                assert!(
                    d != NO_CELL && crate::map::dist(c as u8, d) <= unit.stats().speed as u32,
                    "declared a destination beyond the unit's speed"
                );
            }
        }
    }

    #[test]
    fn expand_returns_distinct_scored_candidates() {
        let Some(m) = model(SamplerConfig { k_candidates: 12, seed: 9, ..Default::default() })
        else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping");
            return;
        };
        let s = Minions::initial_state();
        let (v, cands) = m.expand(&s, 12).unwrap();
        assert!((-1.0..=1.0).contains(&v), "value out of range: {v}");
        assert!(!cands.is_empty());
        assert!(cands.iter().all(|(_, p, _)| *p >= 0.0 && p.is_finite()));

        // Merged by resulting position, so no two candidates reach the same board.
        let mut keys = std::collections::HashSet::new();
        for (spec, _, _) in &cands {
            let next = MinionsTurns::apply(&s, spec);
            assert!(keys.insert(Minions::encode_state(&next)), "duplicate position survived");
        }
    }

    #[test]
    fn t_equals_one_degenerates_to_mean_field() {
        // Sanity on the dial: T=1 must still produce complete, legal declarations
        // — it is the mean-field baseline, not a broken configuration.
        let Some(m) = model(SamplerConfig { t_steps: 1, seed: 4, ..Default::default() }) else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping");
            return;
        };
        let s = Minions::initial_state();
        let (_, cands) = m.expand(&s, 6).unwrap();
        assert!(!cands.is_empty());
        for (spec, _, _) in &cands {
            let r = realize(&s, spec);
            let mut check = s.clone();
            for mv in &r.actions {
                assert!(Minions::legal_moves(&check).contains(mv));
                check = Minions::apply(&check, mv);
            }
        }
    }
}

/// Paired CPU microbenchmark for [`commit_round`], kept out of the default test
/// run (`--ignored`) because it is a timing measurement, not an assertion.
///
/// The pre-optimisation implementation is duplicated verbatim as `legacy` so the
/// comparison is a real A/B rather than a before/after across two builds. Delete
/// both once the change is settled.
#[cfg(test)]
mod commit_bench {
    use super::*;
    use playground_core::SampledGame;
    use rand::SeedableRng;

    fn legacy(
        chain: &mut Partial,
        claimed: &mut Bb,
        cx: &GameCtx,
        logits: &[f32],
        want: usize,
        temperature: f32,
        rng: &mut ChaCha8Rng,
    ) {
        let temp = temperature.max(1e-3);
        let mut picks: Vec<(f32, u8, u8, u8)> = Vec::new();
        let consider = |cell: u8,
                        field: u8,
                        off: usize,
                        n_cls: usize,
                        allowed: &dyn Fn(usize) -> bool,
                        picks: &mut Vec<(f32, u8, u8, u8)>,
                        rng: &mut ChaCha8Rng| {
            if chain.is_committed(cell, field) {
                return;
            }
            let row = cell as usize * DECL_LOGITS + off;
            let (mut best_v, mut best_g) = (0usize, f32::NEG_INFINITY);
            let mut max_l = f32::NEG_INFINITY;
            for v in 0..n_cls {
                if !allowed(v) {
                    continue;
                }
                let l = logits[row + v] / temp;
                if l > max_l {
                    max_l = l;
                }
                let u: f32 = rng.gen_range(1e-9f32..1.0);
                let gm = l - (-u.ln()).ln();
                if gm > best_g {
                    best_g = gm;
                    best_v = v;
                }
            }
            if max_l == f32::NEG_INFINITY {
                return;
            }
            let mut z = 0.0f32;
            for v in 0..n_cls {
                if allowed(v) {
                    z += (logits[row + v] / temp - max_l).exp();
                }
            }
            let p = ((logits[row + best_v] / temp - max_l).exp()) / z.max(1e-9);
            picks.push((p, cell, field, best_v as u8));
        };
        for cell in bitboard::cells(cx.live.moves) {
            let opts = cx.move_opts[cell as usize];
            let taken = *claimed;
            let allow = |v: usize| {
                if opts & (1 << v) == 0 {
                    return false;
                }
                let d = MOVE_DEST[cell as usize][v];
                v == 0 || (d != NO_CELL && !bitboard::has(taken, d))
            };
            consider(cell, C_MOVE, OFF_MOVES, N_OFFSETS, &allow, &mut picks, rng);
        }
        for cell in bitboard::cells(cx.live.targets) {
            let opts = cx.target_opts[cell as usize];
            let allow = |v: usize| opts & (1 << v) != 0;
            consider(cell, C_TARGET, OFF_TARGETS, N_TARGETS, &allow, &mut picks, rng);
        }
        for cell in bitboard::cells(cx.live.spawns) {
            let so = cx.spawn_opts;
            let allow = |v: usize| so & (1 << v) != 0;
            consider(cell, C_SPAWN, OFF_SPAWNS, N_SPAWNS, &allow, &mut picks, rng);
        }
        let already = chain.committed.iter().map(|c| c.count_ones() as usize).sum::<usize>();
        let take = want.saturating_sub(already).max(1).min(picks.len());
        picks.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, cell, field, value) in picks.iter().take(take) {
            if field == C_MOVE && value != 0 {
                let d = MOVE_DEST[cell as usize][value as usize];
                if d == NO_CELL || bitboard::has(*claimed, d) {
                    continue;
                }
                *claimed |= bit(d);
            }
            chain.commit(cell, field, value);
        }
    }

    /// A mid-game position has far more live variables than the opening, which
    /// is where commit time actually accumulates.
    fn position() -> MinionsState {
        let mut st = MinionsState::default();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let plies: usize =
            std::env::var("COMMIT_BENCH_PLIES").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        for _ in 0..plies {
            if crate::game::Minions::outcome(&st).is_terminal() {
                break;
            }
            let spec = crate::turn::TurnSpec::random_playout(&st, &mut rng);
            st = <crate::turn::MinionsTurns as SampledGame>::apply(&st, &spec);
        }
        st
    }

    #[test]
    #[ignore = "timing measurement; run with --ignored --nocapture"]
    fn commit_round_is_faster_than_legacy() {
        let st = position();
        let cx = MinionsModel::context(&st);
        let width = NUM_CELLS * DECL_LOGITS;
        let mut lrng = ChaCha8Rng::seed_from_u64(3);
        let logits: Vec<f32> = (0..width).map(|_| lrng.gen_range(-4.0f32..4.0)).collect();
        let steps = 4usize;
        let reps = 4000usize;

        // Interleaved ABBA, same as the GPU harness: a single ordering biases
        // whichever arm runs second once the core clocks settle.
        let run = |legacy_arm: bool| -> f64 {
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                let mut chain = Partial::new();
                let mut claimed: Bb = 0;
                let mut rng = ChaCha8Rng::seed_from_u64(r as u64);
                let mut picks = Vec::new();
                for t in 1..=steps {
                    let want = MinionsModel::committed_after(t, steps, cx.n_live);
                    if legacy_arm {
                        legacy(&mut chain, &mut claimed, &cx, &logits, want, 1.0, &mut rng);
                    } else {
                        commit_round(&mut chain, &mut claimed, &cx, &logits, want, 1.0, &mut rng, &mut picks);
                    }
                }
                std::hint::black_box(&chain);
            }
            t0.elapsed().as_secs_f64() * 1e3
        };

        run(true);
        run(false); // warm
        let (mut la, mut nu) = (0.0, 0.0);
        for _ in 0..4 {
            la += run(true);
            nu += run(false);
            nu += run(false);
            la += run(true);
        }
        let per = |ms: f64| ms * 1e6 / (reps as f64 * steps as f64 * 8.0);
        println!(
            "commit_round  legacy {:.0} ns/call   new {:.0} ns/call   {:+.1}%   (n_live={})",
            per(la),
            per(nu),
            (la - nu) / la * 100.0,
            cx.n_live
        );
    }
}

/// Paired CPU microbenchmark: random commit order vs the old confidence order.
///
/// Random ordering was adopted for correctness — it makes `q(a)` a plain product
/// of conditionals, which is what ESS and a GFlowNet `log P_F` both need — but it
/// also deletes a whole pass over each variable's classes (the softmax
/// normaliser that confidence needed). This measures that second effect, since
/// `commit` is 13-16% of self-play.
#[cfg(test)]
mod order_bench {
    use super::*;
    use playground_core::Game;
    use rand::SeedableRng;

    /// `commit_round` as it was: score every candidate's confidence, then commit
    /// the most confident. Kept only for this comparison.
    fn legacy(
        chain: &mut Partial,
        claimed: &mut Bb,
        cx: &GameCtx,
        logits: &[f32],
        want: usize,
        temperature: f32,
        rng: &mut ChaCha8Rng,
        picks: &mut Vec<(f32, u8, u8, u8)>,
    ) {
        let temp = temperature.max(1e-3);
        picks.clear();
        let mut consider = |cell: u8, field: u8, off: usize, n_cls: usize,
                            allowed: &dyn Fn(usize) -> bool,
                            picks: &mut Vec<(f32, u8, u8, u8)>,
                            rng: &mut ChaCha8Rng| {
            if chain.is_committed(cell, field) {
                return;
            }
            let row = cell as usize * DECL_LOGITS + off;
            let mut scaled = [f32::NEG_INFINITY; N_OFFSETS];
            let (mut best_v, mut best_g) = (0usize, f32::NEG_INFINITY);
            let mut max_l = f32::NEG_INFINITY;
            for v in 0..n_cls {
                if !allowed(v) {
                    continue;
                }
                let l = logits[row + v] / temp;
                scaled[v] = l;
                if l > max_l {
                    max_l = l;
                }
                let u: f32 = rng.gen_range(1e-9f32..1.0);
                let gm = l - (-u.ln()).ln();
                if gm > best_g {
                    best_g = gm;
                    best_v = v;
                }
            }
            if max_l == f32::NEG_INFINITY {
                return;
            }
            // The extra pass the new version does not need.
            let mut z = 0.0f32;
            for v in 0..n_cls {
                if scaled[v] != f32::NEG_INFINITY {
                    z += (scaled[v] - max_l).exp();
                }
            }
            let p = (scaled[best_v] - max_l).exp() / z.max(1e-9);
            picks.push((p, cell, field, best_v as u8));
        };
        for cell in bitboard::cells(cx.live.moves) {
            let opts = cx.move_opts[cell as usize];
            let taken = *claimed;
            let allow = |v: usize| {
                if opts & (1 << v) == 0 {
                    return false;
                }
                let d = MOVE_DEST[cell as usize][v];
                v == 0 || (d != NO_CELL && !bitboard::has(taken, d))
            };
            consider(cell, C_MOVE, OFF_MOVES, N_OFFSETS, &allow, picks, rng);
        }
        for cell in bitboard::cells(cx.live.targets) {
            let opts = cx.target_opts[cell as usize];
            let allow = |v: usize| opts & (1 << v) != 0;
            consider(cell, C_TARGET, OFF_TARGETS, N_TARGETS, &allow, picks, rng);
        }
        for cell in bitboard::cells(cx.live.spawns) {
            let so = cx.spawn_opts;
            let allow = |v: usize| so & (1 << v) != 0;
            consider(cell, C_SPAWN, OFF_SPAWNS, N_SPAWNS, &allow, picks, rng);
        }
        let already = chain.committed.iter().map(|c| c.count_ones() as usize).sum::<usize>();
        let take = want.saturating_sub(already).max(1).min(picks.len());
        picks.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, cell, field, value) in picks.iter().take(take) {
            if field == C_MOVE && value != 0 {
                let d = MOVE_DEST[cell as usize][value as usize];
                if d == NO_CELL || bitboard::has(*claimed, d) {
                    continue;
                }
                *claimed |= bit(d);
            }
            chain.commit(cell, field, value);
        }
    }

    #[test]
    #[ignore = "timing measurement; run with --ignored --nocapture"]
    fn random_order_is_cheaper_than_confidence_order() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut st = MinionsState::default();
        for _ in 0..12 {
            if Minions::outcome(&st).is_terminal() {
                break;
            }
            let spec = crate::turn::TurnSpec::random_playout(&st, &mut rng);
            st = crate::turn::realize(&st, &spec).next;
        }
        let cx = MinionsModel::context(&st);
        let width = NUM_CELLS * DECL_LOGITS;
        let mut lrng = ChaCha8Rng::seed_from_u64(3);
        let logits: Vec<f32> = (0..width).map(|_| lrng.gen_range(-4.0f32..4.0)).collect();
        let (steps, reps) = (4usize, 4000usize);

        let run = |legacy_arm: bool| -> f64 {
            let t0 = std::time::Instant::now();
            let mut new_picks = Vec::new();
            let mut old_picks = Vec::new();
            for r in 0..reps {
                let mut chain = Partial::new();
                let mut claimed: Bb = 0;
                let mut rng = ChaCha8Rng::seed_from_u64(r as u64);
                for t in 1..=steps {
                    let want = MinionsModel::committed_after(t, steps, cx.n_live);
                    if legacy_arm {
                        legacy(&mut chain, &mut claimed, &cx, &logits, want, 1.0, &mut rng,
                               &mut old_picks);
                    } else {
                        commit_round(&mut chain, &mut claimed, &cx, &logits, want, 1.0, &mut rng,
                                     &mut new_picks);
                    }
                }
                std::hint::black_box(&chain);
            }
            t0.elapsed().as_secs_f64() * 1e3
        };

        run(true);
        run(false);
        let (mut la, mut nu) = (0.0, 0.0);
        for _ in 0..4 {
            la += run(true);
            nu += run(false);
            nu += run(false);
            la += run(true);
        }
        let per = |ms: f64| ms * 1e6 / (reps as f64 * steps as f64 * 8.0);
        println!(
            "commit_round  confidence {:.0} ns/call   random {:.0} ns/call   {:+.1}%  (n_live={})",
            per(la), per(nu), (la - nu) / la * 100.0, cx.n_live
        );
    }
}
