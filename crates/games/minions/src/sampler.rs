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

use half::f16;
use half::slice::HalfFloatSliceExt;

use playground_core::sampled_mcts::Expansion;
use playground_core::{Game, SampledModel, SampledParams};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::bitboard::{self, bit, Bb, FULL};
use crate::game::{Minions, MinionsState};
use crate::map::{NO_CELL, NUM_CELLS};
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
    /// Sampler draws per node (`D`, see `SampledParams::k_draws`): the node is
    /// asked for exactly this many proposals, and a shortfall of distinct ones
    /// against `k_candidates` is filled with random playouts.
    pub k_draws: usize,
    /// Draws per node for *fast* searches.
    pub k_draws_fast: usize,
    /// Share of candidates drawn uniformly instead of from the net.
    pub random_fraction: f32,
    /// Softmax temperature on the declaration logits.
    pub temperature: f32,
    /// Diffusion rounds that redraw the masked part of a turn when the search
    /// resamples (`SampledParams::resample_every`).
    pub resample_t: usize,
    /// Most of a turn a resample masks again: the share is uniform in
    /// `[0, resample_frac]` of its live variables, and at least one.
    pub resample_frac: f32,
    /// Choose *which* variables a resample masks from the model's own belief
    /// rather than uniformly at random.
    ///
    /// One extra diffusion pass runs over the fully committed turn first, and
    /// each live variable is then unmasked independently with probability
    /// `1 - p(value already there)` -- the mass the head puts on proposing
    /// something else. Fields it would have changed get rerolled; fields it is
    /// confident about stay. `resample_frac` plays no part: the number masked
    /// is whatever the beliefs say.
    ///
    /// This is the use the committed-field consistency loss (`--w-unmask`)
    /// exists for. Without that training a committed field's logits are
    /// unconstrained, so this probability means nothing.
    ///
    /// Costs one diffusion call per resample batch: 3 instead of 2 at
    /// `resample_t = 2`.
    pub resample_bias: bool,
    /// Completions drawn from the *final* diffusion step instead of one.
    ///
    /// 1 (default) is the existing sampler. Above 1, the last round is replayed
    /// this many times from the same partial declaration against the same
    /// logits — CPU only, so no extra network call and, critically, no new
    /// tensor shape — and the NNUE picks which completion the chain keeps. The
    /// others are handed back as extra candidates, so they cost their
    /// realization once and are never re-evaluated.
    pub last_stage_samples: usize,
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
            k_draws: 12,
            k_draws_fast: 6,
            random_fraction: 0.0,
            temperature: 1.0,
            resample_t: 2,
            resample_frac: 0.5,
            resample_bias: false,
            last_stage_samples: 1,
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
        if full {
            self.k_candidates
        } else {
            self.k_candidates_fast.min(self.k_candidates)
        }
    }

    pub fn draws_for(&self, full: bool) -> usize {
        if full {
            self.k_draws
        } else {
            self.k_draws_fast
        }
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
/// The mass the diffusion head puts on the value already committed at one
/// field, over that field's legal options.
///
/// Legality is the option mask only -- deliberately *not* the claimed-hex rule
/// `commit_round` applies while building a turn. Every move is committed when
/// this is called, so every destination is already claimed and that rule would
/// leave almost nothing legal, making the probability meaningless.
///
/// A field holding a value outside its class range scores 0, so it is always
/// rerolled: a realized turn can carry `NO_MOVE` for a unit that never acted,
/// and the head cannot express that, so it has no opinion to keep.
fn p_committed_value(lg: &[f32], cx: &GameCtx, spec: &TurnSpec, cell: u8, field: u8) -> f32 {
    let ci = cell as usize;
    let (off, n_cls, opts, cur) = match field {
        C_MOVE => (
            OFF_MOVES,
            N_OFFSETS,
            cx.move_opts[ci] as u64,
            spec.moves[ci] as usize,
        ),
        C_TARGET => (
            OFF_TARGETS,
            N_TARGETS,
            cx.target_opts[ci] as u64,
            spec.targets[ci] as usize,
        ),
        _ => (
            OFF_SPAWNS,
            N_SPAWNS,
            cx.spawn_opts as u64,
            spec.spawns[ci] as usize,
        ),
    };
    let row = ci * DECL_LOGITS + off;
    let mut max_l = f32::NEG_INFINITY;
    for v in 0..n_cls {
        if opts & (1 << v) != 0 && lg[row + v] > max_l {
            max_l = lg[row + v];
        }
    }
    if max_l == f32::NEG_INFINITY {
        // Nothing legal here, so there is nothing to reroll into.
        return 1.0;
    }
    let (mut sum, mut cur_e) = (0.0f32, 0.0f32);
    for v in 0..n_cls {
        if opts & (1 << v) == 0 {
            continue;
        }
        let e = (lg[row + v] - max_l).exp();
        sum += e;
        if v == cur {
            cur_e = e;
        }
    }
    if sum <= 0.0 {
        1.0
    } else {
        cur_e / sum
    }
}

/// Whether the sampler takes its lean paths: diffusion logits gathered to the
/// live cells inside the backend's read, and inverse-CDF draws in `consider`.
/// Off is the previous behaviour (whole logits converted, Gumbel-max draws);
/// self-play flips it per window for an in-process A/B (`lean_ab`).
pub static LEAN_USE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn lean_on() -> bool {
    LEAN_USE.load(std::sync::atomic::Ordering::Relaxed)
}

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
    if !lean_on() {
        let (mut best_v, mut best_g) = (0usize, f32::NEG_INFINITY);
        for v in 0..n_cls {
            if !allowed(v) {
                continue;
            }
            let l = logits[row + v] / temp;
            let u: f32 = rng.gen_range(1e-9f32..1.0);
            let gm = l - (-u.ln()).ln();
            if gm > best_g {
                best_g = gm;
                best_v = v;
            }
        }
        if best_g > f32::NEG_INFINITY {
            picks.push((cell, field, best_v as u8));
        }
        return;
    }
    // One categorical draw by inverse CDF: a uniform against the running sum
    // of the softmax weights. It draws from the same distribution the Gumbel-max
    // trick did, for one random number and one `exp` per class instead of one
    // random number and two `ln`s.
    let mut w = [0f32; N_OFFSETS];
    let mut max_l = f32::NEG_INFINITY;
    for v in 0..n_cls {
        if allowed(v) {
            let l = logits[row + v] / temp;
            w[v] = l;
            if l > max_l {
                max_l = l;
            }
        } else {
            w[v] = f32::NEG_INFINITY;
        }
    }
    if max_l == f32::NEG_INFINITY {
        return; // nothing legal here
    }
    let mut sum = 0.0f32;
    for x in &mut w[..n_cls] {
        *x = if *x == f32::NEG_INFINITY {
            0.0
        } else {
            (*x - max_l).exp()
        };
        sum += *x;
    }
    let mut u = rng.gen::<f32>() * sum;
    let mut pick = 0;
    for (v, &x) in w[..n_cls].iter().enumerate() {
        if x > 0.0 {
            pick = v; // the last weighted class, should rounding run past it
            if u < x {
                break;
            }
            u -= x;
        }
    }
    picks.push((cell, field, pick as u8));
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
        consider(
            chain, cell, C_MOVE, OFF_MOVES, N_OFFSETS, allow, logits, temp, picks, rng,
        );
    }
    for cell in bitboard::cells(cx.live.targets) {
        let opts = cx.target_opts[cell as usize];
        let allow = |v: usize| opts & (1 << v) != 0;
        consider(
            chain,
            cell,
            C_TARGET,
            OFF_TARGETS,
            N_TARGETS,
            allow,
            logits,
            temp,
            picks,
            rng,
        );
    }
    for cell in bitboard::cells(cx.live.spawns) {
        let so = cx.spawn_opts;
        let allow = |v: usize| so & (1 << v) != 0;
        consider(
            chain, cell, C_SPAWN, OFF_SPAWNS, N_SPAWNS, allow, logits, temp, picks, rng,
        );
    }

    let already = chain
        .committed
        .iter()
        .map(|c| c.count_ones() as usize)
        .sum::<usize>();
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

/// The best-move head's turn for one position: every live variable at its
/// highest-scoring legal class.
///
/// Moves are settled most-confident first, so when two units want the same hex
/// the surer one gets it and the other takes its best remaining option -- the
/// no-shared-destination rule `commit_round` enforces with `claimed`.
fn decode_best(cx: &GameCtx, out: &[f32]) -> TurnSpec {
    let mut chain = Partial::new();
    let at = |cell: u8, off: usize, v: usize| out[cell as usize * DECL_LOGITS + off + v];
    let mut claimed: Bb = 0;
    let mut pending: Vec<u8> = bitboard::cells(cx.live.moves).collect();
    while !pending.is_empty() {
        let mut pick: Option<(usize, usize, f32)> = None;
        for (i, &cell) in pending.iter().enumerate() {
            let opts = cx.move_opts[cell as usize];
            for v in 0..N_OFFSETS {
                if opts & (1 << v) == 0 {
                    continue;
                }
                let d = MOVE_DEST[cell as usize][v];
                // Offset 0 is "stay", always legal, and never claims a hex.
                if v != 0 && (d == NO_CELL || bitboard::has(claimed, d)) {
                    continue;
                }
                let s = at(cell, OFF_MOVES, v);
                if pick.is_none_or(|(_, _, best)| s > best) {
                    pick = Some((i, v, s));
                }
            }
        }
        let Some((i, v, _)) = pick else { break };
        let cell = pending.swap_remove(i);
        if v != 0 {
            claimed |= bit(MOVE_DEST[cell as usize][v]);
        }
        chain.commit(cell, C_MOVE, v as u8);
    }
    let mut argmax =
        |cell: u8, field: u8, off: usize, n: usize, allowed: &dyn Fn(usize) -> bool| {
            let best = (0..n)
                .filter(|&v| allowed(v))
                .max_by(|&a, &b| at(cell, off, a).total_cmp(&at(cell, off, b)));
            if let Some(v) = best {
                chain.commit(cell, field, v as u8);
            }
        };
    for cell in bitboard::cells(cx.live.targets) {
        let opts = cx.target_opts[cell as usize];
        argmax(cell, C_TARGET, OFF_TARGETS, N_TARGETS, &|v| {
            opts & (1 << v) != 0
        });
    }
    for cell in bitboard::cells(cx.live.spawns) {
        let so = cx.spawn_opts;
        argmax(cell, C_SPAWN, OFF_SPAWNS, N_SPAWNS, &|v| so & (1 << v) != 0);
    }
    chain.spec
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
    /// Drawing extra completions at the last diffusion step and scoring them
    /// with the NNUE (`SamplerConfig::last_stage_samples`).
    ///
    /// Counted apart from `commit` and `realize` even though it is made of
    /// both: the whole question this feature asks is what the extra draws
    /// cost, and folding them into the phases they resemble would hide it.
    pub last_stage_ns: AtomicU64,
}

impl Phases {
    fn add(counter: &AtomicU64, t: std::time::Instant) {
        counter.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Zeroes every counter, so the next report covers one window only.
    ///
    /// `report_wall` divides by one window's worker time; without this the
    /// counters stay cumulative and `unaccounted` clamps to 0 from the second
    /// window on, hiding whatever the workers do outside the phases.
    pub fn reset(&self) {
        for c in [
            &self.features_ns,
            &self.trunk_ns,
            &self.decl_write_ns,
            &self.diffusion_ns,
            &self.commit_ns,
            &self.realize_ns,
            &self.energy_ns,
            &self.playout_ns,
            &self.last_stage_ns,
        ] {
            c.store(0, Ordering::Relaxed);
        }
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
            + g(&self.playout_ns)
            + g(&self.last_stage_ns);
        let worker_ms = secs * 1000.0 * workers.max(1) as f64;
        if worker_ms <= 0.0 {
            return body;
        }
        let unaccounted = (1.0 - measured_ms / worker_ms).max(0.0) * 100.0;
        format!("{body}  [unaccounted {unaccounted:.0}% — search/backup/dedupe]")
    }

    pub fn report(&self) -> String {
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        let (f, t, w, d, c, r, e, p, l) = (
            g(&self.features_ns),
            g(&self.trunk_ns),
            g(&self.decl_write_ns),
            g(&self.diffusion_ns),
            g(&self.commit_ns),
            g(&self.realize_ns),
            g(&self.energy_ns),
            g(&self.playout_ns),
            g(&self.last_stage_ns),
        );
        let total = (f + t + w + d + c + r + e + p + l).max(1e-9);
        let pct = |x: f64| 100.0 * x / total;
        format!(
            "features {:.0}% trunk {:.0}% decl-write {:.0}% diffusion {:.0}% \
commit {:.0}% realize {:.0}% playout {:.0}% energy {:.0}% last-stage {:.0}%  \
(GPU {:.0}%, CPU {:.0}%)",
            pct(f),
            pct(t),
            pct(w),
            pct(d),
            pct(c),
            pct(r),
            pct(p),
            pct(e),
            pct(l),
            pct(t + d + e),
            pct(f + w + c + r + p + l),
        )
    }
}

pub struct MinionsModel {
    /// Candle or ORT — the sampler does not care which; see [`Backend`].
    pub net: Box<dyn Backend>,
    pub cfg: SamplerConfig,
    pub phases: Phases,
    /// Distilled value net used *only* to order candidates at the last
    /// diffusion step (`SamplerConfig::last_stage_samples`).
    ///
    /// Behind `&self` and scored into caller-owned scratch, so the worker
    /// threads do not contend. Its values never reach `w`, `q`, a prior or a
    /// training target: a second, weaker evaluator inside the backup would be
    /// a silent regression in everything the search reports.
    nnue: Option<crate::nnue::Evaluator>,
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
        Self {
            net,
            cfg,
            phases: Phases::default(),
            nnue: None,
            seed,
            expand_seq: AtomicU64::new(0),
        }
    }

    /// Attach the candidate-ordering NNUE. Separate from `new` so the five
    /// construction sites that do not want it stay unchanged.
    pub fn with_nnue(mut self, nnue: Option<crate::nnue::Evaluator>) -> Self {
        self.nnue = nnue;
        self
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
            let Some((_, unit)) = state.piece_at(c) else {
                continue;
            };
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
            let Some((_, unit)) = state.piece_at(c) else {
                continue;
            };
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
        h_all: &[f16],
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
        let steps = if full {
            self.cfg.t_steps
        } else {
            self.cfg.t_steps_fast
        }
        .max(1);

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
        // Completions the last step drew but did not keep, per game.
        let mut extras: Vec<Vec<TurnSpec>> = vec![Vec::new(); g];
        let n_last = self.cfg.last_stage_samples.max(1);
        // One scratch for the whole batch: the evaluator is behind `&self` so
        // worker threads never contend for it.
        let mut scratch = self
            .nnue
            .as_ref()
            .filter(|_| n_last > 1)
            .map(|e| e.scratch());
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
            // Only the final round has anything to choose between: every
            // earlier one leaves fields uncommitted, so its completions are not
            // turns yet.
            let best_of = t == steps && scratch.is_some();
            for row in 0..rows {
                let cx = &ctx[row / k];
                if cx.n_live == 0 {
                    continue;
                }
                let want = Self::committed_after(t, steps, cx.n_live);
                let lg = &logits[row * width..(row + 1) * width];
                if !best_of {
                    commit_round(
                        &mut chains[row],
                        &mut claimed[row],
                        cx,
                        lg,
                        want,
                        temp,
                        &mut rngs[row],
                        &mut picks,
                    );
                    continue;
                }
                let t_last = Instant::now();
                let gi = row / k;
                let nnue = self.nnue.as_ref().expect("best_of implies an evaluator");
                let sc = scratch.as_mut().expect("best_of implies scratch");
                // The chain as it stands, replayed from here `n_last` times.
                // `Partial` and the claim mask are both `Copy`, so a completion
                // costs a stack copy and a commit -- no network call, and no new
                // tensor shape for TensorRT to build an engine for.
                let base_chain = chains[row];
                let base_claimed = claimed[row];
                let mut seen: std::collections::HashSet<Vec<u8>> = Default::default();
                let mut best: Option<(f32, Partial)> = None;
                let mut rest: Vec<TurnSpec> = Vec::new();
                for _ in 0..n_last {
                    let mut ch = base_chain;
                    let mut cl = base_claimed;
                    commit_round(
                        &mut ch,
                        &mut cl,
                        cx,
                        lg,
                        want,
                        temp,
                        &mut rngs[row],
                        &mut picks,
                    );
                    let r = realize(&states[gi], &ch.spec);
                    // Repeats are expected and cost only their realization --
                    // the point of drawing many is that the distribution, not
                    // the draw, decides.
                    if !seen.insert(Minions::encode_state(&r.next)) {
                        continue;
                    }
                    let outcome = Minions::outcome(&r.next);
                    let v = if outcome.is_terminal() {
                        // A finished game is worth its result, not a guess. The
                        // student never trained on terminal positions, and
                        // scoring a winning turn by approximation is how you
                        // throw away the only turns that are certain.
                        let first_to_move = states[gi].side == 0;
                        match outcome {
                            playground_core::Outcome::Draw => 0.0,
                            playground_core::Outcome::FirstWins => {
                                if first_to_move {
                                    1.0
                                } else {
                                    -1.0
                                }
                            }
                            playground_core::Outcome::SecondWins => {
                                if first_to_move {
                                    -1.0
                                } else {
                                    1.0
                                }
                            }
                            playground_core::Outcome::InProgress => unreachable!("terminal"),
                        }
                    } else {
                        // `to_features` is canonical to the side to move, and a
                        // realized turn hands the move over -- so the student's
                        // value is the *opponent's*. Negated, or this would
                        // pick the turn that helps them most.
                        -nnue.value_with(&Minions::to_features(&r.next), sc)
                    };
                    match best {
                        Some((bv, bp)) if bv >= v => rest.push(ch.spec),
                        Some((_, bp)) => {
                            rest.push(bp.spec);
                            best = Some((v, ch));
                        }
                        None => best = Some((v, ch)),
                    }
                }
                if let Some((_, bp)) = best {
                    chains[row] = bp;
                }
                // Capped at what a node can hold: `provide_widen` stops adding
                // at `k_candidates`, so anything past that is realized, scored
                // and thrown away.
                let room = self.cfg.k_candidates.saturating_sub(extras[gi].len());
                extras[gi].extend(rest.into_iter().take(room));
                Phases::add(&self.phases.last_stage_ns, t_last);
            }
            if !best_of {
                Phases::add(&self.phases.commit_ns, clk);
            }
        }
        // Kept specs first, then the extras: the leading candidate is the one
        // the search visits (see `provide_widen`), and it must be the pick.
        Ok(chains
            .chunks(k)
            .zip(extras)
            .map(|(c, ex)| c.iter().map(|p| p.spec).chain(ex).collect())
            .collect())
    }

    /// Single-position shim over [`Self::sample_declarations_batch`], used by
    /// the tests. Self-play always goes through the batched path.
    #[cfg(test)]
    fn sample_declarations(
        &self,
        state: &MinionsState,
        h: &[f16],
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
        self.finish_candidates_keyed(state, declared, false).0
    }

    /// [`Self::finish_candidates`], also returning each kept candidate's
    /// `state_key` in the real frame -- `state` is canonical, and the real
    /// position is its rotation when `flipped`. `realize` commutes with the
    /// rotation, so this is exactly what the search's own `apply` would key.
    fn finish_candidates_keyed(
        &self,
        state: &MinionsState,
        declared: Vec<(TurnSpec, bool)>,
        flipped: bool,
    ) -> (Vec<(TurnSpec, bool)>, Vec<u128>) {
        let mut keys: Vec<u128> = Vec::with_capacity(declared.len());
        // Distinct declarations routinely converge on one board; unmerged they
        // would fragment the visit counts they are supposed to concentrate.
        let mut seen: std::collections::HashMap<u128, usize> = Default::default();
        let mut specs: Vec<(TurnSpec, bool)> = Vec::with_capacity(declared.len());
        for (spec, explore) in declared {
            let r = realize(state, &spec);
            let key = r.next.key();
            match seen.get(&key) {
                Some(&i) => specs[i].1 |= explore,
                None => {
                    seen.insert(key, specs.len());
                    specs.push((r.spec, explore));
                    keys.push(if flipped { r.next.rotated().key() } else { key });
                }
            }
        }
        (specs, keys)
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
    /// The value at `state` and the energy of each of `specs` there, scored as
    /// an expansion scores its candidates -- for turns that came from anywhere
    /// but the sampler, like a reader adding one by hand. Same canonical frame
    /// and graphs as `expand_batch_cached`, and rows padded to a multiple of
    /// the trunk batch so a single turn reuses the engine a widen builds.
    pub fn score_turns(
        &self,
        state: &MinionsState,
        specs: &[TurnSpec],
    ) -> anyhow::Result<(f32, Vec<f32>)> {
        let flipped = state.side == 1;
        let canon = state.canonical();
        let g = self.cfg.pad_to.max(1);
        let pstride = crate::game::PLANES * NUM_CELLS;
        let mut planes = vec![0f32; g * pstride];
        for (o, b) in planes[..pstride]
            .iter_mut()
            .zip(Minions::to_features(&canon))
        {
            *o = b as f32;
        }
        for j in 1..g {
            planes.copy_within(0..pstride, j * pstride);
        }
        let tr = self.net.trunk(&planes, g)?;
        let value = tr.value[0];
        if specs.is_empty() {
            return Ok((value, Vec::new()));
        }
        if !self.cfg.use_energy {
            // What the expansion path hands the search with the head off.
            return Ok((value, vec![1.0; specs.len()]));
        }
        let hstride = self.net.channels() * NUM_CELLS;
        let dstride = crate::turn::ENERGY_PLANES * NUM_CELLS;
        let live = LiveMask::of(&canon);
        let rows = g * specs.len().div_ceil(g);
        let mut decl = vec![0f32; rows * dstride];
        for (at, spec) in specs.iter().enumerate() {
            let c = if flipped { spec.rotated() } else { *spec };
            Partial::fully_committed(&c, &live)
                .write_energy_planes(&mut decl[at * dstride..(at + 1) * dstride]);
        }
        let mut h_b = Vec::with_capacity(rows * hstride);
        for _ in 0..rows {
            h_b.extend_from_slice(&tr.h[..hstride]);
        }
        let mut e = self.net.energy(&h_b, &decl, rows)?;
        e.truncate(specs.len());
        Ok((value, self.priors_from(&e)))
    }

    fn priors_from(&self, energies: &[f32]) -> Vec<f32> {
        energies.to_vec()
    }

    /// Whether this net proposes with a best-move head rather than diffusion.
    pub fn best_mode(&self) -> bool {
        self.net.has_best()
    }

    /// `params` as this net's search must run them. A best-move head proposes
    /// one turn per position, so a node draws once and fills every other
    /// candidate with random playouts: asked for more draws, it would only
    /// repeat that turn until `draw_retries` gave up.
    pub fn tune_params(&self, params: &SampledParams) -> SampledParams {
        // The seat's own sampler sets its width, so one match can pit two
        // candidate counts against each other (`eval-match --candidates-b`).
        let p = SampledParams {
            k_candidates: self.cfg.k_candidates,
            k_draws: self.cfg.k_draws,
            ..*params
        };
        if self.best_mode() {
            SampledParams { k_draws: 1, ..p }
        } else {
            p
        }
    }

    /// The best-move head's turn for each of `states`, from their trunk rows
    /// `h_all`. One call over every row, so the shape is the batch's.
    fn best_declarations_batch(
        &self,
        states: &[MinionsState],
        h_all: &[f16],
    ) -> anyhow::Result<Vec<TurnSpec>> {
        let g = states.len();
        let clk = Instant::now();
        let out = self.net.best(h_all, g)?;
        Phases::add(&self.phases.diffusion_ns, clk);
        let width = NUM_CELLS * DECL_LOGITS;
        anyhow::ensure!(
            out.len() == g * width,
            "best.onnx returned {} values for {g} positions",
            out.len()
        );
        let clk = Instant::now();
        let specs = states
            .iter()
            .enumerate()
            .map(|(gi, s)| decode_best(&Self::context(s), &out[gi * width..(gi + 1) * width]))
            .collect();
        Phases::add(&self.phases.commit_ns, clk);
        Ok(specs)
    }
}

/// The model as one kind of self-play search calls it.
///
/// `full` is the self-play pool's `want_full`: it picks the diffusion depth
/// (`t_steps` or `t_steps_fast`) and whether random candidates are mixed in. A
/// search driven through the `SampledModel` trait has no such argument, so
/// this carries it. `MinionsModel`'s own impl is the full search.
pub struct AsSearch<'a> {
    pub model: &'a MinionsModel,
    pub full: bool,
}

impl AsSearch<'_> {
    /// A fresh stream per call. Still reproducible for a given `cfg.seed`: the
    /// sequence of expansions is what varies, not the starting entropy.
    fn next_seq(&self) -> u64 {
        self.model.expand_seq.fetch_add(1, Ordering::Relaxed)
    }
}

impl SampledModel<MinionsTurns> for AsSearch<'_> {
    /// Expand many positions at once: one trunk forward for all of them, one
    /// diffusion forward per round across every chain of every game, and one
    /// energy forward over every candidate of every game.
    fn expand_batch(
        &self,
        states: &[MinionsState],
        k: usize,
    ) -> anyhow::Result<Vec<Expansion<MinionsTurns>>> {
        self.model
            .expand_batch_mode(states, k, self.full, self.next_seq())
    }

    fn expand(&self, state: &MinionsState, k: usize) -> anyhow::Result<Expansion<MinionsTurns>> {
        Ok(self
            .expand_batch(std::slice::from_ref(state), k)?
            .pop()
            .expect("one state in, one expansion out"))
    }

    /// Proposals and fills in one call, as the self-play pool makes it.
    fn widen_batch(
        &self,
        states: &[MinionsState],
        fill: &[bool],
        k: usize,
    ) -> anyhow::Result<Vec<Vec<(TurnSpec, f32, bool)>>> {
        let (out, _) = self.model.expand_batch_cached(
            states,
            k,
            self.full,
            self.next_seq(),
            &[],
            &|n| n,
            false,
            fill,
        )?;
        Ok(out.into_iter().map(|(_, c)| c).collect())
    }

    fn resample_batch(
        &self,
        states: &[MinionsState],
        from: &[TurnSpec],
    ) -> anyhow::Result<Vec<Vec<(TurnSpec, f32, bool)>>> {
        self.model
            .resample_batch_seq(states, from, self.next_seq(), &[], &|n| n)
    }
}

impl SampledModel<MinionsTurns> for MinionsModel {
    fn expand_batch(
        &self,
        states: &[MinionsState],
        k: usize,
    ) -> anyhow::Result<Vec<Expansion<MinionsTurns>>> {
        AsSearch {
            model: self,
            full: true,
        }
        .expand_batch(states, k)
    }

    fn expand(&self, state: &MinionsState, k: usize) -> anyhow::Result<Expansion<MinionsTurns>> {
        AsSearch {
            model: self,
            full: true,
        }
        .expand(state, k)
    }

    fn widen_batch(
        &self,
        states: &[MinionsState],
        fill: &[bool],
        k: usize,
    ) -> anyhow::Result<Vec<Vec<(TurnSpec, f32, bool)>>> {
        AsSearch {
            model: self,
            full: true,
        }
        .widen_batch(states, fill, k)
    }

    fn resample_batch(
        &self,
        states: &[MinionsState],
        from: &[TurnSpec],
    ) -> anyhow::Result<Vec<Vec<(TurnSpec, f32, bool)>>> {
        AsSearch {
            model: self,
            full: true,
        }
        .resample_batch(states, from)
    }
}

/// Trunk output from an `expand_batch_cached` call, for the caller to keep:
/// `h` for every row of the call (`stride` floats each), and `fresh[i]` =
/// `Some(value)` for each real row the trunk actually ran on.
///
/// One buffer rather than a `Vec` per row: per-row allocations interleaved
/// with the long-lived cache are what grew glibc's thread arenas to 17 GB
/// around an 8 GB cache.
/// One cached trunk row, as [`MinionsModel::run_wave`] takes it.
#[derive(Clone, Copy)]
pub struct CachedRow<'a> {
    pub value: f32,
    pub h: &'a [f16],
    /// The diffusion head's first-step logits at this position, when held:
    /// fp16, canonical frame, and only the live cells' rows, in cell order
    /// (`n_live_cells * DECL_LOGITS`). Nothing else is ever read, and the
    /// whole 49-cell row was as large as `h` itself.
    pub step1: Option<&'a [f16]>,
}

/// A self-play wave's three kinds of request, served together by
/// [`MinionsModel::run_wave`].
pub struct Wave<'a> {
    /// Value-only expansions.
    pub exp: &'a [MinionsState],
    /// Widens, `k` candidates each, fills (`wid_fill`) by playout only.
    pub wid: &'a [MinionsState],
    pub wid_fill: &'a [bool],
    pub k: usize,
    /// Full search: `t_steps` and exploration draws, else `t_steps_fast`.
    pub full: bool,
    /// Refinements of `rs_from[i]` at `rs[i]`.
    pub rs: &'a [MinionsState],
    pub rs_from: &'a [TurnSpec],
    /// Per request, `exp ++ wid ++ rs`; empty when nothing is cached.
    pub cached: &'a [Option<CachedRow<'a>>],
}

/// What [`MinionsModel::run_wave`] returns.
pub struct WaveOut {
    /// Per value-only expansion.
    pub values: Vec<f32>,
    pub wid: Vec<Expansion<MinionsTurns>>,
    pub rs: Vec<Vec<(TurnSpec, f32, bool)>>,
    /// Trunk rows for every request (`exp ++ wid ++ rs`), `fresh` where
    /// computed this wave.
    pub trunk: TrunkRows,
    /// First-step logits computed this wave, per widen, in
    /// [`CachedRow::step1`]'s live-cells layout.
    pub step1: Vec<Option<Vec<f16>>>,
    /// `state_key` of the position each candidate in `wid` / `rs` reaches,
    /// real frame, parallel to them -- for `Search::provide_*_keyed`.
    pub wid_keys: Vec<Vec<u128>>,
    pub rs_keys: Vec<Vec<u128>>,
}

pub struct TrunkRows {
    pub stride: usize,
    pub h: Vec<f16>,
    pub fresh: Vec<Option<f32>>,
}

impl TrunkRows {
    pub fn row(&self, i: usize) -> &[f16] {
        &self.h[i * self.stride..(i + 1) * self.stride]
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
        self.expand_batch_cached(states, k, full, batch_seq, &[], &|n| n, false, &[])
            .map(|(out, _)| out)
    }

    /// `expand_batch_mode`, reusing trunk output the caller already holds.
    ///
    /// `cached[i]` is `(value, h)` for `states[i]` from an earlier call, in the
    /// canonical frame this function computes it in; `None` means compute it.
    /// Rows past `cached.len()` are the caller's padding, which repeats its
    /// last real row, so they share that row's entry. The trunk runs only over
    /// the misses, padded to `trunk_rows(misses)` so TensorRT stays on shapes
    /// it has already built. With nothing cached this is the original single
    /// trunk call, unchanged.
    ///
    /// With `want_fresh`, the returned `TrunkRows` marks each real row the trunk
    /// ran on (hits are `None`), with its `h` in the shared buffer.
    ///
    /// Why: a widen request is always for a node expanded earlier, so its trunk
    /// output was computed then and discarded. Widens are ~60% of requests,
    /// which made recomputing it 45% of every FLOP self-play spent.
    ///
    /// `fill[i]` marks a widen for a node whose proposals have repeated (see
    /// `SampledModel::widen_batch`): its row gets `k` random playouts, all
    /// explore, and none of its diffusion draws. Those draws are still made,
    /// so the batch keeps its shape. Rows past `fill.len()` are proposals.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn expand_batch_cached(
        &self,
        states: &[MinionsState],
        k: usize,
        full: bool,
        batch_seq: u64,
        cached: &[Option<(f32, &[f16])>],
        trunk_rows: &dyn Fn(usize) -> usize,
        want_fresh: bool,
        fill: &[bool],
    ) -> anyhow::Result<(Vec<Expansion<MinionsTurns>>, TrunkRows)> {
        let real = states.len();
        if real == 0 {
            return Ok((
                Vec::new(),
                TrunkRows {
                    stride: 0,
                    h: Vec::new(),
                    fresh: Vec::new(),
                },
            ));
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
        let mut rng =
            ChaCha8Rng::seed_from_u64(self.seed ^ batch_seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let pstride = crate::game::PLANES * NUM_CELLS;

        // ── trunk, for every position the caller does not already hold ────────
        let hstride = self.net.channels() * NUM_CELLS;
        let hit = |gi: usize| {
            cached
                .get(gi.min(cached.len().saturating_sub(1)))
                .copied()
                .flatten()
        };
        // Features for `rows`, padded to `width` by repeating the last one.
        let features = |rows: &[usize], width: usize| -> Vec<f32> {
            let mut planes = vec![0f32; width * pstride];
            for (j, &gi) in rows.iter().enumerate() {
                for (o, b) in planes[j * pstride..(j + 1) * pstride]
                    .iter_mut()
                    .zip(Minions::to_features(&states[gi]))
                {
                    *o = b as f32;
                }
            }
            let n = rows.len();
            for j in n..width {
                planes.copy_within((n - 1) * pstride..n * pstride, j * pstride);
            }
            planes
        };
        let miss: Vec<usize> = (0..g).filter(|&gi| hit(gi).is_none()).collect();
        let (values, h): (Vec<f32>, Vec<f16>) = if miss.len() == g {
            let t = Instant::now();
            let planes = features(&miss, g);
            Phases::add(&self.phases.features_ns, t);
            let t = Instant::now();
            let tr = self.net.trunk(&planes, g)?;
            Phases::add(&self.phases.trunk_ns, t);
            (tr.value, tr.h)
        } else {
            let mut values = vec![0f32; g];
            let mut h = vec![f16::ZERO; g * hstride];
            for gi in 0..g {
                if let Some((v, hc)) = hit(gi) {
                    values[gi] = v;
                    h[gi * hstride..(gi + 1) * hstride].copy_from_slice(hc);
                }
            }
            if !miss.is_empty() {
                let width = trunk_rows(miss.len()).max(miss.len());
                let t = Instant::now();
                let planes = features(&miss, width);
                Phases::add(&self.phases.features_ns, t);
                let t = Instant::now();
                let tr = self.net.trunk(&planes, width)?;
                Phases::add(&self.phases.trunk_ns, t);
                for (j, &gi) in miss.iter().enumerate() {
                    values[gi] = tr.value[j];
                    h[gi * hstride..(gi + 1) * hstride]
                        .copy_from_slice(&tr.h[j * hstride..(j + 1) * hstride]);
                }
            }
            (values, h)
        };
        let n_fresh = if cached.is_empty() {
            real
        } else {
            cached.len().min(real)
        };
        let fresh: Vec<Option<f32>> = (0..n_fresh)
            .map(|gi| (want_fresh && hit(gi).is_none()).then(|| values[gi]))
            .collect();

        // `k == 0` is a value-only expansion: run the trunk, skip the heads.
        //
        // A node's first visit needs a value and nothing else — it will not
        // choose a move until something descends through it, and 77% of nodes
        // never are. Generating a candidate there was the whole of the residual
        // waste after chunking (43.4% of candidates, one per frontier node).
        if k == 0 {
            let out = (0..real).map(|gi| (values.get(gi).copied().unwrap_or(0.0), Vec::new()));
            return Ok((
                out.collect(),
                TrunkRows {
                    stride: hstride,
                    h,
                    fresh,
                },
            ));
        }
        let k = k.max(1);
        // Forced exploration keeps its *expected* share even when candidates are
        // generated one at a time. Rounding here is what broke it: chunked
        // generation calls with `k = 1`, and `round(1 * 0.25) = 0`, so every
        // call took the diffusion path and injection silently vanished — the
        // mechanism that took kill fidelity from 3.8% to 99.6%. Taking the
        // fractional part as a coin keeps the long-run fraction exact at any
        // chunk size.
        // A best-move head proposes one turn per position; every other
        // candidate is a random playout, so there is no separate injection.
        let best_mode = self.best_mode();
        let k_rand = if full && !best_mode {
            let want = (k as f32) * self.cfg.random_fraction;
            let floor = want.floor();
            let extra = if rng.gen::<f32>() < (want - floor) {
                1
            } else {
                0
            };
            floor as usize + extra
        } else {
            0
        };
        let k_diff = k.saturating_sub(k_rand);

        // ── proposals: diffusion, batched across every chain of every game, or
        // the best-move head's one turn per position ──────────────────────────
        let is_fill = |gi: usize| fill.get(gi).copied().unwrap_or(false);
        let mut per_game: Vec<Vec<(TurnSpec, bool)>> = if best_mode {
            if (0..g).any(|gi| !is_fill(gi)) {
                self.best_declarations_batch(states, &h)?
                    .into_iter()
                    .map(|s| vec![(s, false)])
                    .collect()
            } else {
                vec![Vec::new(); g]
            }
        } else {
            self.sample_declarations_batch(states, &h, k_diff, full, &mut rng)?
                .into_iter()
                .map(|v| v.into_iter().map(|s| (s, false)).collect())
                .collect()
        };
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
            if is_fill(gi) {
                // A fill: this node's proposals already repeat, so it takes
                // exploration draws only.
                //
                // Under a best-move head these playouts are not exploration
                // but the ordinary candidates -- all but one of every node's
                // -- so they are not flagged. Flagged, each would claim an
                // explore quota (`forced_visits` apiece), and fifteen of them
                // would spend a root's whole budget on forced visits.
                decls.clear();
                for _ in 0..k {
                    decls.push((TurnSpec::random_playout(&states[gi], &mut rng), !best_mode));
                }
                continue;
            }
            if best_mode {
                // A draw asking for more than one turn: the head's, then
                // playouts.
                while decls.len() < k {
                    decls.push((TurnSpec::random_playout(&states[gi], &mut rng), false));
                }
                continue;
            }
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
        // The energy pad comes from the config, never from this batch's
        // candidate counts: best-of-N appends up to `k_candidates` extras per
        // game, and a ceiling that moved with the data would give TensorRT a
        // fresh shape per wave -- the failure mode measured at 1605 -> 52
        // expansions/s. Fixed per configuration, so still one engine.
        let extra_cap = if self.cfg.last_stage_samples > 1 && self.nnue.is_some() {
            self.cfg.k_candidates
        } else {
            0
        };
        let priors = self.score_lists(states, &h, &specs, g * (k + extra_cap), None)?;

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
        Ok((
            out,
            TrunkRows {
                stride: hstride,
                h,
                fresh,
            },
        ))
    }

    /// Raw energies for every candidate of every game, in one energy forward.
    /// `states` and `h` (one trunk row per state) are in the canonical frame.
    fn score_lists(
        &self,
        states: &[MinionsState],
        h: &[f16],
        specs: &[Vec<(TurnSpec, bool)>],
        padded: usize,
        // Each list's row in the device `h` store, when the wave leased one:
        // then `h` is not sent at all, and nothing is padded.
        store: Option<&[u32]>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if !self.cfg.use_energy {
            return Ok(specs.iter().map(|l| vec![1.0; l.len()]).collect());
        }
        let hstride = self.net.channels() * NUM_CELLS;
        let total: usize = specs.iter().map(|s| s.len()).sum();
        // A ceiling, not a hint: the rows below are resized *down* to `padded`
        // and every candidate is then assumed to have an energy. A caller that
        // grows its candidate lists past this silently loses the tail and the
        // per-list slicing indexes off the end.
        debug_assert!(
            padded >= total,
            "score_lists: padded {padded} < {total} candidates"
        );
        let dstride = crate::turn::ENERGY_PLANES * NUM_CELLS;
        let mut decl = vec![0f32; total * dstride];
        // Each candidate's `h` row, by position: the backend gathers them into
        // its input rather than this building a tiled copy first.
        let mut rows: Vec<u32> = Vec::with_capacity(padded);
        let mut at = 0usize;
        for (gi, list) in specs.iter().enumerate() {
            let live = LiveMask::of(&states[gi]);
            for (spec, _) in list {
                Partial::fully_committed(spec, &live)
                    .write_energy_planes(&mut decl[at * dstride..(at + 1) * dstride]);
                rows.push(gi as u32);
                at += 1;
            }
        }
        debug_assert!(h.len() >= specs.len() * hstride);
        // Pad to a fixed `padded` rows. Candidate counts vary after merging,
        // and TensorRT builds (and caches) a separate engine per input shape —
        // a ragged batch would rebuild constantly. The padding is a few
        // percent of wasted work against seconds per rebuild.
        let energies = if total == 0 {
            Vec::new()
        } else if let Some(srows) = store {
            let rows: Vec<u32> = rows.iter().map(|&gi| srows[gi as usize]).collect();
            let clk = Instant::now();
            let e = self.net.energy_store(&rows, &decl)?;
            Phases::add(&self.phases.energy_ns, clk);
            e
        } else {
            decl.resize(padded * dstride, 0.0);
            let last = *rows.last().expect("total > 0");
            rows.resize(padded, last);
            let clk = Instant::now();
            let mut e = self.net.energy_rows(h, &rows, &decl)?;
            Phases::add(&self.phases.energy_ns, clk);
            e.truncate(total);
            e
        };
        let mut out = Vec::with_capacity(specs.len());
        let mut off = 0usize;
        for list in specs {
            out.push(self.priors_from(&energies[off..off + list.len()]));
            off += list.len();
        }
        Ok(out)
    }

    /// One refinement chain: `spec` (canonical frame) with a share of its live
    /// variables masked again, as `resample_batch_seq` draws it. Returns the
    /// chain, the hexes its kept moves claim, and how many variables were kept
    /// and masked. `bias` is the head's logits over the fully committed turn,
    /// for biased unmasking (`SamplerConfig::resample_bias`).
    fn refine_chain(
        &self,
        cx: &GameCtx,
        spec: &TurnSpec,
        bias: Option<&[f32]>,
        rng: &mut ChaCha8Rng,
    ) -> (Partial, Bb, usize, usize) {
        let mut p = Partial::fully_committed(spec, &cx.live);
        let mut vars: Vec<(u8, u8)> = bitboard::cells(cx.live.moves)
            .map(|c| (c, C_MOVE))
            .chain(bitboard::cells(cx.live.targets).map(|c| (c, C_TARGET)))
            .chain(bitboard::cells(cx.live.spawns).map(|c| (c, C_SPAWN)))
            .collect();
        let n = vars.len();
        let m = if n == 0 {
            0
        } else if let Some(lg) = bias {
            // One independent Bernoulli per variable, at the head's own
            // probability of proposing something other than what is there.
            let mut pc: Vec<f32> = vars
                .iter()
                .map(|&(c, f)| 1.0 - p_committed_value(lg, cx, &p.spec, c, f))
                .collect();
            let mut k = 0usize;
            for i in 0..n {
                if rng.gen::<f32>() < pc[i] {
                    vars.swap(k, i);
                    pc.swap(k, i);
                    k += 1;
                }
            }
            if k == 0 {
                // Masking nothing returns the turn unchanged and wastes the
                // visit, so the least certain variable goes in regardless.
                let best = (0..n).max_by(|&a, &b| pc[a].total_cmp(&pc[b])).unwrap_or(0);
                vars.swap(0, best);
                k = 1;
            }
            k
        } else {
            let share = rng.gen::<f32>() * self.cfg.resample_frac;
            ((share * n as f32).round() as usize).clamp(1, n)
        };
        // The biased path already moved its picks to the front; only the
        // uniform path needs a shuffle to choose which `m` they are.
        if bias.is_none() {
            for i in 0..m {
                let j = i + rng.gen_range(0..n - i);
                vars.swap(i, j);
            }
        }
        let empty = TurnSpec::empty();
        for &(c, f) in &vars[..m] {
            let ci = c as usize;
            match f {
                C_MOVE => p.spec.moves[ci] = empty.moves[ci],
                C_TARGET => p.spec.targets[ci] = empty.targets[ci],
                _ => p.spec.spawns[ci] = empty.spawns[ci],
            }
            p.committed[ci] &= !f;
        }
        // The hexes the kept moves already claim, as `commit_round` would
        // have recorded them. A realized turn can hold `NO_MOVE` for a
        // unit that did not act; it claims nothing, as `write_planes`
        // gives it no offset.
        let mut cl: Bb = 0;
        for c in bitboard::cells(cx.live.moves) {
            let v = p.spec.moves[c as usize];
            if p.is_committed(c, C_MOVE) && v != 0 && (v as usize) < N_OFFSETS {
                let d = MOVE_DEST[c as usize][v as usize];
                if d != NO_CELL {
                    cl |= bit(d);
                }
            }
        }
        (p, cl, n - m, m)
    }

    /// A partial resample of each `from[i]` at `states[i]`: the search's
    /// refinement step (`SampledParams::resample_every`).
    ///
    /// A share drawn uniformly from `[0, resample_frac]` of the turn's live
    /// variables (at least one) is masked again and redrawn by `resample_t`
    /// diffusion rounds conditioned on the rest; the result is realized and
    /// scored like any candidate. One chain per position, so every forward has
    /// the shape a one-candidate widen already built.
    pub fn resample_batch_seq(
        &self,
        states: &[MinionsState],
        from: &[TurnSpec],
        batch_seq: u64,
        // Trunk rows the caller already holds, and the trunk batch width for
        // the rest -- as `expand_batch_cached` takes them.
        cached: &[Option<(f32, &[f16])>],
        trunk_rows: &dyn Fn(usize) -> usize,
    ) -> anyhow::Result<Vec<Vec<(TurnSpec, f32, bool)>>> {
        let real = states.len();
        anyhow::ensure!(from.len() == real, "resample_batch: one turn per state");
        if real == 0 {
            return Ok(Vec::new());
        }
        // Trunk rows, padded and canonical, exactly as a value-only expansion
        // gets them.
        let (_, tr) =
            self.expand_batch_cached(states, 0, true, batch_seq, cached, trunk_rows, false, &[])?;
        let g = tr.h.len() / tr.stride;
        let src = |i: usize| i.min(real - 1);
        let flipped: Vec<bool> = (0..g).map(|i| states[src(i)].side == 1).collect();
        let canon: Vec<MinionsState> = (0..g).map(|i| states[src(i)].canonical()).collect();
        let ctx: Vec<GameCtx> = canon.iter().map(Self::context).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.seed ^ batch_seq.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 0x5E5A,
        );

        // Biased unmasking needs the head's opinion of the turn as it stands,
        // so the pass is built here and probed once over the fully committed
        // declaration before anything is masked. The redraw loop below reuses
        // the same pass, so this costs exactly one extra diffusion call.
        let probe_stride = DECL_PLANES * NUM_CELLS;
        let mut pass = self.net.diffusion_pass(&tr.h, g)?;
        let bias_logits: Option<Vec<f32>> = if self.cfg.resample_bias {
            let mut decl0 = vec![0f32; g * probe_stride];
            for gi in 0..g {
                let cx = &ctx[gi];
                let spec = if flipped[gi] {
                    from[src(gi)].rotated()
                } else {
                    from[src(gi)]
                };
                Partial::fully_committed(&spec, &cx.live)
                    .write_planes(&mut decl0[gi * probe_stride..(gi + 1) * probe_stride]);
            }
            let clk = Instant::now();
            let l = pass.step(&decl0)?;
            Phases::add(&self.phases.diffusion_ns, clk);
            Some(l)
        } else {
            None
        };

        let mut chains = Vec::with_capacity(g);
        let mut claimed: Vec<Bb> = Vec::with_capacity(g);
        let mut kept = Vec::with_capacity(g);
        let mut masked = Vec::with_capacity(g);
        let mut rngs: Vec<ChaCha8Rng> = Vec::with_capacity(g);
        for gi in 0..g {
            let spec = if flipped[gi] {
                from[src(gi)].rotated()
            } else {
                from[src(gi)]
            };
            let bias = bias_logits.as_ref().map(|bl| {
                let row0 = gi * NUM_CELLS * DECL_LOGITS;
                &bl[row0..row0 + NUM_CELLS * DECL_LOGITS]
            });
            let (p, cl, n_kept, m) = self.refine_chain(&ctx[gi], &spec, bias, &mut rng);
            chains.push(p);
            claimed.push(cl);
            kept.push(n_kept);
            masked.push(m);
            rngs.push(ChaCha8Rng::seed_from_u64(rng.gen::<u64>() ^ gi as u64));
        }

        let steps = self.cfg.resample_t.max(1);
        let stride = DECL_PLANES * NUM_CELLS;
        let width = NUM_CELLS * DECL_LOGITS;
        let mut decl = vec![0f32; g * stride];
        let mut picks: Vec<(u8, u8, u8)> = Vec::with_capacity(NUM_CELLS);
        for t in 1..=steps {
            for (i, c) in chains.iter().enumerate() {
                c.write_planes(&mut decl[i * stride..(i + 1) * stride]);
            }
            let clk = Instant::now();
            let logits = pass.step(&decl)?;
            Phases::add(&self.phases.diffusion_ns, clk);
            let clk = Instant::now();
            for row in 0..g {
                if masked[row] == 0 {
                    continue;
                }
                let want = kept[row] + Self::committed_after(t, steps, masked[row]);
                commit_round(
                    &mut chains[row],
                    &mut claimed[row],
                    &ctx[row],
                    &logits[row * width..(row + 1) * width],
                    want,
                    self.cfg.temperature,
                    &mut rngs[row],
                    &mut picks,
                );
            }
            Phases::add(&self.phases.commit_ns, clk);
        }

        let specs: Vec<Vec<(TurnSpec, bool)>> = (0..g)
            .map(|gi| self.finish_candidates(&canon[gi], vec![(chains[gi].spec, false)]))
            .collect();
        let energies = self.score_lists(&canon, &tr.h, &specs, g, None)?;
        let mut out: Vec<Vec<(TurnSpec, f32, bool)>> = specs
            .into_iter()
            .zip(energies)
            .enumerate()
            .map(|(gi, (list, es))| {
                list.into_iter()
                    .zip(es)
                    .map(|((s, x), e)| (if flipped[gi] { s.rotated() } else { s }, e, x))
                    .collect()
            })
            .collect();
        out.truncate(real);
        Ok(out)
    }

    /// Whether [`Self::run_wave`] can serve this sampler. It covers the
    /// production configuration only; the rest keep the per-kind path.
    pub fn joint_ok(&self) -> bool {
        !self.best_mode()
            && !self.cfg.resample_bias
            && !(self.cfg.last_stage_samples > 1 && self.nnue.is_some())
    }

    /// One self-play wave -- value-only expansions, widens and refines --
    /// with every network call shared between them.
    ///
    /// The per-kind path (`expand_batch_cached` twice, `resample_batch_seq`)
    /// ran up to three trunk calls, `T + resample_t` diffusion calls and two
    /// energy calls per wave. Here there is one trunk call for every miss, one
    /// energy call, and one diffusion call per step of a joint schedule:
    /// widens run steps `1..=T`, refines the last `resample_t` of them. A call
    /// costs ~150 us before its first row, so 6 calls became 4.
    ///
    /// **Step-1 logits are cached.** A widen chain starts from the empty
    /// declaration, so its first step depends on `h` alone -- a node widened
    /// again (a root is widened ~200 times) recomputed the same logits every
    /// time, and a node with `k` chains computed them `k` times. Step 1 runs
    /// one row per *position* without cached logits (`CachedRow::step1`), and
    /// returns what it computed in `WaveOut::step1` for the caller to keep.
    /// Fill requests skip the diffusion head entirely, since their proposals
    /// are thrown away.
    ///
    /// The draws differ from the per-kind path's (different RNG streams, same
    /// distributions), so the two are compared by an A/B, not for equality.
    pub fn run_wave(
        &self,
        w: &Wave<'_>,
        batch_seq: u64,
        trunk_rows: &dyn Fn(usize) -> usize,
    ) -> anyhow::Result<WaveOut> {
        let (ne, nw, nr) = (w.exp.len(), w.wid.len(), w.rs.len());
        anyhow::ensure!(w.rs_from.len() == nr, "run_wave: one turn per refine");
        anyhow::ensure!(
            self.joint_ok(),
            "run_wave: configuration needs the per-kind path"
        );
        let n = ne + nw + nr;
        let real: Vec<&MinionsState> = w.exp.iter().chain(w.wid).chain(w.rs).collect();
        let flipped: Vec<bool> = real.iter().map(|s| s.side == 1).collect();
        let canon: Vec<MinionsState> = real.iter().map(|s| s.canonical()).collect();
        let hstride = self.net.channels() * NUM_CELLS;
        let hit = |i: usize| w.cached.get(i).copied().flatten();
        let mut rng =
            ChaCha8Rng::seed_from_u64(self.seed ^ batch_seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let round16 = |x: usize| (x + 15) / 16 * 16;

        // ── trunk: one call for every request without a cached row ───────────
        let mut values = vec![0f32; n];
        let mut h = vec![f16::ZERO; n * hstride];
        let mut fresh: Vec<Option<f32>> = vec![None; n];
        let mut miss: Vec<usize> = Vec::new();
        for i in 0..n {
            match hit(i) {
                Some(c) => {
                    values[i] = c.value;
                    h[i * hstride..(i + 1) * hstride].copy_from_slice(c.h);
                }
                None => miss.push(i),
            }
        }
        if !miss.is_empty() {
            let pstride = crate::game::PLANES * NUM_CELLS;
            let width = trunk_rows(miss.len()).max(miss.len());
            let t = Instant::now();
            let mut planes = vec![0f32; width * pstride];
            for (j, &i) in miss.iter().enumerate() {
                for (o, b) in planes[j * pstride..(j + 1) * pstride]
                    .iter_mut()
                    .zip(Minions::to_features(&canon[i]))
                {
                    *o = b as f32;
                }
            }
            let nm = miss.len();
            for j in nm..width {
                planes.copy_within((nm - 1) * pstride..nm * pstride, j * pstride);
            }
            Phases::add(&self.phases.features_ns, t);
            let t = Instant::now();
            let tr = self.net.trunk(&planes, width)?;
            Phases::add(&self.phases.trunk_ns, t);
            for (j, &i) in miss.iter().enumerate() {
                values[i] = tr.value[j];
                h[i * hstride..(i + 1) * hstride]
                    .copy_from_slice(&tr.h[j * hstride..(j + 1) * hstride]);
                fresh[i] = Some(values[i]);
            }
        }

        // ── chains ──────────────────────────────────────────────────────────
        // Widen game `gi` is request `ne + gi`; refine `ri` is `ne + nw + ri`.
        let k = w.k.max(1);
        let is_fill = |gi: usize| w.wid_fill.get(gi).copied().unwrap_or(false);
        let k_rand = if w.full && nw > 0 {
            let want = (k as f32) * self.cfg.random_fraction;
            let floor = want.floor();
            floor as usize + usize::from(rng.gen::<f32>() < (want - floor))
        } else {
            0
        };
        let k_diff = k.saturating_sub(k_rand);
        let tw = if w.full {
            self.cfg.t_steps
        } else {
            self.cfg.t_steps_fast
        }
        .max(1);
        let trs = self.cfg.resample_t.max(1);
        let ctx: Vec<GameCtx> = canon[ne..].iter().map(Self::context).collect();

        struct Chain {
            p: Partial,
            claimed: Bb,
            rng: ChaCha8Rng,
        }
        // Widen chains, game-major: game `gi`'s are `wc[gi * k_diff..][..k_diff]`
        // (empty for a fill, or a position with nothing to declare).
        let active_game = |gi: usize| !is_fill(gi) && k_diff > 0 && ctx[gi].n_live > 0;
        let mut wc: Vec<Chain> = Vec::with_capacity(nw * k_diff);
        for gi in 0..nw {
            for c in 0..k_diff {
                wc.push(Chain {
                    p: Partial::new(),
                    claimed: 0,
                    rng: ChaCha8Rng::seed_from_u64(rng.gen::<u64>() ^ (gi * k_diff + c) as u64),
                });
            }
        }
        // Refine chains, one per request, with `(kept, masked)`.
        let mut rc: Vec<(Chain, usize, usize)> = Vec::with_capacity(nr);
        for ri in 0..nr {
            let spec = if flipped[ne + nw + ri] {
                w.rs_from[ri].rotated()
            } else {
                w.rs_from[ri]
            };
            let (p, claimed, kept, masked) =
                self.refine_chain(&ctx[nw + ri], &spec, None, &mut rng);
            let crng = ChaCha8Rng::seed_from_u64(rng.gen::<u64>() ^ ri as u64);
            rc.push((
                Chain {
                    p,
                    claimed,
                    rng: crng,
                },
                kept,
                masked,
            ));
        }
        let any_wid = (0..nw).any(|gi| active_game(gi));
        let any_rs = rc.iter().any(|(_, _, m)| *m > 0);
        let steps = match (any_wid, any_rs) {
            (true, true) => tw.max(trs),
            (true, false) => tw,
            (false, true) => trs,
            (false, false) => 0,
        };
        let rs_from_step = steps.saturating_sub(trs) + 1;

        // Every widen and refine position's `h`, uploaded once for the wave when
        // the backend keeps a device store: the steps below then send indices,
        // and get back only the live cells' logits.
        let lease = if steps > 0 || nw + nr > 0 {
            self.net.h_lease(&h[ne * hstride..])?
        } else {
            None
        };
        // Cells a position's chains read logits at: its live move, target and
        // spawn cells, in the canonical frame.
        let live_cells: Vec<Bb> = ctx
            .iter()
            .map(|c| c.live.moves | c.live.targets | c.live.spawns)
            .collect();

        // ── joint diffusion schedule ─────────────────────────────────────────
        enum Target {
            Step1(usize),
            Wid(usize),
            Rs(usize),
        }
        let dstride = DECL_PLANES * NUM_CELLS;
        let lwidth = NUM_CELLS * DECL_LOGITS;
        let temp = self.cfg.temperature;
        let mut picks: Vec<(u8, u8, u8)> = Vec::with_capacity(NUM_CELLS);
        let mut step1: Vec<Option<Vec<f16>>> = vec![None; nw];
        let mut scratch = vec![0f32; lwidth];
        let mut rowbuf = vec![0f32; lwidth];
        let (mut decl, mut cells, mut starts) = (Vec::new(), Vec::new(), Vec::new());
        for s in 1..=steps {
            let mut rows: Vec<u32> = Vec::new();
            let mut targets: Vec<Target> = Vec::new();
            let wid_on = any_wid && s <= tw;
            let rs_on = any_rs && s >= rs_from_step;
            if wid_on {
                for gi in (0..nw).filter(|&gi| active_game(gi)) {
                    if s == 1 {
                        if hit(ne + gi).and_then(|c| c.step1).is_none() {
                            rows.push((ne + gi) as u32);
                            targets.push(Target::Step1(gi));
                        }
                    } else {
                        for c in gi * k_diff..(gi + 1) * k_diff {
                            rows.push((ne + gi) as u32);
                            targets.push(Target::Wid(c));
                        }
                    }
                }
            }
            if rs_on {
                for (ri, (_, _, m)) in rc.iter().enumerate() {
                    if *m > 0 {
                        rows.push((ne + nw + ri) as u32);
                        targets.push(Target::Rs(ri));
                    }
                }
            }
            // The live cells' logits, gathered by the backend, and where each
            // target's cells start in them.
            let logits = if rows.is_empty() {
                Vec::new()
            } else {
                let clk = Instant::now();
                decl.clear();
                decl.resize(rows.len() * dstride, 0.0);
                cells.clear();
                starts.clear();
                for (j, t) in targets.iter().enumerate() {
                    let out = &mut decl[j * dstride..(j + 1) * dstride];
                    match *t {
                        Target::Step1(_) => {} // the empty declaration
                        Target::Wid(c) => wc[c].p.write_planes(out),
                        Target::Rs(ri) => rc[ri].0.p.write_planes(out),
                    }
                    starts.push(cells.len());
                    let pos = rows[j] as usize - ne;
                    cells.extend(
                        bitboard::cells(live_cells[pos]).map(|c| (j * NUM_CELLS) as u32 + c as u32),
                    );
                }
                starts.push(cells.len());
                Phases::add(&self.phases.decl_write_ns, clk);
                let clk = Instant::now();
                let g = match &lease {
                    Some(l) => {
                        let srows: Vec<u32> =
                            rows.iter().map(|&r| l.rows[r as usize - ne]).collect();
                        self.net.diffusion_store(&srows, &decl, &cells)?
                    }
                    None if lean_on() => self.net.diffusion_cells(&h, &rows, &decl, &cells)?,
                    None => {
                        let width = round16(rows.len());
                        let mut prows = rows.clone();
                        prows.resize(width, *rows.last().unwrap());
                        let mut pdecl = decl.clone();
                        pdecl.resize(width * dstride, 0.0);
                        let full = self.net.diffusion_rows(&h, &prows, &pdecl)?;
                        crate::net::gather_rows(&full, &cells, DECL_LOGITS)
                    }
                };
                Phases::add(&self.phases.diffusion_ns, clk);
                g
            };
            let clk = Instant::now();
            // Step 1 from cached logits: no row was spent on these games.
            if wid_on && s == 1 {
                for gi in (0..nw).filter(|&gi| active_game(gi)) {
                    let Some(l1) = hit(ne + gi).and_then(|c| c.step1) else {
                        continue;
                    };
                    for (k, c) in bitboard::cells(live_cells[gi]).enumerate() {
                        let c = c as usize;
                        l1[k * DECL_LOGITS..(k + 1) * DECL_LOGITS].convert_to_f32_slice(
                            &mut scratch[c * DECL_LOGITS..(c + 1) * DECL_LOGITS],
                        );
                    }
                    let want = Self::committed_after(1, tw, ctx[gi].n_live);
                    for c in gi * k_diff..(gi + 1) * k_diff {
                        let ch = &mut wc[c];
                        commit_round(
                            &mut ch.p,
                            &mut ch.claimed,
                            &ctx[gi],
                            &scratch,
                            want,
                            temp,
                            &mut ch.rng,
                            &mut picks,
                        );
                    }
                }
            }
            for (j, t) in targets.iter().enumerate() {
                // Only live cells are ever read, and all of this row's are
                // written, so the buffer needs no clearing.
                for k in starts[j]..starts[j + 1] {
                    let c = cells[k] as usize % NUM_CELLS;
                    rowbuf[c * DECL_LOGITS..(c + 1) * DECL_LOGITS]
                        .copy_from_slice(&logits[k * DECL_LOGITS..(k + 1) * DECL_LOGITS]);
                }
                let lg: &[f32] = &rowbuf;
                match *t {
                    Target::Step1(gi) => {
                        // Committed from the fp16 values it is cached as, so
                        // a later hit reproduces this draw exactly. A no-op
                        // with fp16 graphs, whose outputs are fp16 already.
                        let l16: Vec<f16> = logits
                            [starts[j] * DECL_LOGITS..starts[j + 1] * DECL_LOGITS]
                            .iter()
                            .map(|&x| f16::from_f32(x))
                            .collect();
                        for (k, c) in bitboard::cells(live_cells[gi]).enumerate() {
                            let c = c as usize;
                            l16[k * DECL_LOGITS..(k + 1) * DECL_LOGITS].convert_to_f32_slice(
                                &mut scratch[c * DECL_LOGITS..(c + 1) * DECL_LOGITS],
                            );
                        }
                        step1[gi] = Some(l16);
                        let want = Self::committed_after(1, tw, ctx[gi].n_live);
                        for c in gi * k_diff..(gi + 1) * k_diff {
                            let ch = &mut wc[c];
                            commit_round(
                                &mut ch.p,
                                &mut ch.claimed,
                                &ctx[gi],
                                &scratch,
                                want,
                                temp,
                                &mut ch.rng,
                                &mut picks,
                            );
                        }
                    }
                    Target::Wid(c) => {
                        let gi = c / k_diff;
                        let want = Self::committed_after(s, tw, ctx[gi].n_live);
                        let ch = &mut wc[c];
                        commit_round(
                            &mut ch.p,
                            &mut ch.claimed,
                            &ctx[gi],
                            lg,
                            want,
                            temp,
                            &mut ch.rng,
                            &mut picks,
                        );
                    }
                    Target::Rs(ri) => {
                        let (ch, kept, masked) = &mut rc[ri];
                        let want =
                            *kept + Self::committed_after(s + 1 - rs_from_step, trs, *masked);
                        commit_round(
                            &mut ch.p,
                            &mut ch.claimed,
                            &ctx[nw + ri],
                            lg,
                            want,
                            temp,
                            &mut ch.rng,
                            &mut picks,
                        );
                    }
                }
            }
            Phases::add(&self.phases.commit_ns, clk);
        }

        // ── candidates: playouts, realize and merge ──────────────────────────
        let t_play = Instant::now();
        let mut declared: Vec<Vec<(TurnSpec, bool)>> = Vec::with_capacity(nw + nr);
        for gi in 0..nw {
            let st = &canon[ne + gi];
            if is_fill(gi) {
                declared.push(
                    (0..k)
                        .map(|_| (TurnSpec::random_playout(st, &mut rng), true))
                        .collect(),
                );
                continue;
            }
            let mut d: Vec<(TurnSpec, bool)> = wc[gi * k_diff..(gi + 1) * k_diff]
                .iter()
                .map(|c| (c.p.spec, false))
                .collect();
            for _ in 0..k_rand {
                d.push((TurnSpec::random_playout(st, &mut rng), true));
            }
            declared.push(d);
        }
        Phases::add(&self.phases.playout_ns, t_play);
        for (ch, _, _) in &rc {
            declared.push(vec![(ch.p.spec, false)]);
        }
        let t = Instant::now();
        let (specs, mut keys): (Vec<Vec<(TurnSpec, bool)>>, Vec<Vec<u128>>) = declared
            .into_iter()
            .enumerate()
            .map(|(j, d)| self.finish_candidates_keyed(&canon[ne + j], d, flipped[ne + j]))
            .unzip();
        Phases::add(&self.phases.realize_ns, t);

        // ── one energy call for every candidate of the wave ──────────────────
        let total: usize = specs.iter().map(|l| l.len()).sum();
        let priors = self.score_lists(
            &canon[ne..],
            &h[ne * hstride..],
            &specs,
            round16(total),
            lease.as_ref().map(|l| &l.rows[..]),
        )?;
        let mut lists: Vec<Vec<(TurnSpec, f32, bool)>> = specs
            .into_iter()
            .zip(priors)
            .enumerate()
            .map(|(j, (sp, pr))| {
                let flip = flipped[ne + j];
                sp.into_iter()
                    .zip(pr)
                    .map(|((spec, x), e)| (if flip { spec.rotated() } else { spec }, e, x))
                    .collect()
            })
            .collect();
        let rs = lists.split_off(nw);
        let rs_keys = keys.split_off(nw);
        let wid = lists
            .into_iter()
            .enumerate()
            .map(|(gi, c)| (values[ne + gi], c))
            .collect();
        Ok(WaveOut {
            values: values[..ne].to_vec(),
            wid,
            rs,
            trunk: TrunkRows {
                stride: hstride,
                h,
                fresh,
            },
            step1,
            wid_keys: keys,
            rs_keys,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::TARGET_KILL;
    use crate::units::Unit;
    use playground_core::SampledGame;

    /// `consider` draws each legal class with its tempered softmax probability,
    /// never an illegal one.
    #[test]
    fn consider_draws_the_tempered_softmax() {
        use rand::SeedableRng;
        let mut logits = vec![0f32; NUM_CELLS * DECL_LOGITS];
        let cell = 9u8;
        let row = cell as usize * DECL_LOGITS + OFF_MOVES;
        for v in 0..N_OFFSETS {
            logits[row + v] = ((v * 7) % 11) as f32 * 0.35 - 1.5;
        }
        let allowed = |v: usize| v % 5 != 3 && v < 30;
        let temp = 0.8f32;
        let z: f64 = (0..N_OFFSETS)
            .filter(|&v| allowed(v))
            .map(|v| (f64::from(logits[row + v]) / f64::from(temp)).exp())
            .sum();
        let chain = Partial::new();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut picks = Vec::new();
        let n = 400_000;
        let mut hist = [0usize; N_OFFSETS];
        for _ in 0..n {
            picks.clear();
            consider(
                &chain, cell, C_MOVE, OFF_MOVES, N_OFFSETS, allowed, &logits, temp, &mut picks,
                &mut rng,
            );
            hist[picks[0].2 as usize] += 1;
        }
        for v in 0..N_OFFSETS {
            let p = if allowed(v) {
                (f64::from(logits[row + v]) / f64::from(temp)).exp() / z
            } else {
                0.0
            };
            let f = hist[v] as f64 / n as f64;
            let se = (p * (1.0 - p) / n as f64).sqrt().max(1e-9);
            assert!(
                (f - p).abs() <= 5.0 * se,
                "class {v}: drawn {f:.5}, softmax {p:.5}"
            );
        }
    }

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
                let extra = if rng.gen::<f32>() < (want - floor) {
                    1
                } else {
                    0
                };
                total += floor as usize + extra;
            }
            let observed = total as f32 / (trials * k) as f32;
            assert!(
                (observed - fraction).abs() < 0.01,
                "k={k}: injected {observed:.3} of candidates, wanted {fraction}"
            );
        }
    }

    /// A fill row gets exploration draws only, every candidate flagged explore,
    /// while the proposal rows of the same call are untouched by it.
    #[test]
    fn fill_rows_draw_only_exploration() {
        let Some(m) = model(SamplerConfig {
            k_candidates: 4,
            random_fraction: 0.0,
            seed: 5,
            ..Default::default()
        }) else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping");
            return;
        };
        let states = vec![MinionsState::default(); 2];
        let (out, _) = m
            .expand_batch_cached(&states, 4, true, 3, &[], &|n| n, false, &[false, true])
            .unwrap();
        assert!(
            !out[0].1.is_empty() && out[0].1.iter().all(|c| !c.2),
            "with random_fraction 0 a proposal row holds no explore candidates"
        );
        assert!(
            !out[1].1.is_empty() && out[1].1.iter().all(|c| c.2),
            "a fill row holds exploration draws only"
        );
    }

    /// A widen served from cached trunk output must be exactly the widen that
    /// recomputes it: same candidates, same priors, same values, row for row.
    ///
    /// The cache is replayed by row, across the rotation that canonicalises
    /// the second seat and the padding that repeats the last row. Each is a way
    /// to hand a candidate set the wrong position's features, and a wrong `h`
    /// still produces plausible candidates -- so compare against the uncached
    /// path on the same RNG stream, with the cache full and half full (the
    /// misses then run through a padded trunk call of their own).
    #[test]
    fn cached_trunk_output_matches_recomputing_it() {
        let Some(m) = model(SamplerConfig {
            k_candidates: 4,
            seed: 5,
            ..Default::default()
        }) else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping");
            return;
        };
        let mut states = Vec::new();
        let mut st = MinionsState::default();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        while states.len() < 7 && !crate::game::Minions::outcome(&st).is_terminal() {
            states.push(st.clone());
            let spec = crate::turn::TurnSpec::random_playout(&st, &mut rng);
            st = <crate::turn::MinionsTurns as playground_core::sampled_mcts::SampledGame>::apply(
                &st, &spec,
            );
        }
        assert!(
            states.iter().any(|s| s.side == 0) && states.iter().any(|s| s.side == 1),
            "need both seats: the second is canonicalised by rotation"
        );

        // A value-only expansion, keeping what the trunk computed.
        let none: Vec<Option<(f32, &[f16])>> = vec![None; states.len()];
        let (exp, rows) = m
            .expand_batch_cached(&states, 0, true, 1, &none, &|n| n, true, &[])
            .unwrap();
        let kept: Vec<(f32, Vec<f16>)> = rows
            .fresh
            .iter()
            .enumerate()
            .map(|(i, f)| (f.expect("every miss is returned"), rows.row(i).to_vec()))
            .collect();
        for (e, (v, _)) in exp.iter().zip(&kept) {
            assert_eq!(e.0, *v);
        }

        let want = m.expand_batch_mode(&states, 2, true, 42).unwrap();
        let all: Vec<Option<(f32, &[f16])>> =
            kept.iter().map(|(v, h)| Some((*v, &h[..]))).collect();
        let half: Vec<Option<(f32, &[f16])>> = all
            .iter()
            .enumerate()
            .map(|(i, c)| if i % 2 == 0 { *c } else { None })
            .collect();
        // Rows must be far apart, or a misplaced one could pass unnoticed.
        let dist = |x: &[f16], y: &[f16]| {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p.to_f32() - q.to_f32()).abs())
                .fold(0f32, f32::max)
        };
        for i in 0..kept.len() {
            for j in i + 1..kept.len() {
                assert!(
                    dist(&kept[i].1, &kept[j].1) > 0.1,
                    "rows {i} and {j} are indistinguishable"
                );
            }
        }

        // All cached: the heads see bit-identical inputs, so everything matches exactly.
        let (got, fresh) = m
            .expand_batch_cached(
                &states,
                2,
                true,
                42,
                &all,
                &|n| n.next_multiple_of(4),
                true,
                &[],
            )
            .unwrap();
        assert!(
            fresh.fresh.iter().all(|f| f.is_none()),
            "all hits: nothing comes back fresh"
        );
        assert_eq!(want.len(), got.len());
        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            assert_eq!(w.0, g.0, "row {i} value");
            assert_eq!(w.1.len(), g.1.len(), "row {i} candidate count");
            for ((ws, wp, we), (gs, gp, ge)) in w.1.iter().zip(&g.1) {
                assert!(
                    ws == gs && wp == gp && we == ge,
                    "row {i} candidate differs"
                );
            }
        }

        // Half cached: the misses run through a trunk batch of their own, so
        // they round differently -- ~1e-7 in f32, which can tip an fp16 `h` to
        // the neighbouring value (one ulp, <0.01 at these magnitudes) -- as rows
        // do in production, where batch composition changes every wave. Hits
        // must still be exact; a miss on the wrong row would be off by >0.1.
        let (got, fresh) = m
            .expand_batch_cached(
                &states,
                2,
                true,
                42,
                &half,
                &|n| n.next_multiple_of(4),
                true,
                &[],
            )
            .unwrap();
        for (i, (c, f)) in half.iter().zip(&fresh.fresh).enumerate() {
            if c.is_some() {
                assert!(f.is_none(), "row {i}: a hit must not come back fresh");
                assert_eq!(got[i].0, kept[i].0, "row {i}: hit value");
            } else {
                let v = f.expect("a miss comes back fresh");
                let h = fresh.row(i);
                assert!(
                    (v - kept[i].0).abs() < 1e-4,
                    "row {i}: miss value {v} vs {}",
                    kept[i].0
                );
                assert!(
                    (got[i].0 - kept[i].0).abs() < 1e-4,
                    "row {i}: returned value"
                );
                assert!(
                    dist(h, &kept[i].1) < 2e-2,
                    "row {i}: miss computed for the wrong position"
                );
            }
        }
    }

    fn fixture_net() -> Option<crate::net::LocalNet> {
        let dir = std::env::var("MINIONS_PARITY_DIR").ok()?;
        crate::net::LocalNet::load(
            &std::path::Path::new(&dir).join("checkpoint.safetensors"),
            false,
        )
        .ok()
    }

    fn model(cfg: SamplerConfig) -> Option<MinionsModel> {
        Some(MinionsModel::new(Box::new(fixture_net()?), cfg))
    }

    /// A backend that emulates the device store on the CPU: a lease copies the
    /// rows, and the store calls gather them, run the plain call, and gather
    /// the named cells. `run_wave` through it must match `run_wave` without it
    /// exactly -- the row and cell bookkeeping is the only difference.
    struct StoreEmu {
        inner: crate::net::LocalNet,
        store: std::sync::Mutex<Vec<f16>>,
    }

    impl crate::net::Backend for StoreEmu {
        fn channels(&self) -> usize {
            self.inner.channels()
        }
        fn trunk(&self, planes: &[f32], b: usize) -> anyhow::Result<crate::net::TrunkOut> {
            self.inner.trunk(planes, b)
        }
        fn diffusion(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
            self.inner.diffusion(h, decl, b)
        }
        fn energy(&self, h: &[f16], decl: &[f32], b: usize) -> anyhow::Result<Vec<f32>> {
            self.inner.energy(h, decl, b)
        }
        fn diffusion_pass<'a>(
            &'a self,
            h: &'a [f16],
            b: usize,
        ) -> anyhow::Result<Box<dyn crate::net::DiffusionPass + 'a>> {
            Ok(Box::new(crate::net::NaivePass { net: self, h, b }))
        }
        fn h_lease<'a>(&'a self, h: &[f16]) -> anyhow::Result<Option<crate::net::StoreLease<'a>>> {
            let stride = self.channels() * NUM_CELLS;
            let mut st = self.store.lock().unwrap();
            // Leased rows start at an offset, as a real store's would.
            let base = st.len() / stride + 3;
            st.resize((base) * stride, f16::ZERO);
            st.extend_from_slice(h);
            let n = h.len() / stride;
            Ok(Some(crate::net::StoreLease {
                rows: (0..n).map(|i| (base + i) as u32).collect(),
                hold: Box::new(()),
            }))
        }
        fn diffusion_store(
            &self,
            rows: &[u32],
            decl: &[f32],
            cells: &[u32],
        ) -> anyhow::Result<Vec<f32>> {
            let stride = self.channels() * NUM_CELLS;
            let h = crate::net::gather_rows(&self.store.lock().unwrap(), rows, stride);
            let full = self.inner.diffusion(&h, decl, rows.len())?;
            Ok(crate::net::gather_rows(&full, cells, DECL_LOGITS))
        }
        fn energy_store(&self, rows: &[u32], decl: &[f32]) -> anyhow::Result<Vec<f32>> {
            let stride = self.channels() * NUM_CELLS;
            let h = crate::net::gather_rows(&self.store.lock().unwrap(), rows, stride);
            self.inner.energy(&h, decl, rows.len())
        }
    }

    #[test]
    fn run_wave_store_path_matches_the_plain_path() {
        let cfg = SamplerConfig {
            k_candidates: 2,
            t_steps: 4,
            resample_t: 2,
            resample_frac: 1.0,
            random_fraction: 0.25,
            seed: 13,
            ..Default::default()
        };
        let (Some(plain), Some(inner)) = (model(cfg.clone()), fixture_net()) else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping");
            return;
        };
        let emu = MinionsModel::new(
            Box::new(StoreEmu {
                inner,
                store: Default::default(),
            }),
            cfg,
        );
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut states = vec![Minions::initial_state()];
        for _ in 0..8 {
            let s = states.last().unwrap().clone();
            if Minions::outcome(&s).is_terminal() {
                break;
            }
            states.push(realize(&s, &TurnSpec::random_playout(&s, &mut rng)).next);
        }
        let n = states.len();
        let (exp, wid, rs) = (&states[..2], &states[2..n - 2], &states[n - 3..]);
        let rs_from: Vec<TurnSpec> = rs
            .iter()
            .map(|s| TurnSpec::random_playout(s, &mut rng))
            .collect();
        let fill: Vec<bool> = (0..wid.len()).map(|i| i == 1).collect();
        for full in [true, false] {
            let run = |m: &MinionsModel| {
                m.run_wave(
                    &Wave {
                        exp,
                        wid,
                        wid_fill: &fill,
                        k: 2,
                        full,
                        rs,
                        rs_from: &rs_from,
                        cached: &[],
                    },
                    9,
                    &|x| x,
                )
                .unwrap()
            };
            let (a, b) = (run(&plain), run(&emu));
            assert_eq!(a.values, b.values);
            assert_eq!(a.wid, b.wid, "widens differ (full {full})");
            assert_eq!(a.rs, b.rs, "refines differ (full {full})");
            for (x, y) in a.step1.iter().zip(&b.step1) {
                assert_eq!(x.is_some(), y.is_some());
            }
        }
    }

    /// Cached step-1 logits must reproduce the wave exactly: same seed, same
    /// candidates, energies and values whether step 1 was computed or read
    /// back from `WaveOut::step1` -- and whether the trunk ran or its rows came
    /// from the cache. A cache that changed the draws would bias self-play
    /// toward whatever the cached nodes happened to sample.
    #[test]
    fn run_wave_step1_cache_reproduces_the_wave() {
        let Some(m) = model(SamplerConfig {
            k_candidates: 2,
            t_steps: 4,
            resample_t: 2,
            resample_frac: 1.0,
            random_fraction: 0.25,
            seed: 11,
            ..Default::default()
        }) else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping");
            return;
        };
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut states = vec![Minions::initial_state()];
        for _ in 0..6 {
            let s = states.last().unwrap().clone();
            if Minions::outcome(&s).is_terminal() {
                break;
            }
            states.push(realize(&s, &TurnSpec::random_playout(&s, &mut rng)).next);
        }
        let n = states.len();
        let exp = &states[..1];
        let wid = &states[1..n - 1];
        let rs = &states[n - 2..];
        let rs_from: Vec<TurnSpec> = rs
            .iter()
            .map(|s| TurnSpec::random_playout(s, &mut rng))
            .collect();
        let fill = vec![false; wid.len()];
        let run = |cached: &[Option<CachedRow>]| {
            m.run_wave(
                &Wave {
                    exp,
                    wid,
                    wid_fill: &fill,
                    k: 2,
                    full: true,
                    rs,
                    rs_from: &rs_from,
                    cached,
                },
                7,
                &|x| x,
            )
            .unwrap()
        };
        let a = run(&[]);
        assert!(
            a.step1.iter().all(|l| l.is_some()),
            "every widen computes step 1"
        );
        // The keys handed to the search are exactly what its own `apply`
        // would compute, for both movers (the states alternate sides).
        let pairs = wid
            .iter()
            .zip(a.wid.iter().map(|(_, c)| c).zip(&a.wid_keys));
        let pairs = pairs.chain(rs.iter().zip(a.rs.iter().zip(&a.rs_keys)));
        let mut checked = 0;
        for (st, (c, k)) in pairs {
            assert_eq!(c.len(), k.len());
            for ((spec, _, _), &key) in c.iter().zip(k) {
                assert_eq!(key, MinionsTurns::state_key(&MinionsTurns::apply(st, spec)));
                checked += 1;
            }
        }
        assert!(checked > 0);
        let rows = |with_step1: bool| -> Vec<Option<CachedRow>> {
            (0..exp.len() + wid.len() + rs.len())
                .map(|i| {
                    let step1 = if with_step1 && i >= exp.len() && i < exp.len() + wid.len() {
                        a.step1[i - exp.len()].as_deref()
                    } else {
                        None
                    };
                    Some(CachedRow {
                        value: a.trunk.fresh[i].unwrap(),
                        h: a.trunk.row(i),
                        step1,
                    })
                })
                .collect()
        };
        for with_step1 in [false, true] {
            let b = run(&rows(with_step1));
            assert_eq!(a.values, b.values);
            assert_eq!(a.wid, b.wid, "widens differ (step 1 cached: {with_step1})");
            assert_eq!(a.rs, b.rs, "refines differ (step 1 cached: {with_step1})");
            if with_step1 {
                assert!(
                    b.step1.iter().all(|l| l.is_none()),
                    "a hit computes nothing"
                );
            }
        }
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
            assert!(
                opts[c as usize] & (1 << TARGET_KILL) != 0,
                "kill is always offered"
            );
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
            assert!(
                m & (1 << u as u8) != 0,
                "{} should be affordable",
                u.stats().name
            );
        }
    }

    /// A resample of a realized turn -- which can hold `NO_MOVE` for a unit
    /// that did not act -- returns one legal, scored candidate per position,
    /// from either seat, whatever share it masks.
    #[test]
    fn resampling_realized_turns_is_legal() {
        let Some(m) = model(SamplerConfig {
            seed: 5,
            ..Default::default()
        }) else {
            eprintln!("MINIONS_PARITY_DIR unset — skipping");
            return;
        };
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut states = Vec::new();
        let mut s = Minions::initial_state();
        for _ in 0..24 {
            if Minions::outcome(&s).is_terminal() {
                break;
            }
            states.push(s.clone());
            s = realize(&s, &TurnSpec::random_playout(&s, &mut rng)).next;
        }
        let from: Vec<TurnSpec> = states
            .iter()
            .map(|st| realize(st, &TurnSpec::random_playout(st, &mut rng)).spec)
            .collect();
        for seq in 0..4 {
            let out = m
                .resample_batch_seq(&states, &from, seq, &[], &|n| n)
                .unwrap();
            assert_eq!(out.len(), states.len());
            for (st, cands) in states.iter().zip(&out) {
                assert_eq!(cands.len(), 1, "one candidate per position");
                let (spec, energy, explore) = cands[0];
                assert!(energy.is_finite() && !explore);
                let r = realize(st, &spec);
                let mut check = st.clone();
                for mv in &r.actions {
                    assert!(Minions::legal_moves(&check).contains(mv), "illegal {mv}");
                    check = Minions::apply(&check, mv);
                }
            }
        }
    }

    #[test]
    fn sampled_declarations_are_collision_free_and_realizable() {
        let Some(m) = model(SamplerConfig {
            k_candidates: 8,
            seed: 3,
            ..Default::default()
        }) else {
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
                let Some((_, unit)) = s.piece_at(c as u8) else {
                    continue;
                };
                assert!(
                    d != NO_CELL && crate::map::dist(c as u8, d) <= unit.stats().speed as u32,
                    "declared a destination beyond the unit's speed"
                );
            }
        }
    }

    /// The best-move decoder commits every live variable to a class its masks
    /// offer and never sends two units to one hex -- even when every unit's
    /// top choice is the same hex, where the surest unit gets it and the rest
    /// take their next best.
    #[test]
    fn best_decoding_is_legal_and_collision_free() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut s = Minions::initial_state();
        for turn in 0..30 {
            if Minions::outcome(&s).is_terminal() {
                break;
            }
            let cx = MinionsModel::context(&s);
            let mut out: Vec<f32> = (0..NUM_CELLS * DECL_LOGITS)
                .map(|_| rng.gen_range(-1.0f32..1.0))
                .collect();
            if turn % 2 == 1 {
                let hot = rng.gen_range(0..NUM_CELLS) as u8;
                for c in 0..NUM_CELLS {
                    for v in 1..N_OFFSETS {
                        if MOVE_DEST[c][v] == hot {
                            out[c * DECL_LOGITS + OFF_MOVES + v] = 10.0 + rng.gen::<f32>();
                        }
                    }
                }
            }
            let spec = decode_best(&cx, &out);
            let mut dests = std::collections::HashSet::new();
            for c in bitboard::cells(cx.live.moves) {
                let k = spec.moves[c as usize];
                assert!(
                    k != crate::turn::NO_MOVE && cx.move_opts[c as usize] & (1 << k) != 0,
                    "cell {c}: move {k} is not on offer"
                );
                if k != 0 {
                    assert!(
                        dests.insert(MOVE_DEST[c as usize][k as usize]),
                        "two units sent to one hex"
                    );
                }
            }
            for c in bitboard::cells(cx.live.targets) {
                assert!(cx.target_opts[c as usize] & (1 << spec.targets[c as usize]) != 0);
            }
            for c in bitboard::cells(cx.live.spawns) {
                assert!(cx.spawn_opts & (1 << spec.spawns[c as usize]) != 0);
            }
            let r = realize(&s, &spec);
            let mut check = s.clone();
            for mv in &r.actions {
                assert!(Minions::legal_moves(&check).contains(mv), "illegal {mv}");
                check = Minions::apply(&check, mv);
            }
            s = realize(&s, &TurnSpec::random_playout(&s, &mut rng)).next;
        }
    }

    #[test]
    fn expand_returns_distinct_scored_candidates() {
        let Some(m) = model(SamplerConfig {
            k_candidates: 12,
            seed: 9,
            ..Default::default()
        }) else {
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
            assert!(
                keys.insert(Minions::encode_state(&next)),
                "duplicate position survived"
            );
        }
    }

    #[test]
    fn t_equals_one_degenerates_to_mean_field() {
        // Sanity on the dial: T=1 must still produce complete, legal declarations
        // — it is the mean-field baseline, not a broken configuration.
        let Some(m) = model(SamplerConfig {
            t_steps: 1,
            seed: 4,
            ..Default::default()
        }) else {
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

/// Diagnostic, run by hand: what the diffusion depth `T` does to proposals.
///
/// Positions come from 32 games the net plays against itself, each turn drawn
/// from its own candidates in proportion to `exp f`. At every third turn each
/// `T` draws 16 declarations per position from the same trunk features, and the
/// report compares them on what the realizer and the net's own heads say:
/// distinct turns; realization fidelity -- declared moves, kills and spawns that
/// actually happen, which is where two correlated variables committed
/// independently show up; the best energy and best one-turn value among the
/// draws; and how often a `T`'s draws contain the best turn of every `T`'s draws
/// pooled. `T = 100` commits one variable per round: exact any-order
/// autoregressive sampling.
///
/// `MINIONS_SWEEP_DIR=<absolute net dir> cargo test --release t_sweep --
/// --ignored --nocapture`
#[cfg(test)]
mod t_sweep {
    use super::*;
    use crate::net::TrunkOut;
    use crate::turn::ENERGY_PLANES;
    use playground_core::SampledGame;
    use std::collections::HashSet;

    const GAMES: usize = 32;
    const K: usize = 16;

    /// The depths to compare: `MINIONS_SWEEP_TS` as a comma list, else
    /// 4, 8, 16, 100. Repeating one (`4,4`) measures the sweep's own noise.
    fn ts() -> Vec<usize> {
        std::env::var("MINIONS_SWEEP_TS")
            .ok()
            .map(|v| v.split(',').map(|t| t.trim().parse().expect("T")).collect())
            .unwrap_or_else(|| vec![4, 8, 16, 100])
    }

    /// One `T`'s draws at one position.
    #[derive(Default, Clone)]
    struct Probe {
        /// Each distinct resulting position: (key, energy, one-turn value).
        distinct: Vec<(Vec<u8>, f32, f32)>,
        /// Declared / done moves, kills, spawns, summed over the draws.
        fid: [u32; 6],
    }

    fn trunk_of(m: &MinionsModel, canon: &[MinionsState]) -> TrunkOut {
        let pstride = crate::game::PLANES * NUM_CELLS;
        let mut planes = vec![0f32; canon.len() * pstride];
        for (i, s) in canon.iter().enumerate() {
            for (o, b) in planes[i * pstride..(i + 1) * pstride]
                .iter_mut()
                .zip(Minions::to_features(s))
            {
                *o = b as f32;
            }
        }
        m.net.trunk(&planes, canon.len()).unwrap()
    }

    fn score(
        m: &MinionsModel,
        canon: &[MinionsState],
        h: &[f16],
        decls: &[Vec<TurnSpec>],
    ) -> Vec<Probe> {
        let hstride = m.net.channels() * NUM_CELLS;
        let dstride = ENERGY_PLANES * NUM_CELLS;
        let mut probes = vec![Probe::default(); canon.len()];
        let mut rows: Vec<(usize, TurnSpec, MinionsState)> = Vec::new();
        for (i, (s, ds)) in canon.iter().zip(decls).enumerate() {
            let mut seen = HashSet::new();
            for d in ds {
                let r = realize(s, d);
                let f = &r.fidelity;
                let add = [
                    f.declared_moves,
                    f.done_moves,
                    f.declared_kills,
                    f.done_kills,
                    f.declared_spawns,
                    f.done_spawns,
                ];
                for (slot, v) in probes[i].fid.iter_mut().zip(add) {
                    *slot += v;
                }
                if seen.insert(Minions::encode_state(&r.next)) {
                    rows.push((i, r.spec, r.next));
                }
            }
        }
        // Energies and one-turn values in batches of the sampler's own shape.
        let batch = GAMES * K;
        for part in rows.chunks(batch) {
            let mut decl = vec![0f32; batch * dstride];
            let mut h_b = Vec::with_capacity(batch * hstride);
            for (j, (i, spec, _)) in part.iter().enumerate() {
                Partial::fully_committed(spec, &LiveMask::of(&canon[*i]))
                    .write_energy_planes(&mut decl[j * dstride..(j + 1) * dstride]);
                h_b.extend_from_slice(&h[i * hstride..(i + 1) * hstride]);
            }
            h_b.resize(batch * hstride, f16::ZERO);
            let e = m.net.energy(&h_b, &decl, batch).unwrap();
            let next: Vec<MinionsState> = (0..batch)
                .map(|j| part[j.min(part.len() - 1)].2.canonical())
                .collect();
            let tr = trunk_of(m, &next);
            for (j, (i, _, nx)) in part.iter().enumerate() {
                let v = MinionsTurns::terminal_value(nx).unwrap_or(tr.value[j]);
                let v = if nx.side != canon[*i].side { -v } else { v };
                probes[*i]
                    .distinct
                    .push((Minions::encode_state(nx), e[j], v));
            }
        }
        probes
    }

    #[test]
    #[ignore]
    fn t_sweep() {
        let dir = std::path::PathBuf::from(
            std::env::var("MINIONS_SWEEP_DIR").expect("MINIONS_SWEEP_DIR=<absolute net dir>"),
        );
        let cache = dir.join("trt_cache").to_string_lossy().into_owned();
        let net = crate::ort_net::OrtNet::new(&dir, "trt", true, &cache).unwrap();
        let mut m = MinionsModel::new(
            Box::new(net),
            SamplerConfig {
                t_steps: 4,
                t_steps_fast: 4,
                k_candidates: K,
                k_candidates_fast: K,
                k_draws: K,
                k_draws_fast: K,
                random_fraction: 0.0,
                pad_to: GAMES,
                seed: 7,
                ..Default::default()
            },
        );

        // ── positions: the net against itself, turns drawn by exp f ──────────
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut games: Vec<MinionsState> = (0..GAMES)
            .map(|g| MinionsState::with_bid(g as i16 % 7))
            .collect();
        let mut pool: Vec<MinionsState> = Vec::new();
        for turn in 0..45u64 {
            let live: Vec<usize> = (0..GAMES)
                .filter(|&g| !Minions::outcome(&games[g]).is_terminal())
                .collect();
            if live.is_empty() {
                break;
            }
            let states: Vec<MinionsState> = live.iter().map(|&g| games[g].clone()).collect();
            if turn % 3 == 1 {
                pool.extend(states.iter().cloned());
            }
            let exps = m.expand_batch_mode(&states, K, true, turn).unwrap();
            for (&g, (_, cands)) in live.iter().zip(exps) {
                let spec = if cands.is_empty() {
                    TurnSpec::empty()
                } else {
                    let mx = cands.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
                    let w: Vec<f64> = cands.iter().map(|c| f64::from(c.1 - mx).exp()).collect();
                    let mut r = rng.gen::<f64>() * w.iter().sum::<f64>();
                    let mut pick = cands.len() - 1;
                    for (i, wi) in w.iter().enumerate() {
                        if r < *wi {
                            pick = i;
                            break;
                        }
                        r -= wi;
                    }
                    cands[pick].0
                };
                let next = realize(&games[g], &spec).next;
                games[g] = if Minions::encode_state(&next) == Minions::encode_state(&games[g]) {
                    realize(&games[g], &TurnSpec::empty()).next
                } else {
                    next
                };
            }
        }

        // ── every T on the same positions and trunk features ─────────────────
        let ts = ts();
        let mut probes: Vec<Vec<Probe>> = vec![Vec::new(); ts.len()];
        let mut secs = vec![0f64; ts.len()];
        let mut batches = 0;
        for chunk in pool.chunks(GAMES) {
            let n = chunk.len();
            let canon: Vec<MinionsState> = (0..GAMES)
                .map(|i| chunk[i.min(n - 1)].canonical())
                .collect();
            let h = trunk_of(&m, &canon).h;
            for (ti, &t) in ts.iter().enumerate() {
                m.cfg.t_steps = t;
                let clk = Instant::now();
                let decls = m
                    .sample_declarations_batch(&canon, &h, K, true, &mut rng)
                    .unwrap();
                secs[ti] += clk.elapsed().as_secs_f64();
                probes[ti].extend(score(&m, &canon[..n], &h, &decls[..n]));
            }
            batches += 1;
        }

        // ── report ───────────────────────────────────────────────────────────
        let npos = probes[0].len();
        println!("{npos} positions from {GAMES} games, {K} draws per position per T");
        let best = |key: fn(&(Vec<u8>, f32, f32)) -> f32| -> Vec<(Vec<u8>, f32)> {
            (0..npos)
                .map(|i| {
                    probes
                        .iter()
                        .flat_map(|p| p[i].distinct.iter())
                        .map(|d| (d.0.clone(), key(d)))
                        .fold((Vec::new(), f32::NEG_INFINITY), |a, b| {
                            if b.1 > a.1 {
                                b
                            } else {
                                a
                            }
                        })
                })
                .collect()
        };
        let best_e = best(|d| d.1);
        let best_v = best(|d| d.2);
        for (ti, &t) in ts.iter().enumerate() {
            let p = &probes[ti];
            let mean = |f: &dyn Fn(usize) -> f64| (0..npos).map(f).sum::<f64>() / npos as f64;
            let top = |i: usize, key: fn(&(Vec<u8>, f32, f32)) -> f32| {
                p[i].distinct
                    .iter()
                    .map(key)
                    .fold(f32::NEG_INFINITY, f32::max)
            };
            let has = |i: usize, want: &[(Vec<u8>, f32)]| {
                f64::from(u8::from(p[i].distinct.iter().any(|d| d.0 == want[i].0)))
            };
            let mut fid = [0u64; 6];
            for q in p {
                for (a, b) in fid.iter_mut().zip(q.fid) {
                    *a += u64::from(b);
                }
            }
            let rate = |d: u64, n: u64| n as f64 / d.max(1) as f64;
            println!(
                "T={t:>3}: {:4.1} distinct | done/declared moves {:.3} kills {:.3} spawns \
                 {:.3} | energy: best-of-draws gap {:.2} nats, has pooled best {:.2} | \
                 one-turn value: best-of-draws {:+.3} (gap {:.3}), mean {:+.3}, has pooled \
                 best {:.2} | sampling {:.0} ms per {GAMES}x{K}",
                mean(&|i| p[i].distinct.len() as f64),
                rate(fid[0], fid[1]),
                rate(fid[2], fid[3]),
                rate(fid[4], fid[5]),
                mean(&|i| f64::from(best_e[i].1 - top(i, |d| d.1))),
                mean(&|i| has(i, &best_e)),
                mean(&|i| f64::from(top(i, |d| d.2))),
                mean(&|i| f64::from(best_v[i].1 - top(i, |d| d.2))),
                mean(&|i| {
                    let d = &p[i].distinct;
                    d.iter().map(|x| f64::from(x.2)).sum::<f64>() / d.len().max(1) as f64
                }),
                mean(&|i| has(i, &best_v)),
                secs[ti] * 1e3 / f64::from(batches),
            );
        }

        // ── per-call latency at the shapes the strategies would run at ───────
        let hstride = m.net.channels() * NUM_CELLS;
        let pstride = crate::game::PLANES * NUM_CELLS;
        let time = |f: &dyn Fn()| {
            f();
            let clk = Instant::now();
            for _ in 0..50 {
                f();
            }
            clk.elapsed().as_secs_f64() / 50.0 * 1e3
        };
        let s0 = Minions::initial_state();
        let feat: Vec<f32> = Minions::to_features(&s0)
            .iter()
            .map(|&b| b as f32)
            .collect();
        let h1 = m.net.trunk(&feat, 1).unwrap().h;
        for rows in [GAMES, GAMES * K] {
            let planes: Vec<f32> = (0..rows).flat_map(|_| feat.iter().copied()).collect();
            let h_b: Vec<f16> = (0..rows)
                .flat_map(|_| h1[..hstride].iter().copied())
                .collect();
            let decl = vec![0f32; rows * DECL_PLANES * NUM_CELLS];
            let edecl = vec![0f32; rows * ENERGY_PLANES * NUM_CELLS];
            let _ = pstride;
            println!(
                "per call at {rows} rows: trunk {:.2} ms, diffusion step {:.2} ms, energy {:.2} ms",
                time(&|| {
                    m.net.trunk(&planes, rows).unwrap();
                }),
                time(&|| {
                    m.net.diffusion(&h_b, &decl, rows).unwrap();
                }),
                time(&|| {
                    m.net.energy(&h_b, &edecl, rows).unwrap();
                }),
            );
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
            consider(
                cell,
                C_TARGET,
                OFF_TARGETS,
                N_TARGETS,
                &allow,
                &mut picks,
                rng,
            );
        }
        for cell in bitboard::cells(cx.live.spawns) {
            let so = cx.spawn_opts;
            let allow = |v: usize| so & (1 << v) != 0;
            consider(cell, C_SPAWN, OFF_SPAWNS, N_SPAWNS, &allow, &mut picks, rng);
        }
        let already = chain
            .committed
            .iter()
            .map(|c| c.count_ones() as usize)
            .sum::<usize>();
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
        let plies: usize = std::env::var("COMMIT_BENCH_PLIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);
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
                        commit_round(
                            &mut chain,
                            &mut claimed,
                            &cx,
                            &logits,
                            want,
                            1.0,
                            &mut rng,
                            &mut picks,
                        );
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
        let mut consider = |cell: u8,
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
        let already = chain
            .committed
            .iter()
            .map(|c| c.count_ones() as usize)
            .sum::<usize>();
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
                        legacy(
                            &mut chain,
                            &mut claimed,
                            &cx,
                            &logits,
                            want,
                            1.0,
                            &mut rng,
                            &mut old_picks,
                        );
                    } else {
                        commit_round(
                            &mut chain,
                            &mut claimed,
                            &cx,
                            &logits,
                            want,
                            1.0,
                            &mut rng,
                            &mut new_picks,
                        );
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
            per(la),
            per(nu),
            (la - nu) / la * 100.0,
            cx.n_live
        );
    }
}
