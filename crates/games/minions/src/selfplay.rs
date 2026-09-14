//! Self-play actor: plays games with the sampled search and appends training
//! examples to a replay buffer the Python learner reads.
//!
//! Structure follows the Hex actor (`crates/games/hex/src/selfplay.rs`), which is
//! the only training loop in this repo with real mileage: flat binary chunks
//! written `.tmp` then renamed, a sliding window pruned by **example count**
//! first, and weights hot-reloaded by mtime between flushes.
//!
//! # Record format (`MIN1`, version 1)
//!
//! Header: `magic u32 | version u32 | count u64`, then `count` fixed-stride
//! examples of [`EXAMPLE_BYTES`]:
//!
//! | field | bytes | meaning |
//! |---|---|---|
//! | `planes` | 3100 `u8` | `Minions::to_features`, already `u8` — 4× smaller than `f32` |
//! | `to_move` | 1 | 1 or 2 |
//! | `value` | 4 `f32` | game result from this position's mover's view |
//! | `live` | 100 | which variables exist here: bit 0 move, 1 target, 2 spawn |
//! | `n_cands` | 1 | how many candidate slots are populated (0 for a fast search) |
//! | `cands` | 16 × (147 + 4) | realized [`TurnSpec`] bytes and its visit share, forced visits removed |
//!
//! The candidate list is *every* child of the root, in descending target order,
//! zero targets included. Shares are visits with the forced-exploration quota
//! subtracted, so the learner never sees exploration as if the search had chosen
//! it; a child the search examined and never came back to stores 0, which is a
//! negative the energy head trains against rather than missing data. Value
//! targets come from every visit.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use playground_core::sampled_mcts::{SampledGame, SampledParams, Search, Step};
use playground_core::{Game, Outcome};

use crate::game::{Minions, MinionsState};
use crate::map::NUM_CELLS;
use crate::sampler::{MinionsModel, SamplerConfig};
use crate::turn::{MinionsTurns, TurnSpec};

pub const MAGIC: u32 = 0x4d_49_4e_33; // "MIN3"
pub const VERSION: u32 = 9;
/// Horizon for the necromancer-survival target, in side-turns.
pub const NECRO_HORIZON: usize = 10;
/// Buyable unit classes (zombie, skeleton, serpent, warg, ghost).
pub const N_BUYABLE: usize = 5;
/// Candidate slots stored per example: every child of a full-search root, zero
/// targets included. The record is fixed-stride, so this is a capacity, not a
/// selection -- it must be at least the full-search candidate count, which the
/// actor checks at startup. It was 8, and recording kept only the top 8 children
/// with non-zero targets: the energy head never saw a move the search declined,
/// and on 89% of positions it also lost children that had earned visits (up to
/// ~24% of the target's mass).
pub const TOP_K: usize = 16;
pub const SPEC_BYTES: usize = 3 * NUM_CELLS;
pub const PLANE_BYTES: usize = crate::game::PLANES * NUM_CELLS;
/// Auxiliary "rest of the game" targets, all in the canonical frame — index 0 is
/// the side to move, index 1 the opponent — so they match the feature planes and
/// the net never has to learn the mapping.
///
/// Hex control is quantised to a byte (1/255 is far finer than the noise in a
/// fraction over at most 40 turns); necromancer survival is 0/1 per horizon;
/// unit selection is a small non-negative rate and stays f32.
pub const HEX_CONTROL_BYTES: usize = 2 * NUM_CELLS;
pub const NECRO_SAFETY_BYTES: usize = 2 * NECRO_HORIZON;

pub const EXAMPLE_BYTES: usize = PLANE_BYTES
    + 1
    + 4
    + NUM_CELLS
    + 1
    + TOP_K * (SPEC_BYTES + 4)
    + HEX_CONTROL_BYTES
    + NECRO_SAFETY_BYTES
    + 1 // is_full
    + 4; // score

pub struct Example {
    pub planes: Vec<u8>,
    pub to_move: u8,
    /// Whether this position came from a *full* search and so carries a policy
    /// target. Fast positions still carry value and auxiliary targets.
    pub is_full: bool,
    pub value: f32,
    /// Which declaration variables are live here — bit 0 move, 1 target, 2
    /// spawn. Recorded rather than re-derived, so the learner never has to
    /// reimplement spawn-zone or reachability rules in Python.
    pub live: Vec<u8>,
    pub candidates: Vec<(TurnSpec, f32)>,
    /// Fraction of the remaining turns each side ends holding each hex.
    /// `[own; NUM_CELLS]` then `[enemy; NUM_CELLS]`, quantised to `0..=255`.
    pub hex_control: Vec<u8>,
    /// For each side and each horizon `m = 1..=NECRO_HORIZON`, whether that
    /// necromancer dies on or before `m` more side-turns. Monotone in `m`.
    pub necro_safety: Vec<u8>,
    /// Terminal [`crate::game::game_score`] from this example's mover, in raw
    /// points. Unlike `value`, which saturates at +/-1 the moment the game is
    /// decided, this keeps saying *how well* it was won -- so a net that
    /// already wins reliably still has a gradient telling it to win sooner and
    /// by more.
    pub score: f32,
}

impl Example {
    fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&self.planes);
        out.push(self.to_move);
        out.extend_from_slice(&self.value.to_le_bytes());
        out.extend_from_slice(&self.live);
        out.push(self.candidates.len().min(TOP_K) as u8);
        for i in 0..TOP_K {
            match self.candidates.get(i) {
                Some((spec, p)) => {
                    out.extend_from_slice(&spec.to_bytes());
                    out.extend_from_slice(&p.to_le_bytes());
                }
                None => out.extend(std::iter::repeat_n(0u8, SPEC_BYTES + 4)),
            }
        }
        out.extend_from_slice(&self.hex_control);
        out.extend_from_slice(&self.necro_safety);
        out.push(u8::from(self.is_full));
        out.extend_from_slice(&self.score.to_le_bytes());
        debug_assert_eq!(out.len() - start, EXAMPLE_BYTES);
    }
}

#[derive(Clone, Debug)]
pub struct ActorConfig {
    pub sims: u32,
    /// KR-PUCT kernel bandwidth; 0 keeps the flat PUCT search (the default).
    pub kernel_sigma: f32,
    /// Progressive widening at interior nodes; `pw_beta = 0` disables it, and
    /// `ceil(pw_beta)` is the count a node starts with.
    pub pw_alpha: f32,
    pub pw_beta: f32,
    /// Candidates per model call. 1 matches the one-per-visit widening rate;
    /// 0 disables chunking.
    pub pw_chunk: usize,
    /// Repeated sampler draws in a row before an interior widen fills with a
    /// random playout instead; 0 = only when the draw budget is spent.
    pub draw_retries: u32,
    pub fast_sims: u32,
    /// Probability a turn gets the full search — only full searches produce a
    /// policy target (KataGo's playout-cap randomization).
    pub full_search_prob: f64,
    pub c_puct: f32,
    pub k_forced: f32,
    /// **Random openings** (0 = off): each game plays `N` uniformly random legal
    /// *turns* before the search takes over, and none of them become training
    /// examples.
    ///
    /// The point is to keep the training distribution moving. Self-play comes
    /// from one frozen net, so without this every game starts from the same
    /// position and the data only changes when the net does — a Hex run stalled
    /// at parity exactly this way, and random openings took it from 0.000 to
    /// 0.625 against MoHex. Turns here, not plies, because a Minions turn is the
    /// unit of decision.
    pub random_opening_max: u32,
    /// Shape of the opening length: `P(N = n) ∝ decay^-n`, truncated to
    /// `0..=random_opening_max`. 1.0 = uniform.
    ///
    /// Tilts toward short prefixes, which sit closer to positions real games
    /// reach, while still covering deep ones often enough to keep the tail alive.
    pub random_opening_decay: f32,
    /// Visits guaranteed to each forced-exploration candidate at the root. These
    /// are subtracted again before the policy target forms, so they cost search
    /// budget but leave no trace in what the policy learns.
    pub min_explore_visits: f32,
    /// Flat forced-exploration quota per covered candidate.
    pub forced_visits: u32,
    /// Non-explore candidates drawn at random to also receive a quota.
    pub forced_extra: u32,
    /// Pool workers. Each owns a disjoint slice of the pool and runs the whole
    /// gather → expand → provide loop, so one worker's CPU phases overlap
    /// another's GPU phases. 1 restores the single-threaded actor.
    pub workers: usize,
    pub temp_moves: u32,
    pub backend: crate::net::BackendKind,
}

impl ActorConfig {
    /// The parameters of one search -- the only place a real search's
    /// parameters are built. Self-play calls it for every turn; eval and the
    /// UI call it with the config self-play recorded (see `search_config`), so
    /// none of them can search differently by accident.
    ///
    /// `full` is playout-cap randomization: a full search produces a policy
    /// target, a fast one only advances the game.
    pub fn search_params(&self, sampler: &SamplerConfig, full: bool, seed: u64) -> SampledParams {
        SampledParams {
            simulations: if full { self.sims } else { self.fast_sims },
            c_puct: self.c_puct,
            k_candidates: sampler.k_for(full),
            k_draws: sampler.draws_for(full),
            draw_retries: self.draw_retries,
            fills: true,
            // Fast searches are not training data, so they get no forced
            // exploration to correct for either.
            k_forced: if full { self.k_forced } else { 0.0 },
            // Fast searches produce no training data, so there is nothing to
            // keep the support of — and no pruning to correct it either.
            min_explore_visits: if full { self.min_explore_visits } else { 0.0 },
            // Fast searches buy no forced exploration: they carry no policy
            // target, and at 100 simulations the quotas would eat the budget.
            forced_visits: if full { self.forced_visits } else { 0 },
            forced_extra: self.forced_extra,
            temp_moves: self.temp_moves,
            pw_alpha: self.pw_alpha,
            pw_beta: self.pw_beta,
            pw_chunk: self.pw_chunk,
            kernel_sigma: self.kernel_sigma,
            seed,
        }
    }
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            sims: 200,
            fast_sims: 50,
            full_search_prob: 0.25,
            c_puct: 1.5,
            k_forced: 2.0,
            min_explore_visits: 2.0,
            forced_visits: 30,
            forced_extra: 4,
            workers: 4,
            kernel_sigma: 0.0,
            pw_alpha: 0.5,
            pw_beta: 2.0,
            pw_chunk: 1,
            draw_retries: 3,
            random_opening_max: 22,
            // Chosen so the truncated geometric on 0..=22 has mean 7.5 — a
            // quarter of `MAX_TURNS`, deep enough to reach genuinely mid-game
            // positions and tilted so short prefixes stay the most common
            // single outcome.
            //
            // These two are only meaningful against the turn cap, and the max
            // has to move with it as well as the decay: holding the max at 15
            // and solving for a mean of 7.5 drives the decay to ~1.0008, which
            // is uniform, and the tilt is the point. Scaled from 15/1.1335 when
            // the cap was 20, and 30/1.069 when it was 40.
            random_opening_decay: 1.0884,
            temp_moves: 6,
            backend: crate::net::BackendKind::Candle { cuda: false },
        }
    }
}

/// Graveyard control at game end, from `side`'s point of view: the dense
/// auxiliary target. Cheap to compute and, on Hex, one of the four changes that
/// turned a failed run into a working one.

/// A turn awaiting its value target: planes, side, live mask, candidates.
/// One recorded position, plus where in the game it sat — the auxiliary targets
/// are all forward-looking averages, so they need the index into [`TurnTrace`].
type Pending = (Vec<u8>, u8, Vec<u8>, Vec<(TurnSpec, f32)>, usize, bool);

/// What each turn leaves behind, kept for the whole game.
///
/// Only what the auxiliary targets need, decoded once here rather than re-derived
/// from a stored board: who ends the turn holding each hex, whether each
/// necromancer is still alive, and what was bought.
#[derive(Clone)]
struct TurnTrace {
    /// 0 empty, 1 first player, 2 second player.
    owner: [u8; NUM_CELLS],
    necro_alive: [bool; 2],
}

impl TurnTrace {
    fn of(state: &MinionsState, _side: u8) -> Self {
        let mut owner = [0u8; NUM_CELLS];
        let mut necro_alive = [false; 2];
        for c in 0..NUM_CELLS {
            if let Some((s, unit)) = state.piece_at(c as u8) {
                owner[c] = s + 1;
                if matches!(unit, crate::units::Unit::Necromancer) {
                    necro_alive[s as usize] = true;
                }
            }
        }
        Self { owner, necro_alive }
    }
}

/// Pack [`LiveMask`] into one byte per cell for the record.
fn live_bytes(state: &MinionsState) -> Vec<u8> {
    let live = crate::turn::LiveMask::of(state);
    let mut out = vec![0u8; NUM_CELLS];
    for c in crate::bitboard::cells(live.moves) {
        out[c as usize] |= crate::turn::C_MOVE;
    }
    for c in crate::bitboard::cells(live.targets) {
        out[c as usize] |= crate::turn::C_TARGET;
    }
    for c in crate::bitboard::cells(live.spawns) {
        out[c as usize] |= crate::turn::C_SPAWN;
    }
    out
}

/// A game in the pool: its position, the search currently running on it, and
/// the training examples it has banked so far.
pub struct Slot {
    state: MinionsState,
    search: Search<MinionsTurns>,
    /// Whether the running search is a full one (and so yields a policy target).
    full: bool,
    pending: Vec<Pending>,
    /// End-of-turn summary for every turn played so far, including the fast
    /// searches that record no example — the auxiliary targets average over the
    /// whole remaining game, not just the recorded positions.
    trace: Vec<TurnTrace>,
    turn: u32,
    rng: rand_chacha::ChaCha8Rng,
    alive: bool,
    /// Trunk output `(value, h)` for positions this slot's current search has
    /// already evaluated, keyed by `state_key`. A widen request is always for a
    /// node expanded earlier; without this it re-ran the trunk on a position
    /// whose output had just been discarded -- ~60% of trunk rows, and 45% of
    /// the FLOPs self-play spent. Cleared when the search ends, and on reload.
    trunk_cache: std::collections::HashMap<u128, (f32, Box<[half::f16]>)>,
}

/// Play a **pool** of games, batching expansions across whichever ones happen to
/// need one.
///
/// This is the Hex actor's shape: keep the pool far larger than the batch, take
/// whatever is ready, and replace finished games immediately. Lockstep — where
/// every game advances in the same round — was simpler but had two problems the
/// pool fixes:
///
/// * **The batch narrowed.** As games ended, the round shrank toward batch 1,
///   which is where the GPU does 817 positions/s instead of 115,000.
/// * **Time-to-first-flush scaled with the batch.** The whole batch finished at
///   once, so a large `concurrency` never flushed inside `--chunk-secs` at all.
///
/// Batches are also **padded to a fixed width**, because TensorRT builds and
/// caches an engine per input shape; a ragged batch would rebuild constantly.
#[allow(clippy::too_many_arguments)]
/// Drive the pool with `workers` threads, each owning a disjoint slice of it.
///
/// The phases inside one worker are inherently alternating — a diffusion round
/// cannot start until the previous round's commit lands — so a single worker
/// leaves the GPU idle for every commit, realize and playout it does. Measured,
/// that is a third of wall time. Independent workers fill each other's gaps:
/// while one is on CPU the other is on GPU, and the backend's per-session mutex
/// serialises the device without any scheduling of our own.
///
/// Slices are disjoint, so no slot is touched by two workers, and every worker
/// gets its own seed stream and stats. This is only sound because an expansion
/// is now a pure function of its arguments — see
/// [`MinionsModel::expand_batch_mode`], which took the search mode and the RNG
/// out of shared state precisely so batches could interleave.
pub fn drive_pool(
    model: &MinionsModel,
    cfg: &ActorConfig,
    slots: &mut [Slot],
    max_batch: usize,
    next_seed: &mut u64,
    deadline: Instant,
    stats: &mut PoolStats,
) -> anyhow::Result<Vec<Example>> {
    let workers = cfg.workers.max(1);
    if workers == 1 {
        return drive_pool_worker(model, cfg, slots, max_batch, next_seed, deadline, stats);
    }
    // Each worker needs `2 * max_batch` slots of its own for `pick_mode` to keep
    // guaranteeing zero padding; fall back to one worker rather than pad.
    let per = slots.len() / workers;
    if per < 2 * max_batch {
        return drive_pool_worker(model, cfg, slots, max_batch, next_seed, deadline, stats);
    }

    let base = *next_seed;
    let mut chunks: Vec<&mut [Slot]> = slots.chunks_mut(per).collect();
    chunks.truncate(workers);

    let results: Vec<anyhow::Result<(Vec<Example>, PoolStats, u64)>> =
        std::thread::scope(|sc| {
            let handles: Vec<_> = chunks
                .into_iter()
                .enumerate()
                .map(|(w, chunk)| {
                    sc.spawn(move || {
                        // Disjoint seed streams: worker `w` counts from its own
                        // base, so two workers never draw the same batch seed.
                        let mut seed = base.wrapping_add((w as u64) << 40);
                        let mut st = PoolStats::default();
                        let out = drive_pool_worker(
                            model, cfg, chunk, max_batch, &mut seed, deadline, &mut st,
                        )?;
                        Ok((out, st, seed))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("pool worker panicked")).collect()
        });

    let mut out = Vec::new();
    for r in results {
        let (examples, st, seed) = r?;
        out.extend(examples);
        stats.merge(&st);
        *next_seed = (*next_seed).max(seed);
    }
    Ok(out)
}

fn drive_pool_worker(
    model: &MinionsModel,
    cfg: &ActorConfig,
    slots: &mut [Slot],
    max_batch: usize,
    next_seed: &mut u64,
    deadline: Instant,
    stats: &mut PoolStats,
) -> anyhow::Result<Vec<Example>> {
    use rand::{Rng, SeedableRng};
    let pool_size = slots.len().max(1);
    let max_batch = max_batch.max(1).min(pool_size);
    // Trunk-output cache budget for this worker's slice, in bytes.
    // `MINIONS_TRUNK_CACHE_MB=0` turns it off; it is read at runtime, like
    // `MINIONS_SHAPE_Q`, so both arms of an A/B run one binary.
    let cache_budget: usize = std::env::var("MINIONS_TRUNK_CACHE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8192)
        .saturating_mul(1 << 20)
        / cfg.workers.max(1);
    let use_cache = cache_budget > 0 && TRUNK_CACHE_USE.load(std::sync::atomic::Ordering::Relaxed);
    // Slots keep their caches across flush windows, so count what they hold.
    let mut cache_bytes: usize = slots.iter().map(cache_size).sum();
    let seed = *next_seed;

    let start_search = |state: &MinionsState,
                        turn: u32,
                        rng: &mut rand_chacha::ChaCha8Rng|
     -> (Search<MinionsTurns>, bool) {
        let full = rng.gen_bool(cfg.full_search_prob);
        let params = cfg.search_params(&model.cfg, full, seed ^ (turn as u64));
        (Search::new(state.clone(), &params, turn), full)
    };

    let mut out: Vec<Example> = Vec::new();
    let mut cursor = 0usize;
    // Full and fast searches want different diffusion depths, so a batch must
    // hold one kind. Which kind to gather is chosen per batch by *supply* — see
    // the selection below; strict alternation starves one side.
    let mut want_full = true;
    let mut gathered_either = false;

    // Run until the flush deadline. Games still in progress stay in their slots
    // and continue into the next chunk — abandoning them would throw away work
    // *and* leave them without the outcome their value target needs.
    while Instant::now() < deadline {
        // Pick this batch's kind by supply rather than strict alternation.
        //
        // A slot sits in full mode ~4x longer than fast (200 sims per turn
        // against 50), so alternating 1:1 over-serves fast: fast slots churn
        // through turns, and every turn re-rolls the mode with 25% landing on
        // full, so the pool ratchets toward full until fewer than `max_batch`
        // slots are left in fast. Every fast batch after that is padded —
        // measured at 30-40%, i.e. that share of GPU work discarded. Slots are
        // never retired (`alive` is only ever set true), so these counts are
        // exact and the scan is cheap next to an expansion.
        let (mut n_full, mut n_fast) = (0usize, 0usize);
        for s in slots.iter() {
            if !s.alive {
                continue;
            }
            if s.full {
                n_full += 1;
            } else {
                n_fast += 1;
            }
        }
        want_full = pick_mode(n_full, n_fast, max_batch, want_full);

        // Gather up to `max_batch` positions that need expanding, sweeping the
        // pool from where we left off so no slot is starved.
        let mut ids: Vec<usize> = Vec::with_capacity(max_batch);
        let mut states: Vec<MinionsState> = Vec::with_capacity(max_batch);
        // Parallel to `ids`: whether this request is a widen (more candidates
        // for a node that earned them) rather than a first expansion. Both are
        // the same shape, so they share one batch — which is the whole point,
        // since it lets a wave cover more nodes instead of over-serving few.
        let mut is_widen: Vec<bool> = Vec::with_capacity(max_batch);
        // Per request: a widen for a node whose proposals have repeated, which
        // takes exploration draws instead (`SampledModel::widen_batch`).
        let mut fills: Vec<bool> = Vec::with_capacity(max_batch);
        let mut scanned = 0usize;
        // Each kind gets its own fixed width, and the two sum to `max_batch`.
        // Padding both to the full width — which is what a single width forces —
        // ran the trunk twice over `max_batch` rows and the diffusion head at
        // 46% filler. Complementary widths keep both shapes fixed while halving
        // the trunk rows per wave and taking the head padding to ~0.
        // Batch-shape strategy, switchable at runtime so the two can be paired in
        // one ABBA run rather than compared across rebuilds (this box drifts
        // ~30% with heat, which is larger than either effect being measured).
        //
        // 0  = pad both calls to `max_batch`: one shape, but the head call runs
        //      ~46% filler because the widen share of a wave varies.
        // >0 = round each group up to this multiple: padding bounded by the
        //      quantum, at the cost of an engine per shape.
        //
        // Measured ABBA (3 reversed pairs, cooldown gate): quantising to 16 is
        // **+13.6% / +16.1% / +20.8%** over the single shape. The engine per
        // shape is a one-time build that TensorRT caches; the head padding it
        // removes is paid every wave. Unpaired runs said the opposite by a wide
        // margin — this box drifts ~30% with heat, so do not trust a comparison
        // that is not paired and cooled.
        let shape_q: usize = std::env::var("MINIONS_SHAPE_Q")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let width_for = |n: usize| {
            if shape_q == 0 {
                max_batch
            } else {
                ((n + shape_q - 1) / shape_q * shape_q).max(shape_q).min(max_batch)
            }
        };
        while ids.len() < max_batch && scanned < pool_size {
            let i = cursor % pool_size;
            cursor = cursor.wrapping_add(1);
            scanned += 1;
            if !slots[i].alive || slots[i].full != want_full {
                continue;
            }
            // Advance this slot until it wants an expansion, finishing as many
            // turns (and games) as it can without touching the GPU.
            loop {
                match slots[i].search.step() {
                    Step::NeedExpansion(st) => {
                        ids.push(i);
                        states.push(st);
                        is_widen.push(false);
                        fills.push(false);
                        break;
                    }
                    Step::NeedWiden { state, fill } => {
                        ids.push(i);
                        states.push(state);
                        is_widen.push(true);
                        fills.push(fill);
                        break;
                    }
                    Step::Done => {
                        // The tree this cache described is gone.
                        cache_bytes = cache_bytes.saturating_sub(cache_drop(&mut slots[i]));
                        if !advance_turn(&mut slots[i], &start_search, stats) {
                            // Game over: bank it and refill the slot, so the
                            // batch never narrows while work remains.
                            out.extend(examples_from(
                                std::mem::take(&mut slots[i].pending),
                                &slots[i].trace,
                                &slots[i].state,
                            ));
                            // Refill immediately, so the batch never narrows
                            // while there is still work to do.
                            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(*next_seed);
                            *next_seed = next_seed.wrapping_add(1);
                            let (state, opening) = random_opening(
                                cfg.random_opening_max,
                                cfg.random_opening_decay,
                                &mut rng,
                            );
                            let (search, full) = start_search(&state, opening, &mut rng);
                            slots[i] = Slot {
                                state,
                                search,
                                full,
                                pending: Vec::new(),
                                trace: Vec::new(),
                                turn: opening,
                                rng,
                                alive: true,
                                trunk_cache: Default::default(),
                            };
                        }
                    }
                }
            }
        }
        if ids.is_empty() {
            // Nothing of this kind is ready; try the other before giving up.
            want_full = !want_full;
            if !gathered_either {
                break;
            }
            gathered_either = false;
            continue;
        }
        gathered_either = true;
        let keys: Vec<u128> = if cache_budget > 0 {
            states.iter().map(<MinionsTurns as SampledGame>::state_key).collect()
        } else {
            Vec::new()
        };

        // Each kind is padded to its own width below; padding the combined list
        // here as well would only clone states nothing reads.
        let real = ids.len();
        stats.batches += 1;
        stats.expansions += real as u64;
        stats.padded += (max_batch.saturating_sub(real)) as u64;
        // One `k` for the whole wave keeps the tensor shape fixed — the
        // constraint that killed an earlier overlap scheme, which set TensorRT
        // rebuilding an engine per shape and cost 97% of throughput.
        // Request sizes come from the same rule the single-game search uses
        // (`chunk_k` / `expansion_k`), applied to this wave's kind of search.
        let wave = cfg.search_params(&model.cfg, want_full, 0);
        let k_call = playground_core::sampled_mcts::chunk_k(&wave);
        // Two calls, split by what each request actually needs. Both are padded
        // to `max_batch`, so both keep a fixed tensor shape and TensorRT builds
        // one engine apiece.
        //
        // The split is the point: an expansion needs only a value, so it runs
        // the trunk and stops. Only widens — nodes that something is actually
        // descending through — pay for the diffusion and energy heads, which are
        // ~70% of the loop. A widen's trunk output comes from the slot's cache
        // when it has one: recomputing it was 51% of GPU time at the
        // 2026-09-12 profile, not the ~5% this split was designed around.
        let exp_idx: Vec<usize> = (0..real).filter(|&i| !is_widen[i]).collect();
        let wid_idx: Vec<usize> = (0..real).filter(|&i| is_widen[i]).collect();
        stats.req_expand += exp_idx.len() as u64;
        stats.req_widen += wid_idx.len() as u64;
        if !wid_idx.is_empty() {
            stats.head_real += wid_idx.len() as u64;
            stats.head_run += width_for(wid_idx.len()) as u64;
        }

        let mut values = vec![0.0f32; real];
        if !exp_idx.is_empty() {
            let mut st: Vec<MinionsState> = exp_idx.iter().map(|&i| states[i].clone()).collect();
            let n = st.len();
            let w = width_for(n);
            while st.len() < w {
                st.push(st[n - 1].clone());
            }
            let exp_k = playground_core::sampled_mcts::expansion_k(&wave);
            let cached = if use_cache { cache_lookup(&*slots, &ids, &keys, &exp_idx) } else { Vec::new() };
            stats.cache_hits += cached.iter().filter(|c| c.is_some()).count() as u64;
            let (r, fresh) = model.expand_batch_cached(
                &st, exp_k, want_full, *next_seed, &cached, &width_for, cache_budget > 0, &[],
            )?;
            drop(cached);
            *next_seed = next_seed.wrapping_add(1);
            for (slot, (v, _)) in exp_idx.iter().zip(r) {
                values[*slot] = v;
            }
            cache_store(slots, &ids, &keys, &exp_idx, &fresh, cache_budget, &mut cache_bytes, stats);
        }
        let mut cands_for: Vec<Option<Vec<(crate::turn::TurnSpec, f32, bool)>>> =
            (0..real).map(|_| None).collect();
        if !wid_idx.is_empty() {
            let mut st: Vec<MinionsState> = wid_idx.iter().map(|&i| states[i].clone()).collect();
            let n = st.len();
            let w = width_for(n);
            while st.len() < w {
                st.push(st[n - 1].clone());
            }
            let cached = if use_cache { cache_lookup(&*slots, &ids, &keys, &wid_idx) } else { Vec::new() };
            stats.cache_hits += cached.iter().filter(|c| c.is_some()).count() as u64;
            // Padding rows are proposals; their output is dropped either way.
            let fill: Vec<bool> = wid_idx.iter().map(|&i| fills[i]).collect();
            let (r, fresh) = model.expand_batch_cached(
                &st, k_call, want_full, *next_seed, &cached, &width_for, use_cache, &fill,
            )?;
            drop(cached);
            *next_seed = next_seed.wrapping_add(1);
            for (slot, (_, c)) in wid_idx.iter().zip(r) {
                cands_for[*slot] = Some(c);
            }
            cache_store(slots, &ids, &keys, &wid_idx, &fresh, cache_budget, &mut cache_bytes, stats);
        }
        for (n, (i, widen)) in ids.into_iter().zip(is_widen).enumerate() {
            if widen {
                slots[i].search.provide_widen(cands_for[n].take().unwrap_or_default());
            } else {
                slots[i].search.provide(values[n], Vec::new());
            }
        }
        // The next kind is chosen from supply at the top of the loop.
    }
    Ok(out)
}

/// Whether this flush window reads the trunk cache. `run_loop` sets it per
/// window; it is only ever false while an in-generation A/B is running (see
/// `TRUNK_CACHE_AB_FILE`).
static TRUNK_CACHE_USE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static WINDOWS_RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// While this file exists in the live dir, windows alternate cache off/on in
/// A B B A order, so a normal generation doubles as a paired benchmark: the
/// data is exactly production data, and adjacent windows share the box's
/// thermal state (which drifts ~30%, more than most effects measured here).
/// Off windows still *store* expansion output and only skip the lookups, so an
/// on window never starts cold. Delete the file and every window uses it.
const TRUNK_CACHE_AB_FILE: &str = "trunk_cache_ab";
/// The same A B B A switch between the fp16-edge graphs and an fp32 export of
/// the same net in `best/fp32/` (`ort_net::GRAPH_F16_USE`). Needs that
/// directory; without it the switch has nothing to switch to.
const GRAPH_F16_AB_FILE: &str = "graph_f16_ab";

fn entry_bytes(h_len: usize) -> usize {
    h_len * std::mem::size_of::<half::f16>() + 64
}

fn cache_size(slot: &Slot) -> usize {
    slot.trunk_cache.values().map(|(_, h)| entry_bytes(h.len())).sum()
}

/// Freed cache blocks, reused rather than handed back to the allocator.
///
/// Entries are all one size and churn at ~10k/s. Allocating each one
/// interleaved them with the per-wave multi-MB buffers in glibc's thread
/// arenas, which grew to 17.2 GB (276 x 64 MB heaps) around an 8 GB cache.
/// Recycling caps the footprint at the peak live cache, which the budget bounds.
static BLOCK_POOL: std::sync::Mutex<Vec<Box<[half::f16]>>> = std::sync::Mutex::new(Vec::new());

fn take_blocks(n: usize) -> Vec<Box<[half::f16]>> {
    if n == 0 {
        return Vec::new();
    }
    let mut pool = BLOCK_POOL.lock().unwrap_or_else(|e| e.into_inner());
    let keep = pool.len().saturating_sub(n);
    pool.split_off(keep)
}

fn return_blocks(mut blocks: Vec<Box<[half::f16]>>) {
    if !blocks.is_empty() {
        BLOCK_POOL.lock().unwrap_or_else(|e| e.into_inner()).append(&mut blocks);
    }
}

/// Empty a slot's trunk cache into the block pool, returning the bytes it held.
fn cache_drop(slot: &mut Slot) -> usize {
    let b = cache_size(slot);
    if !slot.trunk_cache.is_empty() {
        return_blocks(slot.trunk_cache.drain().map(|(_, (_, h))| h).collect());
    }
    b
}

/// Cached trunk output for each request in `idx` (indices into `ids`/`keys`).
/// Empty when the cache is off, which the sampler reads as "nothing cached".
fn cache_lookup<'a>(
    slots: &'a [Slot],
    ids: &[usize],
    keys: &[u128],
    idx: &[usize],
) -> Vec<Option<(f32, &'a [half::f16])>> {
    if keys.is_empty() {
        return Vec::new();
    }
    idx.iter()
        .map(|&j| slots[ids[j]].trunk_cache.get(&keys[j]).map(|(v, h)| (*v, &h[..])))
        .collect()
}

/// Keep the trunk output the sampler just computed, while the budget allows.
/// Over budget the entry is simply not kept: a miss recomputes, so the budget
/// bounds memory without ever changing a result.
#[allow(clippy::too_many_arguments)]
fn cache_store(
    slots: &mut [Slot],
    ids: &[usize],
    keys: &[u128],
    idx: &[usize],
    rows: &crate::sampler::TrunkRows,
    budget: usize,
    bytes: &mut usize,
    stats: &mut PoolStats,
) {
    let want = rows.fresh.iter().take(idx.len()).filter(|f| f.is_some()).count();
    let mut blocks = take_blocks(want);
    let b = entry_bytes(rows.stride);
    for (n, &j) in idx.iter().enumerate() {
        let Some(v) = rows.fresh.get(n).copied().flatten() else { continue };
        stats.cache_misses += 1;
        if *bytes + b > budget {
            stats.cache_full += 1;
            continue;
        }
        let mut blk = match blocks.pop() {
            Some(blk) if blk.len() == rows.stride => blk,
            _ => vec![half::f16::ZERO; rows.stride].into_boxed_slice(),
        };
        blk.copy_from_slice(rows.row(n));
        match slots[ids[j]].trunk_cache.insert(keys[j], (v, blk)) {
            Some((_, old)) => return_blocks(vec![old]),
            None => *bytes += b,
        }
    }
    return_blocks(blocks);
    stats.cache_peak_bytes = stats.cache_peak_bytes.max(*bytes as u64);
}

/// Steady-state progress, independent of whether games happen to finish inside
/// a flush window. With a large pool a cold start produces no completed games
/// for a long time, so example counts alone measure the transient rather than
/// throughput.
#[derive(Default, Clone, Copy, Debug)]
pub struct PoolStats {
    pub batches: u64,
    /// Forced-exploration candidates that reached a recorded (full-search) root.
    pub explore_seen: u64,
    /// ...and how many survived policy-target pruning, i.e. earned enough organic
    /// visits to be worth teaching. Near zero means the injected candidates are
    /// wasted budget; near one means the sampler is badly under-covering and the
    /// search prefers what exploration finds. Either extreme is a signal.
    pub explore_promoted: u64,
    /// Share of the policy target's mass that landed on explore candidates.
    pub explore_mass: f64,
    pub explore_nodes: u64,
    /// Real (unpadded) positions expanded.
    pub expansions: u64,
    /// Padding rows — wasted work, traded for a fixed TensorRT input shape.
    pub padded: u64,
    /// Requests that only wanted a value (a node's first visit) versus those
    /// that wanted a candidate. The split decides how much of each wave's
    /// generative-head work is padding: the head call is padded to `max_batch`,
    /// so a wave that is mostly expansions runs the diffusion head almost
    /// entirely on filler.
    pub req_expand: u64,
    pub req_widen: u64,
    /// Head rows actually wanted, and head rows run. The gap is the padding
    /// waste in the most expensive phase.
    pub head_real: u64,
    pub head_run: u64,
    /// Trunk-cache lookups that hit, rows the trunk ran on with the cache on,
    /// and of those the ones not kept because the budget was full.
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_full: u64,
    /// Largest cache held, summed over workers.
    pub cache_peak_bytes: u64,
    pub turns: u64,
    pub games: u64,
}

impl PoolStats {
    /// Fold another worker's counters in.
    pub fn merge(&mut self, o: &PoolStats) {
        self.batches += o.batches;
        self.expansions += o.expansions;
        self.padded += o.padded;
        self.turns += o.turns;
        self.games += o.games;
        self.explore_seen += o.explore_seen;
        self.explore_promoted += o.explore_promoted;
        self.explore_mass += o.explore_mass;
        self.explore_nodes += o.explore_nodes;
        self.req_expand += o.req_expand;
        self.req_widen += o.req_widen;
        self.head_real += o.head_real;
        self.head_run += o.head_run;
        self.cache_hits += o.cache_hits;
        self.cache_misses += o.cache_misses;
        self.cache_full += o.cache_full;
        self.cache_peak_bytes += o.cache_peak_bytes;
    }

    /// Where each wave's requests went, and how much of the generative-head
    /// work was filler. `head` padding is the expensive kind: the diffusion and
    /// energy heads are the bulk of the loop.
    pub fn split_report(&self) -> String {
        let reqs = (self.req_expand + self.req_widen).max(1);
        let head_waste = if self.head_run == 0 {
            0.0
        } else {
            100.0 * (1.0 - self.head_real as f64 / self.head_run as f64)
        };
        let mut s = format!(
            "requests: {:.0}% value-only, {:.0}% candidate  |  head rows {} of {} run ({:.0}% padding)",
            100.0 * self.req_expand as f64 / reqs as f64,
            100.0 * self.req_widen as f64 / reqs as f64,
            self.head_real,
            self.head_run,
            head_waste
        );
        let looked = self.cache_hits + self.cache_misses;
        if looked > 0 {
            s += &format!(
                "  |  trunk cache {:.1}% hit ({} of {}), {} not kept, peak {:.0} MB",
                100.0 * self.cache_hits as f64 / looked as f64,
                self.cache_hits,
                looked,
                self.cache_full,
                self.cache_peak_bytes as f64 / (1u64 << 20) as f64
            );
        }
        s
    }

    pub fn report(&self, secs: f64) -> String {
        format!(
            "{:.0} expansions/s, {:.1} turns/s, {:.2} games/s, avg batch {:.1} ({:.0}% padding)",
            self.expansions as f64 / secs,
            self.turns as f64 / secs,
            self.games as f64 / secs,
            self.expansions as f64 / self.batches.max(1) as f64,
            100.0 * self.padded as f64 / (self.expansions + self.padded).max(1) as f64,
        )
    }

    /// Whether forced exploration is paying for itself.
    ///
    /// `promoted` is the share of injected candidates that survived policy-target
    /// pruning — i.e. earned organic visits *after* their guaranteed ones were
    /// subtracted. Read it alongside `mass`, the share of the target's
    /// probability that landed on them. Near zero on both means the injection is
    /// wasted budget; consistently high means the learned sampler is
    /// under-covering and search prefers what exploration turns up.
    pub fn explore_report(&self) -> String {
        if self.explore_nodes == 0 {
            return "explore: none recorded".to_string();
        }
        format!(
            "explore: {:.1}% promoted ({} of {}), {:.1}% of target mass, {:.1} per node",
            100.0 * self.explore_promoted as f64 / self.explore_seen.max(1) as f64,
            self.explore_promoted,
            self.explore_seen,
            100.0 * self.explore_mass / self.explore_nodes as f64,
            self.explore_seen as f64 / self.explore_nodes as f64,
        )
    }
}

/// Which kind of search the next batch should serve.
///
/// Prefers a kind that can fill a whole batch, alternating when both can so
/// neither side is starved. Since every slot is counted in exactly one of
/// `n_full`/`n_fast`, the padding case is unreachable whenever the pool holds at
/// least `2 * max_batch` slots.
fn pick_mode(n_full: usize, n_fast: usize, max_batch: usize, prev_full: bool) -> bool {
    match (n_full >= max_batch, n_fast >= max_batch) {
        (true, true) => !prev_full,
        (true, false) => true,
        (false, true) => false,
        // Neither can fill one; keep the current kind and accept padding.
        (false, false) => prev_full,
    }
}

/// Play `n` uniformly random legal turns from the opening, resampling if the
/// prefix ends the game — a finished prefix leaves nothing to train on.
///
/// Uses [`TurnSpec::random_playout`] so each opening turn is a real sequence of
/// legal moves rather than a declaration that mostly fails to realize.
fn random_opening(
    max: u32,
    decay: f32,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> (MinionsState, u32) {
    // Every game also draws its own starting-money handicap, so the value head
    // sees the whole range it will be asked to price during bidding rather than
    // one fixed number. See `MinionsState::with_bid`.
    let bid = draw_bid(rng);
    if max == 0 {
        return (MinionsState::with_bid(bid), 0);
    }
    for _ in 0..16 {
        let n = playground_core::batch_search::sample_geometric_len(max, decay, rng);
        let mut state = MinionsState::with_bid(bid);
        let mut played = 0u32;
        for _ in 0..n {
            if Minions::outcome(&state).is_terminal() {
                break;
            }
            let spec = TurnSpec::random_playout(&state, rng);
            state = crate::turn::realize(&state, &spec).next;
            played += 1;
        }
        if !Minions::outcome(&state).is_terminal() {
            return (state, played);
        }
    }
    (MinionsState::with_bid(bid), 0)
}

/// Draw a starting-money handicap for one game, uniform over the bid range.
fn draw_bid(rng: &mut rand_chacha::ChaCha8Rng) -> i16 {
    use rand::Rng;
    rng.gen_range(crate::game::BID_MIN..=crate::game::BID_MAX)
}

/// Build a fresh pool of `n` games.
pub fn new_pool(model: &MinionsModel, cfg: &ActorConfig, n: usize, seed: u64) -> Vec<Slot> {
    use rand::{Rng, SeedableRng};
    (0..n.max(1))
        .map(|i| {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));
            let (state, opening) =
                random_opening(cfg.random_opening_max, cfg.random_opening_decay, &mut rng);
            let full = rng.gen_bool(cfg.full_search_prob);
            let params = cfg.search_params(&model.cfg, full, seed ^ i as u64);
            let search = Search::new(state.clone(), &params, opening);
            // `turn` starts at the opening length: it drives the temperature
            // schedule, and an opening already spent those plies of exploration.
            Slot {
                state,
                search,
                full,
                pending: Vec::new(),
                trace: Vec::new(),
                turn: opening,
                rng,
                alive: true,
                trunk_cache: Default::default(),
            }
        })
        .collect()
}

/// Starts the next search on a slot: `(search, is_full_search)`.
type StartSearch<'a> =
    &'a dyn Fn(&MinionsState, u32, &mut rand_chacha::ChaCha8Rng) -> (Search<MinionsTurns>, bool);

/// A slot's search finished: record its policy target, play the move, and start
/// the next search. Returns false when the game is over.
fn advance_turn(
    slot: &mut Slot,
    start_search: StartSearch,
    stats: &mut PoolStats,
) -> bool {
    let finished = std::mem::replace(
        &mut slot.search,
        Search::new(slot.state.clone(), &SampledParams { simulations: 0, ..Default::default() }, 0),
    );
    // Taken from the search itself, so the target can only ever subtract the
    // floor this search actually ran with.
    let ran_with = *finished.params();
    let result = finished.finish();
    if result.candidates.is_empty() {
        return false;
    }
    // Empty for a fast search: no policy target, but the position is still
    // recorded for its value and auxiliary targets.
    let mut policy: Vec<(TurnSpec, f32)> = Vec::new();
    if slot.full {
        // Subtract the floor this search actually granted, not a default one:
        // selection and pruning must agree exactly, or forced visits survive into
        // the target as if they were search's own judgement.
        let target = result.policy_target_with(&ran_with);
        let scored = policy_record(
            result.candidates.iter().map(|(spec, _)| *spec).zip(target.iter().copied()),
        );
        // Promotion rate: did forced exploration actually contribute anything the
        // policy learns from? The pruning has already subtracted the guaranteed
        // visits, so surviving here means the candidate earned organic visits on
        // merit — which is the only evidence that injecting it was worth the
        // budget.
        let n_explore = result.explore.iter().filter(|&&e| e).count();
        if n_explore > 0 {
            stats.explore_nodes += 1;
            stats.explore_seen += n_explore as u64;
            let total: f32 = target.iter().sum();
            let mut mass = 0.0f32;
            for (i, &e) in result.explore.iter().enumerate() {
                if e && target.get(i).copied().unwrap_or(0.0) > 0.0 {
                    stats.explore_promoted += 1;
                    mass += target[i];
                }
            }
            if total > 0.0 {
                stats.explore_mass += (mass / total) as f64;
            }
        }

        policy = scored;
    }

    // Record *every* position, full or fast — playout-cap randomization.
    //
    // A fast search's visit counts are far too noisy to be a policy target, so
    // only full searches carry one. But the value and auxiliary targets come
    // from the game's outcome, not from the search, so a fast position is worth
    // exactly as much as a full one for them. Discarding fast positions entirely
    // threw away ~75% of the value data at `full_search_prob 0.25` for no
    // saving. `is_full` tells the learner which rows may train the policy.
    {
        // The sampler plans in the mover's canonical frame, so the planes, the
        // live mask and the declaration targets must all be in that frame too —
        // otherwise the net is trained on one board and run on another.
        let canon = slot.state.canonical();
        let flip = slot.state.side == 1;
        let policy = if flip {
            policy.into_iter().map(|(spec, w)| (spec.rotated(), w)).collect()
        } else {
            policy
        };
        slot.pending.push((
            Minions::to_features(&canon),
            slot.state.side,
            live_bytes(&canon),
            policy,
            // This turn is about to become `trace[len]`, and the auxiliary
            // targets average from here to the end of the game.
            slot.trace.len(),
            slot.full,
        ));
    }
    // Realize once and keep the actions: `apply` would discard them, and the
    // purchases are only visible in the atomic sequence — a spawn may draw from
    // reinforcements (free, e.g. an unsummoned unit returning) rather than buy.
    let mover = slot.state.side;
    let realized = crate::turn::realize(&slot.state, &result.candidates[result.chosen].0);
    let mut buys = [[0u8; N_BUYABLE]; 2];
    for mv in &realized.actions {
        if let crate::game::MinionsMove::Buy(unit) = mv {
            if let Some(slot_i) = unit.reinforcement_slot() {
                buys[mover as usize][slot_i] = buys[mover as usize][slot_i].saturating_add(1);
            }
        }
    }
    slot.state = realized.next;
    slot.trace.push(TurnTrace::of(&slot.state, mover));
    slot.turn += 1;
    if Minions::outcome(&slot.state).is_terminal() {
        return false;
    }
    let (search, full) = start_search(&slot.state, slot.turn, &mut slot.rng);
    slot.search = search;
    slot.full = full;
    true
}

/// Attach the value and ownership targets a finished game determines.
/// Auxiliary targets for a position at trace index `t0`, in the canonical frame
/// (index 0 = `side`, the player to move).
///
/// All three look *forward* over the rest of the game, which is why the whole
/// trajectory is kept rather than just the final board:
///
/// - **hex control** — the fraction of remaining turns each side ends holding
///   the hex. A soft occupancy map; a hex a side holds throughout scores 1.
/// - **necromancer survival** — for each horizon `m`, whether that necromancer is
///   dead within `m` more turns. Monotone in `m` by construction, so a net that
///   learns it is forced to learn a coherent hazard curve rather than 10
///   unrelated bits.
/// - **unit selection** — mean purchases of each buyable per *own* turn.
///   Dividing by that side's own turns rather than all turns keeps the quantity
///   independent of who happens to move next.
fn aux_targets(
    trace: &[TurnTrace],
    t0: usize,
    side: u8,
) -> (Vec<u8>, Vec<u8>) {
    let rest = &trace[t0.min(trace.len())..];
    let sides = [side, 1 - side];

    // ── hex control ──────────────────────────────────────────────────────────
    let mut hex = vec![0u8; 2 * NUM_CELLS];
    if !rest.is_empty() {
        for (slot, &p) in sides.iter().enumerate() {
            for c in 0..NUM_CELLS {
                let held = rest.iter().filter(|t| t.owner[c] == p + 1).count();
                let frac = held as f32 / rest.len() as f32;
                hex[slot * NUM_CELLS + c] = (frac * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    // ── necromancer survival ─────────────────────────────────────────────────
    let mut safety = vec![0u8; 2 * NECRO_HORIZON];
    for (slot, &p) in sides.iter().enumerate() {
        // First remaining turn at whose end this necromancer is gone.
        let death = rest.iter().position(|t| !t.necro_alive[p as usize]);
        for m in 1..=NECRO_HORIZON {
            let dead = death.is_some_and(|d| d < m);
            safety[slot * NECRO_HORIZON + m - 1] = u8::from(dead);
        }
    }

    (hex, safety)
}

fn examples_from(
    pending: Vec<Pending>,
    trace: &[TurnTrace],
    final_state: &MinionsState,
) -> Vec<Example> {
    let outcome = Minions::outcome(final_state);
    pending
        .into_iter()
        .map(|(planes, side, live, candidates, t0, is_full)| {
            let (hex_control, necro_safety) = aux_targets(trace, t0, side);
            // `aux_targets` and `ownership_of` already order the two sides from
            // the mover's view; the geometry still has to be rotated to match
            // the canonical planes above. `necro_safety` and `unit_select` are
            // not cell-indexed, so they are unaffected.
            let flip = side == 1;
            let rot_cells = |v: &[u8], planes: usize| -> Vec<u8> {
                let mut out = vec![0u8; v.len()];
                for pl in 0..planes {
                    for c in 0..NUM_CELLS {
                        out[pl * NUM_CELLS + crate::map::rot(c)] = v[pl * NUM_CELLS + c];
                    }
                }
                out
            };
            let hex_control = if flip { rot_cells(&hex_control, 2) } else { hex_control };
            let z = match outcome {
                Outcome::FirstWins => {
                    if side == 0 { 1.0 } else { -1.0 }
                }
                Outcome::SecondWins => {
                    if side == 1 { 1.0 } else { -1.0 }
                }
                _ => 0.0,
            };
            Example {
                planes,
                to_move: side + 1,
                is_full,
                value: z,
                live,
                candidates,
                hex_control,
                necro_safety,
                // Terminal score from this example's mover. Every position in
                // the game gets the same final number, exactly as `value` does.
                score: crate::game::game_score(final_state, side),
            }
        })
        .collect()
}

fn write_chunk(dir: &Path, index: usize, examples: &[Example]) -> anyhow::Result<PathBuf> {
    let mut buf = Vec::with_capacity(16 + examples.len() * EXAMPLE_BYTES);
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(examples.len() as u64).to_le_bytes());
    for e in examples {
        e.write(&mut buf);
    }
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("chunk_{index:06}.bin"));
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, &buf)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Drop oldest chunks until the buffer fits. Bounded by **examples first**: the
/// Hex run learned the hard way that a chunk-count bound silently collapses the
/// window when games get shorter, spiking reuse and with it the value loss.
/// The policy target as recorded: every child of the root, strongest first,
/// zero targets included.
///
/// Nothing is filtered and nothing is renormalised. `policy_target_with` already
/// normalises over all children, and keeping the zeros is the point: a child the
/// search examined and declined is the negative the energy head most needs, and
/// dropping it made `softmax f` train over a set the search never considered.
/// Order matters to the learner, whose trajectory-balance step takes the first
/// *k* slots, so the strongest children must come first.
///
/// More children than `TOP_K` cannot happen -- `run_loop` refuses a full-search
/// candidate count above it, and widening never touches the root -- so overflow
/// is a bug: an assertion in debug, and in release a one-time `WARNING` (which
/// the cycle monitor catches) rather than an actor that dies mid-generation.
fn policy_record(children: impl Iterator<Item = (TurnSpec, f32)>) -> Vec<(TurnSpec, f32)> {
    let mut v: Vec<(TurnSpec, f32)> = children.collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    if v.len() > TOP_K {
        debug_assert!(false, "{} root children exceed TOP_K = {}", v.len(), TOP_K);
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "WARNING: {} root children exceed TOP_K = {}; the weakest are not recorded",
                v.len(),
                TOP_K
            );
        }
        v.truncate(TOP_K);
    }
    v
}

fn prune_buffer(dir: &Path, max_examples: usize, max_chunks: usize) -> anyhow::Result<()> {
    let mut chunks: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    chunks.sort();
    // Read the count out of the chunk header rather than dividing by
    // `EXAMPLE_BYTES`. The stride changes whenever the record format does, and
    // a sliding window holds both versions at once — after `TOP_K` 8 -> 16 a
    // size-based count read every older chunk as 0.69 of its true size, so the
    // pruner under-pruned and the buffer silently overshot its cap. The header
    // has always carried the exact count.
    let count_of = |p: &Path| -> usize {
        let mut buf = [0u8; 16];
        match std::fs::File::open(p).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut buf)) {
            Ok(()) => u64::from_le_bytes(buf[8..16].try_into().unwrap_or_default()) as usize,
            Err(_) => 0,
        }
    };
    let mut total: usize = chunks.iter().map(|p| count_of(p)).sum();
    let mut n = chunks.len();
    for p in &chunks {
        if total <= max_examples && n <= max_chunks {
            break;
        }
        total -= count_of(p);
        n -= 1;
        let _ = std::fs::remove_file(p);
    }
    Ok(())
}

/// Continuous actor: play games, flush chunks, hot-reload weights.
#[allow(clippy::too_many_arguments)]
pub fn run_loop(
    live: &str,
    cfg: ActorConfig,
    sampler: SamplerConfig,
    concurrency: usize,
    max_batch: usize,
    chunk_secs: u64,
    max_examples: usize,
    max_chunks: usize,
    target_examples: usize,
    seed: u64,
) -> anyhow::Result<()> {
    let live = Path::new(live);
    let buffer = live.join("buffer");
    std::fs::create_dir_all(&buffer)?;
    // TensorRT engine cache; `content_keyed` shards it per checkpoint so a
    // stale engine can never be reused across a promotion.
    let cache_dir = live.join("trt_cache").to_string_lossy().into_owned();
    let weights = live.join("latest.safetensors");
    let best = live.join("best.safetensors");

    // Generate from the champion when there is one — that indirection *is* the
    // promotion gate, so data never comes from a net that lost its match.
    let resolve = || if best.exists() { best.clone() } else { weights.clone() };

    // Every root child is recorded, and a record holds `TOP_K` of them. Widening
    // stops at `k_candidates`, so this bound is the whole guarantee that nothing
    // the search considered is left out of the target -- refuse to start rather
    // than truncate silently for a whole generation.
    anyhow::ensure!(
        sampler.k_candidates <= TOP_K,
        "--candidates {} exceeds the {} root children a record can hold; raise TOP_K \
         (and the record VERSION, in every file scripts/check-record-layout.sh checks)",
        sampler.k_candidates,
        TOP_K
    );

    // Falling below the pruning threshold is silent and ruinous for the energy
    // head, so say so at startup rather than leaving it to be discovered from a
    // flat loss curve days later.
    let per_candidate = cfg.sims as f32 / sampler.k_candidates.max(1) as f32;
    let need = playground_core::sampled_mcts::min_visits_per_candidate(cfg.k_forced);
    if per_candidate < need {
        eprintln!(
            "WARNING: {} sims over {} candidates = {:.1} visits each, below the {:.1} \n\
             needed to survive policy-target pruning. Recorded nodes will hold a single \n\
             candidate, and the energy head will receive no gradient. Raise --sims or \n\
             lower --candidates.",
            cfg.sims, sampler.k_candidates, per_candidate, need
        );
    }

    // Record the search this run uses, so eval and the UI run the same one.
    // Rewritten on every start, before any game is played.
    crate::search_config::save(&live.join(crate::search_config::FILE), &cfg, &sampler)?;

    println!("waiting for {} …", weights.display());
    while !weights.exists() && !best.exists() {
        std::thread::sleep(Duration::from_secs(5));
    }

    // Continue from the highest index present, not the file *count*.
    //
    // Counting is wrong whenever pruning has removed chunks from the middle or
    // front of the range: with `chunk_000047..077` surviving, a count of 32
    // restarts numbering at 32, and since `prune_buffer` treats the sort order
    // as age, every freshly written chunk sorts *before* the survivors and is
    // deleted as though it were the oldest data in the buffer. Self-play then
    // runs continuously while the learner trains on a frozen set — losses keep
    // falling because it is overfitting, so nothing looks wrong.
    let mut chunk_index = std::fs::read_dir(&buffer)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .filter_map(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("chunk_"))
                .and_then(|s| s.parse::<usize>().ok())
        })
        .max()
        .map_or(0, |m| m + 1);
    let mut last_mtime = None;
    let mut model =
        MinionsModel::new(crate::net::load_backend(live, &cfg.backend, &cache_dir)?, sampler);
    let mut produced = 0usize;
    // One persistent pool. Games that do not finish inside a chunk carry over —
    // discarding them would waste the work and leave them with no outcome for
    // their value target.
    let mut pool = new_pool(&model, &cfg, concurrency, seed);
    let mut pool_seed = seed.wrapping_add(concurrency as u64);

    loop {
        // A PAUSE file parks the actor so an evaluator isn't fighting it for the
        // GPU — same handshake as the Hex cycle script.
        if live.join("PAUSE").exists() {
            let _ = std::fs::write(live.join("actor.paused"), b"1");
            std::thread::sleep(Duration::from_secs(5));
            continue;
        }
        let _ = std::fs::remove_file(live.join("actor.paused"));

        let window = WINDOWS_RUN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let ab = live.join(TRUNK_CACHE_AB_FILE).exists();
        let cache_on = !ab || matches!(window % 4, 1 | 2);
        TRUNK_CACHE_USE.store(cache_on, std::sync::atomic::Ordering::Relaxed);
        let f16_ab = live.join(GRAPH_F16_AB_FILE).exists();
        let f16_on = !f16_ab || matches!(window % 4, 1 | 2);
        crate::ort_net::GRAPH_F16_USE.store(f16_on, std::sync::atomic::Ordering::Relaxed);
        let deadline = Instant::now() + Duration::from_secs(chunk_secs);
        let mut stats = PoolStats::default();
        let t0 = Instant::now();
        let examples =
            drive_pool(&model, &cfg, &mut pool, max_batch, &mut pool_seed, deadline, &mut stats)?;
        println!("  {}", stats.report(t0.elapsed().as_secs_f64()));
        println!(
            "  {}",
            model.phases.report_wall(t0.elapsed().as_secs_f64(), cfg.workers)
        );
        println!("  {}", stats.explore_report());
        println!("  {}", stats.split_report());
        // Raw counts, so a benchmark can measure fixed work rather than a rate:
        // throughput swings ~4x over a generation as the pool moves in step.
        println!(
            "  window: {} expansions in {:.3}s  cache {}{}  f16 {}{}",
            stats.expansions,
            t0.elapsed().as_secs_f64(),
            if cache_on { "on" } else { "off" },
            if ab { " (ab)" } else { "" },
            if f16_on { "on" } else { "off" },
            if f16_ab { " (ab)" } else { "" }
        );
        if !examples.is_empty() {
            let path = write_chunk(&buffer, chunk_index, &examples)?;
            chunk_index += 1;
            produced += examples.len();
            println!(
                "wrote {} ({} examples, {} total)",
                path.file_name().unwrap().to_string_lossy(),
                examples.len(),
                produced
            );
            prune_buffer(&buffer, max_examples, max_chunks)?;
        }

        if target_examples > 0 && produced >= target_examples {
            println!("reached target of {target_examples} examples; exiting");
            return Ok(());
        }

        // Hot-reload if the learner published new weights.
        let path = resolve();
        let mt = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        if mt != last_mtime {
            match crate::net::load_backend(live, &cfg.backend, &cache_dir) {
                Ok(n) => {
                    model = MinionsModel::new(n, sampler);
                    // Cached trunk output belongs to the weights that computed it.
                    for slot in pool.iter_mut() {
                        cache_drop(slot);
                    }
                    last_mtime = mt;
                    println!("reloaded weights from {}", path.display());
                }
                Err(e) => eprintln!("weight reload failed: {e}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// The shipped opening distribution must have the mean it claims, and must
    /// stay a sane fraction of the game.
    ///
    /// `max`, `decay` and `MAX_TURNS` are only meaningful together — a decay
    /// tuned for one horizon gives a different mean at another, and an opening
    /// longer than the cap would consume the whole game. Nothing else in the
    /// system would notice either drifting.
    #[test]
    fn default_opening_length_is_a_quarter_of_the_game() {
        use rand::SeedableRng;
        let cfg = ActorConfig::default();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(21);
        let n = 60_000;
        let mut total = 0u64;
        let mut seen_max = 0u32;
        for _ in 0..n {
            let k = playground_core::batch_search::sample_geometric_len(
                cfg.random_opening_max,
                cfg.random_opening_decay,
                &mut rng,
            );
            total += k as u64;
            seen_max = seen_max.max(k);
        }
        let mean = total as f64 / n as f64;
        let want = crate::game::MAX_TURNS as f64 / 4.0;
        assert!((mean - want).abs() < 0.3, "opening length mean {mean:.2}, wanted ~{want}");
        assert_eq!(seen_max, cfg.random_opening_max, "the deepest opening must occur");
        assert!(
            cfg.random_opening_max < crate::game::MAX_TURNS as u32,
            "an opening cannot be longer than the game"
        );
    }

    /// Random openings must be geometrically tilted toward short prefixes, and
    /// must never hand back a finished game (there would be nothing to train on).
    #[test]
    fn random_openings_are_short_tilted_and_playable() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4);
        let (max, decay) = (8u32, 1.15f32);

        let mut counts = vec![0u32; max as usize + 1];
        for _ in 0..4000 {
            counts[playground_core::batch_search::sample_geometric_len(max, decay, &mut rng) as usize] += 1;
        }
        assert!(counts.iter().all(|&c| c > 0), "every length must be reachable: {counts:?}");
        assert!(counts[0] > counts[max as usize],
                "decay must favour short openings: {counts:?}");
        // A tilt, not a cutoff — the deep tail has to stay usable.
        assert!(counts[max as usize] * 6 > counts[0],
                "decay {decay} truncates the tail too hard: {counts:?}");

        // Uniform when the decay is 1.
        let mut flat = vec![0u32; max as usize + 1];
        for _ in 0..4000 {
            flat[playground_core::batch_search::sample_geometric_len(max, 1.0, &mut rng) as usize] += 1;
        }
        let (lo, hi) = (*flat.iter().min().unwrap(), *flat.iter().max().unwrap());
        assert!(hi < lo * 2, "decay 1.0 must be uniform: {flat:?}");

        for _ in 0..40 {
            let (state, n) = random_opening(max, decay, &mut rng);
            assert!(n <= max);
            assert!(!Minions::outcome(&state).is_terminal(),
                    "an opening that ends the game leaves nothing to train on");
            assert_eq!(state.turn as u32, n, "turn counter must follow the opening");
        }

        // Openings off still yields the opening *board* — but with a drawn
        // handicap, since every game prices its own starting money.
        let (s0, n0) = random_opening(0, decay, &mut rng);
        assert_eq!(n0, 0);
        assert_eq!(s0.turn, 0);
        assert_eq!(s0.cells, Minions::initial_state().cells, "board must be the opening one");
        let bid = s0.money[1] as i16 - s0.money[0] as i16;
        assert!(
            (crate::game::BID_MIN..=crate::game::BID_MAX).contains(&bid),
            "handicap {bid} outside the bid range"
        );

        // Across many draws the handicap must actually span the range, or the
        // value head never learns to price the ends of it.
        let mut lo = i16::MAX;
        let mut hi = i16::MIN;
        for _ in 0..400 {
            let (st, _) = random_opening(0, decay, &mut rng);
            let b = st.money[1] as i16 - st.money[0] as i16;
            lo = lo.min(b);
            hi = hi.max(b);
        }
        assert_eq!((lo, hi), (crate::game::BID_MIN, crate::game::BID_MAX));
    }

    /// The auxiliary targets are forward-looking averages in the canonical frame,
    /// which is easy to get subtly wrong in ways nothing else would catch.
    #[test]
    fn aux_targets_read_the_rest_of_the_game() {
        let blank = TurnTrace {
            owner: [0u8; NUM_CELLS],
            necro_alive: [true, true],
        };
        // Four turns, alternating sides. Side 0 holds cell 5 for the last two;
        // side 1's necromancer dies at the third; side 0 buys two zombies on its
        // own turns.
        let mut trace = vec![blank.clone(), blank.clone(), blank.clone(), blank.clone()];
        for (i, t) in trace.iter_mut().enumerate() {
        }
        trace[2].owner[5] = 1;
        trace[3].owner[5] = 1;
        trace[2].necro_alive[1] = false;
        trace[3].necro_alive[1] = false;

        // From the start, as side 0: it holds cell 5 for 2 of 4 turns.
        let (hex, safety) = aux_targets(&trace, 0, 0);
        assert_eq!(hex[5], 128, "own control of cell 5 should be 2/4");
        assert_eq!(hex[NUM_CELLS + 5], 0, "the enemy never holds cell 5");

        // Side 1's necromancer dies at remaining-turn index 2, so horizons 1 and
        // 2 are still alive and 3+ are dead. Enemy is slot 1 for side 0.
        assert_eq!(safety[NECRO_HORIZON], 0, "m=1: not yet dead");
        assert_eq!(safety[NECRO_HORIZON + 1], 0, "m=2: not yet dead");
        assert_eq!(safety[NECRO_HORIZON + 2], 1, "m=3: dead by now");
        assert_eq!(safety[NECRO_HORIZON + 9], 1, "monotone in m");
        assert!(safety[..NECRO_HORIZON].iter().all(|&v| v == 0), "own necro survives");

        // Side 0 moved on 2 of the 4 turns and bought 2 zombies across them.

        // Viewed from side 1, the frame flips: cell 5 is *enemy*-held.
        let (hex1, safety1) = aux_targets(&trace, 0, 1);
        assert_eq!(hex1[NUM_CELLS + 5], 128, "canonical frame must swap sides");
        assert_eq!(hex1[5], 0);
        assert_eq!(safety1[2], 1, "side 1's own necromancer dies by m=3");

        // Starting later, only the tail counts: side 0 holds cell 5 throughout.
        let (hex_late, _) = aux_targets(&trace, 2, 0);
        assert_eq!(hex_late[5], 255, "held for all remaining turns");
    }

    fn example_stride_matches_the_declared_layout() {
        let e = Example {
            is_full: true,
            planes: vec![7u8; PLANE_BYTES],
            to_move: 1,
            value: -0.5,
            live: vec![7u8; NUM_CELLS],
            candidates: vec![(TurnSpec::empty(), 1.0)],
            hex_control: vec![0u8; HEX_CONTROL_BYTES],
            necro_safety: vec![0u8; NECRO_SAFETY_BYTES],
            score: 0.0,
        };
        let mut buf = Vec::new();
        e.write(&mut buf);
        assert_eq!(buf.len(), EXAMPLE_BYTES);
        // The learner reads this stride, so pin the constant itself too:
        // planes, to_move, value, ownership, live, n_cands, TOP_K candidate
        // slots, then the auxiliary targets (hex control, necromancer survival,
        // unit selection).
        assert_eq!(
            EXAMPLE_BYTES,
            3100 + 1 + 4 + 100 + 100 + 1 + 8 * 304 + 200 + 20 + 40
        );
    }

    /// Every root child is recorded, zero targets included, strongest first.
    ///
    /// Regression: recording filtered out zero-target children and then kept
    /// only the top 8 of the rest, so the energy head never trained against a
    /// move the search declined -- and on 89% of positions it also lost children
    /// that had earned visits. The stored vector must be the search's own
    /// target, whole. Calls the real rule rather than restating it, which is
    /// what the test it replaces did.
    #[test]
    fn every_root_child_is_recorded() {
        let mut target = vec![0.0f32; TOP_K];
        target[1] = 0.30;
        target[2] = 0.05;
        target[4] = 0.40;
        target[5] = 0.25;
        let zeros = target.iter().filter(|&&p| p == 0.0).count();
        assert!(zeros > TOP_K / 2, "fixture must be mostly zeros, like a real root");

        let rec = policy_record(target.iter().map(|&p| (TurnSpec::empty(), p)));
        assert_eq!(rec.len(), TOP_K, "no child may be dropped");
        assert_eq!(rec.iter().filter(|(_, p)| *p == 0.0).count(), zeros, "zeros are kept");
        assert!(
            (rec.iter().map(|(_, p)| p).sum::<f32>() - 1.0).abs() < 1e-6,
            "the search's own normalisation is stored unchanged"
        );
        assert!(rec.windows(2).all(|w| w[0].1 >= w[1].1), "strongest first");
        assert_eq!(rec[0].1, 0.40);
    }

    /// The batch gatherer must never schedule a kind that cannot fill a batch
    /// while the other kind can.
    ///
    /// Regression: it used to alternate full/fast unconditionally. A slot sits
    /// in full mode ~4x longer than fast (200 sims per turn against 50), so
    /// alternating over-serves fast; fast slots churn through turns, each turn
    /// re-rolls the mode with 25% landing on full, and the pool ratchets toward
    /// full until fewer than `max_batch` slots remain in fast. Every fast batch
    /// after that was padded — 30-40% of GPU work thrown away in a real run.
    #[test]
    fn batch_kind_follows_supply_not_alternation() {
        // Fast is starved: never schedule fast, whatever we served last.
        assert!(pick_mode(480, 8, 32, true));
        assert!(pick_mode(480, 8, 32, false));
        // Symmetrically for full.
        assert!(!pick_mode(8, 480, 32, true));
        assert!(!pick_mode(8, 480, 32, false));
        // Both well supplied: alternate, so neither side starves.
        assert!(!pick_mode(256, 256, 32, true));
        assert!(pick_mode(256, 256, 32, false));
        // Exactly at the threshold still counts as able to fill.
        assert!(pick_mode(32, 31, 32, true));

        // The padding branch is unreachable for any pool of at least
        // 2 * max_batch slots, since every slot is counted exactly once.
        let (pool, max_batch) = (512usize, 32usize);
        for n_full in 0..=pool {
            let n_fast = pool - n_full;
            assert!(
                n_full >= max_batch || n_fast >= max_batch,
                "both kinds short at {n_full}/{n_fast} — padding would be forced"
            );
        }
    }

    fn a_full_candidate_list_still_fits_one_stride() {
        let e = Example {
            is_full: true,
            planes: vec![0u8; PLANE_BYTES],
            to_move: 2,
            value: 1.0,
            live: vec![3u8; NUM_CELLS],
            candidates: (0..TOP_K).map(|_| (TurnSpec::empty(), 1.0 / TOP_K as f32)).collect(),
            hex_control: vec![0u8; HEX_CONTROL_BYTES],
            necro_safety: vec![0u8; NECRO_SAFETY_BYTES],
            score: 0.0,
        };
        let mut buf = Vec::new();
        e.write(&mut buf);
        assert_eq!(buf.len(), EXAMPLE_BYTES);
    }

    #[test]
    /// The chunk counter must continue past the highest surviving index, not
    /// restart at the file count.
    ///
    /// Regression guard: pruning leaves gaps, so a count-based index collides
    /// with survivors and — because `prune_buffer` reads sort order as age —
    /// every new chunk is deleted as the "oldest" the moment it is written.
    #[test]
    fn chunk_index_resumes_past_the_highest_surviving_chunk() {
        let uniq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("minions_idx_{}_{uniq}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        assert!(dir.is_dir(), "temp dir {dir:?} was not created");
        // A buffer that has been pruned: a gap at the front, survivors at 47-49.
        for i in [47usize, 48, 49] {
            std::fs::write(dir.join(format!("chunk_{i:06}.bin")), b"x").expect("write");
        }
        let next = std::fs::read_dir(&dir)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "bin"))
            .filter_map(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix("chunk_"))
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |m| m + 1);
        assert_eq!(next, 50, "must continue past 49, not restart at the count (3)");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn chunk_round_trips_through_the_header() {
        let dir = std::env::temp_dir().join(format!("minions_chunk_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ex: Vec<Example> = (0..3)
            .map(|i| Example {
                is_full: true,
                planes: vec![i as u8; PLANE_BYTES],
                to_move: 1,
                value: 0.25,
                    live: vec![1u8; NUM_CELLS],
                candidates: vec![(TurnSpec::empty(), 1.0)],
            hex_control: vec![0u8; HEX_CONTROL_BYTES],
            necro_safety: vec![0u8; NECRO_SAFETY_BYTES],
            score: 0.0,
            })
            .collect();
        let p = write_chunk(&dir, 0, &ex).unwrap();
        let raw = std::fs::read(&p).unwrap();
        assert_eq!(u32::from_le_bytes(raw[0..4].try_into().unwrap()), MAGIC);
        assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), VERSION);
        assert_eq!(u64::from_le_bytes(raw[8..16].try_into().unwrap()), 3);
        assert_eq!(raw.len(), 16 + 3 * EXAMPLE_BYTES);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
