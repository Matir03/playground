//! Sampled AlphaZero: PUCT over a *sampled* candidate set.
//!
//! The flat `BatchGame` path indexes a small fixed action space; the `MoveGame`
//! path enumerates a per-state legal-move list. Both assume the legal set can be
//! written down. For a game whose action is a whole turn — on the order of
//! 10^15–10^25 legal turns in Minions — neither can be, so this path never asks
//! for one: a node's children are whatever the model *proposed*, and the engine
//! is authoritative only in that every proposal is already a legal move.
//!
//! That is the Sampled MuZero / Sampled AlphaZero shape (Hubert et al. 2021):
//! sample `k` actions, search over those, and treat their visit counts as the
//! improved policy. Note the deliberate absence of a `legal_moves` in
//! [`SampledGame`] — it is what stops this path from quietly acquiring the
//! assumption it exists to avoid.
//!
//! Two exploration mechanisms, kept separate because they are corrected for
//! differently:
//!
//! - **Forced playouts** at the root ([`SampledParams::k_forced`]), KataGo-style,
//!   so a low-prior candidate cannot be starved before it is tried.
//! - **Forced exploration** ([`SampledParams::min_explore_visits`]): candidates
//!   the sampler drew *without* the model — the Minions sampler builds them by
//!   playing a random legal turn — are floored at a fixed number of visits
//!   regardless of prior. KataGo's floor cannot do this job alone: it scales with
//!   `P(a)` and only protects candidates already tried once, so the candidates
//!   most worth a look, the ones the energy head scores near zero, can go
//!   unvisited entirely. Support collapse is self-reinforcing without it — the
//!   sampler stops proposing a region, so the energy head never sees it, so its
//!   prior goes to zero.
//!
//! Both would bias the training target, so
//! [`SampledSearchResult::policy_target_with`] gates on them: a child whose
//! visits were all bought by a floor drops out entirely, and a child that
//! earned even one visit on its own merits keeps its *whole* count — the forced
//! visits looked at that move too, and their results came back. A candidate
//! that was *forced* into the tree and then earned organic visits survives on
//! merit; one that didn't, vanishes. Value targets deliberately keep every visit — a random candidate is
//! cheap value data even when it is worthless policy data.

use crate::puct;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// A game whose action space is too large to enumerate.
///
/// Note what is *not* here: no `legal_moves`. Candidates come from the model.
pub trait SampledGame: Send + Sync + 'static {
    type State: Clone + Send + Sync;
    type Move: Clone + Send + Sync + std::fmt::Debug;

    /// Network input width, in floats.
    const PLANES: usize;

    fn initial_state() -> Self::State;
    /// 1 or 2 — the side to move, used to tag training examples.
    fn to_move(state: &Self::State) -> u8;
    fn apply(state: &Self::State, mv: &Self::Move) -> Self::State;
    /// Terminal value for the side to move; `None` while in progress.
    fn terminal_value(state: &Self::State) -> Option<f32>;
    fn encode_planes(state: &Self::State, out: &mut [f32]);
    /// Key for merging candidates that lead to the same position. Distinct turns
    /// often converge on one board, and without merging their visits fragment.
    fn state_key(state: &Self::State) -> u128;

    /// Distance between two candidate moves at `state`, for the KR-PUCT kernel.
    ///
    /// The default is a hard delta, which makes the kernel an identity matrix and
    /// KR-PUCT collapse to exactly the PUCT it generalises — so a game that has
    /// no meaningful metric over its actions is unaffected by any of this.
    /// Games with structured actions override it (Minions: `turn::distance`).
    fn move_distance(_state: &Self::State, _a: &Self::Move, _b: &Self::Move) -> f32 {
        1.0
    }
}

/// The network, as this search sees it.
///
/// One call does everything a node expansion needs: the value, and the candidate
/// moves with their priors. Sampling (diffusion steps, Gumbel top-k), random
/// injection, realization and energy scoring all live behind this seam, which
/// keeps the search itself testable against a mock.
/// A node expansion: the value, and the candidate moves with their priors.
/// A value, plus candidates as `(move, energy, explore)`.
///
/// The f32 is the **raw energy** `f(s, a)`, on whatever scale the model's head
/// emits — *not* a probability. Turning energies into priors is the search's
/// job because only the search knows the whole candidate set: with chunked
/// generation a model call sees one candidate at a time, and a softmax over one
/// element is `1.0` regardless of what the head said. Normalising per chunk
/// discards the head's opinion entirely and leaves priors that encode arrival
/// order — every candidate 0.5, halving as the next one lands.
///
/// `explore` marks a candidate drawn by **forced exploration** rather than by the
/// model — support coverage that does not depend on the sampler being any good.
/// It is carried all the way to [`SampledSearchResult::explore`] because both the
/// selection floor and the training-target pruning have to agree about which
/// visits were forced; if they disagree, forced visits leak into the policy
/// target and actively corrupt it.
pub type Expansion<G> = (f32, Vec<(<G as SampledGame>::Move, f32, bool)>);

/// What a resumable search is waiting for.
pub enum Step<G: SampledGame> {
    /// An expanded node has earned more candidates than it holds. Answer with
    /// [`SampledModel::widen_batch`] and feed the candidates to
    /// [`Search::provide_widen`].
    ///
    /// `fill` is set when the node has made all its sampler draws
    /// ([`SampledParams::k_draws`]) and is still short: the request is for
    /// exploration candidates to make up the shortfall, not another proposal.
    NeedWiden { state: G::State, fill: bool },
    /// This visit refines rather than descends: answer with
    /// [`SampledModel::resample_batch`] on `from` and feed the result to
    /// [`Search::provide_resample`]. See [`SampledParams::resample_every`].
    NeedResample { state: G::State, from: G::Move },
    /// The search reached an unexpanded node and needs its value and candidates.
    /// Feed them back with [`Search::provide`].
    NeedExpansion(G::State),
    /// Simulation budget spent; call [`Search::finish`].
    Done,
}

pub trait SampledModel<G: SampledGame>: Send + Sync {
    /// Value for the side to move in `[-1, 1]`, and up to `k` distinct candidate
    /// `k == 0` means **value only**: return the value and an empty candidate
    /// list, skipping candidate generation entirely. A node's first visit needs
    /// nothing else — it will not choose a move until something descends
    /// through it, and most nodes never are.
    ///
    /// moves with priors. Priors need not be normalised; the search normalises
    /// over whatever comes back. Returning an empty candidate list is treated as
    /// a pass-less dead end and the node is scored on value alone.
    fn expand(&self, state: &G::State, k: usize) -> anyhow::Result<Expansion<G>>;

    /// Expand several positions at once.
    ///
    /// This is where throughput lives. At batch 1 a GPU spends essentially all
    /// its time on launch latency — measured on this net, the trunk runs 817
    /// positions/s at batch 1 and 115,000/s at batch 512, a 141x difference for
    /// the same arithmetic. The default here is a loop for correctness and for
    /// mock models; every real backend should override it.
    fn expand_batch(&self, states: &[G::State], k: usize) -> anyhow::Result<Vec<Expansion<G>>> {
        states.iter().map(|s| self.expand(s, k)).collect()
    }

    /// Up to `k` more candidates for each of `states`, nodes that asked to
    /// widen.
    ///
    /// `fill[i]` marks a node that has made all its sampler draws and holds
    /// fewer distinct candidates than it earned -- the draws repeated. It gets
    /// **exploration candidates** instead, made without the learned proposal
    /// and flagged explore, to fill the shortfall. A fill that adds nothing new
    /// is retried; after as many misses as the node has earned candidates, the
    /// node is marked exhausted.
    ///
    /// The default answers proposal widens with `expand_batch` and has no
    /// exploration source, so a fill gets nothing and the node stops short.
    fn widen_batch(
        &self,
        states: &[G::State],
        fill: &[bool],
        k: usize,
    ) -> anyhow::Result<Vec<Vec<(G::Move, f32, bool)>>> {
        anyhow::ensure!(
            fill.len() == states.len(),
            "widen_batch: one fill flag per state"
        );
        let ask: Vec<G::State> = states
            .iter()
            .zip(fill)
            .filter(|(_, f)| !**f)
            .map(|(s, _)| s.clone())
            .collect();
        let mut got = if ask.is_empty() {
            Vec::new()
        } else {
            self.expand_batch(&ask, k)?
        }
        .into_iter();
        fill.iter()
            .map(|&f| {
                if f {
                    Ok(Vec::new())
                } else {
                    got.next()
                        .map(|(_, c)| c)
                        .ok_or_else(|| anyhow::anyhow!("expand_batch returned too few results"))
                }
            })
            .collect()
    }

    /// A candidate near each `from[i]` at `states[i]`: part of the turn kept,
    /// the rest redrawn. The default has no way to do that and returns
    /// nothing, and the visit then descends into `from` as it would have.
    fn resample_batch(
        &self,
        states: &[G::State],
        from: &[G::Move],
    ) -> anyhow::Result<Vec<Vec<(G::Move, f32, bool)>>> {
        let _ = from;
        Ok(states.iter().map(|_| Vec::new()).collect())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SampledParams {
    pub simulations: u32,
    pub c_puct: f32,
    /// Candidates a node holds once fully widened (`N`).
    pub k_candidates: usize,
    /// Sampler draws per node (`D`). A node draws `D` proposals; the distinct
    /// ones become its candidates, and a shortfall to `k_candidates` is filled
    /// with exploration candidates (see [`SampledModel::widen_batch`]). The
    /// root is asked exactly `D` times whether or not the sampler repeats
    /// itself.
    ///
    /// Pro-rated under progressive widening: a node that has earned `n` of its
    /// `k_candidates` draws `ceil(n * D / k_candidates)` before it fills, so
    /// the root, which earns them all at once, draws exactly `D`. An interior
    /// node may fill sooner, after [`Self::draw_retries`] repeats in a row.
    pub k_draws: usize,
    /// Sampler draws in a row that may add nothing at an interior node before
    /// its widen falls back to a fill (a random playout) instead. 0 = keep
    /// drawing until the node's draw budget is spent.
    ///
    /// A widen at an interior node holds up the simulation that asked for it
    /// until something new arrives, and a sharp sampler can repeat itself for
    /// the node's whole budget. The root is exempt: it draws its `D` once per
    /// search, before any visit.
    pub draw_retries: u32,
    /// Fill a node's shortfall with exploration candidates (random playouts)
    /// once its draws are spent. Off, the node keeps only what its sampler
    /// draws brought and stops widening there.
    pub fills: bool,
    /// Every this many visits to a node, a visit refines instead of
    /// descending: it takes the candidate selection picked, asks the model to
    /// redraw part of it ([`SampledModel::resample_batch`]), adds what comes
    /// back and visits that. 0 = off. A fraction spreads refines evenly: 1.5
    /// alternates gaps of 1 and 2 visits (see [`resample_gap`]).
    pub resample_every: f32,
    /// Deepest node that refines (the root is depth 0); `u32::MAX` = all.
    pub resample_max_depth: u32,
    /// KataGo forced-playout constant; 0 disables forced playouts.
    pub k_forced: f32,
    /// Visits guaranteed to every *explore*-origin root candidate, regardless of
    /// its prior — the part KataGo's prior-scaled floor cannot provide, since
    /// that floor is proportional to `P(a)` and skips candidates never tried.
    /// Like forced playouts, these are counted and gate the training target:
    /// a candidate that earned nothing beyond them contributes nothing.
    /// 0 disables it.
    pub min_explore_visits: f32,
    /// Visits guaranteed to every candidate holding a forced-exploration quota,
    /// as a flat constant rather than a function of the prior.
    ///
    /// The prior-proportional KataGo floor `sqrt(k*p*N)` gave the *most* forced
    /// visits to the candidates the policy already rated highest, which is
    /// backwards for a mechanism whose job is to measure the ones it does not.
    /// A constant spends the same evidence on every candidate it examines.
    ///
    /// These visits are excluded from PUCT entirely (see [`Entry::n_eff`]) and
    /// from the policy target; they buy a value estimate, nothing else.
    pub forced_visits: u32,
    /// Non-explore candidates chosen at random to also receive a quota.
    ///
    /// Explore candidates are always covered. These extras widen the set of
    /// moves that get a trustworthy value estimate beyond the ones the sampler
    /// happened to flag.
    pub forced_extra: u32,
    /// Plies sampled proportional to visits before switching to argmax.
    pub temp_moves: u32,
    /// Progressive widening for **non-root** nodes: how many of the generated
    /// candidates are selectable at `n` visits, `ceil(pw_beta * n^pw_alpha)`,
    /// capped at `k`.
    ///
    /// There is deliberately no separate "initial count" knob: at `n = 1` the
    /// schedule is `ceil(pw_beta)`, so `pw_beta` *is* the initial count.
    ///
    /// Measured motivation: 69% of expanded nodes receive exactly one visit and
    /// 91.6% of generated candidates are never tried, so spreading a node's
    /// whole candidate set across a couple of visits wastes the search's
    /// discrimination. The root is exempt — it has the visits to use them, and
    /// its full set is the training target.
    ///
    /// `pw_beta <= 0` disables widening and every candidate is selectable, which
    /// is the pre-existing behaviour.
    pub pw_alpha: f32,
    pub pw_beta: f32,
    /// Candidates generated per model call. `0` disables chunking and produces
    /// a node's whole set at once.
    ///
    /// `1` is the natural setting, because widening reveals at most one new
    /// candidate per visit: generating more than that produces candidates the
    /// node may never be allowed to select, which is precisely the waste
    /// chunking exists to remove (69% of nodes are visited exactly once, and
    /// 91.6% of generated candidates are never tried).
    ///
    /// Generating one at a time is only affordable because a wave batches
    /// across *nodes* — `max_batch` separate nodes asking for one candidate
    /// each fills the same batch that one node asking for `max_batch` would.
    /// That coupling matters: with a small chunk the wave must cover
    /// proportionally more nodes, or the batch shrinks and the GPU starves.
    pub pw_chunk: usize,
    /// KR-PUCT kernel bandwidth over `SampledGame::move_distance`. Candidates
    /// within roughly this distance share visit counts and values, which is the
    /// point: with ~16 visits per root child, per-candidate value estimates are
    /// far too noisy to stand alone.
    ///
    /// **0 disables the kernel entirely and reproduces plain PUCT bit-for-bit**,
    /// which is what keeps the flat search available as a mode rather than being
    /// replaced. Asserted by `zero_bandwidth_matches_plain_puct`.
    pub kernel_sigma: f32,
    pub seed: u64,
}

impl Default for SampledParams {
    fn default() -> Self {
        Self {
            simulations: 200,
            c_puct: 1.5,
            k_candidates: 32,
            k_draws: 32,
            draw_retries: 3,
            fills: true,
            resample_every: 0.0,
            resample_max_depth: u32::MAX,
            k_forced: 2.0,
            min_explore_visits: 2.0,
            forced_visits: 30,
            forced_extra: 4,
            temp_moves: 8,
            pw_alpha: 0.5,
            pw_beta: 2.0,
            pw_chunk: 1,
            kernel_sigma: 0.0,
            seed: 0,
        }
    }
}

/// Visits per candidate below which policy-target pruning erases almost the
/// whole candidate set.
///
/// The historical rule this solves: pruning subtracted a forced-playout floor
/// of `sqrt(k·p·N)` from every non-best child and dropped whatever was left at
/// one visit or fewer. With uniform priors `p = 1/K` and `x = N/K` visits per
/// candidate, a candidate survived only when `x − 1 > sqrt(k·x)`, i.e.
/// `x² − (2+k)x + 1 > 0`. For the default `k_forced = 2` that is `x > 3.73`.
///
/// This matters because falling below it is *silent*: search still works, the
/// value and sampler heads still train, but every recorded node has a single
/// candidate — and a softmax over one element has zero gradient, so the energy
/// head (the policy of record) quietly learns nothing at all.
/// Approximate only, since the target now gates on *counted* forced visits
/// rather than this estimate, and so prunes strictly less. Retained for the
/// startup warning, where the question is the rough one of whether
/// `--sims / --candidates` leaves anything to learn from.
pub fn min_visits_per_candidate(k_forced: f32) -> f32 {
    if k_forced <= 0.0 {
        return 0.0;
    }
    let b = 2.0 + k_forced;
    (b + (b * b - 4.0).sqrt()) / 2.0
}

struct Entry<G: SampledGame> {
    mv: G::Move,
    /// Raw energy `f(s, a)` as the model reported it. Kept alongside `p`
    /// because widening grows the candidate set, and the softmax has to be
    /// recomputed over *all* of it — which needs the pre-normalisation values.
    energy: f32,
    /// Prior: `softmax(energy)` over this node's entries. Derived, never
    /// supplied — see [`renormalise`].
    p: f32,
    n: u32,
    w: f32,
    child: i32,
    /// Drawn by forced exploration rather than proposed by the model.
    explore: bool,
    /// Visits this entry received *because* a floor demanded them, counted in
    /// `backup` together with `n` -- so never more than `n`. The target compares against this exact number instead of
    /// the prior-proportional estimate `sqrt(k*p*N)`, which over-subtracted
    /// from high-prior candidates — the floor grows with the prior, so the
    /// moves the policy rates highest were the most likely to be zeroed.
    forced_n: u32,
    /// Forced visits this entry is entitled to, set once when the node expands.
    /// 0 for candidates outside the forced-exploration set.
    forced_quota: u32,
    /// Sampler draws merged into this entry: 1 for the draw that created it,
    /// plus one for every later draw that reached the same position. 0 for an
    /// entry made by a fill, which is not a sampler draw. The record of why a
    /// node needed filling.
    draws: u32,
    /// Put here by hand (`Search::add_candidate`) rather than by the model.
    manual: bool,
    /// Value-head estimate of where this entry leads, from this node's side to
    /// move, for an entry whose child node does not exist yet. Only a
    /// hand-added candidate has one: the search learns a child's value by
    /// expanding it, but a reader adding a turn wants to see it at once.
    eval: Option<f32>,
    /// Added by a resample ([`SampledParams::resample_every`]).
    resampled: bool,
}

impl<G: SampledGame> Entry<G> {
    /// Visits PUCT is allowed to see: everything the search *chose*, with the
    /// forced quota removed.
    ///
    /// The forced visits still count toward `w`, and so toward `q` below --
    /// measuring the candidate's value is the entire reason they were spent.
    /// What they must not do is make the candidate look *explored*, because
    /// then PUCT would treat evidence we insisted on buying as evidence the
    /// search had chosen to buy, and stop looking.
    fn n_eff(&self) -> u32 {
        self.n.saturating_sub(self.forced_n)
    }
}

struct Node<G: SampledGame> {
    state: G::State,
    entries: Vec<Entry<G>>,
    /// Set when fills have missed as many times as the node has earned
    /// candidates: the position has run out of distinct ones to offer, and
    /// without this the node would keep asking for ones it can never get,
    /// spinning the search forever.
    widen_exhausted: bool,
    /// Sampler draws requested at this node so far -- requested, not returned,
    /// so a model that proposes nothing still uses them up. Once they reach the
    /// node's draw target (see [`SampledParams::k_draws`]), a node still short
    /// of candidates is filled instead.
    draws: u32,
    /// Sampler draws in a row that brought nothing new; reset by anything that
    /// does. See [`SampledParams::draw_retries`].
    draw_misses: u32,
    /// Fills that brought nothing new. Random fills repeat themselves well
    /// before a position runs out of turns, so one miss is not the signal to
    /// stop; this many, against the node's candidate target, is.
    fill_misses: u32,
    /// `total_n` when this node last asked for a resample, and how many it
    /// has asked for.
    resampled_at: u32,
    resample_count: u32,
    /// Row-major `k x k` kernel matrix over `entries`, empty when the kernel is
    /// disabled. Built once at expansion: the distances depend only on the
    /// candidates and the state, neither of which changes afterwards, so paying
    /// `O(k^2)` distance evaluations per *node* keeps selection down to a plain
    /// `O(k^2)` weighted sum with no geometry in the inner loop.
    kern: Vec<f32>,
    total_n: u32,
    w_sum: f32,
    visited_p: f32,
    expanded: bool,
    terminal: bool,
    term_v: f32,
    /// The value head at this position, from its side to move: what its
    /// expansion backed up. `None` until expanded.
    nn_value: Option<f32>,
}

/// Recompute every entry's prior as `softmax` over the node's energies.
///
/// The single place priors are made, and it runs over the *whole* candidate set
/// every time that set grows. That is the point: energies from different model
/// calls are only comparable once they meet here, so normalising anywhere
/// earlier — per chunk, inside the model — throws the comparison away.
///
/// Entries also end up sorted by prior, descending. Progressive widening
/// reveals a prefix of this list, so the order decides which candidates a
/// lightly-visited node considers at all; it must be the ones the policy rates
/// highest, not the order the sampler happened to emit them in.
fn renormalise<G: SampledGame>(entries: &mut [Entry<G>]) {
    entries.sort_by(|a, b| b.energy.total_cmp(&a.energy));
    let max = entries
        .iter()
        .map(|e| e.energy)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Every candidate was reported as impossible, or there are none. A
        // uniform prior at least lets the search proceed on its own statistics.
        let u = 1.0 / entries.len().max(1) as f32;
        entries.iter_mut().for_each(|e| e.p = u);
        return;
    }
    let mut sum = 0.0f32;
    for e in entries.iter_mut() {
        e.p = (e.energy - max).exp();
        sum += e.p;
    }
    if sum > 0.0 {
        entries.iter_mut().for_each(|e| e.p /= sum);
    } else {
        let u = 1.0 / entries.len().max(1) as f32;
        entries.iter_mut().for_each(|e| e.p = u);
    }
}

/// `ln(exp(a) + exp(b))`, overflow-safe.
///
/// Merging two candidates that realize to the same position: their prior mass
/// adds, so their energies combine this way rather than by taking one.
fn logaddexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

/// Candidate-budget accounting.
///
/// The question progressive widening exists to answer: a node that receives one
/// visit uses exactly *one* of the `k` candidates it paid the sampler to
/// generate, and the other `k-1` are pure waste. Candidate generation is the
/// dominant cost of self-play (diffusion 56-60%, commit 17%, realize 9%), so if
/// most expanded nodes are visited once, a widening schedule that starts small
/// and grows with visits is worth its complexity — and if they are not, it is
/// not. Measured rather than assumed.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchStats {
    /// Nodes that were expanded (and so paid for a candidate set).
    pub nodes: u64,
    /// Candidates generated across all expanded nodes, after dedupe.
    pub candidates: u64,
    /// Candidates that were never selected even once.
    pub unvisited: u64,
    /// Expanded nodes bucketed by total visits:
    /// `1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+`.
    pub visit_hist: [u32; 8],
    /// Searches counted here.
    pub searches: u64,
    /// Resample requests (see [`SampledParams::resample_every`]), and the new
    /// candidates they added.
    pub resamples: u64,
    pub resamples_root: u64,
    pub resample_new: u64,
    /// Searches whose played move was a candidate a resample added.
    pub chosen_resampled: u64,
}

impl SearchStats {
    /// Bucket index for a visit count, matching `visit_hist`'s documented edges.
    fn bucket(n: u32) -> usize {
        match n {
            0 | 1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            9..=16 => 4,
            17..=32 => 5,
            33..=64 => 6,
            _ => 7,
        }
    }

    /// Accumulate another search's counts into this one.
    pub fn merge(&mut self, other: &SearchStats) {
        self.nodes += other.nodes;
        self.candidates += other.candidates;
        self.unvisited += other.unvisited;
        self.searches += other.searches;
        self.resamples += other.resamples;
        self.resamples_root += other.resamples_root;
        self.resample_new += other.resample_new;
        self.chosen_resampled += other.chosen_resampled;
        for (a, b) in self.visit_hist.iter_mut().zip(other.visit_hist.iter()) {
            *a += b;
        }
    }

    /// Fraction of generated candidates that were never tried. This is the P0
    /// gate for progressive widening.
    pub fn waste(&self) -> f64 {
        if self.candidates == 0 {
            0.0
        } else {
            self.unvisited as f64 / self.candidates as f64
        }
    }
}

pub struct SampledSearchResult<G: SampledGame> {
    /// Root candidates with their raw visit counts.
    pub candidates: Vec<(G::Move, u32)>,
    /// The normalised priors the search actually selected on, parallel to
    /// `candidates`. Returned rather than recomputed because the sampler is
    /// stochastic — asking it again would yield different candidates, and the
    /// search selected against *these*.
    pub priors: Vec<f32>,
    /// Which candidates came from forced exploration, parallel to `candidates`.
    pub explore: Vec<bool>,
    /// Visits each root candidate received because a floor demanded them,
    /// parallel to `candidates`. Counted as they happened rather than estimated,
    /// so the target knows exactly what forced exploration bought.
    pub forced: Vec<u32>,
    /// Index into `candidates` of the move to play.
    pub chosen: usize,
    /// Candidate-budget accounting over the whole tree, not just the root.
    pub stats: SearchStats,
    /// Root value from the side to move's perspective.
    pub value: f64,
    pub nodes: u64,
}

impl<G: SampledGame> SampledSearchResult<G> {
    /// Visit counts with forced-only candidates dropped, normalised.
    ///
    /// KataGo's policy-target pruning in the exact-count form: drop any child
    /// whose visits were all bought by a floor, keep the full count of any that
    /// earned one on merit, and never prune the most-visited child. Without
    /// this, every forced visit and every randomly injected candidate would be
    /// taught to the policy as if the search had chosen it.
    pub fn policy_target(&self, k_forced: f32) -> Vec<f32> {
        self.policy_target_with(&SampledParams {
            k_forced,
            ..Default::default()
        })
    }

    /// As [`Self::policy_target`]; the `params` are accepted for symmetry with
    /// the search and for the `min_explore_visits` warning path.
    ///
    /// Search and target must agree about what was bought. They cannot drift
    /// here, because the target reads the counts the search recorded as it
    /// granted them rather than reconstructing a floor from the priors.
    pub fn policy_target_with(&self, params: &SampledParams) -> Vec<f32> {
        let _ = params;
        let n = self.candidates.len();
        let mut counts: Vec<f32> = Vec::with_capacity(n);
        // Every candidate is in the target, forced-exploration ones included --
        // they are part of the sample the energy head is being fit to, and
        // omitting them would make the softmax normalise over a set the search
        // did not actually consider.
        //
        // What is removed is the forced visits themselves. Those were bought,
        // not chosen, so they say nothing about which move search preferred. A
        // candidate that attracted no chosen visits therefore targets exactly
        // 0: it was examined, and the search declined to come back. That is a
        // real signal rather than missing data, which is why it is trained on
        // rather than pruned.
        //
        // In practice 0 is rare. A forced candidate ends the search with
        // `n_eff = 0`, which makes its PUCT exploration term maximal, so it is
        // among the first things selection reaches for once the quotas are paid
        // -- and by then its value estimate is trustworthy, because paying the
        // quota is what made it so.
        for (i, (_, v)) in self.candidates.iter().enumerate() {
            let forced = self.forced.get(i).copied().unwrap_or(0);
            counts.push(v.saturating_sub(forced) as f32);
        }
        let sum: f32 = counts.iter().sum();
        if sum <= 0.0 {
            // No candidate earned a chosen visit anywhere -- only reachable at
            // absurdly small budgets. Fall back to the most-visited child so a
            // target is never all zeros, which would be a zero-gradient row.
            let best = self
                .candidates
                .iter()
                .enumerate()
                .max_by_key(|(_, (_, v))| *v)
                .map(|(i, _)| i)
                .unwrap_or(0);
            counts.iter_mut().for_each(|c| *c = 0.0);
            if !counts.is_empty() {
                counts[best] = 1.0;
            }
            return counts;
        }
        counts.iter_mut().for_each(|c| *c /= sum);
        counts
    }
}

fn new_node<G: SampledGame>(state: G::State) -> Node<G> {
    let tv = G::terminal_value(&state);
    Node {
        state,
        entries: Vec::new(),
        widen_exhausted: false,
        draws: 0,
        draw_misses: 0,
        fill_misses: 0,
        resampled_at: 0,
        resample_count: 0,
        nn_value: None,
        kern: Vec::new(),
        total_n: 0,
        w_sum: 0.0,
        visited_p: 0.0,
        expanded: false,
        terminal: tv.is_some(),
        term_v: tv.unwrap_or(0.0),
    }
}

/// Assign forced-exploration quotas over a node's candidates.
///
/// Every explore candidate is covered first -- those are the moves the sampler
/// injected precisely because the energy head would not have proposed them, and
/// they are worthless without a value estimate. `forced_extra` more are drawn at
/// random from the rest, so the trustworthy estimates are not confined to the
/// injected set.
///
/// **Budgeted.** The quotas are capped at half the simulation budget. Without
/// that, the measured 4.3 explore candidates per node plus 4 extras at 30 visits
/// each demand ~250 forced visits, which is 62% of a 400-simulation search and
/// more than a 100-simulation fast search has in total -- the fast search would
/// become entirely forced visits and never run PUCT at all, silently disabling
/// playout-cap randomization. Explore candidates keep priority as the budget
/// tightens; the random extras are what get dropped.
fn assign_forced_quotas<G: SampledGame>(
    entries: &mut [Entry<G>],
    params: &SampledParams,
    rng: &mut ChaCha8Rng,
) {
    for e in entries.iter_mut() {
        e.forced_quota = 0;
    }
    let per = params.forced_visits;
    if per == 0 {
        return;
    }
    let budget = quota_budget(params);
    let mut spent = 0u32;

    for e in entries.iter_mut().filter(|e| e.explore) {
        if spent + per > budget {
            return;
        }
        e.forced_quota = per;
        spent += per;
    }

    let mut rest: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.explore)
        .map(|(i, _)| i)
        .collect();
    // Partial Fisher-Yates: only as many draws as we will actually use.
    let want = params.forced_extra as usize;
    for k in 0..want.min(rest.len()) {
        if spent + per > budget {
            break;
        }
        let j = k + (rng.gen::<u32>() as usize) % (rest.len() - k);
        rest.swap(k, j);
        entries[rest[k]].forced_quota = per;
        spent += per;
    }
}

/// Forced visits a node's quotas may add up to: half the simulation budget, or
/// one quota if that is less. Shared by the quotas set at expansion and the
/// top-ups a widen adds, so candidates arriving later cannot overdraw it.
fn quota_budget(params: &SampledParams) -> u32 {
    (params.simulations / 2).max(params.forced_visits)
}

/// Visits from a node's `j`-th refine (counting from zero, the first measured
/// from no visits) to its next: consecutive differences of `floor(i * every)`,
/// so a fractional interval alternates whole gaps averaging `every`, and a
/// whole one is exactly `every` every time. At least 1: one refine per visit
/// at most.
fn resample_gap(every: f32, j: u32) -> u32 {
    let e = every as f64;
    let gap = ((j as f64 + 1.0) * e).floor() - (j as f64 * e).floor();
    (gap as u32).max(1)
}

/// How many of a node's candidates are selectable at `n` visits.
///
/// The root always sees all of them: it is the node whose visit distribution
/// becomes the training target, and every candidate needs a chance to earn its
/// share. Interior nodes widen with visits.
fn visible<G: SampledGame>(node: &Node<G>, params: &SampledParams, is_root: bool) -> usize {
    let m = node.entries.len();
    if is_root || params.pw_beta <= 0.0 {
        return m;
    }
    let visits = node.total_n.max(1) as usize;
    let n = visits as f32;
    let target = (params.pw_beta * n.powf(params.pw_alpha)).ceil() as usize;
    // Never reveal more than one new candidate per visit. In `n` visits a node
    // can have tried at most `n` children, so an `n + 1`st is the most that can
    // be put to use — and this makes the one-per-visit property structural
    // rather than an accident of the default `pw_beta`/`pw_alpha`, whose
    // schedule happens to grow slowly enough only because `beta * alpha == 1`.
    target.min(visits + 1).clamp(1, m)
}

/// How many candidates a node has *earned* — what `visible` would show it if it
/// held the full set. The root earns all of them immediately: its visit
/// distribution is the training target, so every candidate needs its chance.
fn earned<G: SampledGame>(node: &Node<G>, params: &SampledParams, is_root: bool) -> usize {
    let k = params.k_candidates.max(1);
    if is_root || params.pw_beta <= 0.0 {
        return k;
    }
    let n = node.total_n.max(1) as f32;
    let target = (params.pw_beta * n.powf(params.pw_alpha)).ceil() as usize;
    target.clamp(1, k)
}

/// Sampler draws a node makes before it fills, once it has earned `n`
/// candidates: `k_draws` pro-rated over the `k_candidates` it can grow to.
fn draw_target(params: &SampledParams, n: usize) -> u32 {
    let k = params.k_candidates.max(1);
    ((n * params.k_draws).div_ceil(k)) as u32
}

/// Select a child by PUCT, with root-only forced playouts.
/// Returns `(index, forced)` — `forced` when a floor demanded this visit
/// rather than PUCT choosing it, so the target can subtract exactly those.
fn select<G: SampledGame>(node: &Node<G>, params: &SampledParams, is_root: bool) -> (usize, bool) {
    let n = visible(node, params, is_root);
    debug_assert!(n > 0);

    // Sweep the root once: every candidate is visited a single time before PUCT
    // takes over, so each has a real value estimate behind it rather than an FPU
    // guess. That seeded visit is *not* evidence the search chose the move, so
    // `policy_target_with` subtracts it back out of the training target.
    if is_root {
        if let Some(i) = node.entries.iter().position(|e| e.n == 0) {
            // The sweep visit is seeded, not chosen: it counts as forced.
            return (i, true);
        }
    }

    if is_root && params.forced_visits > 0 {
        // Serve the largest outstanding quota. Unlike the KataGo floor this
        // replaces, the quota is flat and is bought from `n = 0` -- the whole
        // point is to measure candidates PUCT would never try, and a floor that
        // only keeps *already-tried* candidates alive cannot do that.
        let mut best_deficit = 0u32;
        let mut forced_pick = usize::MAX;
        for (i, e) in node.entries.iter().enumerate().take(n) {
            let deficit = e.forced_quota.saturating_sub(e.forced_n);
            if deficit > best_deficit {
                best_deficit = deficit;
                forced_pick = i;
            }
        }
        if forced_pick != usize::MAX {
            return (forced_pick, true);
        }
    }

    let parent_q = if node.total_n > 0 {
        node.w_sum / node.total_n as f32
    } else {
        0.0
    };
    let fpu = puct::fpu(parent_q, node.visited_p);

    // KR-PUCT (docs/algorithm.md "Selection"): regress each candidate's value
    // and visit mass over its neighbours before scoring it. Guarded so that the
    // disabled path below is *literally* the code that ran before the kernel
    // existed — not an algebraically equivalent rewrite.
    // `m` is the full candidate count and the kernel matrix's stride; `n` is the
    // widened prefix that may be *selected*. Statistics still regress over every
    // candidate, including ones not yet selectable — a hidden neighbour's visits
    // are perfectly good evidence about a visible candidate.
    let m = node.entries.len();
    if params.kernel_sigma > 0.0 && node.kern.len() == m * m {
        let mut n_hat = vec![0.0f32; n];
        let mut nq_hat = vec![0.0f32; n];
        let mut w_hat = vec![0.0f32; n];
        for i in 0..n {
            let row = &node.kern[i * m..(i + 1) * m];
            for (j, e) in node.entries.iter().enumerate() {
                // Two different visit counts on purpose: `n_hat` drives the
                // exploration term and so excludes forced visits, while
                // `q_hat`'s denominator is the full count that `w` was
                // accumulated over.
                n_hat[i] += row[j] * e.n_eff() as f32;
                nq_hat[i] += row[j] * e.n as f32;
                w_hat[i] += row[j] * e.w;
            }
        }
        let sqrt_total: f32 = n_hat.iter().sum::<f32>().max(1.0).sqrt();
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..n {
            // A candidate with no kernel-weighted visits falls back to the
            // node's own value, never a flat 0 — see `puct::fpu` for why 0 is a
            // trap. Normally a neighbour supplies an estimate and this is dead.
            let q = if nq_hat[i] > 0.0 {
                w_hat[i] / nq_hat[i]
            } else {
                fpu
            };
            let u = params.c_puct * node.entries[i].p * sqrt_total / (1.0 + n_hat[i]);
            let s = q + u;
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        return (best, false);
    }

    // Forced visits are invisible to the exploration term, in the numerator as
    // well as the denominator: a node whose visits were mostly bought rather
    // than chosen has explored less than its raw count suggests, and `sqrt(N)`
    // should say so.
    let eff_total: u32 = node.entries.iter().map(|e| e.n_eff()).sum();
    let sqrt_total = (eff_total as f32).max(1.0).sqrt();
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, e) in node.entries.iter().enumerate().take(n) {
        // `e.w` is accumulated in the *owning node's* perspective (the backup
        // negates on the way up), so it is used as-is here. `q` deliberately
        // uses the *full* visit count: the forced visits were spent measuring
        // exactly this, and discarding them would throw the estimate away.
        let q = if e.n > 0 { e.w / e.n as f32 } else { fpu };
        let u = params.c_puct * e.p * sqrt_total / (1.0 + e.n_eff() as f32);
        let s = q + u;
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    (best, false)
}

/// A resumable search: it runs until it needs an expansion, hands the position
/// out, and continues once the answer comes back.
///
/// Splitting it this way is what lets many games share one batched forward.
/// Each game still has at most one expansion outstanding, so the tree statistics
/// are identical to running them one at a time — no virtual loss, no
/// approximation, exactly the property `batch_search` documents for the discrete
/// games.
pub struct Search<G: SampledGame> {
    tree: Vec<Node<G>>,
    params: SampledParams,
    rng: ChaCha8Rng,
    ply: u32,
    sims_left: u32,
    nodes: u64,
    /// (node, entry, forced) for each edge descended through in the current
    /// simulation. `forced` rides along to `backup` instead of being tallied
    /// at selection, so a quota is only charged for a visit that is backed up.
    path: Vec<(usize, usize, bool)>,
    /// Node awaiting expansion, if `step` returned `NeedExpansion`.
    pending: Option<usize>,
    /// Node awaiting more candidates, if `step` returned `NeedWiden`, and
    /// whether the request was a fill.
    pending_widen: Option<(usize, bool)>,
    /// Where to pick the descent back up after a widen, with `path` intact.
    resume_at: Option<usize>,
    /// `(node, entry)`: the candidate a widen just added to an interior node,
    /// which the resumed descent visits next.
    visit_new: Option<(usize, usize)>,
    /// `(node, entry)` a pending `NeedResample` started from.
    pending_resample: Option<(usize, usize)>,
    /// Resample requests made, for [`SearchStats::resamples`], and how many
    /// of them were at the root.
    resamples: u64,
    resamples_root: u64,
    /// Entry indices, root first, the next simulation must descend through
    /// before PUCT takes over: a visit forced by hand (`force_line`).
    forced_line: std::collections::VecDeque<usize>,
}

/// One candidate edge, flattened for inspection.
///
/// Everything here is already in the tree; this exists so a caller outside the
/// crate can read it without `Node`/`Entry` becoming public API.
pub struct TreeEdge<G: SampledGame> {
    pub mv: G::Move,
    /// Prior probability, normalised over this node's candidate set.
    pub prior: f32,
    /// Raw energy `f(s, a)` as the model reported it, before any softmax.
    pub energy: f32,
    pub visits: u32,
    /// Mean value over this edge's visits, from the perspective of the side to
    /// move at the *parent*. `None` when the edge was never visited.
    pub value: Option<f32>,
    /// Index into the exported node list, when this edge has been expanded.
    pub child: Option<usize>,
    /// Drawn by forced exploration rather than proposed by the model.
    pub explore: bool,
    /// Of `visits`, how many a floor demanded rather than PUCT choosing.
    pub forced: u32,
    /// Sampler draws merged into this candidate: 1 for the draw that created
    /// it, plus each later draw that reached the same position.
    pub draws: u32,
    /// The value head's estimate of the position this edge leads to, from the
    /// side to move at the *parent* -- what the edge's first visit backed up,
    /// before any search below it. `value` is the search's mean over every
    /// visit. `None` until the child has been evaluated.
    pub prior_value: Option<f32>,
    /// Added by hand ([`Search::add_candidate`]) rather than by the model.
    pub manual: bool,
}

/// A node of the search tree, flattened for inspection.
pub struct TreeNode<G: SampledGame> {
    pub parent: Option<usize>,
    pub total_visits: u32,
    /// Mean value over this node's visits, from the side to move here.
    pub value: Option<f32>,
    pub terminal: bool,
    pub term_value: f32,
    /// The value head here, from the side to move here. `None` until expanded.
    pub nn_value: Option<f32>,
    pub edges: Vec<TreeEdge<G>>,
    /// Row-major `k x k` KR-PUCT kernel over `edges`, in `edges` order, or
    /// empty when the kernel is off (`kernel_sigma == 0`). Entry `(i, j)` is
    /// how much of candidate `j`'s visits and value candidate `i` borrows when
    /// selection scores it, so row `i` is exactly the "relative influence" of
    /// the rest of the set on `i`. The diagonal is 1 by construction.
    pub kernel: Vec<f32>,
}

impl<G: SampledGame> Search<G> {
    /// The parameters this search runs with -- for a caller that needs them
    /// after the fact, like self-play forming the target this search's floors
    /// must be subtracted from.
    pub fn params(&self) -> &SampledParams {
        &self.params
    }

    /// Flatten the tree for inspection: analysis views, not the search itself.
    ///
    /// Node 0 is the root and every `child` index refers to this list, so a
    /// caller can walk variations without knowing the internal layout. Nothing
    /// here is recomputed — these are the numbers the search actually selected
    /// on, which is the point: an analysis view that re-derives priors would be
    /// showing a different sampler draw than the one that produced the tree.
    pub fn export_tree(&self) -> Vec<TreeNode<G>> {
        self.tree
            .iter()
            .enumerate()
            .map(|(i, node)| {
                TreeNode {
                    // The root is node 0 and every other node is reached by
                    // exactly one edge, so the parent is recoverable by scan.
                    parent: if i == 0 {
                        None
                    } else {
                        self.tree
                            .iter()
                            .position(|n| n.entries.iter().any(|e| e.child == i as i32))
                    },
                    total_visits: node.total_n,
                    value: (node.total_n > 0).then(|| node.w_sum / node.total_n as f32),
                    nn_value: node.nn_value,
                    terminal: node.terminal,
                    term_value: node.term_v,
                    edges: node
                        .entries
                        .iter()
                        .map(|e| TreeEdge {
                            mv: e.mv.clone(),
                            // Already a normalised softmax over this node's
                            // set — `renormalise` is the only writer of `p`.
                            prior: e.p,
                            energy: e.energy,
                            visits: e.n,
                            value: (e.n > 0).then(|| e.w / e.n as f32),
                            prior_value: self.prior_value(node, e),
                            child: (e.child >= 0).then_some(e.child as usize),
                            explore: e.explore,
                            forced: e.forced_n,
                            draws: e.draws,
                            manual: e.manual,
                        })
                        .collect(),
                    kernel: node.kern.clone(),
                }
            })
            .collect()
    }
    /// The value head's estimate of where `e` leads, from `node`'s side to
    /// move: the child's own evaluation, flipped across a change of side, or a
    /// hand-added entry's `eval` while it has no child yet.
    fn prior_value(&self, node: &Node<G>, e: &Entry<G>) -> Option<f32> {
        if e.child < 0 {
            return e.eval;
        }
        let child = &self.tree[e.child as usize];
        let v = if child.terminal {
            child.term_v
        } else {
            child.nn_value?
        };
        Some(if G::to_move(&node.state) != G::to_move(&child.state) {
            -v
        } else {
            v
        })
    }

    /// Allow `n` more simulations; the next [`drive`] runs exactly those. How
    /// an analysis session steps a search a few simulations at a time.
    pub fn add_simulations(&mut self, n: u32) {
        self.sims_left += n;
    }

    /// Replace the remaining budget. `params.simulations` keeps the configured
    /// one, which the forced-visit quotas are sized against, so a search
    /// started with a single simulation still searches like the configured
    /// one as it is stepped.
    pub fn set_remaining(&mut self, n: u32) {
        self.sims_left = n;
    }

    /// The node reached from the root through `entries`, one entry index per
    /// level.
    pub fn node_at(&self, entries: &[usize]) -> anyhow::Result<usize> {
        let mut cur = 0usize;
        for (depth, &i) in entries.iter().enumerate() {
            let e = self.tree[cur]
                .entries
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("depth {depth}: the node has no candidate {i}"))?;
            anyhow::ensure!(
                e.child >= 0,
                "depth {depth}: that turn has never been visited, so there is no node for it yet"
            );
            cur = e.child as usize;
        }
        Ok(cur)
    }

    /// Entry indices for a line of moves from the root, matched by the position
    /// each reaches -- the key candidates are merged by -- so the caller needs
    /// no knowledge of how entries are ordered. The last move need only be a
    /// candidate; every earlier one must lead to an existing node.
    pub fn resolve_line(&self, line: &[G::Move]) -> anyhow::Result<Vec<usize>> {
        let mut cur = 0usize;
        let mut out = Vec::with_capacity(line.len());
        for (depth, mv) in line.iter().enumerate() {
            let node = &self.tree[cur];
            let key = G::state_key(&G::apply(&node.state, mv));
            let i = node
                .entries
                .iter()
                .position(|e| G::state_key(&G::apply(&node.state, &e.mv)) == key)
                .ok_or_else(|| anyhow::anyhow!("depth {depth}: not a candidate of that node"))?;
            out.push(i);
            if depth + 1 < line.len() {
                let child = node.entries[i].child;
                anyhow::ensure!(
                    child >= 0,
                    "depth {depth}: that turn has never been visited, so there is no node for it yet"
                );
                cur = child as usize;
            }
        }
        Ok(out)
    }

    /// The position at `node`.
    pub fn state_of(&self, node: usize) -> &G::State {
        &self.tree[node].state
    }

    /// The move of `node`'s candidate `i`.
    pub fn candidate(&self, node: usize, i: usize) -> &G::Move {
        &self.tree[node].entries[i].mv
    }

    /// Make the next simulation descend through `entries` (root first) and
    /// continue by PUCT below the last one, adding that simulation to the
    /// budget. Every edge on the line counts as a forced visit: like the
    /// quotas, it buys a value estimate without looking like a choice PUCT made.
    pub fn force_line(&mut self, entries: &[usize]) -> anyhow::Result<()> {
        let (&last, above) = entries
            .split_last()
            .ok_or_else(|| anyhow::anyhow!("an empty line forces nothing"))?;
        anyhow::ensure!(self.tree[0].expanded, "the root has not been expanded yet");
        let parent = self.node_at(above)?;
        anyhow::ensure!(
            last < self.tree[parent].entries.len(),
            "the node has no candidate {last}"
        );
        self.forced_line = entries.iter().copied().collect();
        self.sims_left += 1;
        Ok(())
    }

    /// Add `mv` to `node`'s candidates by hand, with energy `energy` and the
    /// value head's estimate `eval` of where it leads (from `node`'s side to
    /// move). Priors are renormalised over the grown set, as a widen does.
    ///
    /// Returns the entry's index and whether it is new: a move reaching a
    /// position the node already holds *is* that candidate, not a second one.
    pub fn add_candidate(
        &mut self,
        node: usize,
        mv: G::Move,
        energy: f32,
        eval: Option<f32>,
    ) -> anyhow::Result<(usize, bool)> {
        anyhow::ensure!(node < self.tree.len(), "no node {node}");
        anyhow::ensure!(
            self.tree[node].expanded,
            "that position has not been evaluated yet: visit it first"
        );
        anyhow::ensure!(!self.tree[node].terminal, "the game is over there");
        let state = self.tree[node].state.clone();
        let key = G::state_key(&G::apply(&state, &mv));
        let find = |entries: &[Entry<G>]| {
            entries
                .iter()
                .position(|e| G::state_key(&G::apply(&state, &e.mv)) == key)
        };
        if let Some(i) = find(&self.tree[node].entries) {
            return Ok((i, false));
        }
        self.tree[node].entries.push(Entry {
            mv,
            energy,
            p: 0.0,
            n: 0,
            w: 0.0,
            child: -1,
            explore: false,
            forced_n: 0,
            forced_quota: 0,
            draws: 0,
            manual: true,
            eval,
            resampled: false,
        });
        renormalise(&mut self.tree[node].entries);
        // Every prior moved, so the visited mass FPU reads must be re-summed.
        let nd = &mut self.tree[node];
        nd.visited_p = nd.entries.iter().filter(|e| e.n > 0).map(|e| e.p).sum();
        self.build_kernel(node);
        let i = find(&self.tree[node].entries).expect("just added");
        Ok((i, true))
    }

    /// The root's value as `finish` reports it.
    ///
    /// Each candidate's value weighted by its *chosen* visits, not its raw
    /// ones. Each candidate's own `q` still uses every visit it received --
    /// the forced ones were spent measuring it and are good evidence -- but
    /// the position's value is a statement about what search concluded, and
    /// a plain `w_sum / total_n` lets a quota spent proving a move is bad
    /// drag the position's evaluation down with it. At a 100-simulation
    /// budget with one 30-visit quota that was enough to report a won
    /// position at 0.36, which `an_immediate_win_is_taken` caught.
    pub fn root_value(&self) -> f64 {
        let root = &self.tree[0];
        let eff_n: u32 = root.entries.iter().map(|e| e.n_eff()).sum();
        if eff_n > 0 {
            let acc: f64 = root
                .entries
                .iter()
                .filter(|e| e.n > 0)
                .map(|e| e.n_eff() as f64 * (e.w as f64 / e.n as f64))
                .sum();
            acc / eff_n as f64
        } else if root.total_n > 0 {
            // Every visit was forced -- only at budgets below the quota total.
            root.w_sum as f64 / root.total_n as f64
        } else {
            0.0
        }
    }

    /// What `finish` would play now, without finishing: the same rule on a
    /// copy of the RNG, so asking twice gives the same answer.
    pub fn chosen(&self) -> usize {
        self.pick(&mut self.rng.clone())
    }

    /// Play on *chosen* visits, the same counts the policy target is built
    /// from. Forced visits are bought, not preferred, and a candidate given a
    /// 30-visit quota would otherwise win the argmax outright at small budgets
    /// purely because we insisted on measuring it -- which is exactly what
    /// `an_immediate_win_is_taken` caught.
    fn pick(&self, rng: &mut ChaCha8Rng) -> usize {
        let eff: Vec<u32> = self.tree[0].entries.iter().map(|e| e.n_eff()).collect();
        if eff.is_empty() {
            0
        } else if self.ply < self.params.temp_moves {
            // Opening plies: sample proportional to visits, for self-play diversity.
            let total: u32 = eff.iter().sum();
            if total == 0 {
                rng.gen_range(0..eff.len())
            } else {
                let mut r = rng.gen_range(0..total);
                let mut pick = eff.len() - 1;
                for (i, n) in eff.iter().enumerate() {
                    if r < *n {
                        pick = i;
                        break;
                    }
                    r -= *n;
                }
                pick
            }
        } else {
            // Ties go to the earlier entry, which is the higher prior (entries
            // are kept sorted by energy). `max_by_key` alone returns the *last*
            // maximum, so at small budgets -- most root children on one visit
            // each -- it played the lowest-rated of them.
            eff.iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.cmp(b).then(j.cmp(i)))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    pub fn new(root: G::State, params: &SampledParams, ply: u32) -> Self {
        Self {
            tree: vec![new_node::<G>(root)],
            params: *params,
            rng: ChaCha8Rng::seed_from_u64(params.seed ^ (ply as u64).wrapping_mul(0x9E3779B9)),
            ply,
            sims_left: params.simulations,
            nodes: 0,
            path: Vec::new(),
            pending: None,
            pending_widen: None,
            resume_at: None,
            visit_new: None,
            pending_resample: None,
            resamples: 0,
            resamples_root: 0,
            forced_line: Default::default(),
        }
    }

    /// Advance until an expansion is needed or the budget is spent.
    pub fn step(&mut self) -> Step<G> {
        debug_assert!(
            self.pending.is_none(),
            "step called with an expansion outstanding"
        );
        while self.sims_left > 0 {
            let mut cur = match self.resume_at.take() {
                Some(node) => node,
                None => {
                    self.path.clear();
                    0
                }
            };
            let value = loop {
                if self.tree[cur].terminal {
                    break self.tree[cur].term_v;
                }
                if !self.tree[cur].expanded {
                    self.pending = Some(cur);
                    return Step::NeedExpansion(self.tree[cur].state.clone());
                }
                // The candidate a widen just added is where this simulation goes
                // next: widening is the visit, not the promise of one. Left to
                // PUCT, a new candidate starts at the first-play value, below
                // the child already being searched, and at a small `c_puct` it
                // was never tried at all -- nodes of 200 visits held 16
                // candidates and had visited one.
                let (pick, forced) = match self.visit_new.take() {
                    Some((node, entry)) if node == cur => (entry, false),
                    // A visit forced by hand (`force_line`) goes where it was
                    // told, widening nothing on the way: the reader is asking
                    // about this line, not about more candidates.
                    None if !self.forced_line.is_empty() => {
                        let pick = self.forced_line.pop_front().expect("checked non-empty");
                        (pick, true)
                    }
                    _ => {
                        // Ask for the next chunk before selecting, if this node
                        // has earned more candidates than it holds: a sampler
                        // draw while it has draws left, a fill once they are
                        // spent. The descent pauses here and resumes when the
                        // candidates arrive, so a draw that repeats is retried
                        // by the same simulation -- `draw_retries` times at an
                        // interior node, and then it fills.
                        if !self.tree[cur].widen_exhausted {
                            let node = &self.tree[cur];
                            let held = node.entries.len();
                            let want = earned(node, &self.params, cur == 0);
                            let retries = self.params.draw_retries;
                            let fill = node.draws >= draw_target(&self.params, want)
                                || (cur != 0 && retries > 0 && node.draw_misses >= retries);
                            // Unchunked, every draw came with the expansion, so
                            // a fill is the only widen there is.
                            if held < want && fill && !self.params.fills {
                                // No fills: the node keeps what its draws brought.
                                self.tree[cur].widen_exhausted = true;
                            } else if held < want && (self.params.pw_chunk > 0 || fill) {
                                self.pending_widen = Some((cur, fill));
                                // `path` is kept. Sorting this node's entries in
                                // `provide_widen` stays safe because `cur`'s own
                                // edge is not in `path` yet -- only its
                                // ancestors'.
                                self.resume_at = Some(cur);
                                return Step::NeedWiden {
                                    state: self.tree[cur].state.clone(),
                                    fill,
                                };
                            }
                        }
                        if self.tree[cur].entries.is_empty() {
                            // Genuinely nothing to play: either the model
                            // proposed nothing, or widening is exhausted. Score
                            // on value alone.
                            break 0.0;
                        }
                        let (pick, forced) = select(&self.tree[cur], &self.params, cur == 0);
                        // Every `resample_every` visits, refine instead of
                        // descending: redraw part of the picked turn and visit
                        // the result. Not on a forced visit, which exists to
                        // measure the candidate it was given.
                        let every = self.params.resample_every;
                        // `path` holds the edges above `cur`, so its length
                        // is `cur`'s depth (the root is 0).
                        let shallow = self.path.len() as u32 <= self.params.resample_max_depth;
                        let node = &self.tree[cur];
                        if every > 0.0
                            && shallow
                            && !forced
                            && node.total_n
                                >= node.resampled_at + resample_gap(every, node.resample_count)
                        {
                            if cur == 0 {
                                self.resamples_root += 1;
                            }
                            let from = node.entries[pick].mv.clone();
                            let state = node.state.clone();
                            self.tree[cur].resampled_at = self.tree[cur].total_n;
                            self.tree[cur].resample_count += 1;
                            self.pending_resample = Some((cur, pick));
                            self.resamples += 1;
                            // As for a widen: `path` holds only ancestors'
                            // edges, so `provide_resample` may re-sort `cur`.
                            self.resume_at = Some(cur);
                            return Step::NeedResample { state, from };
                        }
                        (pick, forced)
                    }
                };
                // Tallied in `backup`, with the visit itself, so a quota is
                // only ever charged for a visit that happened.
                self.path.push((cur, pick, forced));
                let child = self.tree[cur].entries[pick].child;
                if child >= 0 {
                    cur = child as usize;
                } else {
                    let next = G::apply(&self.tree[cur].state, &self.tree[cur].entries[pick].mv);
                    let ci = self.tree.len();
                    self.tree.push(new_node::<G>(next));
                    self.tree[cur].entries[pick].child = ci as i32;
                    cur = ci;
                }
            };
            self.backup(value);
            self.sims_left -= 1;
            // A forced line belongs to one simulation, even one that ended
            // (at a terminal) before using all of it.
            self.forced_line.clear();
        }
        Step::Done
    }

    /// Supply the value and candidates for the position `step` handed out.
    pub fn provide(&mut self, value: f32, candidates: Vec<(G::Move, f32, bool)>) {
        let cur = self
            .pending
            .take()
            .expect("provide without a pending expansion");
        self.nodes += 1;

        // Merge candidates that reach the same position — distinct turns
        // frequently converge, and unmerged they fragment the visit counts they
        // are supposed to concentrate.
        let mut seen: std::collections::HashMap<u128, usize> = Default::default();
        let mut entries: Vec<Entry<G>> = Vec::with_capacity(candidates.len());
        for (mv, p, explore) in candidates {
            let key = G::state_key(&G::apply(&self.tree[cur].state, &mv));
            match seen.get(&key) {
                Some(&i) => {
                    entries[i].draws += 1;
                    entries[i].energy = logaddexp(entries[i].energy, p);
                    // A position the model also proposed still deserves its
                    // guaranteed look if exploration reached it independently.
                    entries[i].explore |= explore;
                }
                None => {
                    seen.insert(key, entries.len());
                    entries.push(Entry {
                        mv,
                        energy: p,
                        p: 0.0,
                        n: 0,
                        w: 0.0,
                        child: -1,
                        explore,
                        forced_n: 0,
                        forced_quota: 0,
                        draws: 1,
                        manual: false,
                        eval: None,
                        resampled: false,
                    });
                }
            }
        }
        renormalise(&mut entries);
        // More draws than candidates can yield more distinct ones than the node
        // holds; keep the ones the policy rates highest.
        let k_max = self.params.k_candidates.max(1);
        if entries.len() > k_max {
            entries.truncate(k_max);
            renormalise(&mut entries);
        }
        // Quotas are set once, over the node's initial candidate set. Widening
        // appends later; a widened explore candidate is covered by the top-up
        // in `provide_widen`, and widened ordinary candidates are not drawn into
        // the random extras -- the draw would no longer be uniform over the set
        // it was specified against, and re-drawing would revoke quotas already
        // being paid.
        assign_forced_quotas(&mut entries, &self.params, &mut self.rng);
        self.tree[cur].entries = entries;
        self.tree[cur].expanded = true;
        self.tree[cur].nn_value = Some(value);
        self.tree[cur].draws = expansion_k(&self.params) as u32;
        self.build_kernel(cur);

        self.backup(value);
        self.sims_left -= 1;
        self.forced_line.clear();
    }

    /// Negamax backup along the descent path. `value` is from the leaf's side.
    fn backup(&mut self, value: f32) {
        let mut v = value;
        for &(node, entry, forced) in self.path.iter().rev() {
            // Negate only when the side to move actually changed across this
            // edge. Negating unconditionally assumes strict alternation, which
            // does not hold for every game: Minions ends a game the instant a
            // necromancer dies, mid-turn, so the winner is still the side to
            // move in the terminal state. Under the old unconditional flip the
            // winning move was backed up as -1 — the search learned to avoid
            // winning, and could only ever win by running out the turn cap.
            // Guarded by `terminal_child_value_is_negative_for_the_side_that_just_moved`
            // in the minions crate.
            let child = self.tree[node].entries[entry].child;
            let flipped = if child >= 0 {
                G::to_move(&self.tree[node].state) != G::to_move(&self.tree[child as usize].state)
            } else {
                true
            };
            if flipped {
                v = -v;
            }
            let nd = &mut self.tree[node];
            let e = &mut nd.entries[entry];
            if e.n == 0 {
                nd.visited_p += e.p;
            }
            e.n += 1;
            if forced {
                e.forced_n += 1;
            }
            e.w += v;
            nd.total_n += 1;
            nd.w_sum += v;
        }
    }

    /// Append a chunk of candidates to a node that earned them.
    ///
    /// Merges against the candidates already there — the sampler is stochastic
    /// and re-proposes, so without merging a node would accumulate duplicates
    /// that fragment exactly the visits they should concentrate. Priors are
    /// renormalised over the grown set and the kernel matrix rebuilt.
    ///
    /// A proposal widen spends the node's draws whatever it brings, and one that
    /// brings nothing new counts toward [`SampledParams::draw_retries`]. A fill
    /// that brings nothing new is a miss, and the node is exhausted once its
    /// misses reach the candidates it has earned. See [`SampledParams::k_draws`].
    pub fn provide_widen(&mut self, candidates: Vec<(G::Move, f32, bool)>) {
        let (cur, fill) = self
            .pending_widen
            .take()
            .expect("provide_widen without a pending widen");
        let mut seen: std::collections::HashMap<u128, usize> = Default::default();
        for (i, e) in self.tree[cur].entries.iter().enumerate() {
            seen.insert(G::state_key(&G::apply(&self.tree[cur].state, &e.mv)), i);
        }
        let k_max = self.params.k_candidates.max(1);
        let before = self.tree[cur].entries.len();
        // A fill is not a sampler draw, so it adds to no entry's draw count.
        let draw = u32::from(!fill);
        let mut added: Vec<u128> = Vec::new();
        for (mv, p, explore) in candidates {
            if self.tree[cur].entries.len() >= k_max {
                break;
            }
            let key = G::state_key(&G::apply(&self.tree[cur].state, &mv));
            match seen.get(&key) {
                Some(&i) => {
                    self.tree[cur].entries[i].draws += draw;
                    self.tree[cur].entries[i].energy =
                        logaddexp(self.tree[cur].entries[i].energy, p);
                    self.tree[cur].entries[i].explore |= explore;
                }
                None => {
                    seen.insert(key, self.tree[cur].entries.len());
                    added.push(key);
                    self.tree[cur].entries.push(Entry {
                        mv,
                        energy: p,
                        p: 0.0,
                        n: 0,
                        w: 0.0,
                        child: -1,
                        explore,
                        forced_n: 0,
                        forced_quota: 0,
                        draws: draw,
                        manual: false,
                        eval: None,
                        resampled: false,
                    });
                }
            }
        }
        if fill {
            if self.tree[cur].entries.len() == before {
                self.tree[cur].fill_misses += 1;
                let want = earned(&self.tree[cur], &self.params, cur == 0) as u32;
                if self.tree[cur].fill_misses >= want {
                    self.tree[cur].widen_exhausted = true;
                }
            }
        } else {
            self.tree[cur].draws += chunk_k(&self.params) as u32;
            if added.is_empty() {
                self.tree[cur].draw_misses += 1;
            }
        }
        if !added.is_empty() {
            self.tree[cur].draw_misses = 0;
        }
        // Re-softmax over the grown set. Sorting inside is safe here because the
        // paused descent's `path` holds only this node's ancestors' edges.
        renormalise(&mut self.tree[cur].entries);
        // The resumed descent visits what arrived -- the highest-prior new
        // candidate, if a chunk brought several. The root is exempt: it fills
        // its whole set before its first visit and then sweeps every candidate
        // once (see `select`), which gives each its visit anyway.
        if cur != 0 && !added.is_empty() {
            let state = &self.tree[cur].state;
            self.visit_new = self.tree[cur]
                .entries
                .iter()
                .position(|e| added.contains(&G::state_key(&G::apply(state, &e.mv))))
                .map(|i| (cur, i));
        }
        // Top up only: an explore candidate that arrived by widening still needs
        // a value estimate, but the random extras were drawn over the initial
        // set and are not redrawn (see `provide`). Within the same budget as
        // the initial quotas: fills can bring a node a dozen explore
        // candidates, and a quota each would spend the whole search on them.
        let per = self.params.forced_visits;
        if per > 0 {
            let budget = quota_budget(&self.params);
            let mut spent: u32 = self.tree[cur].entries.iter().map(|e| e.forced_quota).sum();
            for e in self.tree[cur].entries.iter_mut() {
                if e.explore && e.forced_quota == 0 {
                    if spent + per > budget {
                        break;
                    }
                    e.forced_quota = per;
                    spent += per;
                }
            }
        }
        self.build_kernel(cur);
    }

    /// Add what a resample brought and send the paused visit to it.
    ///
    /// Merged like a widen: a resample that lands on a position the node
    /// already holds visits that candidate instead, and one that brought
    /// nothing visits the candidate it started from -- the descent it
    /// replaced. Not capped by `k_candidates`: the point is to grow the set
    /// around the turns the search is spending its visits on.
    pub fn provide_resample(&mut self, candidates: Vec<(G::Move, f32, bool)>) {
        let (cur, from) = self
            .pending_resample
            .take()
            .expect("provide_resample without a pending resample");
        let state = self.tree[cur].state.clone();
        let key_of = |mv: &G::Move| G::state_key(&G::apply(&state, mv));
        let mut target = key_of(&self.tree[cur].entries[from].mv);
        if let Some((mv, energy, explore)) = candidates.into_iter().next() {
            target = key_of(&mv);
            let entries = &mut self.tree[cur].entries;
            if !entries.iter().any(|e| key_of(&e.mv) == target) {
                entries.push(Entry {
                    mv,
                    energy,
                    p: 0.0,
                    n: 0,
                    w: 0.0,
                    child: -1,
                    explore,
                    forced_n: 0,
                    forced_quota: 0,
                    draws: 1,
                    manual: false,
                    eval: None,
                    resampled: true,
                });
                renormalise(entries);
                self.build_kernel(cur);
            }
        }
        self.visit_new = self.tree[cur]
            .entries
            .iter()
            .position(|e| key_of(&e.mv) == target)
            .map(|i| (cur, i));
    }

    /// Precompute the node's kernel matrix. A no-op when the kernel is off, so
    /// the disabled path allocates nothing and costs nothing.
    fn build_kernel(&mut self, node: usize) {
        let sigma = self.params.kernel_sigma;
        if sigma <= 0.0 {
            return;
        }
        let n = self.tree[node].entries.len();
        let mut kern = vec![0.0f32; n * n];
        let denom = 2.0 * sigma * sigma;
        for i in 0..n {
            kern[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let d = G::move_distance(
                    &self.tree[node].state,
                    &self.tree[node].entries[i].mv,
                    &self.tree[node].entries[j].mv,
                );
                let k = (-(d * d) / denom).exp();
                kern[i * n + j] = k;
                kern[j * n + i] = k;
            }
        }
        self.tree[node].kern = kern;
    }

    pub fn finish(self) -> SampledSearchResult<G> {
        let mut stats = SearchStats::default();
        for node in self.tree.iter() {
            if !node.expanded {
                continue;
            }
            stats.nodes += 1;
            stats.candidates += node.entries.len() as u64;
            stats.unvisited += node.entries.iter().filter(|e| e.n == 0).count() as u64;
            stats.visit_hist[SearchStats::bucket(node.total_n)] += 1;
            stats.resample_new += node.entries.iter().filter(|e| e.resampled).count() as u64;
        }
        stats.searches = 1;
        stats.resamples = self.resamples;
        stats.resamples_root = self.resamples_root;

        let root = &self.tree[0];
        let candidates: Vec<(G::Move, u32)> =
            root.entries.iter().map(|e| (e.mv.clone(), e.n)).collect();
        let priors: Vec<f32> = root.entries.iter().map(|e| e.p).collect();
        let explore: Vec<bool> = root.entries.iter().map(|e| e.explore).collect();
        let forced: Vec<u32> = root.entries.iter().map(|e| e.forced_n).collect();
        let value = self.root_value();
        let chosen = self.pick(&mut self.rng.clone());
        stats.chosen_resampled = root.entries.get(chosen).map_or(0, |e| e.resampled as u64);
        SampledSearchResult {
            candidates,
            priors,
            explore,
            forced,
            chosen,
            value,
            nodes: self.nodes,
            stats,
        }
    }
}

/// Candidates to request at a node's *first* visit.
///
/// Zero when chunking is on — the node only needs a value, and will ask for a
/// candidate if something descends through it. With chunking off the node's
/// draws all come up front (`k_draws` of them), and only a fill can follow.
pub fn expansion_k(params: &SampledParams) -> usize {
    if params.pw_chunk > 0 {
        0
    } else {
        params.k_draws
    }
}

/// Candidates to request per model call: the chunk when chunked generation is
/// on, otherwise the node's whole set.
pub fn chunk_k(params: &SampledParams) -> usize {
    if params.pw_chunk > 0 {
        params.pw_chunk.min(params.k_candidates.max(1))
    } else {
        params.k_candidates
    }
}

/// Run one search to completion against `model`. The single-game path — used by
/// eval, the interactive engine and the tests; self-play drives many
/// [`Search`]es in lockstep instead, so its forwards are batched.
pub fn search<G: SampledGame, M: SampledModel<G>>(
    model: &M,
    root: G::State,
    params: &SampledParams,
    ply: u32,
) -> anyhow::Result<SampledSearchResult<G>> {
    let mut s = Search::new(root, params, ply);
    drive(model, &mut s)?;
    Ok(s.finish())
}

/// Serve one search's requests from `model` until it is done: the single
/// place a lone search's expansions and widens are answered, so `search` and
/// `search_analyzed` cannot drift apart.
pub fn drive<G: SampledGame, M: SampledModel<G>>(
    model: &M,
    s: &mut Search<G>,
) -> anyhow::Result<()> {
    let params = *s.params();
    loop {
        match s.step() {
            Step::NeedExpansion(state) => {
                // Value only; candidates wait until something descends here.
                let (v, cands) = model.expand(&state, expansion_k(&params))?;
                s.provide(v, cands);
            }
            Step::NeedWiden { state, fill } => {
                let cands = model
                    .widen_batch(std::slice::from_ref(&state), &[fill], chunk_k(&params))?
                    .pop()
                    .unwrap_or_default();
                s.provide_widen(cands);
            }
            Step::NeedResample { state, from } => {
                let cands = model
                    .resample_batch(std::slice::from_ref(&state), std::slice::from_ref(&from))?
                    .pop()
                    .unwrap_or_default();
                s.provide_resample(cands);
            }
            Step::Done => return Ok(()),
        }
    }
}

/// As [`search`], but also returns the tree it built.
///
/// `finish` consumes the search, so a caller that wants both the played move
/// and the tree behind it cannot get them from `search` alone. Analysis wants
/// both; self-play wants neither and should keep using `search_lockstep`.
pub fn search_analyzed<G: SampledGame, M: SampledModel<G>>(
    model: &M,
    root: G::State,
    params: &SampledParams,
    ply: u32,
) -> anyhow::Result<(SampledSearchResult<G>, Vec<TreeNode<G>>)> {
    let mut s = Search::new(root, params, ply);
    drive(model, &mut s)?;
    let tree = s.export_tree();
    Ok((s.finish(), tree))
}

/// Drive `n` independent searches in lockstep so their expansions batch.
///
/// Every game holds at most one outstanding expansion, so each one's tree
/// statistics are bit-identical to running it alone — the batching is purely an
/// inference-scheduling change. Games that finish early simply drop out, and the
/// batch narrows.
pub fn search_lockstep<G: SampledGame, M: SampledModel<G>>(
    model: &M,
    roots: Vec<G::State>,
    params: &SampledParams,
    plies: &[u32],
) -> anyhow::Result<Vec<SampledSearchResult<G>>> {
    let mut searches: Vec<Search<G>> = roots
        .into_iter()
        .enumerate()
        .map(|(i, r)| Search::new(r, params, plies.get(i).copied().unwrap_or(0)))
        .collect();
    let mut done = vec![false; searches.len()];

    loop {
        // Collect this round's pending expansions.
        let mut want: Vec<usize> = Vec::with_capacity(searches.len());
        let mut widen: Vec<bool> = Vec::with_capacity(searches.len());
        let mut fills: Vec<bool> = Vec::new();
        let mut states: Vec<G::State> = Vec::with_capacity(searches.len());
        let mut resample: Vec<(usize, G::State, G::Move)> = Vec::new();
        for (i, s) in searches.iter_mut().enumerate() {
            if done[i] {
                continue;
            }
            match s.step() {
                Step::NeedExpansion(st) => {
                    want.push(i);
                    widen.push(false);
                    states.push(st);
                }
                Step::NeedWiden { state, fill } => {
                    want.push(i);
                    widen.push(true);
                    fills.push(fill);
                    states.push(state);
                }
                Step::NeedResample { state, from } => resample.push((i, state, from)),
                Step::Done => done[i] = true,
            }
        }
        if want.is_empty() && resample.is_empty() {
            break;
        }
        if !resample.is_empty() {
            let (ids, (rs, rf)): (Vec<usize>, (Vec<G::State>, Vec<G::Move>)) =
                resample.into_iter().map(|(i, s, m)| (i, (s, m))).unzip();
            let res = model.resample_batch(&rs, &rf)?;
            anyhow::ensure!(
                res.len() == ids.len(),
                "resample_batch returned {} results for {} states",
                res.len(),
                ids.len()
            );
            for (i, cands) in ids.into_iter().zip(res) {
                searches[i].provide_resample(cands);
            }
        }
        // Split by need: expansions want a value (`k = 0`, trunk only), widens
        // want a candidate. Only widens pay for the generative heads.
        let exp: Vec<G::State> = states
            .iter()
            .zip(&widen)
            .filter(|(_, w)| !**w)
            .map(|(s, _)| s.clone())
            .collect();
        let wid: Vec<G::State> = states
            .iter()
            .zip(&widen)
            .filter(|(_, w)| **w)
            .map(|(s, _)| s.clone())
            .collect();
        let exp_res = if exp.is_empty() {
            Vec::new()
        } else {
            model.expand_batch(&exp, expansion_k(params))?
        };
        let wid_res = if wid.is_empty() {
            Vec::new()
        } else {
            model.widen_batch(&wid, &fills, chunk_k(params))?
        };
        anyhow::ensure!(
            exp_res.len() == exp.len() && wid_res.len() == wid.len(),
            "expand_batch/widen_batch returned {}/{} results for {}/{} states",
            exp_res.len(),
            wid_res.len(),
            exp.len(),
            wid.len()
        );
        let mut exp_out = exp_res.into_iter();
        let mut wid_out = wid_res.into_iter();
        for (i, is_widen) in want.into_iter().zip(widen) {
            if is_widen {
                let cands = wid_out.next().expect("one widen result per widen request");
                searches[i].provide_widen(cands);
            } else {
                let (v, cands) = exp_out.next().expect("one value per expansion request");
                searches[i].provide(v, cands);
            }
        }
    }
    Ok(searches.into_iter().map(|s| s.finish()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A counting game: state is a number, a move subtracts 1..=3, reaching 0
    /// wins for the mover. Small enough to reason about, and it has a known
    /// optimal strategy (leave a multiple of 4).
    #[derive(Clone, Debug)]
    struct CountState {
        n: u32,
        side: u8,
    }
    struct Count;

    impl SampledGame for Count {
        type State = CountState;
        type Move = u32;
        const PLANES: usize = 1;
        fn initial_state() -> CountState {
            CountState { n: 21, side: 1 }
        }
        fn to_move(s: &CountState) -> u8 {
            s.side
        }
        fn apply(s: &CountState, mv: &u32) -> CountState {
            CountState {
                n: s.n.saturating_sub(*mv),
                side: 3 - s.side,
            }
        }
        fn terminal_value(s: &CountState) -> Option<f32> {
            // The player who took the last counter won, so the side now to move
            // has lost.
            (s.n == 0).then_some(-1.0)
        }
        fn encode_planes(s: &CountState, out: &mut [f32]) {
            out[0] = s.n as f32;
        }
        fn state_key(s: &CountState) -> u128 {
            (s.n as u128) << 8 | s.side as u128
        }
    }

    /// Proposes every legal subtraction with a uniform prior and a zero value —
    /// so the search's strength comes purely from tree search, which is exactly
    /// what we want to test.
    struct Mock;
    impl SampledModel<Count> for Mock {
        /// Deliberately ignores `k` and proposes every legal move.
        ///
        /// A perfect proposer, so the tests using it isolate *search* behaviour
        /// from what the sampler happened to draw. Truncating to `k` instead
        /// would make it a prefix-returner, which cannot work one candidate at a
        /// time — asked for one it would offer the same move forever and the
        /// winning line would never be proposed at all. Real samplers are
        /// stochastic; `Pool` is the fixture that models that.
        fn expand(&self, s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
            Ok((
                0.0,
                // Equal energies — any constant is a uniform prior after the
                // search's softmax, so 0.0 states that plainly.
                (1..=3u32)
                    .filter(|m| *m <= s.n)
                    .map(|m| (m, 0.0, false))
                    .collect(),
            ))
        }
    }

    /// Always proposes the same move — exercises candidate merging.
    struct Duplicating;
    impl SampledModel<Count> for Duplicating {
        fn expand(&self, s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
            let _ = s;
            Ok((
                0.0,
                vec![
                    (1, 0.0, false),
                    (1, 0.0, false),
                    (1, 0.0, false),
                    (2, 0.0, false),
                ],
            ))
        }
    }

    struct Empty;
    impl SampledModel<Count> for Empty {
        fn expand(&self, _s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
            Ok((0.25, vec![]))
        }
    }

    fn params(sims: u32) -> SampledParams {
        SampledParams {
            simulations: sims,
            temp_moves: 0,
            seed: 1,
            ..Default::default()
        }
    }

    /// Answer one widen the way `drive` does.
    fn widen<M: SampledModel<Count>>(
        model: &M,
        state: &CountState,
        fill: bool,
        k: usize,
    ) -> Vec<(u32, f32, bool)> {
        model
            .widen_batch(std::slice::from_ref(state), &[fill], k)
            .unwrap()
            .pop()
            .unwrap()
    }

    /// Proposes move 1 and nothing else, however often it is asked. Fills with
    /// the other moves in turn, flagged explore, when `fills` is set;
    /// otherwise it keeps the default `widen_batch`, which has no exploration
    /// source.
    struct Stuck {
        fills: bool,
        cursor: std::sync::atomic::AtomicUsize,
    }
    impl Stuck {
        fn new(fills: bool) -> Self {
            Self {
                fills,
                cursor: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }
    impl SampledModel<Count> for Stuck {
        fn expand(&self, _s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
            Ok((
                0.0,
                if k == 0 {
                    Vec::new()
                } else {
                    vec![(1, 0.0, false)]
                },
            ))
        }
        fn widen_batch(
            &self,
            states: &[CountState],
            fill: &[bool],
            k: usize,
        ) -> anyhow::Result<Vec<Vec<(u32, f32, bool)>>> {
            Ok(states
                .iter()
                .zip(fill)
                .map(|(s, &f)| {
                    if !f {
                        return self.expand(s, k).unwrap().1;
                    }
                    if !self.fills {
                        return Vec::new();
                    }
                    let others: Vec<u32> = (2..=3u32).filter(|m| *m <= s.n).collect();
                    (0..k)
                        .filter(|_| !others.is_empty())
                        .map(|_| {
                            let i = self
                                .cursor
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            (others[i % others.len()], 0.0, true)
                        })
                        .collect()
                })
                .collect())
        }
    }

    /// A node draws exactly its `D` proposals, however often they repeat, and
    /// fills the shortfall to `N` with exploration candidates -- for `D` below,
    /// at and above `N`.
    #[test]
    fn a_node_draws_d_times_then_fills_the_shortfall() {
        for d in [1usize, 3, 5] {
            let params = SampledParams {
                simulations: 60,
                k_candidates: 3,
                k_draws: d,
                pw_chunk: 1,
                forced_visits: 0,
                temp_moves: 0,
                seed: 4,
                ..Default::default()
            };
            let model = Stuck::new(true);
            let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
            let mut proposals: std::collections::HashMap<usize, usize> = Default::default();
            loop {
                match search.step() {
                    Step::NeedExpansion(st) => {
                        let (v, c) = model.expand(&st, expansion_k(&params)).unwrap();
                        search.provide(v, c);
                    }
                    Step::NeedWiden { state, fill } => {
                        let (node, _) = search.pending_widen.expect("a widen names its node");
                        if !fill {
                            *proposals.entry(node).or_default() += 1;
                        }
                        search.provide_widen(widen(&model, &state, fill, chunk_k(&params)));
                    }
                    Step::NeedResample { .. } => unreachable!("resampling is off"),
                    Step::Done => break,
                }
            }
            assert_eq!(
                proposals[&0], d,
                "D={d}: the root must be asked exactly D times"
            );
            for (node, n) in &proposals {
                assert!(*n <= d, "D={d}: node {node} was asked {n} times");
            }
            let mut held: Vec<(u32, u32, bool)> = search.tree[0]
                .entries
                .iter()
                .map(|e| (e.mv, e.draws, e.explore))
                .collect();
            held.sort();
            assert_eq!(
                held,
                vec![(1, d as u32, false), (2, 0, true), (3, 0, true)],
                "D={d}: the one distinct proposal (drawn D times) and two fills"
            );
        }
    }

    /// With `fills` off a node never asks for a fill: once its draws are spent
    /// it keeps what they brought, and the search still runs to the end.
    #[test]
    fn without_fills_a_node_keeps_only_its_draws() {
        let params = SampledParams {
            simulations: 60,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 1,
            fills: false,
            forced_visits: 0,
            temp_moves: 0,
            seed: 4,
            ..Default::default()
        };
        let model = Stuck::new(true);
        let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        loop {
            match search.step() {
                Step::NeedExpansion(st) => {
                    let (v, c) = model.expand(&st, expansion_k(&params)).unwrap();
                    search.provide(v, c);
                }
                Step::NeedWiden { state, fill } => {
                    assert!(!fill, "a fill was requested with fills off");
                    search.provide_widen(widen(&model, &state, fill, chunk_k(&params)));
                }
                Step::NeedResample { .. } => unreachable!("resampling is off"),
                Step::Done => break,
            }
        }
        let held: Vec<u32> = search.tree[0].entries.iter().map(|e| e.mv).collect();
        assert_eq!(held, vec![1], "the root holds its one distinct draw only");
    }

    /// A tie on visits goes to the higher prior. At small budgets most root
    /// children hold one visit each, so the tie decides the move.
    #[test]
    fn visit_ties_go_to_the_higher_prior() {
        let params = SampledParams {
            simulations: 1,
            temp_moves: 0,
            ..Default::default()
        };
        let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        let Step::NeedExpansion(_) = search.step() else {
            panic!("the first step expands the root");
        };
        search.provide(0.0, vec![(3, 0.0, false), (1, 2.0, false), (2, 1.0, false)]);
        for e in search.tree[0].entries.iter_mut() {
            e.n = 1;
        }
        let pick = search.chosen();
        assert_eq!(
            search.tree[0].entries[pick].mv, 1,
            "the highest-energy child"
        );
    }

    /// Proposes move 1 only; a resample of move `m` answers `m % 3 + 1`.
    struct Refiner;
    impl SampledModel<Count> for Refiner {
        fn expand(&self, _s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
            Ok((
                0.0,
                if k == 0 {
                    Vec::new()
                } else {
                    vec![(1, 0.0, false)]
                },
            ))
        }
        fn resample_batch(
            &self,
            states: &[CountState],
            from: &[u32],
        ) -> anyhow::Result<Vec<Vec<(u32, f32, bool)>>> {
            Ok(states
                .iter()
                .zip(from)
                .map(|(s, &m)| {
                    let r = m % 3 + 1;
                    if r <= s.n {
                        vec![(r, 0.0, false)]
                    } else {
                        Vec::new()
                    }
                })
                .collect())
        }
    }

    /// Every `resample_every` visits a node refines instead of descending: the
    /// resampled turn joins the node and takes that visit.
    #[test]
    fn resampling_adds_and_visits_a_neighbour() {
        let params = SampledParams {
            simulations: 40,
            k_candidates: 3,
            k_draws: 1,
            pw_chunk: 1,
            fills: false,
            resample_every: 10.0,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            forced_visits: 0,
            temp_moves: 0,
            seed: 4,
            ..Default::default()
        };
        let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        let mut at_root = Vec::new();
        loop {
            match search.step() {
                Step::NeedExpansion(st) => {
                    let (v, c) = Refiner.expand(&st, expansion_k(&params)).unwrap();
                    search.provide(v, c);
                }
                Step::NeedWiden { state, fill } => {
                    search.provide_widen(widen(&Refiner, &state, fill, chunk_k(&params)));
                }
                Step::NeedResample { state, from } => {
                    let (node, _) = search.pending_resample.expect("a resample names its node");
                    if node == 0 {
                        at_root.push(search.tree[0].total_n);
                    }
                    let c = Refiner
                        .resample_batch(&[state], &[from])
                        .unwrap()
                        .pop()
                        .unwrap();
                    search.provide_resample(c);
                }
                Step::Done => {
                    break;
                }
            }
        }
        assert_eq!(
            at_root,
            vec![10, 20, 30],
            "the root refines every 10 visits"
        );
        let added: Vec<&Entry<Count>> = search.tree[0]
            .entries
            .iter()
            .filter(|e| e.resampled)
            .collect();
        assert!(!added.is_empty(), "a resample added a candidate");
        assert!(
            added.iter().all(|e| e.n > 0),
            "and each one added was visited"
        );
    }

    /// A whole interval refines exactly that often; a fraction alternates
    /// whole gaps that average it.
    #[test]
    fn resample_gaps_average_the_interval() {
        let gaps = |every: f32| (0..6).map(|j| resample_gap(every, j)).collect::<Vec<_>>();
        assert_eq!(gaps(2.0), vec![2, 2, 2, 2, 2, 2]);
        assert_eq!(gaps(10.0), vec![10; 6]);
        assert_eq!(gaps(1.5), vec![1, 2, 1, 2, 1, 2]);
        assert_eq!(gaps(2.5), vec![2, 3, 2, 3, 2, 3]);
        assert_eq!(gaps(1.25), vec![1, 1, 1, 2, 1, 1]);
        assert_eq!(gaps(0.5), vec![1; 6], "never more than one per visit");
    }

    /// At an interior node, a draw that keeps repeating is retried
    /// `draw_retries` times and then the widen fills, however much of the
    /// node's draw budget is left. 0 turns the cap off, and only the budget
    /// ends the run.
    #[test]
    fn repeated_draws_at_an_interior_node_fall_back_to_a_fill() {
        for retries in [3u32, 1, 0] {
            let params = SampledParams {
                simulations: 80,
                k_candidates: 3,
                // A budget the misses never reach, so every early fill is the
                // retry cap's doing.
                k_draws: 60,
                draw_retries: retries,
                pw_chunk: 1,
                forced_visits: 0,
                temp_moves: 0,
                seed: 4,
                ..Default::default()
            };
            let model = Stuck::new(true);
            let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
            // Per interior node: the current run of draws that added nothing,
            // and the longest run seen.
            let mut runs: std::collections::HashMap<usize, (u32, u32)> = Default::default();
            // Fills the cap caused, as opposed to a spent draw budget.
            let mut cap_fills = 0;
            loop {
                match search.step() {
                    Step::NeedExpansion(st) => {
                        let (v, c) = model.expand(&st, expansion_k(&params)).unwrap();
                        search.provide(v, c);
                    }
                    Step::NeedWiden { state, fill } => {
                        let (node, _) = search.pending_widen.expect("a widen names its node");
                        let n = &search.tree[node];
                        let before = n.entries.len();
                        let spent = n.draws >= draw_target(&params, earned(n, &params, node == 0));
                        search.provide_widen(widen(&model, &state, fill, chunk_k(&params)));
                        if node == 0 {
                            continue;
                        }
                        let grew = search.tree[node].entries.len() > before;
                        let run = runs.entry(node).or_default();
                        if fill && !spent {
                            assert_eq!(
                                run.0, retries,
                                "retries={retries}: node {node} filled after {} misses",
                                run.0
                            );
                            cap_fills += 1;
                        }
                        run.0 = if grew { 0 } else { run.0 + u32::from(!fill) };
                        run.1 = run.1.max(run.0);
                    }
                    Step::NeedResample { .. } => unreachable!("resampling is off"),
                    Step::Done => break,
                }
            }
            let longest = runs.values().map(|r| r.1).max().unwrap_or(0);
            if retries > 0 {
                assert!(
                    cap_fills > 5,
                    "retries={retries}: only {cap_fills} fills from the cap"
                );
                assert_eq!(
                    longest, retries,
                    "retries={retries}: a run of {longest} misses"
                );
            } else {
                assert_eq!(
                    cap_fills, 0,
                    "retries=0: filled before the draw budget ran out"
                );
                assert!(
                    longest > 3,
                    "retries=0: runs of misses should run on, got {longest}"
                );
            }
        }
    }

    /// Running a search a few simulations at a time builds exactly the tree
    /// running it in one go does: a session's stepping is only a pause.
    #[test]
    fn stepping_a_search_matches_running_it_at_once() {
        let params = SampledParams {
            simulations: 60,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 1,
            temp_moves: 0,
            seed: 12,
            ..Default::default()
        };
        let root = CountState { n: 21, side: 1 };
        let mut whole = Search::<Count>::new(root.clone(), &params, 0);
        drive(&Mock, &mut whole).unwrap();
        let mut stepped = Search::<Count>::new(root, &params, 0);
        stepped.set_remaining(0);
        for n in [1u32, 1, 5, 13, 40] {
            stepped.add_simulations(n);
            drive(&Mock, &mut stepped).unwrap();
        }
        let shape = |s: &Search<Count>| {
            s.export_tree()
                .iter()
                .map(|n| {
                    n.edges
                        .iter()
                        .map(|e| (e.mv, e.visits, e.forced))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(shape(&whole), shape(&stepped));
        assert_eq!(whole.chosen(), stepped.chosen());
    }

    /// A forced line is descended exactly, one simulation, and every edge on
    /// it counts the visit as forced.
    #[test]
    fn a_forced_line_is_descended_and_counted_as_forced() {
        let params = SampledParams {
            simulations: 40,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 0,
            forced_visits: 0,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            temp_moves: 0,
            seed: 3,
            ..Default::default()
        };
        let mut s = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        drive(&Mock, &mut s).unwrap();
        let before = s.export_tree();
        // The least-visited root candidate -- PUCT's last choice -- and its
        // child's first candidate.
        let (i, e) = before[0]
            .edges
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.visits)
            .unwrap();
        let child = e.child.expect("the root sweep visits every candidate");
        assert!(
            !before[child].edges.is_empty(),
            "unchunked: a visited node holds its set"
        );
        s.force_line(&[i, 0]).unwrap();
        drive(&Mock, &mut s).unwrap();
        let after = s.export_tree();
        assert_eq!(
            after[0].total_visits,
            before[0].total_visits + 1,
            "one simulation"
        );
        assert_eq!(after[0].edges[i].visits, before[0].edges[i].visits + 1);
        assert_eq!(after[0].edges[i].forced, before[0].edges[i].forced + 1);
        assert_eq!(
            after[child].edges[0].forced,
            before[child].edges[0].forced + 1
        );
        assert!(s.force_line(&[]).is_err());
        assert!(
            s.force_line(&[99]).is_err(),
            "a candidate the root does not have"
        );
    }

    /// A hand-added candidate joins its node with its own energy and value,
    /// priors renormalised over the grown set, and a move reaching a position
    /// the node already holds is that candidate. `prior_value` is the value
    /// head's opinion of a child, kept apart from the search's mean and
    /// oriented to the parent's side.
    #[test]
    fn hand_added_candidates_carry_energy_and_value() {
        /// Proposes move 1 only; its value is `n / 100` for the side to move.
        struct Valued;
        impl SampledModel<Count> for Valued {
            fn expand(&self, s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
                let cands = if k == 0 {
                    Vec::new()
                } else {
                    vec![(1u32, 0.0, false)]
                };
                Ok((s.n as f32 / 100.0, cands))
            }
            fn widen_batch(
                &self,
                states: &[CountState],
                fill: &[bool],
                _k: usize,
            ) -> anyhow::Result<Vec<Vec<(u32, f32, bool)>>> {
                Ok(fill
                    .iter()
                    .take(states.len())
                    .map(|&f| {
                        if f {
                            Vec::new()
                        } else {
                            vec![(1u32, 0.0, false)]
                        }
                    })
                    .collect())
            }
        }
        let params = SampledParams {
            simulations: 12,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 1,
            forced_visits: 0,
            temp_moves: 0,
            seed: 2,
            ..Default::default()
        };
        let mut s = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        drive(&Valued, &mut s).unwrap();
        let t = s.export_tree();
        assert_eq!(t[0].edges.len(), 1, "the model only knows move 1");
        // The visited child's prior value: the value head there, 20/100,
        // flipped if the side to move changed.
        let one = &t[0].edges[0];
        let child = one.child.expect("move 1 was visited");
        let flip = Count::to_move(s.state_of(0)) != Count::to_move(s.state_of(child));
        assert_eq!(one.prior_value, Some(if flip { -0.20 } else { 0.20 }));
        assert_eq!(t[child].nn_value, Some(0.20));

        let (i, new) = s.add_candidate(0, 2, 5.0, Some(0.25)).unwrap();
        assert!(new);
        let t = s.export_tree();
        let e = &t[0].edges[i];
        assert!(e.manual && e.mv == 2 && e.visits == 0 && e.child.is_none());
        assert_eq!(
            e.prior_value,
            Some(0.25),
            "before any visit, the value given"
        );
        let sum: f32 = t[0].edges.iter().map(|e| e.prior).sum();
        assert!((sum - 1.0).abs() < 1e-5, "priors renormalised: {sum}");
        assert!(
            t[0].edges.iter().all(|x| x.prior <= e.prior),
            "energy 5 leads"
        );

        let (again, new) = s.add_candidate(0, 2, 0.0, None).unwrap();
        assert!(!new && again == i, "same position: the same candidate");
        // And the search can then visit it.
        s.force_line(&[i]).unwrap();
        drive(&Valued, &mut s).unwrap();
        assert_eq!(s.export_tree()[0].edges[i].visits, 1);
    }

    /// A fill that collides is retried: random fills repeat themselves long
    /// before a position runs out of turns, so a miss must not leave the node
    /// short.
    #[test]
    fn a_colliding_fill_is_retried() {
        /// Proposes move 1; fills with 1, 1 (both misses), then 2 and 3, then
        /// only misses.
        struct Colliding {
            fills: std::sync::Mutex<std::collections::VecDeque<u32>>,
        }
        impl SampledModel<Count> for Colliding {
            fn expand(&self, _s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
                Ok((
                    0.0,
                    if k == 0 {
                        Vec::new()
                    } else {
                        vec![(1, 0.0, false)]
                    },
                ))
            }
            fn widen_batch(
                &self,
                states: &[CountState],
                fill: &[bool],
                k: usize,
            ) -> anyhow::Result<Vec<Vec<(u32, f32, bool)>>> {
                Ok(states
                    .iter()
                    .zip(fill)
                    .map(|(s, &f)| {
                        if !f {
                            return self.expand(s, k).unwrap().1;
                        }
                        let m = self.fills.lock().unwrap().pop_front().unwrap_or(1);
                        vec![(m, 0.0, true)]
                    })
                    .collect())
            }
        }
        let params = SampledParams {
            simulations: 40,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 1,
            forced_visits: 0,
            temp_moves: 0,
            seed: 4,
            ..Default::default()
        };
        let model = Colliding {
            fills: std::sync::Mutex::new([1, 1, 2, 3].into()),
        };
        let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        drive(&model, &mut search).unwrap();
        let root = &search.tree[0];
        let mut held: Vec<(u32, u32, bool)> = root
            .entries
            .iter()
            .map(|e| (e.mv, e.draws, e.explore))
            .collect();
        held.sort();
        assert_eq!(held, vec![(1, 3, true), (2, 0, true), (3, 0, true)]);
        assert_eq!(root.fill_misses, 2);
        assert!(
            !root.widen_exhausted,
            "two misses against a target of three must not exhaust"
        );
    }

    /// With no exploration source every fill misses, and the node is exhausted
    /// after as many misses as it has earned candidates, rather than asking
    /// forever.
    #[test]
    fn a_fill_that_adds_nothing_exhausts_the_node() {
        let params = SampledParams {
            simulations: 40,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 1,
            forced_visits: 0,
            temp_moves: 0,
            seed: 4,
            ..Default::default()
        };
        let model = Stuck::new(false);
        let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        drive(&model, &mut search).unwrap();
        let root = &search.tree[0];
        assert!(root.widen_exhausted, "an empty fill must exhaust the node");
        let held: Vec<(u32, u32)> = root.entries.iter().map(|e| (e.mv, e.draws)).collect();
        assert_eq!(held, vec![(1, 3)]);
    }

    /// Unchunked, the draws come with the expansion and a fill follows it.
    #[test]
    fn unchunked_draws_are_filled_too() {
        let params = SampledParams {
            simulations: 20,
            k_candidates: 3,
            k_draws: 4,
            pw_chunk: 0,
            forced_visits: 0,
            temp_moves: 0,
            seed: 4,
            ..Default::default()
        };
        assert_eq!(
            expansion_k(&params),
            4,
            "unchunked expansion asks for all D draws"
        );
        let (_, tree) = search_analyzed::<Count, _>(
            &Stuck::new(true),
            CountState { n: 21, side: 1 },
            &params,
            0,
        )
        .expect("search");
        let mut held: Vec<(u32, u32, bool)> = tree[0]
            .edges
            .iter()
            .map(|e| (e.mv, e.draws, e.explore))
            .collect();
        held.sort();
        assert_eq!(held, vec![(1, 1, false), (2, 0, true), (3, 0, true)]);
    }

    /// Quotas topped up for explore candidates that arrive by widening stay
    /// inside the budget the initial quotas are held to. Fills bring many
    /// explore candidates, and a quota each would spend the whole search on
    /// them.
    #[test]
    fn explore_quotas_from_widening_stay_within_the_budget() {
        let params = SampledParams {
            simulations: 60,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 1,
            forced_visits: 30,
            temp_moves: 0,
            seed: 4,
            ..Default::default()
        };
        let model = Stuck::new(true);
        let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        drive(&model, &mut search).unwrap();
        let root = &search.tree[0];
        assert_eq!(root.entries.iter().filter(|e| e.explore).count(), 2);
        let quotas: u32 = root.entries.iter().map(|e| e.forced_quota).sum();
        assert_eq!(
            quotas,
            quota_budget(&params),
            "two explore candidates at 30 each would overdraw a budget of {}",
            quota_budget(&params)
        );
    }

    /// The identity that keeps the flat search available as a *mode* rather than
    /// being replaced: with `kernel_sigma = 0` the kernel path is skipped and the
    /// search must be bit-identical to the one that existed before KR-PUCT.
    ///
    /// Compared on the full result — visit counts, priors and the chosen move —
    /// not just the final pick, because a divergence anywhere in the tree would
    /// change the training target even when the played move happens to agree.
    #[test]
    fn zero_bandwidth_matches_plain_puct() {
        // A distance function that is deliberately *not* a delta, so if the
        // sigma = 0 guard ever stopped short-circuiting, the kernel would return
        // different answers and this test would catch it.
        struct Spread;
        impl SampledModel<Count> for Spread {
            fn expand(&self, s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
                // Descending energies, so move 1 is the policy's favourite.
                Ok((0.1, (1..=3).map(|m| (m, -(m as f32), m == 3)).collect()))
            }
        }
        let mut params = SampledParams {
            simulations: 120,
            k_forced: 2.0,
            min_explore_visits: 2.0,
            forced_visits: 30,
            forced_extra: 4,
            temp_moves: 0,
            seed: 7,
            ..Default::default()
        };
        params.kernel_sigma = 0.0;
        let flat = search::<Count, _>(&Spread, CountState { n: 21, side: 1 }, &params, 0)
            .expect("flat search");

        // Same params, same seed, through the same code path a KR run takes.
        let again =
            search::<Count, _>(&Spread, CountState { n: 21, side: 1 }, &params, 0).expect("repeat");
        assert_eq!(
            flat.candidates
                .iter()
                .map(|(m, n)| (*m, *n))
                .collect::<Vec<_>>(),
            again
                .candidates
                .iter()
                .map(|(m, n)| (*m, *n))
                .collect::<Vec<_>>(),
            "sigma = 0 must be deterministic and unchanged"
        );
        assert_eq!(flat.chosen, again.chosen);
        assert_eq!(flat.priors, again.priors);
        assert_eq!(
            flat.policy_target_with(&params),
            again.policy_target_with(&params)
        );

        // And a positive bandwidth must actually do something different, or the
        // test above proves nothing.
        params.kernel_sigma = 1.0;
        let kr = search::<Count, _>(&Spread, CountState { n: 21, side: 1 }, &params, 0)
            .expect("kr search");
        let flat_n: Vec<u32> = flat.candidates.iter().map(|(_, n)| *n).collect();
        let kr_n: Vec<u32> = kr.candidates.iter().map(|(_, n)| *n).collect();
        assert_ne!(
            flat_n, kr_n,
            "a live kernel must change the visit distribution"
        );
    }

    /// The root sweep: every candidate is visited at least once, so each has a
    /// real value estimate rather than an FPU guess behind its prior.
    #[test]
    fn every_root_candidate_is_visited_at_least_once() {
        struct Wide;
        impl SampledModel<Count> for Wide {
            fn expand(&self, _s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
                // Lopsided priors: without the sweep, PUCT would starve move 3.
                Ok((
                    0.0,
                    vec![(1, 0.98, false), (2, 0.01, false), (3, 0.01, false)],
                ))
            }
        }
        let params = SampledParams {
            simulations: 24,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            temp_moves: 0,
            seed: 3,
            ..Default::default()
        };
        let r =
            search::<Count, _>(&Wide, CountState { n: 21, side: 1 }, &params, 0).expect("search");
        for (mv, n) in &r.candidates {
            assert!(*n >= 1, "root candidate {mv:?} was never visited (n={n})");
        }
    }

    /// The seeded visit is the search asking, not choosing, so it must not
    /// survive into the training target.
    #[test]
    fn the_seeded_root_visit_is_stripped_from_the_target() {
        struct Wide;
        impl SampledModel<Count> for Wide {
            fn expand(&self, _s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
                Ok((
                    0.0,
                    vec![(1, 0.98, false), (2, 0.01, false), (3, 0.01, false)],
                ))
            }
        }
        // Budget just large enough to sweep and little else: the two weak moves
        // should hold their single seeded visit and nothing more.
        let params = SampledParams {
            simulations: 6,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            temp_moves: 0,
            seed: 5,
            ..Default::default()
        };
        let r =
            search::<Count, _>(&Wide, CountState { n: 21, side: 1 }, &params, 0).expect("search");
        let target = r.policy_target_with(&params);
        for (i, (mv, n)) in r.candidates.iter().enumerate() {
            if *n == 1 {
                assert_eq!(
                    target[i], 0.0,
                    "move {mv:?} had only its seeded visit but kept target mass"
                );
            }
        }
        let sum: f32 = target.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "target must stay normalised, got {sum}"
        );
    }

    /// Progressive widening exposes only a prefix at interior nodes, and never
    /// restricts the root.
    #[test]
    fn widening_limits_interior_nodes_but_not_the_root() {
        let params = SampledParams {
            pw_alpha: 0.5,
            pw_beta: 2.0,
            ..Default::default()
        };
        let mut node = new_node::<Count>(CountState { n: 21, side: 1 });
        node.entries = (0..12)
            .map(|i| Entry::<Count> {
                mv: i,
                energy: 0.0,
                p: 1.0 / 12.0,
                n: 0,
                w: 0.0,
                child: -1,
                explore: false,
                forced_n: 0,
                forced_quota: 0,
                draws: 1,
                manual: false,
                eval: None,
                resampled: false,
            })
            .collect();

        node.total_n = 1;
        assert_eq!(
            visible(&node, &params, true),
            12,
            "the root always sees every candidate"
        );
        // At one visit the schedule is ceil(pw_beta) — the initial count is
        // pw_beta itself, which is why there is no separate knob for it.
        assert_eq!(
            visible(&node, &params, false),
            2,
            "one visit -> ceil(pw_beta)"
        );
        node.total_n = 16;
        assert_eq!(visible(&node, &params, false), 8, "ceil(2*sqrt(16)) = 8");
        node.total_n = 100;
        assert_eq!(
            visible(&node, &params, false),
            12,
            "clamped to the candidate count"
        );

        let off = SampledParams {
            pw_beta: 0.0,
            ..params
        };
        node.total_n = 1;
        assert_eq!(
            visible(&node, &off, false),
            12,
            "pw_beta = 0 disables widening"
        );
    }

    /// Widening must never reveal more than one new candidate per visit, for
    /// *any* schedule — not just the defaults, whose `beta * alpha == 1` makes
    /// the property hold by accident.
    #[test]
    fn widening_never_adds_more_than_one_candidate_per_visit() {
        for (beta, alpha) in [(2.0f32, 0.5f32), (4.0, 0.5), (8.0, 0.9), (1.0, 0.3)] {
            let params = SampledParams {
                pw_alpha: alpha,
                pw_beta: beta,
                ..Default::default()
            };
            let mut node = new_node::<Count>(CountState { n: 21, side: 1 });
            node.entries = (0..64)
                .map(|i| Entry::<Count> {
                    mv: i,
                    energy: 0.0,
                    p: 1.0 / 64.0,
                    n: 0,
                    w: 0.0,
                    child: -1,
                    explore: false,
                    forced_n: 0,
                    forced_quota: 0,
                    draws: 1,
                    manual: false,
                    eval: None,
                    resampled: false,
                })
                .collect();
            let mut prev = 0usize;
            for n in 1..=200u32 {
                node.total_n = n;
                let v = visible(&node, &params, false);
                assert!(
                    v <= prev + 1 || prev == 0,
                    "beta={beta} alpha={alpha}: jumped {prev} -> {v} at n={n}"
                );
                assert!(v >= 1, "a node must always have something to select");
                prev = v;
            }
        }
    }

    /// The root is not widened: it must be fully populated, and then every
    /// candidate visited once, before any candidate gets a second visit.
    ///
    /// Its visit distribution is the training target, so a candidate that never
    /// got a look would be trained against as if search had rejected it.
    #[test]
    fn the_root_is_fully_populated_and_swept_before_any_second_visit() {
        struct Rot {
            cursor: std::sync::atomic::AtomicUsize,
        }
        impl SampledModel<Count> for Rot {
            fn expand(&self, _s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
                let start = self
                    .cursor
                    .fetch_add(k.max(1), std::sync::atomic::Ordering::Relaxed);
                Ok((
                    0.0,
                    (0..k.max(1))
                        .map(|i| ((start + i) % 8 + 1) as u32)
                        .map(|m| (m, 0.0, false))
                        .collect(),
                ))
            }
        }
        let k = 8usize;
        let params = SampledParams {
            simulations: 200,
            k_candidates: k,
            pw_chunk: 1,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            temp_moves: 0,
            seed: 4,
            ..Default::default()
        };
        let model = Rot {
            cursor: std::sync::atomic::AtomicUsize::new(0),
        };
        let mut search = Search::<Count>::new(CountState { n: 40, side: 1 }, &params, 0);
        let mut saw_full_before_first_visit = false;
        loop {
            // Observe the root between steps.
            let root = &search.tree[0];
            if root.expanded {
                let held = root.entries.len();
                let max_n = root.entries.iter().map(|e| e.n).max().unwrap_or(0);
                let min_n = root.entries.iter().map(|e| e.n).min().unwrap_or(0);
                if max_n == 0 && held == k {
                    saw_full_before_first_visit = true;
                }
                assert!(
                    max_n == 0 || held == k,
                    "root took a visit while holding {held} of {k} candidates"
                );
                assert!(
                    max_n < 2 || min_n >= 1,
                    "a root candidate reached {max_n} visits while another still had {min_n}"
                );
            }
            match search.step() {
                Step::NeedExpansion(st) => {
                    let (v, c) = model.expand(&st, chunk_k(&params)).unwrap();
                    search.provide(v, c);
                }
                Step::NeedWiden { state, fill } => {
                    search.provide_widen(widen(&model, &state, fill, chunk_k(&params)));
                }
                Step::NeedResample { .. } => unreachable!("resampling is off"),
                Step::Done => break,
            }
        }
        assert!(
            saw_full_before_first_visit,
            "root was never fully populated before its sweep"
        );
    }

    /// Resuming a paused descent must not corrupt the tree: visits still sum
    /// correctly through the root, and the search still finds the winning line.
    ///
    /// The risk is `path`. A widen keeps it and `provide_widen` re-sorts the
    /// node's entries, so if the paused node's own edge were already in `path`,
    /// the sort would silently repoint it at a different move.
    #[test]
    fn resuming_after_a_widen_keeps_the_tree_consistent() {
        let params = SampledParams {
            simulations: 120,
            k_candidates: 3,
            pw_chunk: 1,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            temp_moves: 0,
            seed: 9,
            ..Default::default()
        };
        let r =
            search::<Count, _>(&Mock, CountState { n: 5, side: 1 }, &params, 0).expect("search");
        let total: u32 = r.candidates.iter().map(|(_, n)| *n).sum();
        assert!(
            total <= params.simulations,
            "root visits {total} exceed the simulation budget"
        );
        assert!(total > 0, "no visits recorded at all");
        assert_eq!(
            r.candidates[r.chosen].0, 1,
            "should leave a multiple of four"
        );
    }

    /// A candidate a widen adds to an interior node is visited by the very
    /// simulation that asked for it, whatever PUCT would have said. At
    /// `c_puct = 0` PUCT alone never leaves the first child it tries, so
    /// without this every interior node ends with one visited candidate.
    #[test]
    fn a_widened_candidate_is_visited_by_the_widening_simulation() {
        /// One distinct candidate per call, cycling through the legal moves.
        struct Cycle {
            cursor: std::sync::atomic::AtomicUsize,
        }
        impl SampledModel<Count> for Cycle {
            fn expand(&self, s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
                let moves: Vec<u32> = (1..=3u32).filter(|m| *m <= s.n).collect();
                let start = self
                    .cursor
                    .fetch_add(k, std::sync::atomic::Ordering::Relaxed);
                Ok((
                    0.0,
                    (0..k)
                        .map(|i| (moves[(start + i) % moves.len()], 0.0, false))
                        .collect(),
                ))
            }
        }
        let params = SampledParams {
            simulations: 200,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 1,
            c_puct: 0.0,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            forced_visits: 0,
            temp_moves: 0,
            seed: 6,
            ..Default::default()
        };
        let model = Cycle {
            cursor: std::sync::atomic::AtomicUsize::new(0),
        };
        let keys = |s: &Search<Count>, node: usize| -> Vec<u128> {
            let st = &s.tree[node].state;
            s.tree[node]
                .entries
                .iter()
                .map(|e| Count::state_key(&Count::apply(st, &e.mv)))
                .collect()
        };
        let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        let mut checked = 0;
        loop {
            match search.step() {
                Step::NeedExpansion(st) => {
                    let (v, c) = model.expand(&st, expansion_k(&params)).unwrap();
                    search.provide(v, c);
                }
                Step::NeedWiden { state, fill } => {
                    let (node, _) = search.pending_widen.expect("a widen names its node");
                    let before = keys(&search, node);
                    search.provide_widen(widen(&model, &state, fill, chunk_k(&params)));
                    let new: Vec<u128> = keys(&search, node)
                        .into_iter()
                        .filter(|k| !before.contains(k))
                        .collect();
                    let leads_to_terminal = search.tree[node].entries.iter().any(|e| {
                        let next = Count::apply(&search.tree[node].state, &e.mv);
                        new.contains(&Count::state_key(&next))
                            && Count::terminal_value(&next).is_some()
                    });
                    if node == 0 || new.is_empty() || leads_to_terminal {
                        continue;
                    }
                    // The simulation's next request is the new candidate's
                    // expansion: it descended straight into it.
                    match search.step() {
                        Step::NeedExpansion(st) => {
                            assert!(
                                new.contains(&Count::state_key(&st)),
                                "node {node}: the widening simulation went somewhere else"
                            );
                            checked += 1;
                            let (v, c) = model.expand(&st, expansion_k(&params)).unwrap();
                            search.provide(v, c);
                        }
                        _ => panic!(
                            "node {node}: the widening simulation did not visit its new candidate"
                        ),
                    }
                }
                Step::NeedResample { .. } => unreachable!("resampling is off"),
                Step::Done => break,
            }
        }
        assert!(checked > 10, "only {checked} interior widens checked");
        for (i, node) in search.tree.iter().enumerate().skip(1) {
            for e in &node.entries {
                assert!(
                    e.n >= 1,
                    "node {i}: a candidate the search widened to was never visited"
                );
            }
        }
    }

    /// A forced visit is tallied when the visit is backed up, not when the edge
    /// is selected. A descent used to be abandoned for a widen and never backed
    /// up, so a tally taken at selection charged the quota for visits that
    /// never happened -- the analysis view showed 30 forced against 10 real
    /// visits. Descents now pause instead, but the tally stays at backup.
    #[test]
    fn forced_visits_are_counted_only_for_visits_that_happen() {
        /// One candidate per call, like the real sampler at `pw_chunk = 1`, so
        /// interior nodes widen again and again, each widen pausing the descent
        /// that asked for it. Flagged explore, so every root candidate
        /// holds a forced quota. (`Mock` hands over a whole node at once, which
        /// widens each node only once and hides the bug.)
        struct OneAtATime {
            cursor: std::sync::atomic::AtomicUsize,
        }
        impl SampledModel<Count> for OneAtATime {
            fn expand(&self, s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
                let moves: Vec<u32> = (1..=3u32).filter(|m| *m <= s.n).collect();
                let start = self
                    .cursor
                    .fetch_add(k, std::sync::atomic::Ordering::Relaxed);
                Ok((
                    0.0,
                    (0..k)
                        .map(|i| (moves[(start + i) % moves.len()], 0.0, true))
                        .collect(),
                ))
            }
        }
        let params = SampledParams {
            simulations: 300,
            k_candidates: 3,
            pw_chunk: 1,
            forced_visits: 30,
            temp_moves: 0,
            seed: 5,
            ..Default::default()
        };
        let model = OneAtATime {
            cursor: std::sync::atomic::AtomicUsize::new(0),
        };

        // Step by step, because final totals hide it: PUCT's own visits to
        // a candidate cover a few phantom forced ones. Between two `step`
        // calls each forced tally must arrive with its visit, so no root
        // entry's forced count may rise by more than its visit count. An
        // abandoned forced pick used to raise the first without the second.
        let mut search = Search::<Count>::new(CountState { n: 21, side: 1 }, &params, 0);
        let mut widens = 0;
        loop {
            let before: Vec<(u32, u32)> = search.tree[0]
                .entries
                .iter()
                .map(|e| (e.n, e.forced_n))
                .collect();
            let step = search.step();
            for (i, &(n0, f0)) in before.iter().enumerate() {
                let e = &search.tree[0].entries[i];
                assert!(
                    e.forced_n - f0 <= e.n - n0,
                    "root entry {i} gained {} forced but only {} visits",
                    e.forced_n - f0,
                    e.n - n0
                );
            }
            match step {
                Step::NeedExpansion(st) => {
                    let (v, c) = model.expand(&st, expansion_k(&params)).unwrap();
                    search.provide(v, c);
                }
                Step::NeedWiden { state, fill } => {
                    widens += 1;
                    search.provide_widen(widen(&model, &state, fill, chunk_k(&params)));
                }
                Step::NeedResample { .. } => unreachable!("resampling is off"),
                Step::Done => break,
            }
        }
        assert!(
            widens > 50,
            "only {widens} widens -- the test is not exercising them"
        );

        let tree = search.export_tree();
        for (i, node) in tree.iter().enumerate() {
            for (j, e) in node.edges.iter().enumerate() {
                assert!(
                    e.forced <= e.visits,
                    "node {i} edge {j} has {} forced of {} visits",
                    e.forced,
                    e.visits
                );
                assert!(
                    e.forced <= params.forced_visits,
                    "node {i} edge {j} has {} forced, over the {} quota",
                    e.forced,
                    params.forced_visits
                );
            }
        }
    }

    /// `draws` accounts for every proposal: the draw that created a candidate
    /// and each later one that reached the same position and merged into it.
    #[test]
    fn draws_count_every_proposal_merged_into_a_candidate() {
        // `Duplicating` offers [1, 1, 1, 2]: three draws of `1`, one of `2`.
        let params = SampledParams {
            simulations: 1,
            k_candidates: 4,
            k_draws: 4,
            pw_chunk: 0,
            forced_visits: 0,
            temp_moves: 0,
            seed: 3,
            ..Default::default()
        };
        let (_, tree) =
            search_analyzed::<Count, _>(&Duplicating, CountState { n: 21, side: 1 }, &params, 0)
                .expect("search");
        let mut draws: Vec<(u32, u32)> = tree[0].edges.iter().map(|e| (e.mv, e.draws)).collect();
        draws.sort();
        assert_eq!(draws, vec![(1, 3), (2, 1)]);
    }

    /// Chunking off must still leave nodes with candidates. With `pw_chunk = 0`
    /// there is no widen path, so a value-only expansion would leave every node
    /// permanently empty and the search would score everything on value alone.
    #[test]
    fn unchunked_expansion_still_yields_candidates() {
        let params = SampledParams {
            simulations: 30,
            k_candidates: 3,
            k_draws: 3,
            pw_chunk: 0,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            temp_moves: 0,
            seed: 2,
            ..Default::default()
        };
        assert_eq!(
            expansion_k(&params),
            3,
            "unchunked expansion must ask for all its draws"
        );
        let r =
            search::<Count, _>(&Mock, CountState { n: 21, side: 1 }, &params, 0).expect("search");
        assert!(
            !r.candidates.is_empty(),
            "root held no candidates with chunking off"
        );
        assert!(
            r.candidates.iter().any(|(_, n)| *n > 0),
            "no candidate was ever visited with chunking off"
        );
    }

    /// Chunked generation must reach the same root candidate set as one wide
    /// call — it changes *how many calls* a node needs, never what it ends up
    /// holding. The root is the case that matters: its set is the training
    /// target, so a chunk that quietly truncated it would corrupt training.
    #[test]
    fn chunked_generation_reaches_the_same_root_set() {
        /// A stochastic sampler: successive calls return *different* draws from
        /// the same pool, which is what makes chunked generation able to
        /// accumulate a full set. A model that returned the same candidates
        /// every call would saturate instead — the `widen_exhausted` path.
        struct Pool {
            cursor: std::sync::atomic::AtomicUsize,
        }
        impl SampledModel<Count> for Pool {
            fn expand(&self, s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
                let start = self
                    .cursor
                    .fetch_add(k.max(1), std::sync::atomic::Ordering::Relaxed);
                let moves: Vec<u32> = (1..=3u32).filter(|m| *m <= s.n).collect();
                let cands: Vec<(u32, f32, bool)> = (0..k.max(1))
                    .map(|i| moves[(start + i) % moves.len()])
                    .map(|m| (m, -(m as f32), false))
                    .collect();
                Ok((0.0, cands))
            }
        }
        let base = SampledParams {
            simulations: 40,
            k_candidates: 3,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            temp_moves: 0,
            seed: 11,
            ..Default::default()
        };
        let pool = Pool {
            cursor: std::sync::atomic::AtomicUsize::new(0),
        };
        let whole =
            search::<Count, _>(&pool, CountState { n: 21, side: 1 }, &base, 0).expect("unchunked");

        let chunked_params = SampledParams {
            pw_chunk: 1,
            ..base
        };
        let chunked = search::<Count, _>(&pool, CountState { n: 21, side: 1 }, &chunked_params, 0)
            .expect("chunked");

        let mut a: Vec<u32> = whole.candidates.iter().map(|(m, _)| *m).collect();
        let mut b: Vec<u32> = chunked.candidates.iter().map(|(m, _)| *m).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(
            a, b,
            "chunked generation must reach the same root candidates"
        );
        let sum: f32 = chunked.priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "priors must stay normalised, got {sum}"
        );
    }

    /// An explore candidate must be *tried* even when its prior is negligible.
    ///
    /// KataGo's floor is `sqrt(k·P·N)` and only applies to candidates already
    /// visited once, so it can neither buy a first visit nor help a candidate the
    /// energy head scores near zero — exactly the ones forced exploration exists
    /// to cover. Without a prior-independent floor, support collapse is
    /// self-reinforcing: the sampler stops proposing a region, the energy head
    /// never sees it, and its prior goes to zero.
    #[test]
    fn explore_candidates_are_visited_despite_a_negligible_prior() {
        let params = SampledParams {
            simulations: 60,
            k_forced: 0.0, // isolate the explore floor from KataGo's
            min_explore_visits: 3.0,
            temp_moves: 0,
            seed: 5,
            ..Default::default()
        };
        let mut search = Search::<Count>::new(CountState { n: 12, side: 1 }, &params, 0);
        loop {
            match search.step() {
                Step::NeedExpansion(s) => {
                    // Move 3 is the explore candidate and is given a prior a
                    // thousand times smaller than its rivals.
                    let cands: Vec<(u32, f32, bool)> = (1..=3u32)
                        .filter(|m| *m <= s.n)
                        .map(|m| (m, if m == 3 { 0.001 } else { 1.0 }, m == 3))
                        .collect();
                    search.provide(0.0, cands);
                }
                Step::NeedWiden { state: s, .. } => {
                    let cands: Vec<(u32, f32, bool)> = (1..=3u32)
                        .filter(|m| *m <= s.n)
                        .map(|m| (m, if m == 3 { 0.001 } else { 1.0 }, m == 3))
                        .collect();
                    search.provide_widen(cands);
                }
                Step::NeedResample { .. } => unreachable!("resampling is off"),
                Step::Done => break,
            }
        }
        let r = search.finish();
        let (i, &(_, visits)) = r
            .candidates
            .iter()
            .enumerate()
            .find(|(_, (m, _))| *m == 3)
            .expect("the explore candidate must be in the tree");
        assert!(r.explore[i], "provenance must survive into the result");
        assert!(
            visits as f32 >= params.min_explore_visits,
            "explore candidate got {visits} visits, below the guaranteed {}",
            params.min_explore_visits
        );
    }

    /// ...and those guaranteed visits must not reach the training target.
    ///
    /// This is the half that makes forced exploration safe. If the search grants
    /// an explore candidate `m` visits and the target counts them as evidence
    /// on their own, the policy is taught that injected noise was search's
    /// judgement — strictly worse than not exploring at all.
    #[test]
    fn guaranteed_explore_visits_are_pruned_from_the_target() {
        let params = SampledParams {
            k_forced: 0.0,
            min_explore_visits: 4.0,
            ..Default::default()
        };
        // The explore candidate earned exactly its floor and nothing more: all
        // four of its visits were demanded, none chosen.
        let r = SampledSearchResult::<Count> {
            forced: vec![1, 1, 4],
            stats: SearchStats::default(),
            candidates: vec![(1, 40), (2, 20), (3, 4)],
            priors: vec![0.5, 0.4, 0.1],
            explore: vec![false, false, true],
            chosen: 0,
            value: 0.0,
            nodes: 64,
        };
        let t = r.policy_target_with(&params);
        assert_eq!(
            t[2], 0.0,
            "a candidate that only ever got its floor must not survive"
        );
        assert!(
            (t.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "target must stay normalised"
        );

        // Earning well beyond the floor, it survives on merit: four visits were
        // demanded, the other twenty-one were chosen.
        let r2 = SampledSearchResult::<Count> {
            forced: vec![1, 1, 4],
            stats: SearchStats::default(),
            candidates: vec![(1, 40), (2, 20), (3, 25)],
            priors: vec![0.5, 0.4, 0.1],
            explore: vec![false, false, true],
            chosen: 0,
            value: 0.0,
            nodes: 85,
        };
        let t2 = r2.policy_target_with(&params);
        assert!(
            t2[2] > 0.0,
            "organic visits beyond the floor must survive pruning"
        );
    }

    #[test]
    fn lockstep_matches_running_the_searches_alone() {
        // The batching must be a scheduling change and nothing more: each game
        // holds one outstanding expansion, so its tree is unchanged.
        let roots: Vec<CountState> = (3..=9).map(|n| CountState { n, side: 1 }).collect();
        let plies: Vec<u32> = (0..roots.len() as u32).collect();
        let p = params(200);
        let batched = search_lockstep::<Count, _>(&Mock, roots.clone(), &p, &plies).unwrap();
        assert_eq!(batched.len(), roots.len());
        for (i, root) in roots.into_iter().enumerate() {
            let alone = search::<Count, _>(&Mock, root, &p, plies[i]).unwrap();
            assert_eq!(
                batched[i].candidates, alone.candidates,
                "game {i}: visit counts differ under lockstep"
            );
            assert_eq!(
                batched[i].chosen, alone.chosen,
                "game {i}: chose differently"
            );
            assert!(
                (batched[i].value - alone.value).abs() < 1e-9,
                "game {i}: value differs"
            );
        }
    }

    #[test]
    fn search_finds_the_winning_move() {
        // Leaving a multiple of 4 wins. From 5, that means taking 1. Kept
        // shallow on purpose: the mock evaluates every position at 0.0, so the
        // only signal is terminal backups, and the search has to actually
        // resolve the line rather than lean on a value head.
        let s = CountState { n: 5, side: 1 };
        let r = search::<Count, _>(&Mock, s, &params(400), 99).unwrap();
        assert_eq!(
            r.candidates[r.chosen].0, 1,
            "should leave a multiple of four"
        );
        assert!(
            r.value > 0.0,
            "a won position should evaluate positive, got {}",
            r.value
        );
    }

    #[test]
    fn an_immediate_win_is_taken() {
        for n in 1..=3u32 {
            let s = CountState { n, side: 1 };
            let r = search::<Count, _>(&Mock, s, &params(100), 99).unwrap();
            assert_eq!(r.candidates[r.chosen].0, n, "should just take the last {n}");
            assert!(
                r.value > 0.5,
                "a won position should evaluate positive, got {}",
                r.value
            );
        }
    }

    #[test]
    fn a_lost_position_evaluates_negative() {
        // 4 is a multiple of 4: every move leaves the opponent a winning take.
        let r = search::<Count, _>(&Mock, CountState { n: 4, side: 1 }, &params(400), 99).unwrap();
        assert!(
            r.value < 0.0,
            "a lost position should evaluate negative, got {}",
            r.value
        );
    }

    #[test]
    fn visits_are_conserved_across_the_root() {
        let sims = 300;
        let r = search::<Count, _>(&Mock, Count::initial_state(), &params(sims), 99).unwrap();
        let total: u32 = r.candidates.iter().map(|(_, n)| *n).sum();
        // Every simulation after the root's own expansion descends through
        // exactly one root child.
        assert_eq!(total, sims - 1);
    }

    #[test]
    fn duplicate_candidates_are_merged_not_fragmented() {
        let r = search::<Count, _>(&Duplicating, Count::initial_state(), &params(50), 99).unwrap();
        assert_eq!(
            r.candidates.len(),
            2,
            "three copies of `1` should merge into one child"
        );
        let moves: Vec<u32> = r.candidates.iter().map(|(m, _)| *m).collect();
        assert!(moves.contains(&1) && moves.contains(&2));
    }

    #[test]
    fn a_model_that_proposes_nothing_does_not_hang() {
        let r = search::<Count, _>(&Empty, Count::initial_state(), &params(20), 99).unwrap();
        assert!(r.candidates.is_empty());
        assert_eq!(r.chosen, 0);
    }

    #[test]
    fn policy_target_subtracts_forced_visits() {
        // Forced visits are bought, not chosen, so they are removed from every
        // candidate's count -- the winner's included. A candidate left with
        // nothing it earned targets exactly 0, which is a real observation
        // ("we examined this and search declined to return") rather than
        // missing data.
        let r = SampledSearchResult::<Count> {
            forced: vec![1, 3, 4],
            stats: SearchStats::default(),
            candidates: vec![(1, 93), (2, 3), (3, 4)],
            priors: vec![0.9, 0.05, 0.05],
            explore: vec![false; 8],
            chosen: 0,
            value: 0.0,
            nodes: 0,
        };
        let t = r.policy_target(2.0);
        assert!(
            (t.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "target must be normalised"
        );
        assert_eq!(t[1], 0.0, "forced-only visits earn no mass: {t:?}");
        assert_eq!(t[2], 0.0, "forced-only visits earn no mass: {t:?}");
        assert!((t[0] - 1.0).abs() < 1e-6, "92 of 92 chosen visits: {t:?}");
    }

    #[test]
    fn the_leader_is_not_exempt_from_subtraction() {
        // The old rule never subtracted from the most-visited child. It does
        // now, so two candidates with equal raw counts but different quotas do
        // not tie.
        let r = SampledSearchResult::<Count> {
            forced: vec![30, 0],
            stats: SearchStats::default(),
            candidates: vec![(1, 60), (2, 60)],
            priors: vec![0.5, 0.5],
            explore: vec![true, false],
            chosen: 1,
            value: 0.0,
            nodes: 0,
        };
        let t = r.policy_target(2.0);
        assert!((t[0] - 30.0 / 90.0).abs() < 1e-5, "{t:?}");
        assert!((t[1] - 60.0 / 90.0).abs() < 1e-5, "{t:?}");
    }

    #[test]
    fn a_candidate_counts_only_what_it_earned() {
        // Mass is now proportional to `n - forced` rather than gated on it, so
        // a candidate contributes exactly the visits search chose to give it.
        let target = |visits: u32, forced: u32, k: usize| -> Vec<f32> {
            let r = SampledSearchResult::<Count> {
                forced: vec![forced; k],
                stats: SearchStats::default(),
                candidates: (0..k).map(|i| (i as u32, visits)).collect(),
                priors: vec![1.0 / k as f32; k],
                explore: vec![false; k],
                chosen: 0,
                value: 0.0,
                nodes: 0,
            };
            r.policy_target(2.0)
        };
        // Every visit forced: nothing earned anything, so the target falls back
        // to a single child rather than returning all zeros, which would be a
        // zero-gradient row.
        let t = target(4, 4, 8);
        assert_eq!(t.iter().filter(|&&p| p > 0.0).count(), 1, "{t:?}");
        // One earned visit each: equal counts, so a uniform target.
        let t = target(5, 4, 8);
        assert!(t.iter().all(|&p| (p - 0.125).abs() < 1e-6), "{t:?}");
        // Five earned each -- still uniform, and independent of the prior.
        let t = target(6, 1, 8);
        assert!(t.iter().all(|&p| (p - 0.125).abs() < 1e-6), "{t:?}");
    }

    /// The exported tree must be walkable: root first, child indices resolving,
    /// parents pointing back, priors normalised per node.
    #[test]
    fn exported_tree_is_internally_consistent() {
        let params = SampledParams {
            simulations: 60,
            ..Default::default()
        };
        let (_, tree) =
            search_analyzed::<Count, _>(&Mock, CountState { n: 21, side: 0 }, &params, 0)
                .expect("search");

        assert!(
            !tree.is_empty(),
            "a completed search must export at least the root"
        );
        assert_eq!(tree[0].parent, None, "node 0 must be the root");

        for (i, node) in tree.iter().enumerate() {
            if i > 0 {
                let p = node.parent.expect("every non-root node has a parent");
                assert!(
                    tree[p].edges.iter().any(|e| e.child == Some(i)),
                    "node {i} claims parent {p}, which does not point back"
                );
            }
            let sum: f32 = node.edges.iter().map(|e| e.prior).sum();
            if !node.edges.is_empty() {
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "node {i} priors sum to {sum}, not 1"
                );
            }
            for e in &node.edges {
                if let Some(c) = e.child {
                    assert!(c < tree.len(), "node {i} has a child index past the end");
                }
                // An unvisited edge has no value to report, and a visited one
                // must: the alternative is a caller rendering 0.0 as an opinion.
                assert_eq!(
                    e.value.is_some(),
                    e.visits > 0,
                    "node {i} value/visit disagree"
                );
                assert!(e.forced <= e.visits, "node {i} forced visits exceed total");
            }
        }
    }

    /// Priors must follow the energies, not the order candidates arrived in.
    ///
    /// Chunked generation hands the model one candidate per call, so a model
    /// that normalises within its own call returns `exp(f - f) = 1` every time
    /// and the head's opinion never reaches the search. The symptom was priors
    /// that halved — 0.5, 0.25, 0.125 — identical for *any* energies, because
    /// each newcomer took half the mass and pushed the rest down.
    #[test]
    fn widened_priors_follow_energy_not_arrival_order() {
        /// Emits a decisive energy gap: the first candidate should end up with
        /// essentially all the prior mass.
        struct Skewed {
            cursor: std::sync::atomic::AtomicUsize,
        }
        impl SampledModel<Count> for Skewed {
            fn expand(&self, s: &CountState, k: usize) -> anyhow::Result<Expansion<Count>> {
                let start = self
                    .cursor
                    .fetch_add(k.max(1), std::sync::atomic::Ordering::Relaxed);
                let moves: Vec<u32> = (1..=3u32).filter(|m| *m <= s.n).collect();
                Ok((
                    0.0,
                    (0..k.max(1))
                        .map(|i| {
                            let idx = start + i;
                            (moves[idx % moves.len()], -10.0 * idx as f32, false)
                        })
                        .collect(),
                ))
            }
        }
        let params = SampledParams {
            simulations: 60,
            k_candidates: 3,
            pw_chunk: 1,
            k_forced: 0.0,
            min_explore_visits: 0.0,
            temp_moves: 0,
            seed: 5,
            ..Default::default()
        };
        let model = Skewed {
            cursor: std::sync::atomic::AtomicUsize::new(0),
        };
        let (_, tree) =
            search_analyzed::<Count, _>(&model, CountState { n: 21, side: 1 }, &params, 0)
                .expect("search");
        let priors: Vec<f32> = tree[0].edges.iter().map(|e| e.prior).collect();
        assert!(
            priors.len() > 1,
            "need several candidates to compare: {priors:?}"
        );
        assert!(
            priors[0] > 0.98,
            "a 10-nat lead should take nearly all the mass, got {priors:?}"
        );
        // The old bug produced exactly this, whatever the energies were.
        for w in priors.windows(2) {
            assert!(
                (w[1] / w[0] - 0.5).abs() > 1e-3,
                "priors are halving, which means arrival order not energy: {priors:?}"
            );
        }
    }

    #[test]
    fn policy_target_never_returns_all_zeros() {
        let r = SampledSearchResult::<Count> {
            forced: vec![],
            stats: SearchStats::default(),
            candidates: vec![(1, 1), (2, 1)],
            priors: vec![0.5, 0.5],
            explore: vec![false; 8],
            chosen: 0,
            value: 0.0,
            nodes: 0,
        };
        let t = r.policy_target(2.0);
        assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{t:?}");
    }

    #[test]
    fn temperature_sampling_explores_the_opening() {
        // With temp_moves high, early plies should not always pick the argmax.
        let p = SampledParams {
            simulations: 60,
            temp_moves: 100,
            seed: 7,
            ..Default::default()
        };
        let picks: std::collections::HashSet<u32> = (0..30)
            .map(|ply| {
                let r = search::<Count, _>(&Mock, Count::initial_state(), &p, ply).unwrap();
                r.candidates[r.chosen].0
            })
            .collect();
        assert!(
            picks.len() > 1,
            "temperature should diversify the opening, got {picks:?}"
        );
    }
}

#[cfg(test)]
mod forced_quota_tests {
    use super::*;

    /// Minimal game: the quota logic only reads `explore`, never the moves.
    struct G0;
    impl SampledGame for G0 {
        type State = ();
        type Move = u32;
        const PLANES: usize = 1;
        fn initial_state() -> Self::State {}
        fn to_move(_: &Self::State) -> u8 {
            1
        }
        fn apply(_: &Self::State, _: &Self::Move) -> Self::State {}
        fn terminal_value(_: &Self::State) -> Option<f32> {
            None
        }
        fn encode_planes(_: &Self::State, _: &mut [f32]) {}
        fn state_key(_: &Self::State) -> u128 {
            0
        }
    }

    fn entry(explore: bool) -> Entry<G0> {
        Entry {
            mv: 1,
            energy: 0.0,
            p: 0.25,
            n: 0,
            w: 0.0,
            child: -1,
            explore,
            forced_n: 0,
            forced_quota: 0,
            draws: 1,
            manual: false,
            eval: None,
            resampled: false,
        }
    }

    fn quotas(n_explore: usize, n_plain: usize, params: &SampledParams) -> Vec<u32> {
        let mut es: Vec<Entry<G0>> = (0..n_explore)
            .map(|_| entry(true))
            .chain((0..n_plain).map(|_| entry(false)))
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        assign_forced_quotas(&mut es, params, &mut rng);
        es.iter().map(|e| e.forced_quota).collect()
    }

    #[test]
    fn every_explore_candidate_is_covered() {
        let p = SampledParams {
            simulations: 4000,
            forced_visits: 30,
            forced_extra: 4,
            ..Default::default()
        };
        let q = quotas(5, 10, &p);
        assert!(
            q[..5].iter().all(|&x| x == 30),
            "explore candidates must all be covered: {q:?}"
        );
    }

    #[test]
    fn exactly_forced_extra_plain_candidates_are_drawn() {
        let p = SampledParams {
            simulations: 4000,
            forced_visits: 30,
            forced_extra: 4,
            ..Default::default()
        };
        let q = quotas(5, 10, &p);
        assert_eq!(
            q[5..].iter().filter(|&&x| x > 0).count(),
            4,
            "expected 4 extras: {q:?}"
        );
    }

    #[test]
    fn quotas_are_capped_at_half_the_simulation_budget() {
        // 4.3 explore candidates per node plus 4 extras at 30 each is ~250
        // visits, which is 62% of a 400-simulation search. The cap is what
        // stops forced exploration from crowding out the search it informs.
        let p = SampledParams {
            simulations: 400,
            forced_visits: 30,
            forced_extra: 4,
            ..Default::default()
        };
        let q = quotas(5, 10, &p);
        let total: u32 = q.iter().sum();
        assert!(total <= 200, "forced total {total} exceeds half of 400");
    }

    #[test]
    fn explore_candidates_keep_priority_when_the_budget_binds() {
        // Under pressure the random extras are what gets dropped, never the
        // injected candidates -- those are the ones with no value estimate.
        let p = SampledParams {
            simulations: 120,
            forced_visits: 30,
            forced_extra: 4,
            ..Default::default()
        };
        let q = quotas(2, 10, &p);
        assert!(
            q[..2].iter().all(|&x| x == 30),
            "explore dropped before extras: {q:?}"
        );
        assert_eq!(
            q.iter().sum::<u32>(),
            60,
            "budget 60 should buy exactly the two explore quotas"
        );
    }

    #[test]
    fn a_candidate_with_only_forced_visits_targets_zero() {
        // The specified behaviour: forced visits are not evidence the search
        // preferred a move, so a candidate that attracted none of its own is a
        // real zero in the target rather than missing data.
        let r = SampledSearchResult::<G0> {
            candidates: vec![(1, 40), (2, 30), (3, 10)],
            priors: vec![0.4, 0.3, 0.3],
            explore: vec![false, true, false],
            forced: vec![0, 30, 0],
            chosen: 0,
            value: 0.0,
            nodes: 0,
            stats: SearchStats::default(),
        };
        let t = r.policy_target_with(&SampledParams::default());
        assert_eq!(
            t[1], 0.0,
            "a purely-forced candidate must target 0, got {t:?}"
        );
        assert!((t[0] - 0.8).abs() < 1e-6, "40 of 50 chosen visits: {t:?}");
        assert!((t[2] - 0.2).abs() < 1e-6, "10 of 50 chosen visits: {t:?}");
    }

    #[test]
    fn forced_candidates_stay_in_the_target_when_they_earn_visits() {
        // Still part of the sample set -- the softmax normalises over every
        // candidate the search saw, forced ones included.
        let r = SampledSearchResult::<G0> {
            candidates: vec![(1, 40), (2, 40)],
            priors: vec![0.5, 0.5],
            explore: vec![false, true],
            forced: vec![0, 30],
            chosen: 0,
            value: 0.0,
            nodes: 0,
            stats: SearchStats::default(),
        };
        let t = r.policy_target_with(&SampledParams::default());
        assert!((t[0] - 0.8).abs() < 1e-6, "{t:?}");
        assert!((t[1] - 0.2).abs() < 1e-6, "10 chosen of 50: {t:?}");
    }
}
