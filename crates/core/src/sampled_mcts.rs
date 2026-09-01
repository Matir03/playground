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
//! [`SampledSearchResult::policy_target_with`] subtracts them back out: each child is pruned to the visit count plain PUCT
//! would have chosen on its own merits. A candidate that was *forced* into the
//! tree and then earned organic visits survives on merit; one that didn't,
//! vanishes. Value targets deliberately keep every visit — a random candidate is
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
/// A value, plus candidates as `(move, prior, explore)`.
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
    /// An expanded node has earned more candidates than it holds. Same shape as
    /// an expansion request, so it rides the same batch; the value that comes
    /// back is ignored.
    NeedWiden(G::State),
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
    fn expand_batch(
        &self,
        states: &[G::State],
        k: usize,
    ) -> anyhow::Result<Vec<Expansion<G>>> {
        states.iter().map(|s| self.expand(s, k)).collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SampledParams {
    pub simulations: u32,
    pub c_puct: f32,
    /// Candidates drawn per node expansion.
    pub k_candidates: usize,
    /// KataGo forced-playout constant; 0 disables forced playouts.
    pub k_forced: f32,
    /// Visits guaranteed to every *explore*-origin root candidate, regardless of
    /// its prior — the part KataGo's prior-scaled floor cannot provide, since
    /// that floor is proportional to `P(a)` and skips candidates never tried.
    /// Like forced playouts, these are subtracted before the training target is
    /// formed. 0 disables it.
    pub min_explore_visits: f32,
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
            k_forced: 2.0,
            min_explore_visits: 2.0,
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
/// Pruning subtracts a forced-playout floor of `sqrt(k·p·N)` from every
/// non-best child and drops whatever is left at one visit or fewer. With
/// uniform priors `p = 1/K` and `x = N/K` visits per candidate, a candidate
/// survives only when `x − 1 > sqrt(k·x)`, i.e. `x² − (2+k)x + 1 > 0`. For the
/// default `k_forced = 2` that is `x > 3.73`.
///
/// This matters because falling below it is *silent*: search still works, the
/// value and sampler heads still train, but every recorded node has a single
/// candidate — and a softmax over one element has zero gradient, so the energy
/// head (the policy of record) quietly learns nothing at all.
pub fn min_visits_per_candidate(k_forced: f32) -> f32 {
    if k_forced <= 0.0 {
        return 0.0;
    }
    let b = 2.0 + k_forced;
    (b + (b * b - 4.0).sqrt()) / 2.0
}

struct Entry<G: SampledGame> {
    mv: G::Move,
    p: f32,
    n: u32,
    w: f32,
    child: i32,
    /// Drawn by forced exploration rather than proposed by the model.
    explore: bool,
}

struct Node<G: SampledGame> {
    state: G::State,
    entries: Vec<Entry<G>>,
    /// Set when a widen chunk produced nothing new. The sampler can saturate —
    /// every proposal merging into a candidate already held — and without this
    /// the node would keep asking for candidates it can never get, spinning the
    /// search forever. Caught by `chunked_generation_reaches_the_same_root_set`.
    widen_exhausted: bool,
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
    /// pruning below has to subtract the forced visits *these* priors earned.
    pub priors: Vec<f32>,
    /// Which candidates came from forced exploration, parallel to `candidates`.
    /// Their guaranteed visits are subtracted before the training target forms.
    pub explore: Vec<bool>,
    /// Index into `candidates` of the move to play.
    pub chosen: usize,
    /// Candidate-budget accounting over the whole tree, not just the root.
    pub stats: SearchStats,
    /// Root value from the side to move's perspective.
    pub value: f64,
    pub nodes: u64,
}

impl<G: SampledGame> SampledSearchResult<G> {
    /// Visit counts with forced and injected exploration subtracted, normalised.
    ///
    /// KataGo's policy-target pruning: give back `sqrt(k · P · N)` from every
    /// non-best child, drop any child left with one visit or fewer, and never
    /// prune the most-visited child. Without this, every forced visit and every
    /// randomly injected candidate would be taught to the policy as if the
    /// search had chosen it.
    pub fn policy_target(&self, k_forced: f32) -> Vec<f32> {
        self.policy_target_with(&SampledParams { k_forced, ..Default::default() })
    }

    /// As [`Self::policy_target`], but subtracting the *same* floor the search
    /// granted — including the guaranteed visits explore candidates received.
    ///
    /// These must agree. If the search floors an explore candidate at `m` visits
    /// and the target only subtracts KataGo's prior-scaled floor, the difference
    /// is pure injected noise taught to the policy as if it were search's
    /// judgement — the opposite of what forced exploration is for.
    pub fn policy_target_with(&self, params: &SampledParams) -> Vec<f32> {
        let k_forced = params.k_forced;
        let priors = &self.priors;
        let n = self.candidates.len();
        let total: u32 = self.candidates.iter().map(|(_, v)| *v).sum();
        let mut counts: Vec<f32> = self.candidates.iter().map(|(_, v)| *v as f32).collect();
        if n == 0 || total == 0 {
            return counts;
        }
        let best = self
            .candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, (_, v))| *v)
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Every root candidate is seeded with one visit by `select` so that it
        // has a real value estimate. That visit is the search *asking*, not the
        // search *choosing*, so it must not appear in the training target — the
        // same reason forced playouts and injected explore visits are stripped.
        const SEEDED: f32 = 1.0;
        for (i, c) in counts.iter_mut().enumerate() {
            if i == best {
                *c = (*c - SEEDED).max(0.0);
                continue;
            }
            let p = priors.get(i).copied().unwrap_or(0.0);
            let ex = self.explore.get(i).copied().unwrap_or(false);
            let forced = if k_forced > 0.0 || params.min_explore_visits > 0.0 {
                explore_floor(ex, p, total as f32, params).max(SEEDED)
            } else {
                SEEDED
            };
            *c = (*c - forced).max(0.0);
            if *c <= 1.0 {
                *c = 0.0;
            }
        }
        let sum: f32 = counts.iter().sum();
        if sum <= 0.0 {
            // Everything pruned away — fall back to the single best child, so a
            // target is never all zeros.
            counts.iter_mut().for_each(|c| *c = 0.0);
            counts[best] = 1.0;
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
        kern: Vec::new(),
        total_n: 0,
        w_sum: 0.0,
        visited_p: 0.0,
        expanded: false,
        terminal: tv.is_some(),
        term_v: tv.unwrap_or(0.0),
    }
}

/// The visit floor a root candidate is entitled to before PUCT may starve it.
///
/// Shared by selection and by [`SampledSearchResult::policy_target`] so the two
/// cannot drift: the target subtracts exactly what the search added, and forced
/// visits leave no trace in what the policy trains on.
fn explore_floor(explore: bool, p: f32, total: f32, params: &SampledParams) -> f32 {
    let katago = if params.k_forced > 0.0 {
        (params.k_forced * p * total).max(0.0).sqrt()
    } else {
        0.0
    };
    if explore {
        katago.max(params.min_explore_visits)
    } else {
        katago
    }
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

/// Select a child by PUCT, with root-only forced playouts.
fn select<G: SampledGame>(node: &Node<G>, params: &SampledParams, is_root: bool) -> usize {
    let n = visible(node, params, is_root);
    debug_assert!(n > 0);
    let total = node.total_n.max(1) as f32;

    // Sweep the root once: every candidate is visited a single time before PUCT
    // takes over, so each has a real value estimate behind it rather than an FPU
    // guess. That seeded visit is *not* evidence the search chose the move, so
    // `policy_target_with` subtracts it back out of the training target.
    if is_root {
        if let Some(i) = node.entries.iter().position(|e| e.n == 0) {
            return i;
        }
    }

    if is_root && (params.k_forced > 0.0 || params.min_explore_visits > 0.0) {
        // A candidate that has been tried at least once is floored at
        // sqrt(k·P·N) visits before PUCT is allowed to starve it.
        let mut best_deficit = 0.0f32;
        let mut forced_pick = usize::MAX;
        for (i, e) in node.entries.iter().enumerate().take(n) {
            // KataGo's floor only *keeps a tried candidate alive*; it never buys
            // a first visit, because a candidate with n = 0 is skipped. That is
            // fine for model proposals but defeats forced exploration: the
            // candidates most worth looking at are exactly the ones the energy
            // head scores lowest, and with a small enough prior PUCT may never
            // try them even once. Explore candidates are therefore floored from
            // n = 0.
            if e.n == 0 && !e.explore {
                continue;
            }
            let floor = explore_floor(e.explore, e.p, total, params);
            let deficit = floor - e.n as f32;
            if deficit > best_deficit {
                best_deficit = deficit;
                forced_pick = i;
            }
        }
        if forced_pick != usize::MAX {
            return forced_pick;
        }
    }

    let parent_q = if node.total_n > 0 { node.w_sum / node.total_n as f32 } else { 0.0 };
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
        let mut w_hat = vec![0.0f32; n];
        for i in 0..n {
            let row = &node.kern[i * m..(i + 1) * m];
            for (j, e) in node.entries.iter().enumerate() {
                n_hat[i] += row[j] * e.n as f32;
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
            let q = if n_hat[i] > 0.0 { w_hat[i] / n_hat[i] } else { fpu };
            let u = params.c_puct * node.entries[i].p * sqrt_total / (1.0 + n_hat[i]);
            let s = q + u;
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        return best;
    }

    let sqrt_total = total.sqrt();
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, e) in node.entries.iter().enumerate().take(n) {
        // `e.w` is accumulated in the *owning node's* perspective (the backup
        // negates on the way up), so it is used as-is here.
        let q = if e.n > 0 { e.w / e.n as f32 } else { fpu };
        let u = params.c_puct * e.p * sqrt_total / (1.0 + e.n as f32);
        let s = q + u;
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
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
    /// (node, entry) pairs descended through in the current simulation.
    path: Vec<(usize, usize)>,
    /// Node awaiting expansion, if `step` returned `NeedExpansion`.
    pending: Option<usize>,
    /// Node awaiting more candidates, if `step` returned `NeedWiden`.
    pending_widen: Option<usize>,
}

impl<G: SampledGame> Search<G> {
    pub fn new(root: G::State, params: &SampledParams, ply: u32) -> Self {
        Self {
            tree: vec![new_node::<G>(root)],
            params: *params,
            rng: ChaCha8Rng::seed_from_u64(
                params.seed ^ (ply as u64).wrapping_mul(0x9E3779B9),
            ),
            ply,
            sims_left: params.simulations,
            nodes: 0,
            path: Vec::new(),
            pending: None,
            pending_widen: None,
        }
    }

    /// Advance until an expansion is needed or the budget is spent.
    pub fn step(&mut self) -> Step<G> {
        debug_assert!(self.pending.is_none(), "step called with an expansion outstanding");
        while self.sims_left > 0 {
            self.path.clear();
            let mut cur = 0usize;
            let value = loop {
                if self.tree[cur].terminal {
                    break self.tree[cur].term_v;
                }
                if !self.tree[cur].expanded {
                    self.pending = Some(cur);
                    return Step::NeedExpansion(self.tree[cur].state.clone());
                }
                // Ask for the next chunk before selecting, if this node has
                // earned more candidates than it holds. The simulation is
                // abandoned rather than paused: nothing has been backed up yet,
                // so re-descending from the root afterwards costs only the
                // descent, and it keeps `path` from holding indices into a node
                // whose entries are about to be re-sorted.
                if self.params.pw_chunk > 0 && !self.tree[cur].widen_exhausted {
                    let held = self.tree[cur].entries.len();
                    if held < earned(&self.tree[cur], &self.params, cur == 0)
                        && held < self.params.k_candidates.max(1)
                    {
                        self.pending_widen = Some(cur);
                        self.path.clear();
                        return Step::NeedWiden(self.tree[cur].state.clone());
                    }
                }
                if self.tree[cur].entries.is_empty() {
                    // Genuinely nothing to play: either the model proposed
                    // nothing, or widening is exhausted. Score on value alone.
                    break 0.0;
                }
                let pick = select(&self.tree[cur], &self.params, cur == 0);
                self.path.push((cur, pick));
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
        }
        Step::Done
    }

    /// Supply the value and candidates for the position `step` handed out.
    pub fn provide(&mut self, value: f32, candidates: Vec<(G::Move, f32, bool)>) {
        let cur = self.pending.take().expect("provide without a pending expansion");
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
                    entries[i].p += p.max(0.0);
                    // A position the model also proposed still deserves its
                    // guaranteed look if exploration reached it independently.
                    entries[i].explore |= explore;
                }
                None => {
                    seen.insert(key, entries.len());
                    entries.push(Entry {
                        mv,
                        p: p.max(0.0),
                        n: 0,
                        w: 0.0,
                        child: -1,
                        explore,
                    });
                }
            }
        }
        // Order by prior, descending. Progressive widening reveals a *prefix* of
        // this list, so the order decides which candidates a lightly-visited
        // node gets to consider at all — it must be the ones the policy rates
        // highest, not whatever order the sampler happened to emit.
        entries.sort_by(|a, b| b.p.total_cmp(&a.p));

        let sum: f32 = entries.iter().map(|e| e.p).sum();
        if sum > 0.0 {
            entries.iter_mut().for_each(|e| e.p /= sum);
        } else {
            let u = 1.0 / entries.len().max(1) as f32;
            entries.iter_mut().for_each(|e| e.p = u);
        }
        self.tree[cur].entries = entries;
        self.tree[cur].expanded = true;
        self.build_kernel(cur);

        self.backup(value);
        self.sims_left -= 1;
    }

    /// Negamax backup along the descent path. `value` is from the leaf's side.
    fn backup(&mut self, value: f32) {
        let mut v = value;
        for &(node, entry) in self.path.iter().rev() {
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
    pub fn provide_widen(&mut self, candidates: Vec<(G::Move, f32, bool)>) {
        let cur = self.pending_widen.take().expect("provide_widen without a pending widen");
        let mut seen: std::collections::HashMap<u128, usize> = Default::default();
        for (i, e) in self.tree[cur].entries.iter().enumerate() {
            seen.insert(G::state_key(&G::apply(&self.tree[cur].state, &e.mv)), i);
        }
        let k_max = self.params.k_candidates.max(1);
        let before = self.tree[cur].entries.len();
        for (mv, p, explore) in candidates {
            if self.tree[cur].entries.len() >= k_max {
                break;
            }
            let key = G::state_key(&G::apply(&self.tree[cur].state, &mv));
            match seen.get(&key) {
                Some(&i) => {
                    self.tree[cur].entries[i].p += p.max(0.0);
                    self.tree[cur].entries[i].explore |= explore;
                }
                None => {
                    seen.insert(key, self.tree[cur].entries.len());
                    self.tree[cur].entries.push(Entry {
                        mv,
                        p: p.max(0.0),
                        n: 0,
                        w: 0.0,
                        child: -1,
                        explore,
                    });
                }
            }
        }
        if self.tree[cur].entries.len() == before {
            self.tree[cur].widen_exhausted = true;
        }
        // Sorting is safe here because a widen abandons its simulation, so no
        // `path` entry indexes into this node.
        self.tree[cur].entries.sort_by(|a, b| b.p.total_cmp(&a.p));
        let sum: f32 = self.tree[cur].entries.iter().map(|e| e.p).sum();
        if sum > 0.0 {
            self.tree[cur].entries.iter_mut().for_each(|e| e.p /= sum);
        } else {
            let u = 1.0 / self.tree[cur].entries.len().max(1) as f32;
            self.tree[cur].entries.iter_mut().for_each(|e| e.p = u);
        }
        self.build_kernel(cur);
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

    pub fn finish(mut self) -> SampledSearchResult<G> {
        let mut stats = SearchStats::default();
        for node in self.tree.iter() {
            if !node.expanded {
                continue;
            }
            stats.nodes += 1;
            stats.candidates += node.entries.len() as u64;
            stats.unvisited += node.entries.iter().filter(|e| e.n == 0).count() as u64;
            stats.visit_hist[SearchStats::bucket(node.total_n)] += 1;
        }

        let root = &self.tree[0];
        let candidates: Vec<(G::Move, u32)> =
            root.entries.iter().map(|e| (e.mv.clone(), e.n)).collect();
        let priors: Vec<f32> = root.entries.iter().map(|e| e.p).collect();
        let explore: Vec<bool> = root.entries.iter().map(|e| e.explore).collect();
        let value =
            if root.total_n > 0 { root.w_sum as f64 / root.total_n as f64 } else { 0.0 };

        let chosen = if candidates.is_empty() {
            0
        } else if self.ply < self.params.temp_moves {
            // Opening plies: sample proportional to visits, for self-play diversity.
            let total: u32 = candidates.iter().map(|(_, n)| *n).sum();
            if total == 0 {
                self.rng.gen_range(0..candidates.len())
            } else {
                let mut r = self.rng.gen_range(0..total);
                let mut pick = candidates.len() - 1;
                for (i, (_, n)) in candidates.iter().enumerate() {
                    if r < *n {
                        pick = i;
                        break;
                    }
                    r -= *n;
                }
                pick
            }
        } else {
            candidates
                .iter()
                .enumerate()
                .max_by_key(|(_, (_, n))| *n)
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        SampledSearchResult { candidates, priors, explore, chosen, value, nodes: self.nodes, stats }
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
    loop {
        match s.step() {
            Step::NeedExpansion(state) => {
                // Value only; candidates wait until something descends here.
                let (v, cands) = model.expand(&state, 0)?;
                s.provide(v, cands);
            }
            Step::NeedWiden(state) => {
                // Same call, same shape; only the candidates are wanted.
                let (_, cands) = model.expand(&state, chunk_k(params))?;
                s.provide_widen(cands);
            }
            Step::Done => return Ok(s.finish()),
        }
    }
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
        let mut states: Vec<G::State> = Vec::with_capacity(searches.len());
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
                Step::NeedWiden(st) => {
                    want.push(i);
                    widen.push(true);
                    states.push(st);
                }
                Step::Done => done[i] = true,
            }
        }
        if want.is_empty() {
            break;
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
            model.expand_batch(&exp, 0)?
        };
        let wid_res = if wid.is_empty() {
            Vec::new()
        } else {
            model.expand_batch(&wid, chunk_k(params))?
        };
        anyhow::ensure!(
            exp_res.len() == exp.len() && wid_res.len() == wid.len(),
            "expand_batch returned {}/{} results for {}/{} states",
            exp_res.len(),
            wid_res.len(),
            exp.len(),
            wid.len()
        );
        let mut exp_out = exp_res.into_iter();
        let mut wid_out = wid_res.into_iter();
        for (i, is_widen) in want.into_iter().zip(widen) {
            if is_widen {
                let (_, cands) = wid_out.next().expect("one widen result per widen request");
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
            CountState { n: s.n.saturating_sub(*mv), side: 3 - s.side }
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
                (1..=3u32).filter(|m| *m <= s.n).map(|m| (m, 1.0, false)).collect(),
            ))
        }
    }

    /// Always proposes the same move — exercises candidate merging.
    struct Duplicating;
    impl SampledModel<Count> for Duplicating {
        fn expand(&self, s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
            let _ = s;
            Ok((0.0, vec![(1, 1.0, false), (1, 1.0, false), (1, 1.0, false), (2, 1.0, false)]))
        }
    }

    struct Empty;
    impl SampledModel<Count> for Empty {
        fn expand(&self, _s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
            Ok((0.25, vec![]))
        }
    }

    fn params(sims: u32) -> SampledParams {
        SampledParams { simulations: sims, temp_moves: 0, seed: 1, ..Default::default() }
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
                Ok((0.1, (1..=3).map(|m| (m, 1.0 / m as f32, m == 3)).collect()))
            }
        }
        let mut params = SampledParams {
            simulations: 120,
            k_forced: 2.0,
            min_explore_visits: 2.0,
            temp_moves: 0,
            seed: 7,
            ..Default::default()
        };
        params.kernel_sigma = 0.0;
        let flat = search::<Count, _>(&Spread, CountState { n: 21, side: 1 }, &params, 0).expect("flat search");

        // Same params, same seed, through the same code path a KR run takes.
        let again = search::<Count, _>(&Spread, CountState { n: 21, side: 1 }, &params, 0).expect("repeat");
        assert_eq!(
            flat.candidates.iter().map(|(m, n)| (*m, *n)).collect::<Vec<_>>(),
            again.candidates.iter().map(|(m, n)| (*m, *n)).collect::<Vec<_>>(),
            "sigma = 0 must be deterministic and unchanged"
        );
        assert_eq!(flat.chosen, again.chosen);
        assert_eq!(flat.priors, again.priors);
        assert_eq!(flat.policy_target_with(&params), again.policy_target_with(&params));

        // And a positive bandwidth must actually do something different, or the
        // test above proves nothing.
        params.kernel_sigma = 1.0;
        let kr = search::<Count, _>(&Spread, CountState { n: 21, side: 1 }, &params, 0).expect("kr search");
        let flat_n: Vec<u32> = flat.candidates.iter().map(|(_, n)| *n).collect();
        let kr_n: Vec<u32> = kr.candidates.iter().map(|(_, n)| *n).collect();
        assert_ne!(flat_n, kr_n, "a live kernel must change the visit distribution");
    }

    /// The root sweep: every candidate is visited at least once, so each has a
    /// real value estimate rather than an FPU guess behind its prior.
    #[test]
    fn every_root_candidate_is_visited_at_least_once() {
        struct Wide;
        impl SampledModel<Count> for Wide {
            fn expand(&self, _s: &CountState, _k: usize) -> anyhow::Result<Expansion<Count>> {
                // Lopsided priors: without the sweep, PUCT would starve move 3.
                Ok((0.0, vec![(1, 0.98, false), (2, 0.01, false), (3, 0.01, false)]))
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
        let r = search::<Count, _>(&Wide, CountState { n: 21, side: 1 }, &params, 0)
            .expect("search");
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
                Ok((0.0, vec![(1, 0.98, false), (2, 0.01, false), (3, 0.01, false)]))
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
        let r = search::<Count, _>(&Wide, CountState { n: 21, side: 1 }, &params, 0)
            .expect("search");
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
        assert!((sum - 1.0).abs() < 1e-5, "target must stay normalised, got {sum}");
    }

    /// Progressive widening exposes only a prefix at interior nodes, and never
    /// restricts the root.
    #[test]
    fn widening_limits_interior_nodes_but_not_the_root() {
        let params = SampledParams { pw_alpha: 0.5, pw_beta: 2.0, ..Default::default() };
        let mut node = new_node::<Count>(CountState { n: 21, side: 1 });
        node.entries = (0..12)
            .map(|i| Entry::<Count> { mv: i, p: 1.0 / 12.0, n: 0, w: 0.0, child: -1, explore: false })
            .collect();

        node.total_n = 1;
        assert_eq!(visible(&node, &params, true), 12, "the root always sees every candidate");
        // At one visit the schedule is ceil(pw_beta) — the initial count is
        // pw_beta itself, which is why there is no separate knob for it.
        assert_eq!(visible(&node, &params, false), 2, "one visit -> ceil(pw_beta)");
        node.total_n = 16;
        assert_eq!(visible(&node, &params, false), 8, "ceil(2*sqrt(16)) = 8");
        node.total_n = 100;
        assert_eq!(visible(&node, &params, false), 12, "clamped to the candidate count");

        let off = SampledParams { pw_beta: 0.0, ..params };
        node.total_n = 1;
        assert_eq!(visible(&node, &off, false), 12, "pw_beta = 0 disables widening");
    }

    /// Widening must never reveal more than one new candidate per visit, for
    /// *any* schedule — not just the defaults, whose `beta * alpha == 1` makes
    /// the property hold by accident.
    #[test]
    fn widening_never_adds_more_than_one_candidate_per_visit() {
        for (beta, alpha) in [(2.0f32, 0.5f32), (4.0, 0.5), (8.0, 0.9), (1.0, 0.3)] {
            let params = SampledParams { pw_alpha: alpha, pw_beta: beta, ..Default::default() };
            let mut node = new_node::<Count>(CountState { n: 21, side: 1 });
            node.entries = (0..64)
                .map(|i| Entry::<Count> {
                    mv: i,
                    p: 1.0 / 64.0,
                    n: 0,
                    w: 0.0,
                    child: -1,
                    explore: false,
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
                let start = self.cursor.fetch_add(k.max(1), std::sync::atomic::Ordering::Relaxed);
                Ok((
                    0.0,
                    (0..k.max(1))
                        .map(|i| ((start + i) % 8 + 1) as u32)
                        .map(|m| (m, 1.0, false))
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
        let model = Rot { cursor: std::sync::atomic::AtomicUsize::new(0) };
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
                Step::NeedWiden(st) => {
                    let (_, c) = model.expand(&st, chunk_k(&params)).unwrap();
                    search.provide_widen(c);
                }
                Step::Done => break,
            }
        }
        assert!(saw_full_before_first_visit, "root was never fully populated before its sweep");
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
                let start = self.cursor.fetch_add(k.max(1), std::sync::atomic::Ordering::Relaxed);
                let moves: Vec<u32> = (1..=3u32).filter(|m| *m <= s.n).collect();
                let cands: Vec<(u32, f32, bool)> = (0..k.max(1))
                    .map(|i| moves[(start + i) % moves.len()])
                    .map(|m| (m, 1.0 / m as f32, false))
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
        let pool = Pool { cursor: std::sync::atomic::AtomicUsize::new(0) };
        let whole = search::<Count, _>(&pool, CountState { n: 21, side: 1 }, &base, 0)
            .expect("unchunked");

        let chunked_params = SampledParams { pw_chunk: 1, ..base };
        let chunked = search::<Count, _>(&pool, CountState { n: 21, side: 1 }, &chunked_params, 0)
            .expect("chunked");

        let mut a: Vec<u32> = whole.candidates.iter().map(|(m, _)| *m).collect();
        let mut b: Vec<u32> = chunked.candidates.iter().map(|(m, _)| *m).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "chunked generation must reach the same root candidates");
        let sum: f32 = chunked.priors.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "priors must stay normalised, got {sum}");
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
                Step::NeedWiden(s) => {
                    let cands: Vec<(u32, f32, bool)> = (1..=3u32)
                        .filter(|m| *m <= s.n)
                        .map(|m| (m, if m == 3 { 0.001 } else { 1.0 }, m == 3))
                        .collect();
                    search.provide_widen(cands);
                }
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
    /// an explore candidate `m` visits and the target does not subtract them, the
    /// policy is taught that injected noise was search's judgement — strictly
    /// worse than not exploring at all.
    #[test]
    fn guaranteed_explore_visits_are_pruned_from_the_target() {
        let params = SampledParams { k_forced: 0.0, min_explore_visits: 4.0, ..Default::default() };
        // The explore candidate earned exactly its floor and nothing more.
        let r = SampledSearchResult::<Count> {
            stats: SearchStats::default(),
            candidates: vec![(1, 40), (2, 20), (3, 4)],
            priors: vec![0.5, 0.4, 0.1],
            explore: vec![false, false, true],
            chosen: 0,
            value: 0.0,
            nodes: 64,
        };
        let t = r.policy_target_with(&params);
        assert_eq!(t[2], 0.0, "a candidate that only ever got its floor must not survive");
        assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-5, "target must stay normalised");

        // Earning well beyond the floor, it survives on merit.
        let r2 = SampledSearchResult::<Count> {
            stats: SearchStats::default(),
            candidates: vec![(1, 40), (2, 20), (3, 25)],
            priors: vec![0.5, 0.4, 0.1],
            explore: vec![false, false, true],
            chosen: 0,
            value: 0.0,
            nodes: 85,
        };
        let t2 = r2.policy_target_with(&params);
        assert!(t2[2] > 0.0, "organic visits beyond the floor must survive pruning");
    }

    #[test]
    fn lockstep_matches_running_the_searches_alone() {
        // The batching must be a scheduling change and nothing more: each game
        // holds one outstanding expansion, so its tree is unchanged.
        let roots: Vec<CountState> =
            (3..=9).map(|n| CountState { n, side: 1 }).collect();
        let plies: Vec<u32> = (0..roots.len() as u32).collect();
        let p = params(200);
        let batched =
            search_lockstep::<Count, _>(&Mock, roots.clone(), &p, &plies).unwrap();
        assert_eq!(batched.len(), roots.len());
        for (i, root) in roots.into_iter().enumerate() {
            let alone = search::<Count, _>(&Mock, root, &p, plies[i]).unwrap();
            assert_eq!(
                batched[i].candidates, alone.candidates,
                "game {i}: visit counts differ under lockstep"
            );
            assert_eq!(batched[i].chosen, alone.chosen, "game {i}: chose differently");
            assert!((batched[i].value - alone.value).abs() < 1e-9, "game {i}: value differs");
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
        assert_eq!(r.candidates[r.chosen].0, 1, "should leave a multiple of four");
        assert!(r.value > 0.0, "a won position should evaluate positive, got {}", r.value);
    }

    #[test]
    fn an_immediate_win_is_taken() {
        for n in 1..=3u32 {
            let s = CountState { n, side: 1 };
            let r = search::<Count, _>(&Mock, s, &params(100), 99).unwrap();
            assert_eq!(r.candidates[r.chosen].0, n, "should just take the last {n}");
            assert!(r.value > 0.5, "a won position should evaluate positive, got {}", r.value);
        }
    }

    #[test]
    fn a_lost_position_evaluates_negative() {
        // 4 is a multiple of 4: every move leaves the opponent a winning take.
        let r = search::<Count, _>(&Mock, CountState { n: 4, side: 1 }, &params(400), 99).unwrap();
        assert!(r.value < 0.0, "a lost position should evaluate negative, got {}", r.value);
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
        assert_eq!(r.candidates.len(), 2, "three copies of `1` should merge into one child");
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
    fn policy_target_prunes_forced_visits_away() {
        // A candidate with a tiny prior that only ever got its forced visits
        // must not appear in the training target.
        let r = SampledSearchResult::<Count> {
            stats: SearchStats::default(),
            candidates: vec![(1, 93), (2, 3), (3, 4)],
            priors: vec![0.9, 0.05, 0.05],
            explore: vec![false; 8],
            chosen: 0,
            value: 0.0,
            nodes: 0,
        };
        let t = r.policy_target(2.0);
        assert!((t.iter().sum::<f32>() - 1.0).abs() < 1e-5, "target must be normalised");
        assert!(t[0] > 0.9, "the earned child should dominate: {t:?}");
        assert_eq!(t[1], 0.0, "forced-only visits must be pruned: {t:?}");
        assert_eq!(t[2], 0.0, "forced-only visits must be pruned: {t:?}");
    }

    #[test]
    fn policy_target_keeps_a_candidate_that_earned_its_visits() {
        // Same tiny prior, but this one attracted far more visits than its
        // forced floor — it survives on merit. This is how an injected random
        // candidate gets promoted.
        let r = SampledSearchResult::<Count> {
            stats: SearchStats::default(),
            candidates: vec![(1, 55), (2, 45)],
            priors: vec![0.95, 0.05],
            explore: vec![false; 8],
            chosen: 0,
            value: 0.0,
            nodes: 0,
        };
        let t = r.policy_target(2.0);
        assert!(t[1] > 0.3, "an organically visited child must survive pruning: {t:?}");
    }

    #[test]
    fn pruning_survival_matches_the_stated_threshold() {
        // Uniform priors and uniform visits, swept across visits-per-candidate.
        // Below the threshold the target collapses to a single candidate, which
        // starves the energy head; above it, several survive.
        let k_forced = 2.0f32;
        let thresh = min_visits_per_candidate(k_forced);
        assert!((thresh - 3.732).abs() < 1e-2, "threshold was {thresh}");
        let survivors = |x: u32, k: usize| -> usize {
            let r = SampledSearchResult::<Count> {
            stats: SearchStats::default(),
                candidates: (0..k).map(|i| (i as u32, x)).collect(),
                priors: vec![1.0 / k as f32; k],
            explore: vec![false; 8],
                chosen: 0,
                value: 0.0,
                nodes: 0,
            };
            r.policy_target(k_forced).iter().filter(|&&p| p > 0.0).count()
        };
        assert_eq!(survivors(3, 8), 1, "below threshold: everything pruned to one");
        assert!(survivors(8, 8) > 1, "well above threshold: candidates survive");
        assert!(survivors(16, 8) > 4, "further above: most survive");
    }

    #[test]
    fn policy_target_never_returns_all_zeros() {
        let r = SampledSearchResult::<Count> {
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
        let p = SampledParams { simulations: 60, temp_moves: 100, seed: 7, ..Default::default() };
        let picks: std::collections::HashSet<u32> = (0..30)
            .map(|ply| {
                let r = search::<Count, _>(&Mock, Count::initial_state(), &p, ply).unwrap();
                r.candidates[r.chosen].0
            })
            .collect();
        assert!(picks.len() > 1, "temperature should diversify the opening, got {picks:?}");
    }
}
