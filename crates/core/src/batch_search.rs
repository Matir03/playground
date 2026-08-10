//! GPU-saturating batched self-play engine, generic over a game.
//!
//! The per-game search ([`GameSearch`]) is the same sequential PUCT MCTS the
//! games used before (node-budget, ε-greedy forced exploration with random
//! visits excluded from the policy target, root Dirichlet, temperature move
//! pick) — but restructured so it *yields* the next leaf that needs a network
//! evaluation instead of blocking on one. That lets the driver run **many games
//! in lockstep**: each round it pulls one pending leaf from every active game,
//! coalesces them into one big batched forward, and scatters the results back.
//!
//! This decouples batch size from thread count — the old design's ceiling, where
//! a leaf forward could only batch as wide as there were OS threads blocked on
//! eval at that instant (measured `avg_batch` ~36–75 of 256). Because each game
//! still has at most one pending leaf at a time, **search behaviour is identical
//! to the old per-game MCTS** — no virtual loss, no approximation.
//!
//! Three further wins over the old `selfplay.rs` server:
//!   * **leaf-side encoding** — workers encode planes; the inference thread only
//!     does H2D → forward → D2H → scatter (no `fill_planes` on its critical path).
//!   * **pipelining** — workers keep descending/backing-up (and queuing the next
//!     batch) while the inference thread runs a forward; the channel buffers it.
//!   * **eval cache (transposition table)** — `hash(position) → (value, priors,
//!     legal)` memoizes the expensive forward *and* legal-move generation. Safe
//!     because the weights are fixed for a chunk; the cache is per-`run_chunk`.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::mpsc::{channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};

/// What the batched search needs from a game. Action space is a fixed `ACTIONS`
/// indices (legal subset varies by state); planes are `PLANES` f32 per position.
pub trait BatchGame: Send + Sync + 'static {
    type State: Clone + Send + Sync;
    const ACTIONS: usize;
    const PLANES: usize;

    fn initial_state() -> Self::State;
    /// 1 or 2 — used only to tag training examples with the side to move.
    fn to_move(state: &Self::State) -> u8;
    /// A *cheap* superset of the legal actions, used only to mask the policy
    /// priors. May include actions that turn out illegal (e.g. Go suicide/ko) —
    /// those are filtered lazily by `apply_action` when the search first selects
    /// them, so we never pay full legality for the whole action space up front.
    fn candidate_actions(state: &Self::State) -> Vec<usize>;
    /// Apply `action`, or `None` if it is actually illegal (the lazy legality
    /// check). For games where every candidate is legal this is always `Some`.
    fn apply_action(state: &Self::State, action: usize) -> Option<Self::State>;
    /// Terminal value for the side to move (`None` = game in progress).
    fn terminal_value(state: &Self::State) -> Option<f32>;
    /// Write the `PLANES` network-input floats for `state` into `out`.
    fn encode_planes(state: &Self::State, out: &mut [f32]);
    /// A collision-resistant key for the eval cache (position identity).
    fn hash_key(state: &Self::State) -> u128;
    /// A cheap signature of the position's *settled area score* (e.g. Go: packed
    /// black/white area). Used for score-stability early termination: when this is
    /// unchanged for `settle_plies` consecutive plies the borders are settled, so
    /// the game is stopped and adjudicated. `None` (the default) disables
    /// score-stability for the game.
    fn score_signature(_state: &Self::State) -> Option<i64> {
        None
    }
}

/// The network, as the engine sees it: a pure batched forward over planes.
pub trait BatchModel: Send + Sync {
    /// `planes` is `b * PLANES` row-major. Returns `(values[b], priors[b*ACTIONS])`
    /// where priors are a row-major softmax over all `ACTIONS`.
    fn forward(&self, planes: &[f32], b: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)>;
}

/// Search knobs (mirrors the per-game `Params`). Budget: `node_budget` if > 0,
/// else `sims`.
#[derive(Clone, Copy)]
pub struct SearchParams {
    pub sims: u32,
    pub node_budget: u32,
    pub c_puct: f32,
    pub dir_alpha: f32,
    pub dir_eps: f32,
    pub temp_moves: u32,
    pub explore_eps: f32,
    /// Hard ply cap for a self-play game (0 = no cap). With score-stability +
    /// resignation enabled this is just a safety backstop against non-terminating
    /// games (e.g. repetition cycles), not the primary terminator.
    pub max_ply: u32,
    /// Score-stability early stop: end the game once `score_signature` is unchanged
    /// for this many consecutive plies (0 = disabled). The final position is then
    /// adjudicated by the caller (area scoring).
    pub settle_plies: u32,
    /// Resignation: a side resigns (and loses) if its root win-estimate stays at or
    /// below `resign_threshold` for `resign_consecutive` of its own moves in a row
    /// (`resign_consecutive == 0` disables resignation).
    pub resign_threshold: f32,
    pub resign_consecutive: u32,
    /// Fraction of games that *never* resign (play to a real terminal/settle/cap),
    /// even when resignation is enabled — the AlphaZero resign-calibration
    /// safeguard, so a too-aggressive threshold can't silently corrupt training
    /// with false resignations. 0 = every game may resign.
    pub resign_playout_frac: f32,
}

/// One move's training-relevant record (policy target + the move actually played).
pub struct MoveRecord {
    pub policy: Vec<f32>, // length ACTIONS, PUCT-only visit distribution
    pub action: usize,
    pub to_move: u8,
}

/// A finished self-play game: the full state sequence (len = moves+1) plus the
/// per-move records. The caller turns these into game-specific training examples
/// (planes, value-z, reifier next-planes, …) so the engine stays game-agnostic.
pub struct GameRecord<G: BatchGame> {
    pub states: Vec<G::State>,
    pub moves: Vec<MoveRecord>,
    /// If `Some(side)`, that side resigned → it loses. The caller should derive the
    /// value target from this (a resignation loss) instead of scoring the final
    /// position, which may not be fully settled when a side resigns.
    pub resigned_by: Option<u8>,
}

/// Throughput counters. `nodes` = distinct tree nodes built (the honest search-
/// effort unit; excludes terminal re-traversals). `evals` = network forwards
/// actually run (nodes minus `cache_hits`).
#[derive(Default, Clone, Copy)]
pub struct ChunkStats {
    pub nodes: u64,
    pub evals: u64,
    pub cache_hits: u64,
    pub terminal_sims: u64,
    pub batches: u64,
    pub batch_samples: u64,
    pub elapsed: f64,
    /// Wall-clock seconds the inference server spent inside `model.forward` (the
    /// GPU compute). `gpu_secs / elapsed` is the GPU duty cycle: ≪1 = CPU-bound.
    pub gpu_secs: f64,
    /// Summed worker time (over all workers, so > elapsed) in each phase — the
    /// gather:wait:apply ratio shows where the CPU side spends its time.
    pub gather_secs: f64,
    pub wait_secs: f64,
    pub apply_secs: f64,
    /// GPU-thread non-forward time: `gpu_block_secs` waiting for leaves (eval queue
    /// empty → starved), `gpu_dispatch_secs` collecting the batch + scattering
    /// results. (forward time is `gpu_secs`.)
    pub gpu_block_secs: f64,
    pub gpu_dispatch_secs: f64,
}

impl ChunkStats {
    pub fn avg_batch(&self) -> f64 {
        if self.batches > 0 { self.batch_samples as f64 / self.batches as f64 } else { 0.0 }
    }
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.evals + self.cache_hits;
        if total > 0 { self.cache_hits as f64 / total as f64 } else { 0.0 }
    }
    /// Fraction of wall time the GPU was computing (≈1 = GPU-bound).
    pub fn gpu_duty(&self) -> f64 {
        if self.elapsed > 0.0 { self.gpu_secs / self.elapsed } else { 0.0 }
    }
}

// ── Eval cache (transposition table) ─────────────────────────────────────────

/// Cached expansion of a position: leaf value + priors + legal action set, so a
/// hit skips both the network forward and legal-move generation.
struct ExpandOut {
    value: f32,
    p: Vec<f32>,        // length ACTIONS, renormalised over legal (0 elsewhere)
    legal: Vec<usize>,
}

const CACHE_SHARDS: usize = 64;

struct Cache {
    shards: Vec<Mutex<HashMap<u128, Arc<ExpandOut>>>>,
}

impl Cache {
    fn new() -> Self {
        Self { shards: (0..CACHE_SHARDS).map(|_| Mutex::new(HashMap::new())).collect() }
    }
    fn shard(&self, key: u128) -> &Mutex<HashMap<u128, Arc<ExpandOut>>> {
        &self.shards[(key as usize) & (CACHE_SHARDS - 1)]
    }
    fn get(&self, key: u128) -> Option<Arc<ExpandOut>> {
        self.shard(key).lock().unwrap().get(&key).cloned()
    }
    fn insert(&self, key: u128, v: Arc<ExpandOut>) {
        self.shard(key).lock().unwrap().insert(key, v);
    }
}

// ── Inference server (batched forward over leaf-encoded planes) ───────────────

/// One worker's *wave*: the planes for `count` leaves (concatenated) plus a reply
/// channel. Submitting whole waves (instead of one request per leaf) keeps the
/// channel traffic at O(workers/round), not O(leaves), and lets the server
/// coalesce several workers' waves into one big forward.
struct Req {
    planes: Vec<f32>, // count * PLANES
    count: usize,
    resp: SyncSender<Vec<(f32, Vec<f32>)>>,
}

/// Coalesce queued waves into batched forwards until all senders drop.
fn inference_server<G: BatchGame>(
    rx: Receiver<Req>,
    model: &dyn BatchModel,
    max_batch: usize,
    stats: &Mutex<ChunkStats>,
) {
    let mut local_batches = 0u64;
    let mut local_samples = 0u64;
    let mut local_gpu = 0.0f64;
    loop {
        let first = match rx.recv() {
            Ok(r) => r,
            Err(_) => break,
        };
        let mut total = first.count;
        let mut waves = vec![first];
        while total < max_batch {
            match rx.try_recv() {
                Ok(r) => {
                    total += r.count;
                    waves.push(r);
                }
                Err(_) => break,
            }
        }
        let mut planes = Vec::with_capacity(total * G::PLANES);
        for w in &waves {
            planes.extend_from_slice(&w.planes);
        }
        let t_gpu = Instant::now();
        let (vals, priors) = model.forward(&planes, total).expect("batched forward failed");
        local_gpu += t_gpu.elapsed().as_secs_f64();
        let mut off = 0usize;
        for w in waves {
            let mut out = Vec::with_capacity(w.count);
            for j in 0..w.count {
                let gi = off + j;
                out.push((vals[gi], priors[gi * G::ACTIONS..(gi + 1) * G::ACTIONS].to_vec()));
            }
            off += w.count;
            let _ = w.resp.send(out);
        }
        local_batches += 1;
        local_samples += total as u64;
    }
    let mut s = stats.lock().unwrap();
    s.batches += local_batches;
    s.batch_samples += local_samples;
    s.gpu_secs += local_gpu;
}

// ── Per-game search ──────────────────────────────────────────────────────────

struct Node<G: BatchGame> {
    state: G::State,
    p: Vec<f32>,
    n: Vec<u32>,
    rnd: Vec<u32>,
    w: Vec<f32>,
    children: Vec<i32>,
    legal: Vec<bool>,
    expanded: bool,
    terminal: bool,
    term_v: f32,
}

impl<G: BatchGame> Node<G> {
    fn new(state: G::State) -> Self {
        let a = G::ACTIONS;
        Self {
            state,
            p: vec![0.0; a],
            n: vec![0; a],
            rnd: vec![0; a],
            w: vec![0.0; a],
            children: vec![-1; a],
            legal: vec![false; a],
            expanded: false,
            terminal: false,
            term_v: 0.0,
        }
    }
}

/// Result of asking a search for its next action.
enum Step<G: BatchGame> {
    /// This leaf needs a network evaluation; its state is provided.
    NeedEval(G::State),
    /// The move search is complete (budget reached / saturated).
    Done,
}

const SATURATION_PATIENCE: u32 = 512;

/// A single position's PUCT search, restructured to yield leaves for batched
/// evaluation. At most one leaf is pending at a time → identical dynamics to the
/// old blocking MCTS.
struct GameSearch<G: BatchGame> {
    arena: Vec<Node<G>>,
    params: SearchParams,
    pending_path: Vec<(usize, usize, bool)>,
    pending_leaf: Option<usize>,
    pending_nodes_before: usize,
    sims: u32,
    no_growth: u32,
    dirichlet_done: bool,
    nodes_built: u64,
    terminal_sims: u64,
    _g: PhantomData<G>,
}

impl<G: BatchGame> GameSearch<G> {
    fn new(root: G::State, params: SearchParams) -> Self {
        Self {
            arena: vec![Node::new(root)],
            params,
            pending_path: Vec::with_capacity(64),
            pending_leaf: None,
            pending_nodes_before: 0,
            sims: 0,
            no_growth: 0,
            dirichlet_done: false,
            nodes_built: 0,
            terminal_sims: 0,
            _g: PhantomData,
        }
    }

    fn budget_reached(&self) -> bool {
        if self.params.node_budget > 0 {
            self.arena.len() as u32 >= self.params.node_budget
                || self.no_growth >= SATURATION_PATIENCE
        } else {
            self.sims >= self.params.sims
        }
    }

    fn select(&self, node: usize) -> usize {
        let nd = &self.arena[node];
        // One pass for the node's totals, so unvisited children can be scored
        // against what this node is actually worth (see `core::puct`).
        let mut total = 0u32;
        let mut w_sum = 0.0f32;
        let mut visited_p = 0.0f32;
        for a in 0..G::ACTIONS {
            if nd.n[a] > 0 {
                total += nd.n[a];
                w_sum += nd.w[a];
                visited_p += nd.p[a];
            }
        }
        let parent_q = if total > 0 { w_sum / total as f32 } else { 0.0 };
        let fpu = crate::puct::fpu(parent_q, visited_p);
        let sqrt_total = ((total + 1) as f32).sqrt();
        let c_puct = self.params.c_puct;
        let mut best = usize::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for a in 0..G::ACTIONS {
            if !nd.legal[a] {
                continue;
            }
            let q = if nd.n[a] > 0 { nd.w[a] / nd.n[a] as f32 } else { fpu };
            let u = c_puct * nd.p[a] * sqrt_total / (1 + nd.n[a]) as f32;
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
        best
    }

    fn random_legal(&self, node: usize, rng: &mut impl Rng) -> usize {
        let nd = &self.arena[node];
        let legal: Vec<usize> = (0..G::ACTIONS).filter(|&a| nd.legal[a]).collect();
        legal[rng.gen_range(0..legal.len())]
    }

    /// Descend (consuming terminal sims inline) until a leaf needs evaluation or
    /// the budget is spent.
    fn next(&mut self, rng: &mut impl Rng) -> Step<G> {
        loop {
            if self.budget_reached() {
                return Step::Done;
            }
            self.pending_path.clear();
            let nodes_before = self.arena.len();
            let mut node = 0usize;
            'descend: while self.arena[node].expanded && !self.arena[node].terminal {
                // Root-only ε-greedy: forced-random exploration is injected only at
                // the root (node 0), so the deep tree stays pure PUCT (its Q values
                // aren't polluted by random rollouts). The root random visits are
                // excluded from the policy target via `rnd`, so the target stays clean.
                let was_random = node == 0
                    && self.params.explore_eps > 0.0
                    && rng.gen::<f32>() < self.params.explore_eps;
                let a = if was_random { self.random_legal(node, rng) } else { self.select(node) };
                if a == usize::MAX {
                    break; // no legal action left (shouldn't happen: pass is always legal)
                }
                let child = self.arena[node].children[a];
                let next = if child >= 0 {
                    child as usize
                } else {
                    // Lazy legality: only now do we pay the full legal check, and
                    // only for an action the search actually wants to descend.
                    match G::apply_action(&self.arena[node].state, a) {
                        Some(next_state) => {
                            let idx = self.arena.len();
                            self.arena.push(Node::new(next_state));
                            self.arena[node].children[a] = idx as i32;
                            idx
                        }
                        None => {
                            self.arena[node].legal[a] = false; // illegal — never pick again
                            continue 'descend;
                        }
                    }
                };
                self.pending_path.push((node, a, was_random));
                node = next;
            }
            self.sims += 1;
            if let Some(tv) = self.terminal_at(node) {
                // Terminal leaf: back up its exact value, no network eval.
                self.backup(tv);
                if self.arena.len() == nodes_before {
                    self.no_growth += 1;
                } else {
                    self.no_growth = 0;
                }
                self.terminal_sims += 1;
                continue;
            }
            self.pending_leaf = Some(node);
            self.pending_nodes_before = nodes_before;
            return Step::NeedEval(self.arena[node].state.clone());
        }
    }

    /// Mark/get a node's terminal value, memoised on the node.
    fn terminal_at(&mut self, node: usize) -> Option<f32> {
        if self.arena[node].terminal {
            return Some(self.arena[node].term_v);
        }
        if let Some(tv) = G::terminal_value(&self.arena[node].state) {
            self.arena[node].terminal = true;
            self.arena[node].term_v = tv;
            Some(tv)
        } else {
            None
        }
    }

    /// Apply a network/cache evaluation to the pending leaf: write priors+legal,
    /// (root) Dirichlet, then back up the value.
    fn apply(&mut self, out: &ExpandOut, rng: &mut impl Rng) {
        let node = self.pending_leaf.take().expect("apply with no pending leaf");
        {
            let nd = &mut self.arena[node];
            for a in 0..G::ACTIONS {
                nd.legal[a] = false;
                nd.p[a] = 0.0;
            }
            for &a in &out.legal {
                nd.legal[a] = true;
                nd.p[a] = out.p[a];
            }
            nd.expanded = true;
        }
        if node == 0 && self.params.dir_eps > 0.0 && !self.dirichlet_done {
            self.apply_dirichlet(rng);
            self.dirichlet_done = true;
        }
        self.backup(out.value);
        self.nodes_built += 1;
        // A leaf eval always created a fresh node during descent (or expanded the
        // root) → growth; reset the saturation counter.
        if self.arena.len() != self.pending_nodes_before {
            self.no_growth = 0;
        }
    }

    fn apply_dirichlet(&mut self, rng: &mut impl Rng) {
        let legal: Vec<usize> = (0..G::ACTIONS).filter(|&a| self.arena[0].legal[a]).collect();
        if legal.is_empty() {
            return;
        }
        let gamma = Gamma::new(self.params.dir_alpha, 1.0).unwrap();
        let mut g: Vec<f32> = legal.iter().map(|_| gamma.sample(rng) as f32).collect();
        let gsum: f32 = g.iter().sum::<f32>().max(1e-12);
        for x in &mut g {
            *x /= gsum;
        }
        let eps = self.params.dir_eps;
        for (i, &a) in legal.iter().enumerate() {
            self.arena[0].p[a] = (1.0 - eps) * self.arena[0].p[a] + eps * g[i];
        }
    }

    fn backup(&mut self, mut v: f32) {
        for &(n, a, was_random) in self.pending_path.iter().rev() {
            v = -v;
            let nd = &mut self.arena[n];
            nd.n[a] += 1;
            nd.w[a] += v;
            if was_random {
                nd.rnd[a] += 1;
            }
        }
    }

    /// PUCT-only visit counts at the root (forced-random visits excluded).
    fn policy_counts(&self) -> Vec<u32> {
        let root = &self.arena[0];
        (0..G::ACTIONS).map(|a| root.n[a].saturating_sub(root.rnd[a])).collect()
    }

    /// Mean action-value at the root from the side-to-move's perspective, in
    /// [-1, 1] — the search's win estimate, used for the resignation check. 0 when
    /// the root has no visits yet.
    fn root_value(&self) -> f32 {
        let root = &self.arena[0];
        let n: u32 = root.n.iter().sum();
        if n == 0 {
            return 0.0;
        }
        let w: f32 = root.w.iter().sum();
        w / n as f32
    }
}

// ── Driver ───────────────────────────────────────────────────────────────────

/// A game in flight: its current move's search, the states played so far, and the
/// move records. `rng` is per-game for independent, reproducible play.
struct Runner<G: BatchGame> {
    search: GameSearch<G>,
    states: Vec<G::State>,
    moves: Vec<MoveRecord>,
    rng: ChaCha8Rng,
    finished: bool,
    resigned_by: Option<u8>,
    /// Whether this game is allowed to resign (false in the resign-calibration
    /// fraction that always plays to the end).
    resign_enabled: bool,
    /// Score-stability: last seen `score_signature` and how many consecutive plies
    /// it has been unchanged.
    last_sig: Option<i64>,
    stable_plies: u32,
    /// Per-side (index = to_move-1) run of consecutive own-moves at/under the
    /// resignation threshold.
    resign_run: [u32; 2],
}

impl<G: BatchGame> Runner<G> {
    fn new(seed: u64, params: SearchParams) -> Self {
        let s0 = G::initial_state();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        // Decide up front (deterministically per game) whether this game may resign;
        // a `resign_playout_frac` slice always plays to the end for calibration.
        let resign_enabled = rng.gen::<f32>() >= params.resign_playout_frac;
        Self {
            search: GameSearch::new(s0.clone(), params),
            states: vec![s0],
            moves: Vec::new(),
            rng,
            finished: false,
            resigned_by: None,
            resign_enabled,
            last_sig: None,
            stable_plies: 0,
            resign_run: [0, 0],
        }
    }

    /// Finish the current move search: pick a move from the visit counts, advance
    /// the game, and either start the next move's search or finish the game.
    fn complete_move(&mut self, params: &SearchParams) {
        let counts = self.search.policy_counts();
        // The search's win-estimate for the side about to move (used to resign).
        let root_value = self.search.root_value();
        let sum: u32 = counts.iter().sum();
        let cur = self.states.last().unwrap();
        let mut policy = vec![0f32; G::ACTIONS];
        let action;
        if sum > 0 {
            for a in 0..G::ACTIONS {
                policy[a] = counts[a] as f32 / sum as f32;
            }
            let ply = self.states.len() as u32 - 1;
            action = if ply < params.temp_moves {
                sample_categorical(&policy, &mut self.rng)
            } else {
                argmax(&counts)
            };
        } else {
            // Degenerate (no PUCT visits at the root): uniform over candidates.
            let legal = G::candidate_actions(cur);
            let u = 1.0 / legal.len().max(1) as f32;
            for &a in &legal {
                policy[a] = u;
            }
            action = legal.first().copied().unwrap_or(0);
        }
        let to_move = G::to_move(cur);
        // Resignation (KataGo/AZ-style): if the side to move's search win-estimate
        // is hopeless for enough of its own moves in a row, it resigns instead of
        // playing on (e.g. nonsense invasions of the opponent's settled territory,
        // which is exactly the time-wasting we want to cut).
        if params.resign_consecutive > 0 && self.resign_enabled && self.resigned_by.is_none() {
            let si = (to_move - 1) as usize;
            if root_value <= params.resign_threshold {
                self.resign_run[si] += 1;
                self.resign_run[1 - si] = 0;
                if self.resign_run[si] >= params.resign_consecutive {
                    self.resigned_by = Some(to_move);
                }
            } else {
                self.resign_run[si] = 0;
            }
        }
        self.moves.push(MoveRecord { policy, action, to_move });
        // The chosen action came from visited (hence legal) children, so this is
        // legal; fall back to the first candidate if the degenerate branch hit.
        let next = G::apply_action(cur, action)
            .or_else(|| G::candidate_actions(cur).iter().find_map(|&a| G::apply_action(cur, a)))
            .expect("no legal move from a non-terminal state");
        self.states.push(next.clone());

        // Score-stability: once the area-score signature stops changing for
        // `settle_plies` plies in a row, the borders are settled (dame filled,
        // invasions refuted) — stop and let the caller adjudicate by area score
        // rather than grinding score-neutral filler moves out to the ply cap.
        if params.settle_plies > 0 {
            let sig = G::score_signature(&next);
            if sig.is_some() && sig == self.last_sig {
                self.stable_plies += 1;
            } else {
                self.stable_plies = 0;
                self.last_sig = sig;
            }
        }
        let settled = params.settle_plies > 0 && self.stable_plies >= params.settle_plies;
        let capped = params.max_ply > 0 && self.states.len() as u32 - 1 >= params.max_ply;
        let over = G::terminal_value(&next).is_some()
            || self.resigned_by.is_some()
            || settled
            || capped;
        if over {
            self.finished = true;
        } else {
            self.search = GameSearch::new(next, *params);
        }
    }
}

/// Run a chunk of `n_games` self-play games over `threads` workers, batching leaf
/// evaluations across all in-flight games. Returns the finished games plus stats.
#[allow(clippy::too_many_arguments)]
pub fn run_chunk<G: BatchGame>(
    model: &dyn BatchModel,
    n_games: usize,
    params: SearchParams,
    threads: usize,
    max_batch: usize,
    base_seed: u64,
    use_cache: bool,
) -> (Vec<GameRecord<G>>, ChunkStats) {
    let cache = Arc::new(Cache::new());
    let stats = Mutex::new(ChunkStats::default());
    let t0 = Instant::now();

    // Partition games across workers (contiguous ranges).
    let w = threads.max(1).min(n_games.max(1));
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let per = n_games.div_ceil(w);
    let mut start = 0;
    while start < n_games {
        let end = (start + per).min(n_games);
        ranges.push((start, end));
        start = end;
    }

    let (req_tx, req_rx) = channel::<Req>();
    let mut all: Vec<GameRecord<G>> = Vec::with_capacity(n_games);

    std::thread::scope(|scope| {
        let model_ref = model;
        let stats_ref = &stats;
        scope.spawn(move || inference_server::<G>(req_rx, model_ref, max_batch, stats_ref));

        let mut handles = Vec::new();
        for (lo, hi) in ranges {
            let req_tx = req_tx.clone();
            let cache = Arc::clone(&cache);
            let params = params;
            handles.push(scope.spawn(move || {
                worker::<G>(lo, hi, base_seed, params, &req_tx, &cache, use_cache)
            }));
        }
        drop(req_tx); // let the server detect shutdown once workers finish

        for h in handles {
            let (recs, s) = h.join().expect("worker panicked");
            all.extend(recs);
            let mut g = stats.lock().unwrap();
            g.nodes += s.nodes;
            g.evals += s.evals;
            g.cache_hits += s.cache_hits;
            g.terminal_sims += s.terminal_sims;
            g.gather_secs += s.gather_secs;
            g.wait_secs += s.wait_secs;
            g.apply_secs += s.apply_secs;
        }
    });

    let mut s = stats.into_inner().unwrap();
    s.elapsed = t0.elapsed().as_secs_f64();
    (all, s)
}

/// Per-worker accumulator (merged into ChunkStats; batch stats come from server).
#[derive(Default)]
struct WorkerStats {
    nodes: u64,
    evals: u64,
    cache_hits: u64,
    terminal_sims: u64,
    gather_secs: f64,
    wait_secs: f64,
    apply_secs: f64,
}

#[allow(clippy::too_many_arguments)]
fn worker<G: BatchGame>(
    lo: usize,
    hi: usize,
    base_seed: u64,
    params: SearchParams,
    req_tx: &Sender<Req>,
    cache: &Cache,
    use_cache: bool,
) -> (Vec<GameRecord<G>>, WorkerStats) {
    let mut runners: Vec<Runner<G>> =
        (lo..hi).map(|i| Runner::new(base_seed.wrapping_add(i as u64), params)).collect();
    let mut ws = WorkerStats::default();
    // One reply channel, reused for every wave this worker submits.
    let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel::<Vec<(f32, Vec<f32>)>>(1);

    loop {
        // Gather one pending leaf from every active game (resolving cache hits and
        // terminal/short searches inline) into a single wave.
        let t_gather = Instant::now();
        let mut wave_planes: Vec<f32> = Vec::new();
        let mut wave_meta: Vec<(usize, u128, G::State)> = Vec::new();
        let mut any_active = false;
        for (ri, r) in runners.iter_mut().enumerate() {
            if r.finished {
                continue;
            }
            any_active = true;
            loop {
                match r.search.next(&mut r.rng) {
                    Step::Done => {
                        r.complete_move(&params);
                        if r.finished {
                            break;
                        }
                        continue; // new move's search; keep pulling until it needs a leaf
                    }
                    Step::NeedEval(state) => {
                        let key = if use_cache { G::hash_key(&state) } else { 0 };
                        if use_cache {
                            if let Some(out) = cache.get(key) {
                                r.search.apply(&out, &mut r.rng);
                                ws.cache_hits += 1;
                                ws.nodes += 1;
                                continue; // resolved without GPU; pull the next leaf
                            }
                        }
                        let base = wave_planes.len();
                        wave_planes.resize(base + G::PLANES, 0.0);
                        G::encode_planes(&state, &mut wave_planes[base..]);
                        wave_meta.push((ri, key, state));
                        break;
                    }
                }
            }
        }

        ws.gather_secs += t_gather.elapsed().as_secs_f64();
        if !any_active {
            break;
        }
        if wave_meta.is_empty() {
            continue; // every active game resolved via cache this round
        }

        let t_wait = Instant::now();
        req_tx
            .send(Req { planes: wave_planes, count: wave_meta.len(), resp: resp_tx.clone() })
            .expect("inference server gone");
        let results = resp_rx.recv().expect("inference response dropped");
        ws.wait_secs += t_wait.elapsed().as_secs_f64();

        let t_apply = Instant::now();
        for ((ri, key, state), (value, priors)) in wave_meta.into_iter().zip(results) {
            let legal = G::candidate_actions(&state);
            let s: f32 = legal.iter().map(|&a| priors[a]).sum::<f32>().max(1e-12);
            let mut p = vec![0f32; G::ACTIONS];
            for &a in &legal {
                p[a] = priors[a] / s;
            }
            let out = Arc::new(ExpandOut { value, p, legal });
            if use_cache {
                cache.insert(key, Arc::clone(&out));
            }
            let r = &mut runners[ri];
            r.search.apply(&out, &mut r.rng);
            ws.evals += 1;
            ws.nodes += 1;
        }
        ws.apply_secs += t_apply.elapsed().as_secs_f64();
    }

    // Roll per-search terminal-sim counts into worker stats.
    for r in &runners {
        ws.terminal_sims += r.search.terminal_sims;
    }
    let recs = runners
        .into_iter()
        .map(|r| GameRecord { states: r.states, moves: r.moves, resigned_by: r.resigned_by })
        .collect();
    (recs, ws)
}

fn sample_categorical(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen::<f32>();
    let mut acc = 0.0;
    for (a, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return a;
        }
    }
    (0..probs.len()).rev().find(|&a| probs[a] > 0.0).unwrap_or(0)
}

fn argmax(counts: &[u32]) -> usize {
    let mut best = 0;
    for a in 1..counts.len() {
        if counts[a] > counts[best] {
            best = a;
        }
    }
    best
}

// ── Move-indexed search (compound-move games: Amazons, Arimaa) ─────────────────
//
// The flat `BatchGame` path above indexes everything by a small fixed `ACTIONS`.
// Compound-move games (an Amazons turn is queen-move + arrow-shot) have thousands
// of legal moves per position that don't fit a small flat index, so this parallel
// path works over the per-state *legal-move list*: a node's `p/n/w/children` are
// sized to its legal moves (computed once at expansion), and the model emits a
// `POLICY_DIM` policy that the game maps to a prior per move (e.g. a factored
// two-head product). It reuses the same wave protocol (`Req`), inference-server
// batching, and `ChunkStats`/`SearchParams` as the flat path. No eval cache and
// no eps-greedy/score-stability (Go-specific); root Dirichlet + resignation carry
// over. Search dynamics are otherwise identical to the flat MCTS.

/// A game whose moves don't fit a small fixed action index. The model output is a
/// `POLICY_DIM`-wide policy the game maps to per-move priors.
pub trait MoveGame: Send + Sync + 'static {
    type State: Clone + Send + Sync;
    type Move: Copy + Send + Sync;
    const PLANES: usize;
    const POLICY_DIM: usize;

    fn initial_state() -> Self::State;
    /// 1 or 2 — side to move (tags training examples).
    fn to_move(state: &Self::State) -> u8;
    /// Apply a (legal) move.
    fn apply(state: &Self::State, mv: &Self::Move) -> Self::State;
    /// Terminal value for the side to move (`None` = in progress).
    fn terminal_value(state: &Self::State) -> Option<f32>;
    /// Write the `PLANES` network-input floats for `state` into `out`.
    fn encode_planes(state: &Self::State, out: &mut [f32]);
    /// Generate the legal moves AND their priors in one pass, writing into the caller's
    /// reused `moves`/`priors` buffers (cleared first) to avoid per-call allocation.
    /// Priors are the game's combine (e.g. softmax-over-legal of summed two-head
    /// logits); fusing generation with prior computation lets the game compute head
    /// indices during move generation instead of recomputing them — the hot path.
    fn expand(
        state: &Self::State,
        policy: &[f32],
        moves: &mut Vec<Self::Move>,
        priors: &mut Vec<f32>,
    );
}

/// One move's training-relevant record: the legal moves searched, their PUCT visit
/// counts (the policy target, projected to the net's heads by the caller), and the
/// move actually played.
pub struct MvMoveRecord<M: MoveGame> {
    pub moves: Vec<M::Move>,
    pub visits: Vec<u32>,
    pub played: usize,
    pub to_move: u8,
}

/// A finished compound-move self-play game.
pub struct MvGameRecord<M: MoveGame> {
    pub states: Vec<M::State>,
    pub moves: Vec<MvMoveRecord<M>>,
    pub resigned_by: Option<u8>,
}

/// One child edge: the move, its prior, visit count, value sum, and child node
/// index (-1 = unexpanded). Storing all per-move data in ONE contiguous array (vs 5
/// separate Vecs) cuts a node's allocations 5×→1× and makes `select` cache-friendly.
struct MoveEntry<M: MoveGame> {
    mv: M::Move,
    p: f32,
    n: u32,
    w: f32,
    child: i32,
}

struct MvNode<M: MoveGame> {
    state: M::State,
    entries: Vec<MoveEntry<M>>,
    total_n: u32, // cached Σ n over entries, so `select` needn't re-sum each call
    /// Cached Σ w and Σ p-of-visited, for the same reason: first-play urgency
    /// needs both on every `select`.
    w_sum: f32,
    visited_p: f32,
    expanded: bool,
    terminal: bool,
    term_v: f32,
}

impl<M: MoveGame> MvNode<M> {
    fn new(state: M::State) -> Self {
        Self {
            state,
            entries: Vec::new(),
            total_n: 0,
            w_sum: 0.0,
            visited_p: 0.0,
            expanded: false,
            terminal: false,
            term_v: 0.0,
        }
    }
}

enum MvStep<M: MoveGame> {
    NeedEval(M::State),
    Done,
}

struct MvSearch<M: MoveGame> {
    arena: Vec<MvNode<M>>,
    params: SearchParams,
    pending_path: Vec<(usize, usize)>,
    pending_leaf: Option<usize>,
    pending_nodes_before: usize,
    sims: u32,
    no_growth: u32,
    dirichlet_done: bool,
    terminal_sims: u64,
}

impl<M: MoveGame> MvSearch<M> {
    fn new(root: M::State, params: SearchParams) -> Self {
        Self {
            arena: vec![MvNode::new(root)],
            params,
            pending_path: Vec::with_capacity(64),
            pending_leaf: None,
            pending_nodes_before: 0,
            sims: 0,
            no_growth: 0,
            dirichlet_done: false,
            terminal_sims: 0,
        }
    }

    fn budget_reached(&self) -> bool {
        if self.params.node_budget > 0 {
            self.arena.len() as u32 >= self.params.node_budget
                || self.no_growth >= SATURATION_PATIENCE
        } else {
            self.sims >= self.params.sims
        }
    }

    fn select(&self, node: usize) -> usize {
        let nd = &self.arena[node];
        let sqrt_total = ((nd.total_n + 1) as f32).sqrt();
        let c_puct = self.params.c_puct;
        // Unvisited children are scored against this node's own value rather
        // than a flat 0 (see `core::puct`). Both terms are maintained by
        // `backup`, so this stays a single pass over the entries even for games
        // with thousands of moves.
        let parent_q = if nd.total_n > 0 { nd.w_sum / nd.total_n as f32 } else { 0.0 };
        let fpu = crate::puct::fpu(parent_q, nd.visited_p);
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (a, e) in nd.entries.iter().enumerate() {
            let q = if e.n > 0 { e.w / e.n as f32 } else { fpu };
            let u = c_puct * e.p * sqrt_total / (1 + e.n) as f32;
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
        best
    }

    fn next(&mut self) -> MvStep<M> {
        loop {
            if self.budget_reached() {
                return MvStep::Done;
            }
            self.pending_path.clear();
            let nodes_before = self.arena.len();
            let mut node = 0usize;
            while self.arena[node].expanded && !self.arena[node].terminal {
                let a = self.select(node);
                let child = self.arena[node].entries[a].child;
                let next = if child >= 0 {
                    child as usize
                } else {
                    let ns = M::apply(&self.arena[node].state, &self.arena[node].entries[a].mv);
                    let idx = self.arena.len();
                    self.arena.push(MvNode::new(ns));
                    self.arena[node].entries[a].child = idx as i32;
                    idx
                };
                self.pending_path.push((node, a));
                node = next;
            }
            self.sims += 1;
            if let Some(tv) = self.terminal_at(node) {
                self.backup(tv);
                if self.arena.len() == nodes_before {
                    self.no_growth += 1;
                } else {
                    self.no_growth = 0;
                }
                self.terminal_sims += 1;
                continue;
            }
            self.pending_leaf = Some(node);
            self.pending_nodes_before = nodes_before;
            return MvStep::NeedEval(self.arena[node].state.clone());
        }
    }

    fn terminal_at(&mut self, node: usize) -> Option<f32> {
        if self.arena[node].terminal {
            return Some(self.arena[node].term_v);
        }
        if let Some(tv) = M::terminal_value(&self.arena[node].state) {
            self.arena[node].terminal = true;
            self.arena[node].term_v = tv;
            Some(tv)
        } else {
            None
        }
    }

    /// Expand the pending leaf from the freshly-generated `moves`+`priors` (passed by
    /// reference from reused worker scratch — one allocation here for the entries).
    fn apply(&mut self, value: f32, moves: &[M::Move], priors: &[f32], rng: &mut impl Rng) {
        let node = self.pending_leaf.take().expect("apply with no pending leaf");
        {
            let nd = &mut self.arena[node];
            nd.entries.clear();
            nd.entries.reserve(moves.len());
            for (mv, &p) in moves.iter().zip(priors) {
                nd.entries.push(MoveEntry { mv: *mv, p, n: 0, w: 0.0, child: -1 });
            }
            nd.total_n = 0;
            nd.w_sum = 0.0;
            nd.visited_p = 0.0;
            nd.expanded = true;
        }
        if node == 0 && self.params.dir_eps > 0.0 && !self.dirichlet_done {
            self.apply_dirichlet(rng);
            self.dirichlet_done = true;
        }
        self.backup(value);
        if self.arena.len() != self.pending_nodes_before {
            self.no_growth = 0;
        }
    }

    fn apply_dirichlet(&mut self, rng: &mut impl Rng) {
        let m = self.arena[0].entries.len();
        if m == 0 {
            return;
        }
        let gamma = Gamma::new(self.params.dir_alpha, 1.0).unwrap();
        let mut g: Vec<f32> = (0..m).map(|_| gamma.sample(rng) as f32).collect();
        let gsum: f32 = g.iter().sum::<f32>().max(1e-12);
        for x in &mut g {
            *x /= gsum;
        }
        let eps = self.params.dir_eps;
        for (e, &gi) in self.arena[0].entries.iter_mut().zip(g.iter()) {
            e.p = (1.0 - eps) * e.p + eps * gi;
        }
    }

    fn backup(&mut self, mut v: f32) {
        for &(n, a) in self.pending_path.iter().rev() {
            v = -v;
            let nd = &mut self.arena[n];
            let e = &mut nd.entries[a];
            if e.n == 0 {
                nd.visited_p += e.p; // first visit: this child's prior is now "tried"
            }
            e.n += 1;
            e.w += v;
            nd.w_sum += v;
            nd.total_n += 1;
        }
    }

    /// Visit counts at the root (the policy target, parallel to `root_moves`).
    fn policy_counts(&self) -> Vec<u32> {
        self.arena[0].entries.iter().map(|e| e.n).collect()
    }

    fn root_moves(&self) -> Vec<M::Move> {
        self.arena[0].entries.iter().map(|e| e.mv).collect()
    }

    /// State of the leaf awaiting evaluation (set by `next`, consumed by `apply`).
    fn pending_state(&self) -> &M::State {
        &self.arena[self.pending_leaf.expect("pending_state with no pending leaf")].state
    }

    /// Mean action-value at the root from the side-to-move's view, for resignation.
    fn root_value(&self) -> f32 {
        let root = &self.arena[0];
        if root.total_n == 0 {
            return 0.0;
        }
        let w: f32 = root.entries.iter().map(|e| e.w).sum();
        w / root.total_n as f32
    }
}

struct MvRunner<M: MoveGame> {
    search: MvSearch<M>,
    states: Vec<M::State>,
    moves: Vec<MvMoveRecord<M>>,
    rng: ChaCha8Rng,
    finished: bool,
    resigned_by: Option<u8>,
    resign_enabled: bool,
    resign_run: [u32; 2],
    /// Eval only: the side (1/2) that model A plays in this game (0 = self-play, one
    /// model). The leaf's model is A if `state.to_move == eval_a_side` else B.
    eval_a_side: u8,
}

impl<M: MoveGame> MvRunner<M> {
    fn new(seed: u64, params: SearchParams) -> Self {
        let s0 = M::initial_state();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let resign_enabled = rng.gen::<f32>() >= params.resign_playout_frac;
        Self {
            search: MvSearch::new(s0.clone(), params),
            states: vec![s0],
            moves: Vec::new(),
            rng,
            finished: false,
            resigned_by: None,
            resign_enabled,
            eval_a_side: 0,
            resign_run: [0, 0],
        }
    }

    fn complete_move(&mut self, params: &SearchParams) {
        let counts = self.search.policy_counts();
        let root_value = self.search.root_value();
        let sum: u32 = counts.iter().sum();
        let cur = self.states.last().unwrap();
        let to_move = M::to_move(cur);
        let moves: Vec<M::Move> = self.search.root_moves();

        let played = if moves.is_empty() {
            0 // terminal (no legal move) — shouldn't reach here, guarded below
        } else if sum > 0 {
            let ply = self.states.len() as u32 - 1;
            if ply < params.temp_moves {
                let probs: Vec<f32> = counts.iter().map(|&c| c as f32 / sum as f32).collect();
                sample_categorical(&probs, &mut self.rng)
            } else {
                argmax(&counts)
            }
        } else {
            0 // degenerate (no visits): first legal move
        };

        // Resignation (KataGo/AZ-style), with a calibration fraction that never resigns.
        if params.resign_consecutive > 0 && self.resign_enabled && self.resigned_by.is_none() {
            let si = (to_move - 1) as usize;
            if root_value <= params.resign_threshold {
                self.resign_run[si] += 1;
                self.resign_run[1 - si] = 0;
                if self.resign_run[si] >= params.resign_consecutive {
                    self.resigned_by = Some(to_move);
                }
            } else {
                self.resign_run[si] = 0;
            }
        }

        let next = M::apply(cur, &moves[played]);
        self.moves.push(MvMoveRecord { moves, visits: counts, played, to_move });
        self.states.push(next.clone());

        let capped = params.max_ply > 0 && self.states.len() as u32 - 1 >= params.max_ply;
        let over = M::terminal_value(&next).is_some() || self.resigned_by.is_some() || capped;
        if over {
            self.finished = true;
        } else {
            self.search = MvSearch::new(next, *params);
        }
    }
}

// ── Asynchronous pipelined leaf-batching ──────────────────────────────────────
//
// To keep the GPU saturated, games are *overcommitted* (n_games ≫ batch_size) and
// decoupled from workers via a work-queue. A pool of worker threads continuously
// descends games to their next leaf and pushes (game, planes) onto `eval`; the GPU
// thread pulls up to `batch_size` leaves (always a full batch in steady state,
// padded to a fixed shape so TRT never rebuilds), runs one forward, and pushes each
// result back as an `Apply` task. Workers pick up `Apply` tasks (expand + backup)
// and immediately descend the same game to its next leaf. No thread blocks waiting
// on a specific leaf — while some games' leaves are on the GPU, workers advance
// others — so the CPU keeps the eval queue full at the rate the GPU drains it.
// Each game still has exactly one outstanding leaf at a time, so search dynamics
// are identical to the sequential MCTS.

/// CPU work item: games to advance. `Apply` carries a *slice of the batch* —
/// several (game, value, row-offset) triples plus one shared ref to the whole
/// batch's policy logits — so the GPU thread never copies a per-game policy row.
///
/// Chunked rather than one message per game because each send to an idle worker
/// costs a park/unpark (~13 µs measured): at batch 256 that scatter cost more
/// than the forward it was delivering. One chunk per worker keeps every worker
/// busy while cutting the wakeups by the batch/worker ratio.
enum MvTask<M: MoveGame> {
    Seed(MvRunner<M>),
    Apply(Vec<(MvRunner<M>, f32, usize)>, Arc<Vec<f32>>),
}

/// Hand a finished batch back to the workers as at most `n_workers` messages.
fn mv_scatter<M: MoveGame>(
    work_tx: &crossbeam_channel::Sender<MvTask<M>>,
    runners: Vec<MvRunner<M>>,
    vals: &[f32],
    priors: Arc<Vec<f32>>,
    n_workers: usize,
) {
    let n = runners.len();
    if n == 0 {
        return;
    }
    // One move per runner (they're ~500 bytes each): drain straight into the
    // chunk that gets sent, rather than building a combined Vec and splitting it.
    let chunk = n.div_ceil(n_workers.max(1));
    let mut it = runners.into_iter().enumerate();
    loop {
        let mut entries: Vec<(MvRunner<M>, f32, usize)> = Vec::with_capacity(chunk);
        entries.extend(it.by_ref().take(chunk).map(|(i, r)| (r, vals[i], i)));
        if entries.is_empty() {
            return;
        }
        let _ = work_tx.send(MvTask::Apply(entries, Arc::clone(&priors)));
    }
}

/// Apply one chunk: each entry is an independent game, so the order within the
/// chunk is irrelevant to search behaviour.
#[allow(clippy::too_many_arguments)]
fn mv_apply_chunk<M: MoveGame>(
    entries: Vec<(MvRunner<M>, f32, usize)>,
    priors: &Arc<Vec<f32>>,
    params: &SearchParams,
    base_seed: u64,
    seed_ctr: &std::sync::atomic::AtomicU64,
    eval_tx: &crossbeam_channel::Sender<(MvRunner<M>, Vec<f32>)>,
    finished: &Mutex<Vec<MvGameRecord<M>>>,
    moves_buf: &mut Vec<M::Move>,
    prior_buf: &mut Vec<f32>,
    ws: &mut WorkerStats,
) {
    let pd = M::POLICY_DIM;
    for (r, v, off) in entries {
        let slice = &priors[off * pd..(off + 1) * pd];
        mv_advance(
            r, Some((v, slice)), params, base_seed, seed_ctr, eval_tx, finished, moves_buf,
            prior_buf, ws,
        );
    }
}

/// Advance one runner: apply the pending eval (if any), then descend to its next leaf
/// (push to `eval`). When a game ends, record it and immediately start a fresh game
/// on this thread — steady-state refill, so `concurrency` games stay in flight and no
/// worker ever idles. Shutdown is driven by the workers'/GPU's `recv_deadline`, not
/// by game exhaustion, so there's no tail.
#[allow(clippy::too_many_arguments)]
fn mv_advance<M: MoveGame>(
    mut r: MvRunner<M>,
    pending: Option<(f32, &[f32])>,
    params: &SearchParams,
    base_seed: u64,
    seed_ctr: &std::sync::atomic::AtomicU64,
    eval_tx: &crossbeam_channel::Sender<(MvRunner<M>, Vec<f32>)>,
    finished: &Mutex<Vec<MvGameRecord<M>>>,
    moves_buf: &mut Vec<M::Move>,
    prior_buf: &mut Vec<f32>,
    ws: &mut WorkerStats,
) {
    use std::sync::atomic::Ordering;
    if let Some((value, policy)) = pending {
        let t = Instant::now();
        let state = r.search.pending_state().clone();
        M::expand(&state, policy, moves_buf, prior_buf); // fused move-gen + priors (reused bufs)
        r.search.apply(value, moves_buf, prior_buf, &mut r.rng);
        ws.apply_secs += t.elapsed().as_secs_f64(); // expand (move-gen + priors) + backup
        ws.nodes += 1;
        ws.evals += 1;
    }
    loop {
        let t = Instant::now();
        match r.search.next() {
            MvStep::Done => {
                r.complete_move(params);
                if r.finished {
                    ws.gather_secs += t.elapsed().as_secs_f64();
                    finished.lock().unwrap().push(MvGameRecord {
                        states: std::mem::take(&mut r.states),
                        moves: std::mem::take(&mut r.moves),
                        resigned_by: r.resigned_by,
                    });
                    let id = seed_ctr.fetch_add(1, Ordering::Relaxed);
                    r = MvRunner::new(base_seed.wrapping_add(id), *params);
                }
            }
            MvStep::NeedEval(state) => {
                let mut planes = vec![0f32; M::PLANES];
                M::encode_planes(&state, &mut planes);
                ws.gather_secs += t.elapsed().as_secs_f64();
                let _ = eval_tx.send((r, planes));
                return;
            }
        }
    }
}

/// How many assembled batches may be queued ahead of the forward thread. One is
/// enough to overlap gather with compute; depth 2 was measured and made no
/// difference beyond run-to-run noise (holding more batches only starves the
/// queue that fills them).
const BATCH_PIPELINE_DEPTH: usize = 1;

/// One assembled batch: the games it belongs to, their planes padded to the fixed
/// shape, and how long the gatherer blocked for the first leaf.
type GatheredBatch<M> = (Vec<MvRunner<M>>, Vec<f32>, f64);

/// Assemble batches and hand them to the forward thread. Blocks for the first leaf,
/// then takes whatever else is already queued (forward-what's-ready, so the GPU never
/// waits on a straggler); with enough games in flight the queue already holds a full
/// batch. Ends when the leaf channel closes or `deadline` passes.
fn mv_gather_loop<M: MoveGame>(
    eval_rx: crossbeam_channel::Receiver<(MvRunner<M>, Vec<f32>)>,
    batch_tx: crossbeam_channel::Sender<GatheredBatch<M>>,
    batch_size: usize,
    deadline: Option<Instant>,
) {
    loop {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return; // explicit stop: recv won't fire while the queue is never empty
        }
        let mut runners = Vec::with_capacity(batch_size);
        let mut planes = Vec::with_capacity(batch_size * M::PLANES);
        let t_block = Instant::now();
        let first = match deadline {
            Some(d) => eval_rx.recv_deadline(d).ok(),
            None => eval_rx.recv().ok(),
        };
        match first {
            Some((r, pl)) => {
                planes.extend_from_slice(&pl);
                runners.push(r);
            }
            None => return,
        }
        while runners.len() < batch_size {
            match eval_rx.try_recv() {
                Ok((r, pl)) => {
                    planes.extend_from_slice(&pl);
                    runners.push(r);
                }
                Err(_) => break,
            }
        }
        let blocked = t_block.elapsed().as_secs_f64();
        planes.resize(batch_size * M::PLANES, 0.0); // pad a partial batch to fixed shape
        if batch_tx.send((runners, planes, blocked)).is_err() {
            return; // forward thread gone
        }
    }
}

/// GPU thread: block for the next leaf (waking at `deadline`), greedily pull up to
/// `batch_size` more (a full batch in steady state thanks to overcommit), pad to a
/// fixed shape, run ONE forward, then dispatch each result as an `Apply`. The whole
/// policy buffer is shared via `Arc` — no per-game copy on this hot thread.
#[allow(clippy::too_many_arguments)]
fn mv_gpu_loop<M: MoveGame>(
    eval_rx: crossbeam_channel::Receiver<(MvRunner<M>, Vec<f32>)>,
    work_tx: crossbeam_channel::Sender<MvTask<M>>,
    model: &dyn BatchModel,
    batch_size: usize,
    deadline: Instant,
    report_secs: f64,
    stats: &Mutex<ChunkStats>,
    n_workers: usize,
) {
    let mut local = ChunkStats::default();
    // Per-`report_secs` checkpoint so we can see batches stay full the WHOLE run, not
    // just on average. 0 disables.
    let loop_start = Instant::now();
    let mut win_batches = 0u64;
    let mut win_samples = 0u64;
    let mut win_t = Instant::now();
    // Pipeline the gather: a separate thread assembles batches while this one is
    // inside `forward`, so the GPU doesn't idle through the collect.
    let (batch_tx, batch_rx) = crossbeam_channel::bounded::<GatheredBatch<M>>(BATCH_PIPELINE_DEPTH);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            mv_gather_loop::<M>(eval_rx, batch_tx, batch_size, Some(deadline));
        });
        loop {
            let (runners, planes, blocked) = match batch_rx.recv() {
                Ok(b) => b,
                Err(_) => break, // gatherer stopped (deadline / channel closed)
            };
            local.gpu_block_secs += blocked;
            let useful = runners.len();
            let t_gpu = Instant::now();
            let (vals, priors) =
                model.forward(&planes, batch_size).expect("batched forward failed");
            local.gpu_secs += t_gpu.elapsed().as_secs_f64();
            local.batches += 1;
            local.batch_samples += useful as u64;
            win_batches += 1;
            win_samples += useful as u64;
            let t_disp = Instant::now();
            mv_scatter(&work_tx, runners, &vals, Arc::new(priors), n_workers);
            local.gpu_dispatch_secs += t_disp.elapsed().as_secs_f64();
            if report_secs > 0.0 {
                let dt = win_t.elapsed().as_secs_f64();
                if dt >= report_secs {
                    eprintln!(
                        "    [+{:.0}s] avg_batch {:.0}  {:.0} nodes/s  ({} batches)",
                        loop_start.elapsed().as_secs_f64(),
                        win_samples as f64 / win_batches as f64,
                        win_samples as f64 / dt,
                        win_batches,
                    );
                    win_batches = 0;
                    win_samples = 0;
                    win_t = Instant::now();
                }
            }
        }
    });
    let mut s = stats.lock().unwrap();
    s.gpu_secs += local.gpu_secs;
    s.gpu_block_secs += local.gpu_block_secs;
    s.gpu_dispatch_secs += local.gpu_dispatch_secs;
    s.batches += local.batches;
    s.batch_samples += local.batch_samples;
}

/// Steady-state self-play: keep `concurrency` compound-move games in flight (refill
/// finished ones with fresh games) for `max_secs` wall-clock, then stop. Returns
/// every game that *completed* within the window (in-flight games are discarded).
/// `concurrency` should be several × `batch_size` so the GPU always has a full batch
/// queued — the leaf-batching keeps it saturated; workers never idle on a leaf.
pub fn run_selfplay_moves<M: MoveGame>(
    model: &dyn BatchModel,
    concurrency: usize,
    params: SearchParams,
    threads: usize,
    batch_size: usize,
    base_seed: u64,
    max_secs: f64,
    report_secs: f64,
) -> (Vec<MvGameRecord<M>>, ChunkStats) {
    use std::sync::atomic::AtomicU64;
    let stats = Mutex::new(ChunkStats::default());
    let finished: Mutex<Vec<MvGameRecord<M>>> = Mutex::new(Vec::new());
    let seed_ctr = AtomicU64::new(concurrency as u64);
    let deadline = Instant::now() + std::time::Duration::from_secs_f64(max_secs);
    let n_workers = threads.max(1);
    let t0 = Instant::now();

    let (work_tx, work_rx) = crossbeam_channel::unbounded::<MvTask<M>>();
    let (eval_tx, eval_rx) = crossbeam_channel::unbounded::<(MvRunner<M>, Vec<f32>)>();
    for i in 0..concurrency {
        work_tx
            .send(MvTask::Seed(MvRunner::new(base_seed.wrapping_add(i as u64), params)))
            .unwrap();
    }

    std::thread::scope(|scope| {
        {
            let eval_rx = eval_rx.clone();
            let work_tx = work_tx.clone();
            let stats = &stats;
            scope.spawn(move || {
                mv_gpu_loop::<M>(
                    eval_rx, work_tx, model, batch_size, deadline, report_secs, stats, n_workers,
                )
            });
        }
        let mut handles = Vec::new();
        for _ in 0..n_workers {
            let work_rx = work_rx.clone();
            let eval_tx = eval_tx.clone();
            let (finished, seed_ctr, stats) = (&finished, &seed_ctr, &stats);
            handles.push(scope.spawn(move || {
                let mut ws = WorkerStats::default();
                let mut moves_buf: Vec<M::Move> = Vec::new();
                let mut prior_buf: Vec<f32> = Vec::new();
                loop {
                    // Explicit deadline check: recv_deadline alone won't fire while the
                    // work queue is never empty (continuous refill), so check directly.
                    if Instant::now() >= deadline {
                        break;
                    }
                    match work_rx.recv_deadline(deadline) {
                        Ok(MvTask::Seed(r)) => mv_advance(
                            r, None, &params, base_seed, seed_ctr, &eval_tx, finished,
                            &mut moves_buf, &mut prior_buf, &mut ws,
                        ),
                        Ok(MvTask::Apply(entries, pr)) => mv_apply_chunk(
                            entries, &pr, &params, base_seed, seed_ctr, &eval_tx, finished,
                            &mut moves_buf, &mut prior_buf, &mut ws,
                        ),
                        Err(_) => break,
                    }
                }
                let mut g = stats.lock().unwrap();
                g.nodes += ws.nodes;
                g.evals += ws.evals;
                g.gather_secs += ws.gather_secs;
                g.wait_secs += ws.wait_secs;
                g.apply_secs += ws.apply_secs;
            }));
        }
        drop(work_tx);
        drop(work_rx);
        drop(eval_tx);
        drop(eval_rx);
        for h in handles {
            h.join().expect("worker panicked");
        }
    });

    let mut s = stats.into_inner().unwrap();
    s.elapsed = t0.elapsed().as_secs_f64();
    (finished.into_inner().unwrap(), s)
}

/// Continuous self-play actor: keeps `concurrency` games in flight **forever** (refill
/// on completion), and every `flush_secs` calls `on_flush(completed_games)` with the
/// games that finished since the last flush — in-flight games keep running across
/// flushes (no tail, no discarded work). The GPU thread reads the model from `model`
/// each batch, so the caller can hot-swap weights on a learner snapshot by replacing
/// the `Arc` (e.g. inside `on_flush`). Never returns; the process is killed to stop.
/// `report_secs` > 0 logs throughput periodically (0 = quiet).
pub fn run_actor_continuous<M, F>(
    model: &Mutex<Arc<dyn BatchModel>>,
    concurrency: usize,
    params: SearchParams,
    threads: usize,
    batch_size: usize,
    base_seed: u64,
    flush_secs: f64,
    report_secs: f64,
    mut on_flush: F,
) where
    M: MoveGame,
    F: FnMut(Vec<MvGameRecord<M>>),
{
    use std::sync::atomic::AtomicU64;
    let finished: Mutex<Vec<MvGameRecord<M>>> = Mutex::new(Vec::new());
    let seed_ctr = AtomicU64::new(concurrency as u64);
    let n_workers = threads.max(1);

    let (work_tx, work_rx) = crossbeam_channel::unbounded::<MvTask<M>>();
    let (eval_tx, eval_rx) = crossbeam_channel::unbounded::<(MvRunner<M>, Vec<f32>)>();
    for i in 0..concurrency {
        work_tx
            .send(MvTask::Seed(MvRunner::new(base_seed.wrapping_add(i as u64), params)))
            .unwrap();
    }

    std::thread::scope(|scope| {
        // Gather thread: assembles the next batch while the forward thread is inside
        // `forward` (bounded at one batch ahead).
        let (batch_tx, batch_rx) =
            crossbeam_channel::bounded::<GatheredBatch<M>>(BATCH_PIPELINE_DEPTH);
        {
            let eval_rx = eval_rx.clone();
            scope.spawn(move || mv_gather_loop::<M>(eval_rx, batch_tx, batch_size, None));
        }
        // Forward thread: reads the *current* model each batch (hot-swappable).
        {
            let work_tx = work_tx.clone();
            scope.spawn(move || {
                let mut win_b = 0u64;
                let mut win_s = 0u64;
                let mut win_t = Instant::now();
                loop {
                    let (runners, planes, _blocked) = match batch_rx.recv() {
                        Ok(b) => b,
                        Err(_) => break,
                    };
                    let useful = runners.len();
                    let m = model.lock().unwrap().clone(); // current weights (cheap Arc clone)
                    let (vals, priors) = m.forward(&planes, batch_size).expect("batched forward failed");
                    drop(m);
                    mv_scatter(&work_tx, runners, &vals, Arc::new(priors), n_workers);
                    win_b += 1;
                    win_s += useful as u64;
                    if report_secs > 0.0 {
                        let dt = win_t.elapsed().as_secs_f64();
                        if dt >= report_secs {
                            eprintln!(
                                "    [actor gpu] avg_batch {:.0}  {:.0} nodes/s",
                                win_s as f64 / win_b as f64,
                                win_s as f64 / dt,
                            );
                            win_b = 0;
                            win_s = 0;
                            win_t = Instant::now();
                        }
                    }
                }
            });
        }
        // Worker pool: advance games forever (blocking recv; refill on completion).
        for _ in 0..n_workers {
            let work_rx = work_rx.clone();
            let eval_tx = eval_tx.clone();
            let (finished, seed_ctr) = (&finished, &seed_ctr);
            scope.spawn(move || {
                let mut ws = WorkerStats::default();
                let mut moves_buf: Vec<M::Move> = Vec::new();
                let mut prior_buf: Vec<f32> = Vec::new();
                loop {
                    match work_rx.recv() {
                        Ok(MvTask::Seed(r)) => mv_advance(
                            r, None, &params, base_seed, seed_ctr, &eval_tx, finished,
                            &mut moves_buf, &mut prior_buf, &mut ws,
                        ),
                        Ok(MvTask::Apply(entries, pr)) => mv_apply_chunk(
                            entries, &pr, &params, base_seed, seed_ctr, &eval_tx, finished,
                            &mut moves_buf, &mut prior_buf, &mut ws,
                        ),
                        Err(_) => break,
                    }
                }
            });
        }
        drop(work_tx);
        drop(work_rx);
        drop(eval_tx);
        drop(eval_rx);

        // This (caller) thread: flush completed games every `flush_secs`, forever.
        loop {
            std::thread::sleep(std::time::Duration::from_secs_f64(flush_secs));
            let games = std::mem::take(&mut *finished.lock().unwrap());
            on_flush(games);
        }
    });
}

// ── Two-model evaluation (same pipeline as self-play, A vs B) ──────────────────
//
// A head-to-head match A-vs-B reusing the exact async leaf-batching engine — self-
// play is the special case A == B. `n_games` games run to terminal (no resign);
// game i has model A playing side `1 + i%2` (alternate colours). Each leaf is routed
// to the model controlling its side-to-move; the GPU thread partitions the batch by
// model and runs one forward per model, then scatters. Returns A's score in [0,1].

/// Advance one eval runner: apply the pending eval, descend to the next leaf and send
/// it tagged with the model that controls its side-to-move; when the game ends, score
/// it for A (no refill). Sets `done` when the last game finishes.
#[allow(clippy::too_many_arguments)]
fn eval_advance<M: MoveGame>(
    mut r: MvRunner<M>,
    pending: Option<(f32, &[f32])>,
    params: &SearchParams,
    eval_tx: &crossbeam_channel::Sender<(MvRunner<M>, Vec<f32>, usize)>,
    score_x2: &std::sync::atomic::AtomicU64, // Σ 2×(A's game score), so win=2 draw=1 loss=0
    remaining: &std::sync::atomic::AtomicUsize,
    done: &std::sync::atomic::AtomicBool,
    moves_buf: &mut Vec<M::Move>,
    prior_buf: &mut Vec<f32>,
) {
    use std::sync::atomic::Ordering;
    if let Some((value, policy)) = pending {
        let state = r.search.pending_state().clone();
        M::expand(&state, policy, moves_buf, prior_buf);
        r.search.apply(value, moves_buf, prior_buf, &mut r.rng);
    }
    loop {
        match r.search.next() {
            MvStep::Done => {
                r.complete_move(params);
                if r.finished {
                    // Score for A: a finished game's last state is terminal (side to
                    // move lost); A won iff that loser isn't A's side. Ply-cap = draw.
                    let final_state = r.states.last().unwrap();
                    let s2: u64 = match M::terminal_value(final_state) {
                        Some(_) => {
                            let loser = M::to_move(final_state);
                            if loser != r.eval_a_side { 2 } else { 0 }
                        }
                        None => 1, // hit max_ply: draw
                    };
                    score_x2.fetch_add(s2, Ordering::AcqRel);
                    if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                        done.store(true, Ordering::Release);
                    }
                    return;
                }
            }
            MvStep::NeedEval(state) => {
                let model_id = if M::to_move(&state) == r.eval_a_side { 0 } else { 1 };
                let mut planes = vec![0f32; M::PLANES];
                M::encode_planes(&state, &mut planes);
                let _ = eval_tx.send((r, planes, model_id));
                return;
            }
        }
    }
}

/// GPU thread for eval: collect leaves, **partition by model id**, run one forward
/// per model on its sub-batch, and scatter each result as an `Apply`. (For 1 model
/// this is just self-play.) Exits when `done` and the queue has drained.
fn eval_gpu_loop<M: MoveGame>(
    eval_rx: crossbeam_channel::Receiver<(MvRunner<M>, Vec<f32>, usize)>,
    work_tx: crossbeam_channel::Sender<MvTask<M>>,
    models: &[&dyn BatchModel],
    batch_size: usize,
    done: &std::sync::atomic::AtomicBool,
    n_workers: usize,
) {
    use std::sync::atomic::Ordering;
    let timeout = std::time::Duration::from_millis(20);
    loop {
        // Gather up to batch_size leaves (block for the first, then drain).
        let mut leaves: Vec<(MvRunner<M>, Vec<f32>, usize)> = Vec::with_capacity(batch_size);
        match eval_rx.recv_timeout(timeout) {
            Ok(x) => leaves.push(x),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if done.load(Ordering::Acquire) {
                    break;
                }
                continue;
            }
            Err(_) => break,
        }
        while leaves.len() < batch_size {
            match eval_rx.try_recv() {
                Ok(x) => leaves.push(x),
                Err(_) => break,
            }
        }
        // Partition leaf indices by model, forward each model on its sub-batch.
        let mut results: Vec<Option<(f32, std::sync::Arc<Vec<f32>>, usize)>> =
            (0..leaves.len()).map(|_| None).collect();
        for (m, model) in models.iter().enumerate() {
            let idxs: Vec<usize> = (0..leaves.len()).filter(|&i| leaves[i].2 == m).collect();
            if idxs.is_empty() {
                continue;
            }
            let mut planes = Vec::with_capacity(idxs.len() * M::PLANES);
            for &i in &idxs {
                planes.extend_from_slice(&leaves[i].1);
            }
            let (vals, priors) = model.forward(&planes, idxs.len()).expect("eval forward failed");
            let priors = std::sync::Arc::new(priors);
            for (slot, &i) in idxs.iter().enumerate() {
                results[i] = Some((vals[slot], std::sync::Arc::clone(&priors), slot));
            }
        }
        // Chunked scatter, as in self-play: one message per worker rather than per
        // leaf. Each chunk's entries share a `priors` Arc, so group by model first.
        let mut by_model: HashMap<usize, (Vec<(MvRunner<M>, f32, usize)>, Arc<Vec<f32>>)> =
            HashMap::new();
        for ((r, _planes, mid), res) in leaves.into_iter().zip(results) {
            let (value, priors, off) = res.unwrap();
            by_model.entry(mid).or_insert_with(|| (Vec::new(), priors)).0.push((r, value, off));
        }
        for (_mid, (entries, priors)) in by_model {
            let chunk = entries.len().div_ceil(n_workers.max(1));
            let mut entries = entries;
            while !entries.is_empty() {
                let tail = entries.split_off(entries.len().saturating_sub(chunk));
                let _ = work_tx.send(MvTask::Apply(tail, Arc::clone(&priors)));
            }
        }
    }
}

/// Run an `n_games` A-vs-B match on the async pipeline; return A's score in [0,1].
/// `model_a`/`model_b` may be the same model (self-play). Plays to terminal (no
/// resignation) with `params` (use a small `temp_moves` for opening diversity).
pub fn run_eval_moves<M: MoveGame>(
    model_a: &dyn BatchModel,
    model_b: &dyn BatchModel,
    n_games: usize,
    params: SearchParams,
    threads: usize,
    batch_size: usize,
    base_seed: u64,
) -> f64 {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    let score_x2 = AtomicU64::new(0);
    let remaining = AtomicUsize::new(n_games);
    let done = AtomicBool::new(false);
    let n_workers = threads.max(1);
    let pd = M::POLICY_DIM;
    let models: [&dyn BatchModel; 2] = [model_a, model_b];

    let (work_tx, work_rx) = crossbeam_channel::unbounded::<MvTask<M>>();
    let (eval_tx, eval_rx) = crossbeam_channel::unbounded::<(MvRunner<M>, Vec<f32>, usize)>();
    for i in 0..n_games {
        let mut r = MvRunner::new(base_seed.wrapping_add(i as u64), params);
        r.eval_a_side = 1 + (i % 2) as u8; // alternate which colour A plays
        work_tx.send(MvTask::Seed(r)).unwrap();
    }

    std::thread::scope(|scope| {
        {
            let eval_rx = eval_rx.clone();
            let work_tx = work_tx.clone();
            let (models, done) = (&models, &done);
            scope.spawn(move || eval_gpu_loop::<M>(eval_rx, work_tx, models, batch_size, done, n_workers));
        }
        for _ in 0..n_workers {
            let work_rx = work_rx.clone();
            let eval_tx = eval_tx.clone();
            let (score_x2, remaining, done) = (&score_x2, &remaining, &done);
            scope.spawn(move || {
                let mut moves_buf: Vec<M::Move> = Vec::new();
                let mut prior_buf: Vec<f32> = Vec::new();
                let timeout = std::time::Duration::from_millis(20);
                loop {
                    match work_rx.recv_timeout(timeout) {
                        Ok(MvTask::Seed(r)) => eval_advance(
                            r, None, &params, &eval_tx, score_x2, remaining, done,
                            &mut moves_buf, &mut prior_buf,
                        ),
                        Ok(MvTask::Apply(entries, pr)) => {
                            for (r, v, off) in entries {
                                let slice = &pr[off * pd..(off + 1) * pd];
                                eval_advance(
                                    r, Some((v, slice)), &params, &eval_tx, score_x2, remaining,
                                    done, &mut moves_buf, &mut prior_buf,
                                );
                            }
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            if done.load(Ordering::Acquire) {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            });
        }
        drop(work_tx);
        drop(work_rx);
        drop(eval_tx);
        drop(eval_rx);
    });

    score_x2.load(Ordering::Acquire) as f64 / (2.0 * n_games as f64)
}
