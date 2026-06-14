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
    /// Hard ply cap for a self-play game (0 = no cap).
    pub max_ply: u32,
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
}

impl ChunkStats {
    pub fn avg_batch(&self) -> f64 {
        if self.batches > 0 { self.batch_samples as f64 / self.batches as f64 } else { 0.0 }
    }
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.evals + self.cache_hits;
        if total > 0 { self.cache_hits as f64 / total as f64 } else { 0.0 }
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
        let (vals, priors) = model.forward(&planes, total).expect("batched forward failed");
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
        let total: u32 = nd.n.iter().sum();
        let sqrt_total = ((total + 1) as f32).sqrt();
        let c_puct = self.params.c_puct;
        let mut best = usize::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for a in 0..G::ACTIONS {
            if !nd.legal[a] {
                continue;
            }
            let q = if nd.n[a] > 0 { nd.w[a] / nd.n[a] as f32 } else { 0.0 };
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
                let was_random =
                    self.params.explore_eps > 0.0 && rng.gen::<f32>() < self.params.explore_eps;
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
}

impl<G: BatchGame> Runner<G> {
    fn new(seed: u64, params: SearchParams) -> Self {
        let s0 = G::initial_state();
        Self {
            search: GameSearch::new(s0.clone(), params),
            states: vec![s0],
            moves: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            finished: false,
        }
    }

    /// Finish the current move search: pick a move from the visit counts, advance
    /// the game, and either start the next move's search or finish the game.
    fn complete_move(&mut self, params: &SearchParams) {
        let counts = self.search.policy_counts();
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
        self.moves.push(MoveRecord { policy, action, to_move });
        // The chosen action came from visited (hence legal) children, so this is
        // legal; fall back to the first candidate if the degenerate branch hit.
        let next = G::apply_action(cur, action)
            .or_else(|| G::candidate_actions(cur).iter().find_map(|&a| G::apply_action(cur, a)))
            .expect("no legal move from a non-terminal state");
        self.states.push(next.clone());

        let over = G::terminal_value(&next).is_some()
            || (params.max_ply > 0 && self.states.len() as u32 - 1 >= params.max_ply);
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

        if !any_active {
            break;
        }
        if wave_meta.is_empty() {
            continue; // every active game resolved via cache this round
        }

        req_tx
            .send(Req { planes: wave_planes, count: wave_meta.len(), resp: resp_tx.clone() })
            .expect("inference server gone");
        let results = resp_rx.recv().expect("inference response dropped");

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
    }

    // Roll per-search terminal-sim counts into worker stats.
    for r in &runners {
        ws.terminal_sims += r.search.terminal_sims;
    }
    let recs = runners
        .into_iter()
        .map(|r| GameRecord { states: r.states, moves: r.moves })
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
