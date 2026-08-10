//! MCTS in LGS (latent game state) space with on-demand reification.
//!
//! This implements the search described in docs/algorithm.md. All neural
//! network access goes through [`ModelClient`]; there is no tonic/tokio-runtime
//! coupling here, so the algorithm stays unit-testable with a mock client.
//!
//! Shape of one simulation:
//!   1. **Selection** — descend from the root by PUCT until reaching a frontier
//!      node (unexpanded, or a not-yet-reified child of a reified node).
//!   2. **Reification** — if the frontier is a latent child of a *reified*
//!      (concrete) node, decode the edge into a concrete move with `reify`,
//!      apply it to the concrete game, and re-`encode` the result. This is what
//!      keeps the root's chosen edge a real, legal move.
//!   3. **Expansion** — `policy` gives up to B (prior, action-vector, next-LGS)
//!      children; the leaf value comes from `eval` (or the exact game result at
//!      terminal nodes).
//!   4. **Backup** — negamax the leaf value up the visited path.
//!
//! Selection is **KR-UCT** (kernel-regression UCT, docs/algorithm.md step 5):
//! value and visit counts are shared across actions via a Gaussian kernel over
//! their action vectors, so similar actions inform each other. See
//! `kr_uct_select`. (σ → 0 recovers PUCT.)
//!
//! Still deferred (see docs/algorithm.md "open questions"): noisy reified
//! siblings + progressive widening, which matter for large/continuous action
//! spaces (Amazons/Arimaa). For Connect 4's 7 discrete columns the kernel over
//! the column children already gives the KR-UCT smoothing; each edge is reified
//! once.

use bytes::Bytes;

use crate::game::Game;
use crate::latent_state::LatentGameState;
use crate::model::ModelClient;

/// Search configuration. Field list will grow as the algorithm solidifies.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Number of simulations (tree traversals) to run, when `move_millis` is 0.
    pub simulations: u32,
    /// Per-move time budget in milliseconds. If > 0, run uncapped rollouts until
    /// the budget elapses instead of a fixed `simulations` count.
    pub move_millis: u64,
    /// B — number of (action, next_lgs) candidates produced by the policy head
    /// per expansion.
    pub policy_top_b: u32,
    /// K — visits between reifications at the same node. Reserved for the
    /// progressive-widening / sibling schedule (not used yet).
    pub reification_interval: u32,
    /// Multiplier for exploration noise on reified siblings (not used yet).
    pub sibling_noise_base: f64,
    /// Exploration constant (PUCT term, now over kernel-weighted counts).
    pub exploration_c: f64,
    /// KR-UCT kernel bandwidth σ over action vectors: how much value/visits are
    /// shared between similar actions. σ → 0 recovers plain PUCT.
    pub kernel_bandwidth: f64,
    /// Optional RNG seed for deterministic tests.
    pub seed: Option<u64>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 160,
            move_millis: 0,
            policy_top_b: 8,
            reification_interval: 8,
            sibling_noise_base: 0.25,
            exploration_c: 1.4,
            kernel_bandwidth: 0.75,
            seed: None,
        }
    }
}

#[derive(Debug)]
pub struct SearchResult<G: Game> {
    /// Concrete move to play (legal in `root_state`).
    pub chosen: G::Move,
    /// Root value estimate for the side to move, in [-1, 1].
    pub root_value: f64,
    /// Total simulations actually run.
    pub visits: u32,
    /// Number of edges reified into concrete moves during the search.
    pub reifications: u32,
}

// ── Tree ────────────────────────────────────────────────────────────────────

struct Edge {
    prior: f64,
    action_vector: Bytes,
    /// Policy state-head prediction of the child LGS (before reification).
    predicted_lgs: LatentGameState,
    child: usize,
    /// Concrete move, set once this edge is reified.
    mv: Option<()>, // presence flag; the move lives on the child node
}

struct Node<G: Game> {
    lgs: LatentGameState,
    /// Concrete state — `Some` once the node is reified. The root is reified.
    state: Option<G::State>,
    /// Move that reached this node from its parent (set on reification).
    incoming_move: Option<G::Move>,
    terminal: bool,
    /// Terminal value for the player to move at this node.
    terminal_value: f64,
    visits: u32,
    value_sum: f64,
    expanded: bool,
    edges: Vec<Edge>,
}

impl<G: Game> Node<G> {
    fn latent(lgs: LatentGameState) -> Self {
        Self {
            lgs,
            state: None,
            incoming_move: None,
            terminal: false,
            terminal_value: 0.0,
            visits: 0,
            value_sum: 0.0,
            expanded: false,
            edges: Vec::new(),
        }
    }
    fn reified(lgs: LatentGameState, state: G::State) -> Self {
        let mut n = Self::latent(lgs);
        n.state = Some(state);
        n
    }
}

/// Decode an opaque action-vector into a point for the KR-UCT kernel. Action
/// vectors are little-endian f32 (what the model's action head emits).
fn action_point(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Gaussian kernel exp(-||a-b||² / 2σ²) between two action points.
fn kernel(a: &[f32], b: &[f32], sigma: f64) -> f64 {
    let n = a.len().min(b.len());
    let mut sq = 0.0f64;
    for i in 0..n {
        let d = (a[i] - b[i]) as f64;
        sq += d * d;
    }
    (-sq / (2.0 * sigma * sigma)).exp()
}

// ── Search ──────────────────────────────────────────────────────────────────

/// Run a LGS-space MCTS from `root_state` and return the chosen concrete move.
pub async fn search<G: Game, M: ModelClient<G>>(
    model: &M,
    root_state: &G::State,
    config: &SearchConfig,
) -> anyhow::Result<SearchResult<G>> {
    let root_lgs = model.encode(root_state).await?;
    let mut tree: Vec<Node<G>> = vec![Node::reified(root_lgs, root_state.clone())];
    let mut reifications = 0u32;

    if config.move_millis > 0 {
        // Uncapped rollouts until the per-move time budget elapses (with a
        // generous hard cap to bound tree memory).
        const HARD_CAP: u32 = 2_000_000;
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_millis(config.move_millis);
        let mut n = 0u32;
        while n < HARD_CAP && std::time::Instant::now() < deadline {
            simulate(model, &mut tree, config, &mut reifications).await?;
            n += 1;
        }
    } else {
        for _ in 0..config.simulations.max(1) {
            simulate(model, &mut tree, config, &mut reifications).await?;
        }
    }

    // Choose the most-visited reified root edge.
    let root = &tree[0];
    let mut best: Option<usize> = None;
    let mut best_visits = -1i64;
    for (i, e) in root.edges.iter().enumerate() {
        if e.mv.is_none() {
            continue; // only reified edges have a concrete move
        }
        let v = tree[e.child].visits as i64;
        if v > best_visits {
            best_visits = v;
            best = Some(i);
        }
    }

    let chosen = match best.and_then(|i| tree[root.edges[i].child].incoming_move.clone()) {
        Some(mv) if is_legal::<G>(root_state, &mv) => mv,
        // No reified edge (e.g. tiny budget) or an illegal decode: fall back to
        // the first legal move so we always return something playable.
        _ => first_legal::<G>(root_state)?,
    };

    let root_value = if tree[0].visits > 0 {
        tree[0].value_sum / tree[0].visits as f64
    } else {
        0.0
    };

    Ok(SearchResult {
        chosen,
        root_value,
        visits: tree[0].visits,
        reifications,
    })
}

async fn simulate<G: Game, M: ModelClient<G>>(
    model: &M,
    tree: &mut Vec<Node<G>>,
    config: &SearchConfig,
    reifications: &mut u32,
) -> anyhow::Result<()> {
    // 1. Selection — descend by PUCT to a frontier node, recording the path
    //    and the (parent, edge) that led to the frontier.
    let mut path = vec![0usize];
    let mut cur = 0usize;
    let mut last: Option<(usize, usize)> = None; // (parent_idx, edge_idx)
    loop {
        if tree[cur].terminal || !tree[cur].expanded {
            break;
        }
        let Some(edge_idx) = kr_uct_select(tree, cur, config) else {
            break; // expanded but childless — treat as leaf
        };
        let child = tree[cur].edges[edge_idx].child;
        // Stop at a latent child of a reified node: it is the reification
        // frontier for this simulation.
        let stop = tree[child].state.is_none() && tree[cur].state.is_some();
        last = Some((cur, edge_idx));
        path.push(child);
        cur = child;
        if stop {
            break;
        }
    }

    // 2. Reify the frontier if it is a latent child of a reified parent.
    if tree[cur].state.is_none() {
        if let Some((parent, edge_idx)) = last {
            if tree[parent].state.is_some() {
                reify_edge::<G, M>(model, tree, parent, edge_idx, cur).await?;
                *reifications += 1;
            }
        }
    }

    // 3. Frontier value: terminal result, or expand (policy) + eval.
    let value = if tree[cur].terminal {
        tree[cur].terminal_value
    } else if tree[cur].expanded {
        // Re-selected an already-expanded leaf (e.g. childless): just re-eval.
        model.eval(&tree[cur].lgs).await?
    } else {
        expand::<G, M>(model, tree, cur, config.policy_top_b).await?
    };

    // 4. Backup (negamax): players alternate each ply, so flip sign by depth.
    for (depth_from_leaf, &idx) in path.iter().rev().enumerate() {
        let sign = if depth_from_leaf % 2 == 0 { 1.0 } else { -1.0 };
        tree[idx].visits += 1;
        tree[idx].value_sum += value * sign;
    }
    Ok(())
}

/// Decode edge `edge_idx` of reified node `parent` into a concrete move, apply
/// it, and re-encode — turning the latent child `child` into a concrete node.
async fn reify_edge<G: Game, M: ModelClient<G>>(
    model: &M,
    tree: &mut Vec<Node<G>>,
    parent: usize,
    edge_idx: usize,
    child: usize,
) -> anyhow::Result<()> {
    let parent_state = tree[parent].state.clone().expect("reified parent has state");
    let parent_lgs = tree[parent].lgs.clone();
    let to_lgs = tree[parent].edges[edge_idx].predicted_lgs.clone();
    let seed = tree[parent].edges[edge_idx].action_vector.clone();

    let mut mv = model
        .reify(&parent_state, &parent_lgs, &to_lgs, Some(&seed), None)
        .await?;
    if !is_legal::<G>(&parent_state, &mv) {
        // A bad decode shouldn't poison the search; substitute a legal move.
        mv = first_legal::<G>(&parent_state)?;
    }

    let child_state = G::apply(&parent_state, &mv);
    let outcome = G::outcome(&child_state);
    let child_lgs = model.encode(&child_state).await?;

    let node = &mut tree[child];
    node.lgs = child_lgs;
    node.terminal = outcome.is_terminal();
    node.terminal_value = if node.terminal {
        outcome.value_for(G::to_move(&child_state))
    } else {
        0.0
    };
    node.incoming_move = Some(mv);
    node.state = Some(child_state);
    tree[parent].edges[edge_idx].mv = Some(());
    Ok(())
}

/// Expand `node` with up to `top_b` policy children and return its leaf value.
async fn expand<G: Game, M: ModelClient<G>>(
    model: &M,
    tree: &mut Vec<Node<G>>,
    node: usize,
    top_b: u32,
) -> anyhow::Result<f64> {
    let lgs = tree[node].lgs.clone();
    let candidates = model.policy(&lgs, top_b).await?;
    for c in candidates {
        let child_idx = tree.len();
        tree.push(Node::latent(c.next_lgs.clone()));
        tree[node].edges.push(Edge {
            prior: c.prior,
            action_vector: c.action_vector,
            predicted_lgs: c.next_lgs,
            child: child_idx,
            mv: None,
        });
    }
    tree[node].expanded = true;
    Ok(model.eval(&lgs).await?)
}

/// KR-UCT child selection: kernel-regression UCT (Yee et al. 2016). Value and
/// visit counts are shared across actions by a Gaussian kernel over their action
/// vectors, so similar actions inform each other (and unvisited actions inherit
/// a value estimate from their neighbours). With σ → 0 the kernel collapses to
/// the identity and this is exactly PUCT.
///
///   q_b      = child b's mean value, from the parent's perspective
///   W(a)     = Σ_b k(a,b) · n_b                      (kernel-weighted count)
///   Q_kr(a)  = Σ_b k(a,b) · n_b · q_b / W(a)         (kernel-regressed value)
///   score(a) = Q_kr(a) + c · prior_a · √N / (1 + W(a))
fn kr_uct_select<G: Game>(tree: &[Node<G>], node_idx: usize, config: &SearchConfig) -> Option<usize> {
    let node = &tree[node_idx];
    let m = node.edges.len();
    if m == 0 {
        return None;
    }
    let pts: Vec<Vec<f32>> = node.edges.iter().map(|e| action_point(&e.action_vector)).collect();
    let n: Vec<f64> = node.edges.iter().map(|e| tree[e.child].visits as f64).collect();
    let q: Vec<f64> = node
        .edges
        .iter()
        .map(|e| {
            let c = &tree[e.child];
            if c.visits > 0 {
                -(c.value_sum / c.visits as f64) // child value is opponent-perspective
            } else {
                0.0
            }
        })
        .collect();

    let sigma = config.kernel_bandwidth.max(1e-6);
    let total = node.visits as f64;
    // Fallback for an action with no kernel-weighted visits to regress from:
    // this node's own value rather than a flat 0 (see `crate::puct`). Normally
    // the kernel supplies a neighbour's estimate and this never fires; it
    // matters as σ → 0, where KR-UCT collapses to PUCT and the flat 0 becomes
    // the same trap it is there. At a node's first selection every action hits
    // this branch equally, so the choice is still the prior's — unchanged.
    let visited_prior: f32 = node
        .edges
        .iter()
        .enumerate()
        .filter(|(b, _)| n[*b] > 0.0)
        .map(|(_, e)| e.prior as f32)
        .sum();
    let parent_q = if node.visits > 0 { node.value_sum / node.visits as f64 } else { 0.0 };
    let fpu = crate::puct::fpu(parent_q as f32, visited_prior) as f64;
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for a in 0..m {
        let mut w = 0.0;
        let mut num = 0.0;
        for b in 0..m {
            let k = kernel(&pts[a], &pts[b], sigma);
            w += k * n[b];
            num += k * n[b] * q[b];
        }
        let q_kr = if w > 0.0 { num / w } else { fpu };
        let u = config.exploration_c * node.edges[a].prior * total.sqrt() / (1.0 + w);
        let score = q_kr + u;
        if score > best_score {
            best_score = score;
            best = Some(a);
        }
    }
    best
}

fn is_legal<G: Game>(state: &G::State, mv: &G::Move) -> bool {
    let target = G::encode_move(mv);
    G::legal_moves(state).iter().any(|m| G::encode_move(m) == target)
}

fn first_legal<G: Game>(state: &G::State) -> anyhow::Result<G::Move> {
    G::legal_moves(state)
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no legal moves from a non-terminal state"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Outcome, Player};
    use crate::model::PolicyCandidate;
    use async_trait::async_trait;

    // Tiny alternating game: players remove 1 or 2 from a counter; the player
    // who faces an empty counter loses. Enough to drive the search.
    #[derive(Clone, Debug)]
    struct Countdown;
    #[derive(Clone, Debug)]
    struct State {
        remaining: u8,
        ply: u32,
    }

    impl Game for Countdown {
        type State = State;
        type Move = u8;
        fn id() -> u32 {
            99
        }
        fn initial_state() -> State {
            State { remaining: 7, ply: 0 }
        }
        fn to_move(s: &State) -> Player {
            if s.ply % 2 == 0 { Player::First } else { Player::Second }
        }
        fn outcome(s: &State) -> Outcome {
            if s.remaining == 0 {
                // The player to move cannot move and loses.
                match Self::to_move(s) {
                    Player::First => Outcome::SecondWins,
                    Player::Second => Outcome::FirstWins,
                }
            } else {
                Outcome::InProgress
            }
        }
        fn legal_moves(s: &State) -> Vec<u8> {
            (1..=2).filter(|&t| t <= s.remaining).collect()
        }
        fn apply(s: &State, mv: &u8) -> State {
            State { remaining: s.remaining - mv, ply: s.ply + 1 }
        }
        fn ply(s: &State) -> u32 {
            s.ply
        }
        fn to_features(_: &State) -> Vec<u8> {
            vec![]
        }
        fn encode_state(s: &State) -> Vec<u8> {
            let mut v = vec![s.remaining];
            v.extend_from_slice(&s.ply.to_le_bytes());
            v
        }
        fn decode_state(b: &[u8]) -> anyhow::Result<State> {
            Ok(State {
                remaining: b[0],
                ply: u32::from_le_bytes([b[1], b[2], b[3], b[4]]),
            })
        }
        fn encode_move(m: &u8) -> Vec<u8> {
            vec![*m]
        }
        fn decode_move(b: &[u8]) -> anyhow::Result<u8> {
            Ok(b[0])
        }
    }

    // Coherent mock: LGS payload carries the concrete state bytes, so policy can
    // enumerate real children and reify can decode the action seed to a move.
    struct MockModel;

    fn lgs_of(state: &State) -> LatentGameState {
        LatentGameState::new(Countdown::encode_state(state), vec![], "mock".into())
    }

    #[async_trait]
    impl ModelClient<Countdown> for MockModel {
        async fn encode(&self, state: &State) -> anyhow::Result<LatentGameState> {
            Ok(lgs_of(state))
        }
        async fn eval(&self, _lgs: &LatentGameState) -> anyhow::Result<f64> {
            Ok(0.0)
        }
        async fn policy(&self, lgs: &LatentGameState, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>> {
            let state = Countdown::decode_state(&lgs.payload)?;
            let moves = Countdown::legal_moves(&state);
            let n = moves.len().max(1) as f64;
            Ok(moves
                .into_iter()
                .take(top_b as usize)
                .map(|mv| PolicyCandidate {
                    prior: 1.0 / n,
                    // f32 action point (one dim = the move), for the KR kernel.
                    action_vector: (mv as f32).to_le_bytes().to_vec().into(),
                    next_lgs: lgs_of(&Countdown::apply(&state, &mv)),
                })
                .collect())
        }
        async fn reify(
            &self,
            from_state: &State,
            _from_lgs: &LatentGameState,
            _to_lgs: &LatentGameState,
            action_seed: Option<&[u8]>,
            _noise: Option<f64>,
        ) -> anyhow::Result<u8> {
            // Decode the action-head seed back into a move; fall back to legal.
            let legal = Countdown::legal_moves(from_state);
            if let Some(seed) = action_seed {
                if let Ok(mv) = Countdown::decode_move(seed) {
                    if legal.contains(&mv) {
                        return Ok(mv);
                    }
                }
            }
            Ok(legal[0])
        }
        async fn model_version(&self) -> anyhow::Result<String> {
            Ok("mock".into())
        }
    }

    #[tokio::test]
    async fn search_returns_legal_move_and_reifies() {
        let cfg = SearchConfig { simulations: 64, ..Default::default() };
        let state = Countdown::initial_state();
        let res = search(&MockModel, &state, &cfg).await.unwrap();

        assert!(Countdown::legal_moves(&state).contains(&res.chosen), "chosen move must be legal");
        assert_eq!(res.visits, 64);
        assert!(res.reifications > 0, "search should reify at least one edge");
        assert!(res.root_value.abs() <= 1.0);
    }

    #[tokio::test]
    async fn search_works_with_tiny_budget() {
        // One simulation only expands the root; we must still return a legal move.
        let cfg = SearchConfig { simulations: 1, ..Default::default() };
        let state = Countdown::initial_state();
        let res = search(&MockModel, &state, &cfg).await.unwrap();
        assert!(Countdown::legal_moves(&state).contains(&res.chosen));
    }

    #[tokio::test]
    async fn kr_uct_robust_across_bandwidths() {
        // From near-PUCT (σ→0) to heavy smoothing, search stays well-formed.
        let state = Countdown::initial_state();
        for sigma in [0.01, 0.5, 2.0, 50.0] {
            let cfg = SearchConfig { simulations: 48, kernel_bandwidth: sigma, ..Default::default() };
            let res = search(&MockModel, &state, &cfg).await.unwrap();
            assert!(Countdown::legal_moves(&state).contains(&res.chosen), "σ={sigma}");
            assert_eq!(res.visits, 48);
        }
    }

    #[test]
    fn kernel_is_one_at_zero_and_decays() {
        let a = [3.0f32];
        assert!((kernel(&a, &a, 0.75) - 1.0).abs() < 1e-9);
        let near = kernel(&[3.0], &[4.0], 0.75);
        let far = kernel(&[3.0], &[6.0], 0.75);
        assert!(near > far && far >= 0.0, "kernel must decay with distance");
        assert_eq!(action_point(&1.5f32.to_le_bytes()), vec![1.5f32]);
    }
}
