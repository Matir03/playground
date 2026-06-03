//! MCTS in GS (game-state latent) space with on-demand reification.
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
//!   3. **Expansion** — `policy` gives up to B (prior, action-vector, next-GS)
//!      children; the leaf value comes from `eval` (or the exact game result at
//!      terminal nodes).
//!   4. **Backup** — negamax the leaf value up the visited path.
//!
//! Deviations from the full design (deliberate, see docs/algorithm.md "open
//! questions"): selection is PUCT rather than KR-UCT, and each edge is reified
//! once (no noisy reified siblings / progressive widening yet). Those refine
//! search quality but only matter once the network is trained; the structure
//! here — GS-space search that reifies through `encode`/`eval`/`policy`/`reify`
//! — is the planned one. KR-UCT + siblings slot in at the selection/reification
//! steps.

use bytes::Bytes;

use crate::game::Game;
use crate::gs::Gs;
use crate::model::ModelClient;

/// Search configuration. Field list will grow as the algorithm solidifies.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Number of simulations (tree traversals) to run.
    pub simulations: u32,
    /// B — number of (action, next_gs) candidates produced by the policy head
    /// per expansion.
    pub policy_top_b: u32,
    /// K — visits between reifications at the same node. Reserved for the
    /// progressive-widening / sibling schedule (not used yet).
    pub reification_interval: u32,
    /// Multiplier for exploration noise on reified siblings (not used yet).
    pub sibling_noise_base: f64,
    /// PUCT exploration constant.
    pub exploration_c: f64,
    /// Optional RNG seed for deterministic tests.
    pub seed: Option<u64>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 160,
            policy_top_b: 8,
            reification_interval: 8,
            sibling_noise_base: 0.25,
            exploration_c: 1.4,
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
    /// Policy state-head prediction of the child GS (before reification).
    predicted_gs: Gs,
    child: usize,
    /// Concrete move, set once this edge is reified.
    mv: Option<()>, // presence flag; the move lives on the child node
}

struct Node<G: Game> {
    gs: Gs,
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
    fn latent(gs: Gs) -> Self {
        Self {
            gs,
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
    fn reified(gs: Gs, state: G::State) -> Self {
        let mut n = Self::latent(gs);
        n.state = Some(state);
        n
    }
}

/// PUCT score for a child edge, from the parent's perspective. `child` value is
/// stored in the child's own (opponent's) perspective, hence the negation.
fn puct<G: Game>(parent_visits: u32, edge: &Edge, child: &Node<G>, c: f64) -> f64 {
    let q = if child.visits > 0 {
        -(child.value_sum / child.visits as f64)
    } else {
        0.0
    };
    let u = c * edge.prior * (parent_visits as f64).sqrt() / (1.0 + child.visits as f64);
    q + u
}

// ── Search ──────────────────────────────────────────────────────────────────

/// Run a GS-space MCTS from `root_state` and return the chosen concrete move.
pub async fn search<G: Game, M: ModelClient<G>>(
    model: &M,
    root_state: &G::State,
    config: &SearchConfig,
) -> anyhow::Result<SearchResult<G>> {
    let root_gs = model.encode(root_state).await?;
    let mut tree: Vec<Node<G>> = vec![Node::reified(root_gs, root_state.clone())];
    let mut reifications = 0u32;

    for _ in 0..config.simulations.max(1) {
        simulate(model, &mut tree, config, &mut reifications).await?;
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
        let Some(edge_idx) = select_best(tree, cur, config.exploration_c) else {
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
        model.eval(&tree[cur].gs).await?
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
    let parent_gs = tree[parent].gs.clone();
    let to_gs = tree[parent].edges[edge_idx].predicted_gs.clone();
    let seed = tree[parent].edges[edge_idx].action_vector.clone();

    let mut mv = model
        .reify(&parent_state, &parent_gs, &to_gs, Some(&seed), None)
        .await?;
    if !is_legal::<G>(&parent_state, &mv) {
        // A bad decode shouldn't poison the search; substitute a legal move.
        mv = first_legal::<G>(&parent_state)?;
    }

    let child_state = G::apply(&parent_state, &mv);
    let outcome = G::outcome(&child_state);
    let child_gs = model.encode(&child_state).await?;

    let node = &mut tree[child];
    node.gs = child_gs;
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
    let gs = tree[node].gs.clone();
    let candidates = model.policy(&gs, top_b).await?;
    for c in candidates {
        let child_idx = tree.len();
        tree.push(Node::latent(c.next_gs.clone()));
        tree[node].edges.push(Edge {
            prior: c.prior,
            action_vector: c.action_vector,
            predicted_gs: c.next_gs,
            child: child_idx,
            mv: None,
        });
    }
    tree[node].expanded = true;
    Ok(model.eval(&gs).await?)
}

/// Pick the highest-PUCT edge of `node_idx`. Reads child stats from the arena.
fn select_best<G: Game>(tree: &[Node<G>], node_idx: usize, c: f64) -> Option<usize> {
    let node = &tree[node_idx];
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for (i, e) in node.edges.iter().enumerate() {
        let score = puct(node.visits, e, &tree[e.child], c);
        if score > best_score {
            best_score = score;
            best = Some(i);
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

    // Coherent mock: GS payload carries the concrete state bytes, so policy can
    // enumerate real children and reify can decode the action seed to a move.
    struct MockModel;

    fn gs_of(state: &State) -> Gs {
        Gs::new(Countdown::encode_state(state), vec![], "mock".into())
    }

    #[async_trait]
    impl ModelClient<Countdown> for MockModel {
        async fn encode(&self, state: &State) -> anyhow::Result<Gs> {
            Ok(gs_of(state))
        }
        async fn eval(&self, _gs: &Gs) -> anyhow::Result<f64> {
            Ok(0.0)
        }
        async fn policy(&self, gs: &Gs, top_b: u32) -> anyhow::Result<Vec<PolicyCandidate>> {
            let state = Countdown::decode_state(&gs.payload)?;
            let moves = Countdown::legal_moves(&state);
            let n = moves.len().max(1) as f64;
            Ok(moves
                .into_iter()
                .take(top_b as usize)
                .map(|mv| PolicyCandidate {
                    prior: 1.0 / n,
                    action_vector: Countdown::encode_move(&mv).into(),
                    next_gs: gs_of(&Countdown::apply(&state, &mv)),
                })
                .collect())
        }
        async fn reify(
            &self,
            from_state: &State,
            _from_gs: &Gs,
            _to_gs: &Gs,
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
}
