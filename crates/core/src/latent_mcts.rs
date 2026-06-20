//! KR-PUCT MCTS in latent (LGS) space with on-demand reification.
//!
//! Implements the search in docs/algorithm.md. The policy is an MDN over
//! next-LGS ([`MixturePolicy`]); the search samples candidate next-LGS points,
//! selects among them with **KR-PUCT** (kernel-regression PUCT: value/visits
//! shared between nearby candidates by a Gaussian kernel, prior from the MDN
//! density), and **reifies** a candidate into a concrete move on its 2nd visit
//! (decode → apply → re-encode). Candidates that reify to the same concrete
//! state are **aliased** onto one child node, whose merged visits feed the
//! kernel sums. **Progressive widening** grows the candidate set with visits;
//! **forced playouts** guarantee low-prior candidates are explored.
//!
//! Generic over [`LatentModel`]; no I/O, unit-testable with a mock.

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::game::Game;
use crate::latent_state::LatentGameState;
use crate::model::{LatentModel, MixturePolicy};

/// Search configuration. Defaults are tuned for Connect 4 validation.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Number of simulations (playouts) to run.
    pub simulations: u32,
    /// PUCT exploration constant on the MDN-density prior term.
    pub c_puct: f64,
    /// KR kernel bandwidth `h` over next-LGS points (shared with the losses).
    pub kernel_bandwidth: f64,
    /// Progressive widening: target candidates ≈ `pw_beta · n^pw_alpha`.
    pub pw_alpha: f64,
    pub pw_beta: f64,
    /// Hard cap on candidates per node (≤7 distinct reifications for C4 anyway).
    pub max_candidates: usize,
    /// Forced-playout constant: a reified child is forced to `⌈√(k_forced·P·N)⌉`.
    pub k_forced: f64,
    /// First-play-urgency value for unvisited candidates (parent POV).
    pub fpu: f64,
    /// Optional RNG seed for deterministic tests.
    pub seed: Option<u64>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 128,
            c_puct: 1.5,
            kernel_bandwidth: 0.7,
            pw_alpha: 0.5,
            pw_beta: 2.0,
            max_candidates: 16,
            k_forced: 2.0,
            fpu: 0.0,
            seed: None,
        }
    }
}

/// A reified child of the root, for training-target extraction.
#[derive(Debug)]
pub struct RootChild<G: Game> {
    pub mv: G::Move,
    pub state: G::State,
    /// Merged visit count (aliased candidates share this node).
    pub visits: u32,
    /// Mean value from this child's side-to-move POV.
    pub value: f64,
}

#[derive(Debug)]
pub struct SearchResult<G: Game> {
    /// Concrete move to play (legal in `root_state`).
    pub chosen: G::Move,
    /// Root value estimate for the side to move, in [-1, 1].
    pub root_value: f64,
    /// Total simulations run.
    pub visits: u32,
    /// Edges reified into concrete moves during search.
    pub reifications: u32,
    /// Distinct reified children of the root, with merged visit counts.
    pub root_children: Vec<RootChild<G>>,
}

// ── Tree ──────────────────────────────────────────────────────────────────────

/// A sampled next-LGS candidate (an edge). `point` is the kernel/density anchor:
/// the sampled target before reification, the realized child encoding after.
struct Candidate {
    point: Vec<f32>,
    visits: u32,
    /// Child-POV value, accumulated while the candidate is still unreified.
    value_sum: f64,
    /// Child node index once reified (aliases point at a shared node).
    reified: Option<usize>,
}

struct Node<G: Game> {
    lgs: LatentGameState,
    state: G::State,
    incoming_move: Option<G::Move>,
    terminal: bool,
    terminal_value: f64,
    visits: u32,
    value_sum: f64,
    expanded: bool,
    policy: Option<MixturePolicy>,
    candidates: Vec<Candidate>,
    /// Dedup: concrete-state bytes → child node index (for aliasing).
    child_by_key: HashMap<Vec<u8>, usize>,
}

fn make_node<G: Game>(lgs: LatentGameState, state: G::State) -> Node<G> {
    let to_move = G::to_move(&state);
    let outcome = G::outcome(&state);
    let terminal = outcome.is_terminal();
    let terminal_value = if terminal { outcome.value_for(to_move) } else { 0.0 };
    Node {
        lgs,
        state,
        incoming_move: None,
        terminal,
        terminal_value,
        visits: 0,
        value_sum: 0.0,
        expanded: false,
        policy: None,
        candidates: Vec::new(),
        child_by_key: HashMap::new(),
    }
}

/// Gaussian kernel exp(-||a-b||² / 2h²) over next-LGS points.
fn kernel(a: &[f32], b: &[f32], h: f64) -> f64 {
    let n = a.len().min(b.len());
    let mut sq = 0.0f64;
    for i in 0..n {
        let d = (a[i] - b[i]) as f64;
        sq += d * d;
    }
    (-sq / (2.0 * h * h)).exp()
}

// ── Search ──────────────────────────────────────────────────────────────────

/// Run a KR-PUCT latent-space search from `root_state`.
pub async fn search<G: Game, M: LatentModel<G>>(
    model: &M,
    root_state: &G::State,
    config: &SearchConfig,
) -> anyhow::Result<SearchResult<G>> {
    let root_lgs = model.encode(root_state).await?;
    let mut tree: Vec<Node<G>> = vec![make_node::<G>(root_lgs, root_state.clone())];
    let mut rng = match config.seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };
    let mut reifications = 0u32;

    for _ in 0..config.simulations.max(1) {
        simulate(model, &mut tree, config, &mut rng, &mut reifications).await?;
    }

    // Distinct reified children of the root (aliases already merged).
    let mut root_children: Vec<RootChild<G>> = Vec::new();
    for &ci in tree[0].child_by_key.values() {
        let c = &tree[ci];
        if let Some(mv) = c.incoming_move.clone() {
            let value = if c.visits > 0 { c.value_sum / c.visits as f64 } else { 0.0 };
            root_children.push(RootChild { mv, state: c.state.clone(), visits: c.visits, value });
        }
    }
    root_children.sort_by(|a, b| b.visits.cmp(&a.visits));

    let chosen = match root_children.first() {
        Some(rc) if is_legal::<G>(root_state, &rc.mv) => rc.mv.clone(),
        _ => first_legal::<G>(root_state)?,
    };
    let root_value = if tree[0].visits > 0 {
        tree[0].value_sum / tree[0].visits as f64
    } else {
        0.0
    };

    Ok(SearchResult { chosen, root_value, visits: tree[0].visits, reifications, root_children })
}

async fn simulate<G: Game, M: LatentModel<G>>(
    model: &M,
    tree: &mut Vec<Node<G>>,
    config: &SearchConfig,
    rng: &mut ChaCha8Rng,
    reifications: &mut u32,
) -> anyhow::Result<()> {
    let mut path = vec![0usize];
    let mut cur = 0usize;
    let value: f64;
    loop {
        if tree[cur].terminal {
            value = tree[cur].terminal_value;
            break;
        }
        if !tree[cur].expanded {
            tree[cur].policy = Some(model.policy(&tree[cur].lgs).await?);
            tree[cur].expanded = true;
            value = model.eval(&tree[cur].lgs).await?;
            break;
        }

        widen(&mut tree[cur], config, rng);
        let a = kr_puct_select(tree, cur, config);
        tree[cur].candidates[a].visits += 1;

        if let Some(child) = tree[cur].candidates[a].reified {
            path.push(child);
            cur = child;
            continue;
        }
        if tree[cur].candidates[a].visits >= 2 {
            // 2nd visit: reify (decode → apply → re-encode), with aliasing.
            let child = reify_candidate(model, tree, cur, a, reifications).await?;
            path.push(child);
            value = if tree[child].terminal {
                tree[child].terminal_value
            } else {
                model.eval(&tree[child].lgs).await?
            };
            break;
        } else {
            // 1st visit: evaluate the still-latent target (cheap; no concrete game).
            let pt = tree[cur].candidates[a].point.clone();
            let ver = tree[cur].lgs.model_version.clone();
            let v_child = model.eval(&LatentGameState::from_f32(&pt, ver)).await?;
            tree[cur].candidates[a].value_sum += v_child; // child POV
            value = -v_child; // convert to cur's POV (path ends at cur)
            break;
        }
    }

    // Backup (negamax): value is from the POV of the side to move at path.last().
    for (depth_from_leaf, &idx) in path.iter().rev().enumerate() {
        let sign = if depth_from_leaf % 2 == 0 { 1.0 } else { -1.0 };
        tree[idx].visits += 1;
        tree[idx].value_sum += value * sign;
    }
    Ok(())
}

/// Add sampled candidates until the progressive-widening target is met.
fn widen<G: Game>(node: &mut Node<G>, config: &SearchConfig, rng: &mut ChaCha8Rng) {
    let n = node.visits as f64;
    let target = ((config.pw_beta * n.powf(config.pw_alpha)).ceil() as usize)
        .max(1)
        .min(config.max_candidates);
    if node.candidates.len() >= target {
        return;
    }
    // Disjoint field borrows avoid cloning the (k·d) MixturePolicy each widen.
    let Node { policy, candidates, .. } = node;
    if let Some(pol) = policy.as_ref() {
        while candidates.len() < target {
            candidates.push(Candidate {
                point: pol.sample(rng),
                visits: 0,
                value_sum: 0.0,
                reified: None,
            });
        }
    }
}

/// KR-PUCT child selection (with forced playouts). Returns a candidate index.
/// Stack scratch only — this is the search hot path (no per-call allocation).
fn kr_puct_select<G: Game>(tree: &[Node<G>], node_idx: usize, config: &SearchConfig) -> usize {
    const MAX_CAND: usize = 64;
    let node = &tree[node_idx];
    debug_assert!(node.candidates.len() > 0, "select on a node with no candidates");
    debug_assert!(node.candidates.len() <= MAX_CAND, "candidates exceed MAX_CAND");
    let m = node.candidates.len().min(MAX_CAND);

    // Effective counts, parent-POV value, and (unnormalized) MDN priors.
    let pol = node.policy.as_ref().expect("expanded node has a policy");
    let mut n_eff = [0f64; MAX_CAND];
    let mut q = [0f64; MAX_CAND];
    let mut p = [0f64; MAX_CAND];
    let mut mx = f64::NEG_INFINITY;
    for (j, c) in node.candidates.iter().take(m).enumerate() {
        let (n, vsum) = match c.reified {
            Some(ci) => (tree[ci].visits as f64, tree[ci].value_sum),
            None => (c.visits as f64, c.value_sum),
        };
        n_eff[j] = n;
        q[j] = if n > 0.0 { -(vsum / n) } else { config.fpu }; // child POV → parent POV
        let l = pol.log_density(&c.point);
        p[j] = l;
        if l > mx {
            mx = l;
        }
    }
    // Normalize priors (softmax over candidate log-densities).
    let mut psum = 0.0;
    for v in p.iter_mut().take(m) {
        *v = if mx.is_finite() { (*v - mx).exp() } else { 1.0 };
        psum += *v;
    }
    for v in p.iter_mut().take(m) {
        *v = if psum > 0.0 { *v / psum } else { 1.0 / m as f64 };
    }

    // Forced playouts: a reified child below its floor is selected first.
    let nvis = node.visits as f64;
    let mut forced = None;
    let mut best_deficit = 0.0;
    for j in 0..m {
        if let Some(ci) = node.candidates[j].reified {
            let cv = tree[ci].visits as f64;
            if cv >= 1.0 {
                let floor = (config.k_forced * p[j] * nvis).sqrt().ceil();
                let deficit = floor - cv;
                if deficit > best_deficit {
                    best_deficit = deficit;
                    forced = Some(j);
                }
            }
        }
    }
    if let Some(j) = forced {
        return j;
    }

    // KR-PUCT score.
    let h = config.kernel_bandwidth.max(1e-6);
    let sqrt_total = nvis.max(1.0).sqrt();
    let mut best = 0usize;
    let mut best_s = f64::NEG_INFINITY;
    for a in 0..m {
        let (mut w, mut num) = (0.0f64, 0.0f64);
        for b in 0..m {
            let k = kernel(&node.candidates[a].point, &node.candidates[b].point, h);
            w += k * n_eff[b];
            num += k * n_eff[b] * q[b];
        }
        let qkr = if w > 0.0 { num / w } else { config.fpu };
        let u = config.c_puct * p[a] * sqrt_total / (1.0 + w);
        let s = qkr + u;
        if s > best_s {
            best_s = s;
            best = a;
        }
    }
    best
}

/// Reify candidate `a` of node `cur`: decode → apply → re-encode; alias onto an
/// existing child if the concrete state already exists. Returns the child index.
async fn reify_candidate<G: Game, M: LatentModel<G>>(
    model: &M,
    tree: &mut Vec<Node<G>>,
    cur: usize,
    a: usize,
    reifications: &mut u32,
) -> anyhow::Result<usize> {
    let from_state = tree[cur].state.clone();
    let from_lgs = tree[cur].lgs.clone();
    let ver = from_lgs.model_version.clone();
    let to_lgs = LatentGameState::from_f32(&tree[cur].candidates[a].point, ver);

    let mut mv = model.reify(&from_state, &from_lgs, &to_lgs).await?;
    if !is_legal::<G>(&from_state, &mv) {
        mv = first_legal::<G>(&from_state)?; // engine is authoritative on legality
    }
    let child_state = G::apply(&from_state, &mv);
    let key = G::encode_state(&child_state);
    *reifications += 1;

    if let Some(&ci) = tree[cur].child_by_key.get(&key) {
        let anchor = tree[ci].lgs.to_f32();
        let c = &mut tree[cur].candidates[a];
        c.reified = Some(ci);
        c.point = anchor; // realize the kernel anchor onto the child encoding
        return Ok(ci);
    }

    let child_lgs = model.encode(&child_state).await?;
    let mut node = make_node::<G>(child_lgs, child_state);
    node.incoming_move = Some(mv);
    let ci = tree.len();
    let anchor = node.lgs.to_f32();
    tree.push(node);
    tree[cur].child_by_key.insert(key, ci);
    let c = &mut tree[cur].candidates[a];
    c.reified = Some(ci);
    c.point = anchor;
    Ok(ci)
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
    use async_trait::async_trait;

    // Tiny alternating game: remove 1 or 2 from a counter; facing 0 loses.
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
            Ok(State { remaining: b[0], ply: u32::from_le_bytes([b[1], b[2], b[3], b[4]]) })
        }
        fn encode_move(m: &u8) -> Vec<u8> {
            vec![*m]
        }
        fn decode_move(b: &[u8]) -> anyhow::Result<u8> {
            Ok(b[0])
        }
    }

    // Coherent mock latent model. LGS = [remaining, ply]; the MDN puts a tight
    // Gaussian at each legal move's resulting encoding, so sampling lands near
    // real next-states and reify snaps to the nearest legal move.
    struct MockModel;

    fn enc(s: &State) -> Vec<f32> {
        vec![s.remaining as f32, s.ply as f32]
    }

    #[async_trait]
    impl LatentModel<Countdown> for MockModel {
        async fn encode(&self, s: &State) -> anyhow::Result<LatentGameState> {
            Ok(LatentGameState::from_f32(&enc(s), "mock".into()))
        }
        async fn eval(&self, _lgs: &LatentGameState) -> anyhow::Result<f64> {
            Ok(0.0)
        }
        async fn policy(&self, lgs: &LatentGameState) -> anyhow::Result<MixturePolicy> {
            let f = lgs.to_f32();
            let s = State { remaining: f[0].round() as u8, ply: f[1].round() as u32 };
            let moves = Countdown::legal_moves(&s);
            let k = moves.len().max(1);
            let mut means = Vec::with_capacity(k * 2);
            for mv in &moves {
                let nxt = enc(&Countdown::apply(&s, mv));
                means.extend_from_slice(&nxt);
            }
            Ok(MixturePolicy {
                dim: 2,
                weights: vec![1.0 / k as f32; k],
                means,
                stds: vec![0.05; k * 2],
            })
        }
        async fn reify(
            &self,
            from_state: &State,
            _from_lgs: &LatentGameState,
            to_lgs: &LatentGameState,
        ) -> anyhow::Result<u8> {
            let tgt = to_lgs.to_f32();
            // Pick the legal move whose resulting encoding is nearest the target.
            Countdown::legal_moves(from_state)
                .into_iter()
                .min_by(|a, b| {
                    let da = dist(&enc(&Countdown::apply(from_state, a)), &tgt);
                    let db = dist(&enc(&Countdown::apply(from_state, b)), &tgt);
                    da.total_cmp(&db)
                })
                .ok_or_else(|| anyhow::anyhow!("no legal moves"))
        }
        async fn model_version(&self) -> anyhow::Result<String> {
            Ok("mock".into())
        }
    }

    fn dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    #[tokio::test]
    async fn search_returns_legal_move_and_reifies() {
        let cfg = SearchConfig { simulations: 96, seed: Some(1), ..Default::default() };
        let s = Countdown::initial_state();
        let res = search(&MockModel, &s, &cfg).await.unwrap();
        assert!(Countdown::legal_moves(&s).contains(&res.chosen));
        assert_eq!(res.visits, 96);
        assert!(res.reifications > 0, "should reify at least one edge");
        assert!(res.root_value.abs() <= 1.0);
    }

    #[tokio::test]
    async fn aliasing_keeps_children_bounded() {
        // Many sampled candidates, but only ≤2 distinct legal next-states, so
        // aliasing must collapse them to ≤2 reified root children.
        let cfg = SearchConfig {
            simulations: 200,
            max_candidates: 16,
            seed: Some(7),
            ..Default::default()
        };
        let s = Countdown::initial_state();
        let res = search(&MockModel, &s, &cfg).await.unwrap();
        let legal = Countdown::legal_moves(&s).len();
        assert!(
            res.root_children.len() <= legal,
            "aliasing failed: {} children > {} legal moves",
            res.root_children.len(),
            legal
        );
        // Merged child visits should account for most of the root's simulations.
        let child_visits: u32 = res.root_children.iter().map(|c| c.visits).sum();
        assert!(child_visits > 0);
    }

    #[tokio::test]
    async fn tiny_budget_returns_legal_move() {
        let cfg = SearchConfig { simulations: 1, seed: Some(0), ..Default::default() };
        let s = Countdown::initial_state();
        let res = search(&MockModel, &s, &cfg).await.unwrap();
        assert!(Countdown::legal_moves(&s).contains(&res.chosen));
    }

    #[tokio::test]
    async fn robust_across_bandwidths() {
        let s = Countdown::initial_state();
        for h in [0.05, 0.5, 2.0, 20.0] {
            let cfg = SearchConfig { simulations: 64, kernel_bandwidth: h, seed: Some(3), ..Default::default() };
            let res = search(&MockModel, &s, &cfg).await.unwrap();
            assert!(Countdown::legal_moves(&s).contains(&res.chosen), "h={h}");
            assert_eq!(res.visits, 64);
        }
    }

    #[test]
    fn kernel_decays_with_distance() {
        let a = [1.0f32, 2.0];
        assert!((kernel(&a, &a, 0.7) - 1.0).abs() < 1e-9);
        assert!(kernel(&a, &[1.5, 2.0], 0.7) > kernel(&a, &[3.0, 2.0], 0.7));
    }
}
