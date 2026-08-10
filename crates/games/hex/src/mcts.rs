//! Hex search glue for plain AlphaZero.
//!
//! Two entry points, both PUCT over **real** board states with the net's direct
//! policy/value — no latent space, no reifier, no encoding:
//!
//!   * [`MoveGame`] — the batched self-play/eval engine in
//!     `playground_core::batch_search`, which keeps hundreds of games in flight
//!     and coalesces their leaf evaluations into one GPU forward. This is what
//!     training and checkpoint-vs-checkpoint matches run on.
//!   * [`run_search`] — a compact sequential PUCT for playing *one* game against
//!     something outside the pipeline (MoHex, the UI), where batching across
//!     games isn't available.
//!
//! The policy head lives in the canonical frame (see `game::fill_planes_into`),
//! so every logit lookup goes through `canon_index`.

use playground_core::batch_search::MoveGame;

use crate::game::{
    canon_index, fill_planes_into, Hex, HexMove, HexState, N, NN, PLANES as PLANE_COUNT,
};

/// Floats per encoded position (2 planes × 169 cells).
pub const PLANES: usize = PLANE_COUNT * NN;
/// One logit per cell.
pub const POLICY_DIM: usize = NN;

impl MoveGame for Hex {
    type State = HexState;
    type Move = HexMove;
    const PLANES: usize = PLANES;
    const POLICY_DIM: usize = POLICY_DIM;

    fn initial_state() -> HexState {
        HexState::default()
    }

    fn to_move(state: &HexState) -> u8 {
        state.to_move
    }

    fn apply(state: &HexState, mv: &HexMove) -> HexState {
        <Hex as playground_core::Game>::apply(state, mv)
    }

    fn terminal_value(state: &HexState) -> Option<f32> {
        // A win is always scored by the player who just moved, so a decided
        // position is a loss for the side to move. Hex has no draws.
        (state.winner != 0).then_some(-1.0)
    }

    fn encode_planes(state: &HexState, out: &mut [f32]) {
        fill_planes_into(state, out);
    }

    fn expand(state: &HexState, policy: &[f32], moves: &mut Vec<HexMove>, priors: &mut Vec<f32>) {
        moves.clear();
        priors.clear();
        // Softmax over the legal cells only — the net's logits for occupied cells
        // are never normalised in, so no probability mass leaks to illegal moves.
        let mut max = f32::NEG_INFINITY;
        for i in state.empty().iter() {
            let l = policy[canon_index(state, i)];
            if l > max {
                max = l;
            }
            moves.push(HexMove::Play { r: (i / N) as u8, c: (i % N) as u8 });
            priors.push(l);
        }
        let mut sum = 0.0;
        for p in priors.iter_mut() {
            *p = (*p - max).exp();
            sum += *p;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for p in priors.iter_mut() {
            *p *= inv;
        }
    }
}

// ── Sequential PUCT (single game, batch-1 forwards) ──────────────────────────

/// Knobs for [`run_search`] and the eval matches.
#[derive(Clone, Copy)]
pub struct Params {
    /// Simulations per move.
    pub sims: u32,
    pub c_puct: f32,
    /// Root Dirichlet noise (0 = off, as in evaluation).
    pub dir_alpha: f32,
    pub dir_eps: f32,
    /// Sample from the visit distribution for this many opening plies, then argmax.
    pub temp_moves: u32,
}

impl Default for Params {
    fn default() -> Self {
        Self { sims: 400, c_puct: 1.5, dir_alpha: 0.3, dir_eps: 0.0, temp_moves: 0 }
    }
}

/// What [`run_search`] needs from the net: value and raw policy logits for one
/// position, both from the side to move's point of view.
pub trait Evaluator {
    fn evaluate(&self, state: &HexState) -> anyhow::Result<(f32, Vec<f32>)>;
}

struct Node {
    state: HexState,
    moves: Vec<HexMove>,
    priors: Vec<f32>,
    n: Vec<u32>,
    w: Vec<f32>,
    child: Vec<i32>,
    total_n: u32,
    terminal: bool,
    term_v: f32,
}

impl Node {
    fn new(state: HexState) -> Self {
        let terminal = state.winner != 0;
        Self {
            state,
            moves: Vec::new(),
            priors: Vec::new(),
            n: Vec::new(),
            w: Vec::new(),
            child: Vec::new(),
            total_n: 0,
            terminal,
            term_v: -1.0, // decided ⇒ the side to move has lost
        }
    }
}

/// PUCT search from `state`. Returns the per-move visit counts (the policy target)
/// and the root value estimate, both from the side to move's point of view.
pub fn run_search(
    model: &impl Evaluator,
    state: &HexState,
    params: &Params,
    rng: &mut impl rand::Rng,
) -> anyhow::Result<(Vec<HexMove>, Vec<u32>, f32)> {
    let mut nodes: Vec<Node> = vec![Node::new(*state)];
    if nodes[0].terminal {
        return Ok((vec![], vec![], -1.0));
    }
    expand_node(model, &mut nodes, 0)?;
    if params.dir_eps > 0.0 {
        add_dirichlet(&mut nodes[0], params.dir_alpha, params.dir_eps, rng);
    }

    let mut path: Vec<(usize, usize)> = Vec::new();
    for _ in 0..params.sims.max(1) {
        path.clear();
        let mut idx = 0usize;
        // Descend by PUCT until we reach an unexpanded edge or a terminal node.
        let value = loop {
            if nodes[idx].terminal {
                break nodes[idx].term_v;
            }
            let a = select(&nodes[idx], params.c_puct);
            path.push((idx, a));
            let child = nodes[idx].child[a];
            if child < 0 {
                let next = <Hex as playground_core::Game>::apply(&nodes[idx].state, &nodes[idx].moves[a]);
                nodes.push(Node::new(next));
                let new_idx = nodes.len() - 1;
                nodes[idx].child[a] = new_idx as i32;
                if nodes[new_idx].terminal {
                    break nodes[new_idx].term_v;
                }
                break expand_node(model, &mut nodes, new_idx)?;
            }
            idx = child as usize;
        };
        // Back up, flipping sign at every ply (values are side-to-move POV).
        let mut v = value;
        for &(node, a) in path.iter().rev() {
            v = -v;
            let nd = &mut nodes[node];
            nd.n[a] += 1;
            nd.w[a] += v;
            nd.total_n += 1;
        }
    }

    let root = &nodes[0];
    let visits = root.n.clone();
    let total: u32 = visits.iter().sum();
    let value = if total == 0 {
        0.0
    } else {
        root.w.iter().sum::<f32>() / total as f32
    };
    Ok((root.moves.clone(), visits, value))
}

/// Evaluate `idx`, filling in its moves/priors. Returns the leaf value (side-to-move POV).
fn expand_node(model: &impl Evaluator, nodes: &mut [Node], idx: usize) -> anyhow::Result<f32> {
    let (value, logits) = model.evaluate(&nodes[idx].state)?;
    let node = &mut nodes[idx];
    let mut moves = Vec::new();
    let mut priors = Vec::new();
    <Hex as MoveGame>::expand(&node.state, &logits, &mut moves, &mut priors);
    let k = moves.len();
    node.moves = moves;
    node.priors = priors;
    node.n = vec![0; k];
    node.w = vec![0.0; k];
    node.child = vec![-1; k];
    Ok(value)
}

fn select(node: &Node, c_puct: f32) -> usize {
    let sqrt_total = (node.total_n.max(1) as f32).sqrt();
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for a in 0..node.moves.len() {
        let n = node.n[a];
        let q = if n == 0 { 0.0 } else { node.w[a] / n as f32 };
        let u = c_puct * node.priors[a] * sqrt_total / (1.0 + n as f32);
        let score = q + u;
        if score > best_score {
            best_score = score;
            best = a;
        }
    }
    best
}

fn add_dirichlet(root: &mut Node, alpha: f32, eps: f32, rng: &mut impl rand::Rng) {
    use rand_distr::{Distribution, Gamma};
    let gamma = match Gamma::new(alpha, 1.0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let noise: Vec<f32> = (0..root.priors.len()).map(|_| gamma.sample(rng)).collect();
    let sum: f32 = noise.iter().sum();
    if sum <= 0.0 {
        return;
    }
    for (p, n) in root.priors.iter_mut().zip(&noise) {
        *p = (1.0 - eps) * *p + eps * (n / sum);
    }
}

/// Pick a move from a visit distribution: sampled while below `temp_moves`,
/// argmax after (the usual AZ temperature schedule).
pub fn pick_move(
    moves: &[HexMove],
    visits: &[u32],
    ply: u32,
    temp_moves: u32,
    rng: &mut impl rand::Rng,
) -> HexMove {
    let total: u32 = visits.iter().sum();
    if ply < temp_moves && total > 0 {
        let mut t = rng.gen_range(0..total);
        for (mv, &v) in moves.iter().zip(visits) {
            if t < v {
                return *mv;
            }
            t -= v;
        }
    }
    let best = visits
        .iter()
        .enumerate()
        .max_by_key(|&(_, v)| *v)
        .map(|(i, _)| i)
        .unwrap_or(0);
    moves[best]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Both `Game` and `MoveGame` define `initial_state`/`apply`; spell out the
    /// rules one here.
    type G = <Hex as playground_core::Game>::Move;
    fn initial() -> HexState {
        <Hex as playground_core::Game>::initial_state()
    }
    fn apply(state: &HexState, mv: &G) -> HexState {
        <Hex as playground_core::Game>::apply(state, mv)
    }

    /// A net that likes one specific cell and is otherwise flat.
    struct Favours(usize, f32);

    impl Evaluator for Favours {
        fn evaluate(&self, _state: &HexState) -> anyhow::Result<(f32, Vec<f32>)> {
            let mut logits = vec![0f32; POLICY_DIM];
            logits[self.0] = self.1;
            Ok((0.0, logits))
        }
    }

    #[test]
    fn priors_are_a_softmax_over_legal_cells_only() {
        let s = apply(&initial(), &HexMove::Play { r: 6, c: 6 });
        let logits: Vec<f32> = (0..POLICY_DIM).map(|i| i as f32 * 0.01).collect();
        let (mut moves, mut priors) = (Vec::new(), Vec::new());
        <Hex as MoveGame>::expand(&s, &logits, &mut moves, &mut priors);
        assert_eq!(moves.len(), NN - 1, "every empty cell, and only those");
        assert!(!moves.contains(&HexMove::Play { r: 6, c: 6 }));
        assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(priors.iter().all(|p| *p > 0.0));
    }

    #[test]
    fn search_concentrates_visits_on_the_favoured_move() {
        // White to move, so the net's index space is transposed: the search must
        // map logit `canon` back to the real cell (2,1) rather than (1,2).
        let s = apply(&initial(), &HexMove::Play { r: 0, c: 0 });
        let real = 2 * N + 1;
        let canon = crate::game::transpose_index(real);
        let mut rng = rand::rngs::StdRng::from_seed([7; 32]);
        let params = Params { sims: 200, dir_eps: 0.0, ..Default::default() };
        let (moves, visits, _v) = run_search(&Favours(canon, 8.0), &s, &params, &mut rng).unwrap();
        let best = visits.iter().enumerate().max_by_key(|&(_, v)| *v).unwrap().0;
        assert_eq!(moves[best], HexMove::Play { r: 2, c: 1 });
    }

    #[test]
    fn search_finds_the_move_that_completes_a_chain() {
        // Black owns column g except row 7; playing there connects top to bottom.
        let mut s = initial();
        for r in 0..N {
            if r != 6 {
                s.black.set(r * N + 6);
                s.white.set(r * N); // filler in column a so the stone counts stay legal
            }
        }
        s.white.clear(6 * N); // keep white's count one behind
        s.to_move = 1;
        let mut rng = rand::rngs::StdRng::from_seed([3; 32]);
        let params = Params { sims: 400, dir_eps: 0.0, ..Default::default() };
        let (moves, visits, value) = run_search(&Favours(0, 0.0), &s, &params, &mut rng).unwrap();
        let best = visits.iter().enumerate().max_by_key(|&(_, v)| *v).unwrap().0;
        assert_eq!(moves[best], HexMove::Play { r: 6, c: 6 }, "the winning cell");
        assert!(value > 0.5, "a forced win should score near +1, got {value}");
    }
}
