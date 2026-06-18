//! Amazons search glue: the `MoveGame` impl for the batched self-play engine, the
//! shared two-head prior combine, and a small concrete PUCT MCTS used by eval-match
//! (which alternates two models per move, so it can't use the single-model
//! `run_chunk_moves`).

use playground_core::batch_search::MoveGame;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};

use crate::game::{self, AmazonsMove, AmazonsState, CELLS, PLANES as PLANE_COUNT};
use crate::moves::POLICY_DIM;

/// Floats per encoded position (4 planes × 100 cells).
pub const PLANES: usize = PLANE_COUNT * CELLS;

impl MoveGame for crate::game::Amazons {
    type State = AmazonsState;
    type Move = AmazonsMove;
    const PLANES: usize = PLANES;
    const POLICY_DIM: usize = POLICY_DIM;

    fn initial_state() -> AmazonsState {
        <crate::game::Amazons as playground_core::Game>::initial_state()
    }
    fn to_move(state: &AmazonsState) -> u8 {
        state.to_move
    }
    fn apply(state: &AmazonsState, mv: &AmazonsMove) -> AmazonsState {
        game::apply(state, mv)
    }
    fn terminal_value(state: &AmazonsState) -> Option<f32> {
        // No legal move ⇒ side to move loses.
        if game::has_any_move(state) {
            None
        } else {
            Some(-1.0)
        }
    }
    fn encode_planes(state: &AmazonsState, out: &mut [f32]) {
        game::fill_planes_into(state, out);
    }
    fn expand(state: &AmazonsState, policy: &[f32], moves: &mut Vec<AmazonsMove>, priors: &mut Vec<f32>) {
        game::expand_into(state, policy, moves, priors);
    }
}

// ── Concrete PUCT MCTS (eval-match) ───────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Params {
    pub node_budget: u32,
    pub c_puct: f32,
    pub dir_alpha: f32,
    pub dir_eps: f32,
    pub temp_moves: u32,
    pub max_ply: u32,
}

/// Per-leaf evaluator: returns (value for side-to-move, raw policy logits[POLICY_DIM]).
pub trait Evaluator {
    fn eval(&self, state: &AmazonsState) -> anyhow::Result<(f32, Vec<f32>)>;
}

struct CNode {
    state: AmazonsState,
    moves: Vec<AmazonsMove>,
    p: Vec<f32>,
    n: Vec<u32>,
    w: Vec<f32>,
    children: Vec<i32>,
    expanded: bool,
    terminal: bool,
    term_v: f32,
}

impl CNode {
    fn new(state: AmazonsState) -> Self {
        Self {
            state,
            moves: Vec::new(),
            p: Vec::new(),
            n: Vec::new(),
            w: Vec::new(),
            children: Vec::new(),
            expanded: false,
            terminal: false,
            term_v: 0.0,
        }
    }
}

fn select(nd: &CNode, c_puct: f32) -> usize {
    let total: u32 = nd.n.iter().sum();
    let sqrt_total = ((total + 1) as f32).sqrt();
    let mut best = 0;
    let mut best_score = f32::NEG_INFINITY;
    for a in 0..nd.moves.len() {
        let q = if nd.n[a] > 0 { nd.w[a] / nd.n[a] as f32 } else { 0.0 };
        let u = c_puct * nd.p[a] * sqrt_total / (1 + nd.n[a]) as f32;
        let s = q + u;
        if s > best_score {
            best_score = s;
            best = a;
        }
    }
    best
}

/// Run a fixed-budget PUCT search from `state`; return (move, visit-count) pairs.
pub fn run_mcts(
    eval: &impl Evaluator,
    state: &AmazonsState,
    p: &Params,
    rng: &mut ChaCha8Rng,
) -> anyhow::Result<Vec<(AmazonsMove, u32)>> {
    let mut arena: Vec<CNode> = vec![CNode::new(state.clone())];
    let budget = p.node_budget.max(1);
    let mut dirichlet_done = false;
    // Saturation guard: in late-game (few moves) the tree can't grow to `budget`
    // (all reachable leaves terminal/expanded), so without this the loop spins
    // forever. Mirror the batched search's no-growth break.
    let mut no_growth = 0u32;

    while (arena.len() as u32) < budget {
        let len_before = arena.len();
        // Descend to a leaf.
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut node = 0usize;
        while arena[node].expanded && !arena[node].terminal {
            let a = select(&arena[node], p.c_puct);
            let child = arena[node].children[a];
            let nxt = if child >= 0 {
                child as usize
            } else {
                let ns = game::apply(&arena[node].state, &arena[node].moves[a]);
                let idx = arena.len();
                arena.push(CNode::new(ns));
                arena[node].children[a] = idx as i32;
                idx
            };
            path.push((node, a));
            node = nxt;
        }

        // Evaluate / detect terminal.
        let value = if arena[node].terminal {
            arena[node].term_v
        } else if !game::has_any_move(&arena[node].state) {
            arena[node].terminal = true;
            arena[node].term_v = -1.0;
            -1.0
        } else {
            let (v, policy) = eval.eval(&arena[node].state)?;
            let (moves, mut pr) = game::expand(&arena[node].state, &policy);
            let m = moves.len();
            if node == 0 && p.dir_eps > 0.0 && !dirichlet_done {
                let gamma = Gamma::new(p.dir_alpha as f64, 1.0).unwrap();
                let mut g: Vec<f32> = (0..m).map(|_| gamma.sample(rng) as f32).collect();
                let gs: f32 = g.iter().sum::<f32>().max(1e-12);
                for (pi, gi) in pr.iter_mut().zip(g.iter_mut()) {
                    *gi /= gs;
                    *pi = (1.0 - p.dir_eps) * *pi + p.dir_eps * *gi;
                }
                dirichlet_done = true;
            }
            let nd = &mut arena[node];
            nd.moves = moves;
            nd.p = pr;
            nd.n = vec![0; m];
            nd.w = vec![0.0; m];
            nd.children = vec![-1; m];
            nd.expanded = true;
            v
        };

        // Backup (negamax).
        let mut v = value;
        for &(n, a) in path.iter().rev() {
            v = -v;
            arena[n].n[a] += 1;
            arena[n].w[a] += v;
        }

        // No new node this iteration ⇒ the tree is saturated; stop (else infinite loop).
        if arena.len() == len_before {
            no_growth += 1;
            if no_growth >= 512 {
                break;
            }
        } else {
            no_growth = 0;
        }
    }

    let root = &arena[0];
    Ok(root.moves.iter().copied().zip(root.n.iter().copied()).collect())
}
