//! Torch-free PUCT MCTS + self-play game loop — the Rust port of
//! `python/.../training/mcts_core.py`.
//!
//! The search is identical to the Python self-play: expand the root, add
//! Dirichlet noise to the root priors, run a fixed number of PUCT simulations,
//! and return the visit distribution plus the search's root value (the
//! visit-weighted mean of the children's Q). The single difference that matters
//! for performance is that this runs with no GIL, so many games search in
//! parallel across cores (see `selfplay.rs`).
//!
//! Nodes live in a per-game arena (`Vec<Node>`, child indices) rather than boxed
//! pointers: the 6×7 board makes nodes tiny and the arena keeps them contiguous.

use std::time::{Duration, Instant};

use rand::Rng;
use rand_distr::{Distribution, Gamma};

use playground_core::{Game, Outcome};

use crate::game::{Connect4, Connect4Move, Connect4State, C4_COLS};

const PLANES: usize = 2 * 6 * 7;

/// A leaf evaluator: a concrete state -> (value for the side to move, full
/// softmax priors over the 7 columns). Decoupling the search from inference this
/// way (as in Python's `evaluate` callback) lets the same MCTS drive either a
/// direct batch-1 forward or a cross-thread batched GPU inference server. Each
/// search thread owns its own evaluator instance, so this need not be `Sync`.
pub trait Evaluator {
    fn eval(&self, state: &Connect4State) -> anyhow::Result<(f32, [f32; C4_COLS])>;
}

impl<F> Evaluator for F
where
    F: Fn(&Connect4State) -> anyhow::Result<(f32, [f32; C4_COLS])>,
{
    fn eval(&self, state: &Connect4State) -> anyhow::Result<(f32, [f32; C4_COLS])> {
        self(state)
    }
}

/// Parameters mirroring the Python self-play knobs.
#[derive(Clone, Copy)]
pub struct Params {
    /// Fixed simulations per move, used when `move_ms == 0`.
    pub sims: u32,
    /// Per-move time budget in ms; when > 0 it overrides `sims` and the search
    /// runs uncapped rollouts until the budget is spent (mirrors Python).
    pub move_ms: f64,
    pub c_puct: f32,
    pub dir_alpha: f32,
    pub dir_eps: f32,
    pub temp_moves: u32,
}

struct Node {
    state: Connect4State,
    p: [f32; C4_COLS],   // priors, 0 for illegal columns
    n: [u32; C4_COLS],   // visit counts per child
    w: [f32; C4_COLS],   // total value per child (from the parent's perspective)
    children: [i32; C4_COLS], // arena index of each child, -1 = not created
    legal: [bool; C4_COLS],
    expanded: bool,
    terminal: bool,
    term_v: f32,
}

impl Node {
    fn new(state: Connect4State) -> Self {
        Self {
            state,
            p: [0.0; C4_COLS],
            n: [0; C4_COLS],
            w: [0.0; C4_COLS],
            children: [-1; C4_COLS],
            legal: [false; C4_COLS],
            expanded: false,
            terminal: false,
            term_v: 0.0,
        }
    }
}

/// Terminal value for the side to move: a finished game with a winner means the
/// opponent just completed four, so the side to move has lost (-1); a full board
/// is a draw (0). Mirrors `Board.terminal` in connect4.py.
fn terminal_value(state: &Connect4State) -> Option<f32> {
    match Connect4::outcome(state) {
        Outcome::InProgress => None,
        Outcome::Draw => Some(0.0),
        _ => Some(-1.0), // FirstWins / SecondWins: the mover lost
    }
}

/// Expand `arena[idx]`: set it terminal, or evaluate the net and write priors
/// (renormalised over legal columns). Returns the value for the side to move.
fn expand(arena: &mut Vec<Node>, idx: usize, eval: &impl Evaluator) -> anyhow::Result<f32> {
    if let Some(tv) = terminal_value(&arena[idx].state) {
        let node = &mut arena[idx];
        node.terminal = true;
        node.term_v = tv;
        return Ok(tv);
    }
    let (v, probs) = eval.eval(&arena[idx].state)?;
    let legal: Vec<u8> = Connect4::legal_moves(&arena[idx].state)
        .iter()
        .map(|m| m.col)
        .collect();
    let s: f32 = legal.iter().map(|&c| probs[c as usize]).sum::<f32>().max(1e-12);
    let node = &mut arena[idx];
    for &c in &legal {
        let c = c as usize;
        node.legal[c] = true;
        node.p[c] = probs[c] / s;
    }
    node.expanded = true;
    Ok(v)
}

/// PUCT child selection over the legal columns of `node`.
fn select(node: &Node, c_puct: f32) -> usize {
    let total: u32 = node.n.iter().sum();
    let sqrt_total = ((total + 1) as f32).sqrt();
    let mut best = usize::MAX;
    let mut best_score = f32::NEG_INFINITY;
    for c in 0..C4_COLS {
        if !node.legal[c] {
            continue;
        }
        let q = if node.n[c] > 0 { node.w[c] / node.n[c] as f32 } else { 0.0 };
        let u = c_puct * node.p[c] * sqrt_total / (1 + node.n[c]) as f32;
        let score = q + u;
        if score > best_score {
            best_score = score;
            best = c;
        }
    }
    best
}

/// Run PUCT MCTS from `root_state`. Returns `(visit_counts, root_value, nodes)`,
/// where `nodes` is the number of distinct positions in the search tree — the
/// honest measure of search effort (unlike the raw simulation count, which
/// saturated Connect 4 subtrees inflate as sims revisit existing nodes).
pub fn run_mcts(
    eval: &impl Evaluator,
    root_state: &Connect4State,
    p: &Params,
    rng: &mut impl Rng,
) -> anyhow::Result<([u32; C4_COLS], f32, u32)> {
    let mut arena: Vec<Node> = Vec::with_capacity(p.sims as usize + 1);
    arena.push(Node::new(root_state.clone()));
    expand(&mut arena, 0, eval)?;

    // Dirichlet noise on the root's legal priors (exploration), matching Python.
    if p.dir_eps > 0.0 {
        let legal: Vec<usize> = (0..C4_COLS).filter(|&c| arena[0].legal[c]).collect();
        if !legal.is_empty() {
            let gamma = Gamma::new(p.dir_alpha, 1.0).unwrap();
            let mut g: Vec<f32> = legal.iter().map(|_| gamma.sample(rng) as f32).collect();
            let gsum: f32 = g.iter().sum::<f32>().max(1e-12);
            for x in &mut g {
                *x /= gsum;
            }
            for (i, &c) in legal.iter().enumerate() {
                arena[0].p[c] = (1.0 - p.dir_eps) * arena[0].p[c] + p.dir_eps * g[i];
            }
        }
    }

    let use_time = p.move_ms > 0.0;
    let deadline = Instant::now() + Duration::from_secs_f64(p.move_ms / 1000.0);
    let mut path: Vec<(usize, usize)> = Vec::with_capacity(64);
    let mut sim = 0u32;
    loop {
        if use_time {
            if Instant::now() >= deadline {
                break;
            }
        } else if sim >= p.sims {
            break;
        }
        sim += 1;
        path.clear();
        let mut node = 0usize;
        // Descend to a leaf (unexpanded or terminal).
        while arena[node].expanded && !arena[node].terminal {
            let c = select(&arena[node], p.c_puct);
            path.push((node, c));
            let child = arena[node].children[c];
            node = if child >= 0 {
                child as usize
            } else {
                let next_state = Connect4::apply(&arena[node].state, &Connect4Move { col: c as u8 });
                let new_idx = arena.len();
                arena.push(Node::new(next_state));
                arena[node].children[c] = new_idx as i32;
                new_idx
            };
        }
        let mut v = if arena[node].terminal {
            arena[node].term_v
        } else {
            expand(&mut arena, node, eval)?
        };
        // Backprop, flipping sign at each level (value is for the side to move).
        for &(n, c) in path.iter().rev() {
            v = -v;
            arena[n].n[c] += 1;
            arena[n].w[c] += v;
        }
    }

    let nodes = arena.len() as u32;
    let root = &arena[0];
    let total_n: u32 = root.n.iter().sum();
    let root_value = if total_n > 0 {
        root.w.iter().sum::<f32>() / total_n as f32
    } else {
        0.0
    };
    Ok((root.n, root_value, nodes))
}

/// One self-play training example: planes (2×6×7), MCTS policy target, and the
/// MCTS root value target (the local search estimate for the side to move).
/// The benchmark only counts these, but they're the exact targets a Rust
/// trainer would consume — mirroring the tuples Python's `self_play_game` emits.
#[allow(dead_code)] // fields are the training targets; the bench only counts them
pub struct Example {
    pub planes: [f32; PLANES],
    pub policy: [f32; C4_COLS],
    pub value: f32,
}

/// Play one self-play game. Returns `(examples, total_sims, total_nodes)`.
pub fn self_play_game(
    eval: &impl Evaluator,
    p: &Params,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<Example>, u64, u64)> {
    let mut state = Connect4::initial_state();
    let mut examples = Vec::new();
    let mut total_sims: u64 = 0;
    let mut total_nodes: u64 = 0;
    while terminal_value(&state).is_none() {
        let (counts, root_v, nodes) = run_mcts(eval, &state, p, rng)?;
        let sum: u32 = counts.iter().sum();
        total_sims += sum as u64;
        total_nodes += nodes as u64;
        let mut policy = [0f32; C4_COLS];
        for c in 0..C4_COLS {
            policy[c] = counts[c] as f32 / sum as f32;
        }
        examples.push(Example { planes: planes_of(&state), policy, value: root_v });

        let col = if (state.ply as u32) < p.temp_moves {
            sample_categorical(&policy, rng)
        } else {
            argmax(&counts)
        };
        state = Connect4::apply(&state, &Connect4Move { col: col as u8 });
    }
    Ok((examples, total_sims, total_nodes))
}

/// 2×6×7 side-to-move planes (own stones, then opponent's), matching
/// `connect4.py` `Board.planes` and the net's `planes`.
fn planes_of(state: &Connect4State) -> [f32; PLANES] {
    let mover = if state.ply % 2 == 0 { 1u8 } else { 2u8 };
    let cells = 6 * 7;
    let mut data = [0f32; PLANES];
    for r in 0..6 {
        for c in 0..7 {
            let cell = state.cells[r][c];
            let idx = r * 7 + c;
            if cell == mover {
                data[idx] = 1.0;
            } else if cell != 0 {
                data[cells + idx] = 1.0;
            }
        }
    }
    data
}

fn sample_categorical(probs: &[f32; C4_COLS], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen::<f32>();
    let mut acc = 0.0;
    for c in 0..C4_COLS {
        acc += probs[c];
        if r < acc {
            return c;
        }
    }
    // Fall through (rounding): last column with nonzero mass.
    (0..C4_COLS).rev().find(|&c| probs[c] > 0.0).unwrap_or(0)
}

fn argmax(counts: &[u32; C4_COLS]) -> usize {
    let mut best = 0;
    for c in 1..C4_COLS {
        if counts[c] > counts[best] {
            best = c;
        }
    }
    best
}
