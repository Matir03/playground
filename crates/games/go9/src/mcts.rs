//! Torch-free PUCT MCTS + self-play game loop for 9×9 Go — the Rust port of the
//! Connect 4 self-play search (`connect4/src/mcts.rs`), adapted for Go:
//!   * 82 actions (81 board points + pass);
//!   * games end on two passes (rules) or a move cap (adjudicated by area score);
//!   * value targets are the **game outcome z** (±1 from the side-to-move's POV),
//!     backfilled after the game, not the local root value — Go needs a grounded
//!     value signal;
//!   * each example also carries the played action and the next position's planes
//!     so the trainer can fit the reifier (pol_next) with a 1-step consistency
//!     loss without re-implementing Go rules in Python.

use std::time::{Duration, Instant};

use rand::Rng;
use rand_distr::{Distribution, Gamma};

use playground_core::{Game, Outcome};

use crate::game::{area_outcome, Go9, Go9State, N};
use crate::net::{action_to_move, move_to_action, ACTIONS};

const CELLS: usize = N * N; // 81
pub const PLANES: usize = 3 * CELLS; // 243

/// A leaf evaluator: a Go state -> (value for the side to move, softmax priors
/// over the 82 actions). Lets the same search drive a direct batch-1 forward or
/// a cross-thread batched GPU inference server (see selfplay.rs).
pub trait Evaluator {
    fn eval(&self, state: &Go9State) -> anyhow::Result<(f32, [f32; ACTIONS])>;
}

impl<F> Evaluator for F
where
    F: Fn(&Go9State) -> anyhow::Result<(f32, [f32; ACTIONS])>,
{
    fn eval(&self, state: &Go9State) -> anyhow::Result<(f32, [f32; ACTIONS])> {
        self(state)
    }
}

/// Search parameters. Budget precedence: node_budget > move_ms > sims.
#[derive(Clone, Copy)]
pub struct Params {
    pub sims: u32,
    pub move_ms: f64,
    pub node_budget: u32,
    pub c_puct: f32,
    pub dir_alpha: f32,
    pub dir_eps: f32,
    pub temp_moves: u32,
    /// Hard ply cap for a self-play game: at this many moves the game is ended
    /// and adjudicated by area score even if neither side has passed. Keeps weak
    /// nets from dragging games out with noisy filler moves.
    pub max_ply: u32,
}

struct Node {
    state: Go9State,
    p: [f32; ACTIONS],
    n: [u32; ACTIONS],
    w: [f32; ACTIONS],
    children: [i32; ACTIONS],
    legal: [bool; ACTIONS],
    expanded: bool,
    terminal: bool,
    term_v: f32,
}

impl Node {
    fn new(state: Go9State) -> Self {
        Self {
            state,
            p: [0.0; ACTIONS],
            n: [0; ACTIONS],
            w: [0.0; ACTIONS],
            children: [-1; ACTIONS],
            legal: [false; ACTIONS],
            expanded: false,
            terminal: false,
            term_v: 0.0,
        }
    }
}

/// Value for the side to move at a finished position (+1 win / -1 loss / 0 draw).
fn outcome_to_value(o: Outcome, to_move: u8) -> f32 {
    match o {
        Outcome::FirstWins => if to_move == 1 { 1.0 } else { -1.0 },
        Outcome::SecondWins => if to_move == 2 { 1.0 } else { -1.0 },
        _ => 0.0,
    }
}

/// Terminal value for the side to move (None if the game is still in progress).
fn terminal_value(state: &Go9State) -> Option<f32> {
    match Go9::outcome(state) {
        Outcome::InProgress => None,
        o => Some(outcome_to_value(o, state.to_move)),
    }
}

/// Expand `arena[idx]`: set it terminal, or evaluate the net and write priors
/// (renormalised over legal actions). Returns the value for the side to move.
fn expand(arena: &mut Vec<Node>, idx: usize, eval: &impl Evaluator) -> anyhow::Result<f32> {
    if let Some(tv) = terminal_value(&arena[idx].state) {
        let node = &mut arena[idx];
        node.terminal = true;
        node.term_v = tv;
        return Ok(tv);
    }
    let (v, probs) = eval.eval(&arena[idx].state)?;
    let legal: Vec<usize> =
        Go9::legal_moves(&arena[idx].state).iter().map(move_to_action).collect();
    let s: f32 = legal.iter().map(|&a| probs[a]).sum::<f32>().max(1e-12);
    let node = &mut arena[idx];
    for &a in &legal {
        node.legal[a] = true;
        node.p[a] = probs[a] / s;
    }
    node.expanded = true;
    Ok(v)
}

/// PUCT action selection over the legal actions of `node`.
fn select(node: &Node, c_puct: f32) -> usize {
    let total: u32 = node.n.iter().sum();
    let sqrt_total = ((total + 1) as f32).sqrt();
    let mut best = usize::MAX;
    let mut best_score = f32::NEG_INFINITY;
    for a in 0..ACTIONS {
        if !node.legal[a] {
            continue;
        }
        let q = if node.n[a] > 0 { node.w[a] / node.n[a] as f32 } else { 0.0 };
        let u = c_puct * node.p[a] * sqrt_total / (1 + node.n[a]) as f32;
        let score = q + u;
        if score > best_score {
            best_score = score;
            best = a;
        }
    }
    best
}

/// Run PUCT MCTS from `root_state`. Returns `(visit_counts[82], root_value, nodes)`.
pub fn run_mcts(
    eval: &impl Evaluator,
    root_state: &Go9State,
    p: &Params,
    rng: &mut impl Rng,
) -> anyhow::Result<([u32; ACTIONS], f32, u32)> {
    let cap = p.node_budget.max(p.sims) as usize + 1;
    let mut arena: Vec<Node> = Vec::with_capacity(cap.min(4096));
    arena.push(Node::new(root_state.clone()));
    expand(&mut arena, 0, eval)?;

    // Dirichlet noise on the root's legal priors (exploration), matching Python.
    if p.dir_eps > 0.0 {
        let legal: Vec<usize> = (0..ACTIONS).filter(|&a| arena[0].legal[a]).collect();
        if !legal.is_empty() {
            let gamma = Gamma::new(p.dir_alpha, 1.0).unwrap();
            let mut g: Vec<f32> = legal.iter().map(|_| gamma.sample(rng) as f32).collect();
            let gsum: f32 = g.iter().sum::<f32>().max(1e-12);
            for x in &mut g {
                *x /= gsum;
            }
            for (i, &a) in legal.iter().enumerate() {
                arena[0].p[a] = (1.0 - p.dir_eps) * arena[0].p[a] + p.dir_eps * g[i];
            }
        }
    }

    let use_nodes = p.node_budget > 0;
    let use_time = !use_nodes && p.move_ms > 0.0;
    let deadline = Instant::now() + Duration::from_secs_f64(p.move_ms / 1000.0);
    const SATURATION_PATIENCE: u32 = 512;
    let mut path: Vec<(usize, usize)> = Vec::with_capacity(64);
    let mut sim = 0u32;
    let mut no_growth = 0u32;
    loop {
        if use_nodes {
            if arena.len() as u32 >= p.node_budget || no_growth >= SATURATION_PATIENCE {
                break;
            }
        } else if use_time {
            if Instant::now() >= deadline {
                break;
            }
        } else if sim >= p.sims {
            break;
        }
        sim += 1;
        let nodes_before = arena.len();
        path.clear();
        let mut node = 0usize;
        while arena[node].expanded && !arena[node].terminal {
            let a = select(&arena[node], p.c_puct);
            path.push((node, a));
            let child = arena[node].children[a];
            node = if child >= 0 {
                child as usize
            } else {
                let next_state = Go9::apply(&arena[node].state, &action_to_move(a));
                let new_idx = arena.len();
                arena.push(Node::new(next_state));
                arena[node].children[a] = new_idx as i32;
                new_idx
            };
        }
        let mut v = if arena[node].terminal {
            arena[node].term_v
        } else {
            expand(&mut arena, node, eval)?
        };
        for &(n, a) in path.iter().rev() {
            v = -v;
            arena[n].n[a] += 1;
            arena[n].w[a] += v;
        }
        if arena.len() == nodes_before {
            no_growth += 1;
        } else {
            no_growth = 0;
        }
    }

    let nodes = arena.len() as u32;
    let root = &arena[0];
    let total_n: u32 = root.n.iter().sum();
    let root_value =
        if total_n > 0 { root.w.iter().sum::<f32>() / total_n as f32 } else { 0.0 };
    Ok((root.n, root_value, nodes))
}

/// One self-play training example. `planes` is the position; `policy` the MCTS
/// visit distribution; `value` the game-outcome z (side-to-move POV); and
/// `played_action`/`next_planes` the 1-step transition for the reifier loss.
#[allow(dead_code)] // fields are the dumped training targets
pub struct Example {
    pub planes: [f32; PLANES],
    pub policy: [f32; ACTIONS],
    pub value: f32,
    pub played_action: u8,
    pub has_next: bool,
    pub next_planes: [f32; PLANES],
}

/// 3×9×9 side-to-move planes (own stones, opponent stones, side-to-move
/// constant), matching `net.rs` `fill_planes` and `Go9::to_features`.
pub fn planes_of(state: &Go9State) -> [f32; PLANES] {
    let own = state.to_move;
    let side: f32 = if state.to_move == 1 { 1.0 } else { 0.0 };
    let mut data = [0f32; PLANES];
    for r in 0..N {
        for c in 0..N {
            let cell = state.board[r][c];
            let idx = r * N + c;
            if cell == own {
                data[idx] = 1.0;
            } else if cell != 0 {
                data[CELLS + idx] = 1.0;
            }
            data[2 * CELLS + idx] = side;
        }
    }
    data
}

/// Play one self-play game. Returns `(examples, total_sims, total_nodes)`.
pub fn self_play_game(
    eval: &impl Evaluator,
    p: &Params,
    rng: &mut impl Rng,
) -> anyhow::Result<(Vec<Example>, u64, u64)> {
    let mut states = vec![Go9::initial_state()];
    // (visit policy, played action, side-to-move) per move.
    let mut moves: Vec<([f32; ACTIONS], usize, u8)> = Vec::new();
    let mut total_sims: u64 = 0;
    let mut total_nodes: u64 = 0;

    loop {
        let cur = states.last().unwrap();
        if terminal_value(cur).is_some() || states.len() - 1 >= p.max_ply as usize {
            break;
        }
        let (counts, _root_v, nodes) = run_mcts(eval, cur, p, rng)?;
        total_sims += counts.iter().map(|&n| n as u64).sum::<u64>();
        total_nodes += nodes as u64;
        let sum: u32 = counts.iter().sum();
        let mut policy = [0f32; ACTIONS];
        if sum > 0 {
            for a in 0..ACTIONS {
                policy[a] = counts[a] as f32 / sum as f32;
            }
        }
        let a = if (states.len() as u32 - 1) < p.temp_moves {
            sample_categorical(&policy, rng)
        } else {
            argmax(&counts)
        };
        let to_move = cur.to_move;
        moves.push((policy, a, to_move));
        let next = Go9::apply(cur, &action_to_move(a));
        states.push(next);
    }

    // Final result: rules outcome if terminal, else adjudicate the capped game.
    let final_state = states.last().unwrap();
    let outcome = match Go9::outcome(final_state) {
        Outcome::InProgress => area_outcome(final_state),
        o => o,
    };
    let winner: Option<u8> = match outcome {
        Outcome::FirstWins => Some(1),
        Outcome::SecondWins => Some(2),
        _ => None,
    };

    let mut examples = Vec::with_capacity(moves.len());
    for (i, (policy, a, to_move)) in moves.into_iter().enumerate() {
        let z = match winner {
            Some(w) => if w == to_move { 1.0 } else { -1.0 },
            None => 0.0,
        };
        examples.push(Example {
            planes: planes_of(&states[i]),
            policy,
            value: z,
            played_action: a as u8,
            has_next: true,
            next_planes: planes_of(&states[i + 1]),
        });
    }
    Ok((examples, total_sims, total_nodes))
}

fn sample_categorical(probs: &[f32; ACTIONS], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen::<f32>();
    let mut acc = 0.0;
    for a in 0..ACTIONS {
        acc += probs[a];
        if r < acc {
            return a;
        }
    }
    (0..ACTIONS).rev().find(|&a| probs[a] > 0.0).unwrap_or(ACTIONS - 1)
}

fn argmax(counts: &[u32; ACTIONS]) -> usize {
    let mut best = 0;
    for a in 1..ACTIONS {
        if counts[a] > counts[best] {
            best = a;
        }
    }
    best
}
