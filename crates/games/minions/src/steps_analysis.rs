//! The best-first step search (docs/minions-steps.md) in the UI's analysis
//! window, when `serve --steps-net` configured the seat.
//!
//! The window is built for whole-turn edges; the step tree's edges are single
//! decisions inside a turn. They fit without a protocol change:
//!
//! - An edge's opaque `turn` encoding carries its session, its node and the
//!   board to show there: the turn so far *projected* (fated enemies gone,
//!   decided units at their destinations, spawns placed), or the next turn's
//!   position once the edge completes the turn. Walking a line
//!   (`apply_encoded_turn`) shows that board.
//! - Committing an edge (`commit_encoded_turn`, the ▶ button) plays the whole
//!   turn: from the edge's node along the search's best line, greedy on the
//!   net below what the search expanded.
//! - Values are win probabilities mapped to [-1, 1] for the side to move, as
//!   the window expects. There is no policy or energy: every option is
//!   enumerated and evaluated, so `prior` is uniform and `energy` 0.

use std::collections::HashMap;
use std::sync::Mutex;

use playground_core::{
    Analysis, AnalysisEdge, AnalysisNode, Game, SessionOp, SessionResult, SolverBudget,
};

use crate::bfs::{Evaluator, Tree};
use crate::game::{Minions, MinionsState};
use crate::net_seat::StepsSeat;
use crate::steps::StepState;

const MAGIC: &[u8; 4] = b"STP1";
/// Nodes exported per analysis, most visited first; edges of every exported
/// node are all listed.
const MAX_NODES: usize = 600;
const MAX_SESSIONS: usize = 4;

struct Session {
    tree: Tree,
    used: u64,
}

#[derive(Default)]
struct Store {
    next: u64,
    clock: u64,
    live: HashMap<u64, Session>,
    /// The last step edge a line walk applied: the board it showed (as read
    /// back) and its node's value (side to move, [-1, 1]). Walking a line
    /// applies the line and then asks for `eval_only` on the result, so this
    /// is that node. Keyed by node rather than by board: many step nodes
    /// project to the same board (the decided-markers are not part of a game
    /// state), so a board-keyed table returns some other node's value.
    last_walk: Option<(Vec<u8>, f32)>,
}

static STORE: Mutex<Option<Store>> = Mutex::new(None);

fn salt_for(state: &MinionsState) -> u64 {
    Minions::encode_state(state)
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        })
}

/// The board node `n` shows, in the real frame.
pub(crate) fn board_at(tree: &Tree, n: u32) -> MinionsState {
    let st = tree.state_of(n);
    if n == 0 || tree.nodes[n as usize].flip || st.is_terminal() {
        return st.real_base();
    }
    let b = st.view().board;
    if st.flipped {
        b.rotated()
    } else {
        b
    }
}

fn encode_edge(session: u64, node: u32, board: &MinionsState) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&session.to_le_bytes());
    out.extend_from_slice(&node.to_le_bytes());
    out.extend_from_slice(&Minions::encode_state(board));
    out
}

/// `(session, node, board)` if `turn` is a step edge.
fn decode_edge(turn: &[u8]) -> Option<anyhow::Result<(u64, u32, MinionsState)>> {
    if turn.len() < 16 || &turn[..4] != MAGIC {
        return None;
    }
    let session = u64::from_le_bytes(turn[4..12].try_into().unwrap());
    let node = u32::from_le_bytes(turn[12..16].try_into().unwrap());
    Some(Minions::decode_state(&turn[16..]).map(|b| (session, node, b)))
}

/// Walking a line: the board a step edge shows. `None` = not a step edge.
pub fn apply(turn: &[u8]) -> Option<anyhow::Result<MinionsState>> {
    let (session, node, board) = match decode_edge(turn)? {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    let mut guard = STORE.lock().unwrap();
    let store = guard.get_or_insert_with(Store::default);
    store.last_walk = store.live.get(&session).and_then(|s| {
        let n = s.tree.nodes.get(node as usize)?;
        Some((Minions::encode_state(&board), to_pm(n.value)))
    });
    Some(Ok(board))
}

/// Committing a step edge: the real position after the whole turn through it.
pub fn commit(seat: &StepsSeat, turn: &[u8]) -> Option<anyhow::Result<MinionsState>> {
    let (session, node, _) = match decode_edge(turn)? {
        Ok(x) => x,
        Err(e) => return Some(Err(e)),
    };
    let mut store = STORE.lock().unwrap();
    let Some(s) = store
        .get_or_insert_with(Store::default)
        .live
        .get_mut(&session)
    else {
        return Some(Err(anyhow::anyhow!(
            "analysis session {session} has expired: re-analyse, then play the turn"
        )));
    };
    Some(complete_turn(&mut s.tree, node, seat.eval.as_ref()))
}

/// The real position after the turn through node `n`: the search's best line
/// while it has expanded nodes, then greedy on the static values.
fn complete_turn(
    tree: &mut Tree,
    mut n: u32,
    eval: &dyn Evaluator,
) -> anyhow::Result<MinionsState> {
    if n != 0 && (tree.nodes[n as usize].flip || tree.kind(n) == 2) {
        return Ok(tree.state_of(n).real_base());
    }
    loop {
        tree.ensure_expanded(n, eval)?;
        anyhow::ensure!(tree.kind(n) != 3, "the line runs into a dead end");
        let c = tree
            .best_child(n)
            .ok_or_else(|| anyhow::anyhow!("no live option"))?;
        if tree.nodes[c as usize].flip || tree.kind(c) == 2 {
            return Ok(tree.state_of(c).real_base());
        }
        n = c;
    }
}

pub(crate) fn to_pm(v: f32) -> f32 {
    2.0 * v - 1.0
}

/// The session's tree as the window's flattened node list.
fn export(id: u64, tree: &Tree) -> Analysis {
    // Expanded nodes, most descended first (the root always), capped.
    let mut picked: Vec<u32> = vec![0];
    let mut frontier: Vec<u32> = vec![0];
    while let Some(n) = frontier.pop() {
        for c in tree.children(n) {
            if tree.kind(c) == 1 && tree.nodes[c as usize].visits > 0 {
                picked.push(c);
                frontier.push(c);
            }
        }
    }
    picked[1..].sort_by_key(|&c| std::cmp::Reverse(tree.nodes[c as usize].visits));
    picked.truncate(MAX_NODES);
    // Keep only nodes whose parent made the cut, so every node is reachable.
    let mut index: HashMap<u32, usize> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();
    let mut keep: std::collections::HashSet<u32> = picked.iter().copied().collect();
    keep.insert(0);
    let mut stack = vec![0u32];
    while let Some(n) = stack.pop() {
        index.insert(n, order.len());
        order.push(n);
        for c in tree.children(n).rev() {
            if keep.contains(&c) {
                stack.push(c);
            }
        }
    }

    let mut nodes = Vec::with_capacity(order.len());
    for &n in &order {
        let nd = &tree.nodes[n as usize];
        let st = tree.state_of(n);
        let live: Vec<u32> = tree.children(n).filter(|&c| tree.kind(c) != 3).collect();
        let prior = 1.0 / live.len().max(1) as f32;
        let edges = live
            .iter()
            .map(|&c| {
                let cn = &tree.nodes[c as usize];
                let board = board_at(tree, c);
                let mv = encode_edge(id, c, &board);
                let flip = |v: f32| if cn.flip { 1.0 - v } else { v };
                let mut label = st.step_label(cn.step);
                if cn.flip {
                    label.push_str("  ⟶ end of turn");
                } else if tree.kind(c) == 2 {
                    label.push_str("  ⟶ game over");
                }
                AnalysisEdge {
                    mv,
                    label,
                    prior,
                    energy: 0.0,
                    visits: cn.visits,
                    value: (cn.visits > 0 || cn.kind == 1).then(|| to_pm(tree.q_parent(c))),
                    child: index.get(&c).copied(),
                    explore: false,
                    forced: 0,
                    draws: 1,
                    prior_value: Some(to_pm(flip(cn.prior))),
                    manual: false,
                }
            })
            .collect();
        nodes.push(AnalysisNode {
            parent: (n != 0).then(|| index[&nd.parent]),
            total_visits: nd.visits,
            value: Some(to_pm(nd.value)),
            terminal: nd.kind == 2,
            term_value: to_pm(nd.value),
            prior_value: Some(to_pm(nd.prior)),
            edges,
            kernel: Vec::new(),
        });
    }
    let chosen = tree.best_child(0).and_then(|b| {
        tree.children(0)
            .filter(|&c| tree.kind(c) != 3)
            .position(|c| c == b)
    });
    Analysis {
        value: to_pm(tree.nodes[0].value) as f64,
        aux: None,
        nodes,
        chosen,
    }
}

/// Run the search: `secs` seconds (if > 0) or `iterations` iterations.
fn run(tree: &mut Tree, eval: &dyn Evaluator, iterations: u32, secs: f64) -> anyhow::Result<()> {
    tree.ensure_expanded(0, eval)?;
    let t = std::time::Instant::now();
    let mut done = 0u32;
    while !tree.is_solved() {
        if secs > 0.0 {
            if t.elapsed().as_secs_f64() > secs {
                break;
            }
        } else if done >= iterations {
            break;
        }
        if tree.iterate(eval)? == 0 {
            break;
        }
        done += 1;
    }
    Ok(())
}

/// The value shown for a position when walking a line.
pub fn eval_only(seat: &StepsSeat, state: &MinionsState) -> anyhow::Result<Analysis> {
    let here = Minions::encode_state(state);
    let known = STORE
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|s| s.last_walk.as_ref())
        .and_then(|(b, v)| (*b == here).then_some(*v));
    let value = match known {
        Some(v) => v,
        None => {
            let s = StepState::begin(state, salt_for(state));
            if s.is_terminal() {
                to_pm(s.terminal_value())
            } else {
                to_pm(seat.eval.eval(&[s.view()])?[0])
            }
        }
    };
    Ok(Analysis {
        value: value as f64,
        aux: None,
        nodes: Vec::new(),
        chosen: None,
    })
}

/// A one-off analysis (not a session): the configured time.
pub fn analyze(
    seat: &StepsSeat,
    state: &MinionsState,
    budget: SolverBudget,
) -> anyhow::Result<Analysis> {
    let r = session(
        seat,
        SessionOp::Start {
            root: state.clone(),
            simulations: budget.max_visits.unwrap_or(0),
            fast: false,
        },
    )?;
    Ok(r.analysis)
}

/// Start, step or steer a step-search session. `simulations` count search
/// iterations (each expands up to `heads` leaves); 0 at the start runs the
/// seat's time (`--steps-secs`).
pub fn session(
    seat: &StepsSeat,
    op: SessionOp<MinionsState>,
) -> anyhow::Result<SessionResult<MinionsState>> {
    let eval = seat.eval.as_ref();
    let mut guard = STORE.lock().unwrap();
    let store = guard.get_or_insert_with(Store::default);
    store.clock += 1;
    let now = store.clock;
    let mut note: Option<String> = None;
    let id = match op {
        SessionOp::Start {
            root, simulations, ..
        } => {
            let mut tree = Tree::new(
                StepState::begin(&root, salt_for(&root)),
                seat.params.clone(),
                eval,
            )?;
            let secs = if simulations == 0 { seat.secs } else { 0.0 };
            run(&mut tree, eval, simulations.saturating_sub(1), secs)?;
            store.next += 1;
            let id = store.next;
            if store.live.len() >= MAX_SESSIONS {
                let oldest = store
                    .live
                    .iter()
                    .min_by_key(|(_, s)| s.used)
                    .map(|(k, _)| *k);
                if let Some(k) = oldest {
                    store.live.remove(&k);
                }
            }
            store.live.insert(id, Session { tree, used: now });
            id
        }
        SessionOp::Step { id, simulations } => {
            let s = store
                .live
                .get_mut(&id)
                .ok_or_else(|| anyhow::anyhow!("analysis session {id} has expired"))?;
            run(&mut s.tree, eval, simulations, 0.0)?;
            s.used = now;
            id
        }
        SessionOp::Force { id, line } => {
            let s = store
                .live
                .get_mut(&id)
                .ok_or_else(|| anyhow::anyhow!("analysis session {id} has expired"))?;
            let last = line.last().ok_or_else(|| anyhow::anyhow!("empty line"))?;
            let (_, mut n, _) =
                decode_edge(last).ok_or_else(|| anyhow::anyhow!("not a step-search edge"))??;
            // Expand the first unexpanded node on the best line below the edge.
            while s.tree.kind(n) == 1 {
                match s.tree.best_child(n) {
                    Some(c) => n = c,
                    None => break,
                }
            }
            s.tree.ensure_expanded(n, eval)?;
            s.used = now;
            id
        }
        SessionOp::Add { id, .. } => {
            note = Some(
                "the step search enumerates every option; there is nothing to add by hand".into(),
            );
            id
        }
    };
    let s = store
        .live
        .get(&id)
        .ok_or_else(|| anyhow::anyhow!("analysis session {id} has expired"))?;
    let analysis = export(id, &s.tree);
    let st = &s.tree.stats;
    let summary = format!(
        "step search: {} expansions, {} evaluations, {} nodes",
        st.expansions,
        st.evals,
        s.tree.nodes.len()
    );
    Ok(SessionResult {
        id,
        root: s.tree.root.real_base(),
        analysis,
        added: None,
        note: Some(note.map_or(summary.clone(), |n| format!("{n}; {summary}"))),
    })
}
