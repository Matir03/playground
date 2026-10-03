//! Two-tier search: tier 1 over whole turns, tier 2 over the steps within one
//! turn. See `docs/minions-steps.md` ("Two-tier search").
//!
//! Tier 1 is Descent (Cohen-Solal 2020) over turn boundaries with a max
//! backup: each descent follows the best child from the root towards the end
//! of the game (until the budget runs out), and every node on its path then
//! takes the max of its children's values from its mover's point of view.
//! A node's children are the turns found from it so far. It also has a
//! widening pseudo-child standing for every turn not found yet, valued like
//! the turn found last. When a descent prefers the widening, the node's
//! tier-2 search runs another descent for a new turn, which becomes the
//! descent's next node; after `retries` tier-2 descents in a row that end on
//! turns already found, the widening is dropped and the found turns are all
//! the node has.
//!
//! Tier 2 is one MCTS per tier-1 node over the steps of its turn, with UCT
//! selection and a mean (p = 1) backup of the returns. Every tier-2 descent
//! runs from the turn's start to its end, expanding (evaluating the children
//! of) each unexpanded node it meets. The end of a turn -- the step that
//! hands the move over, or that ends the game -- is a tier-2 terminal, valued
//! statically from the opponent's point of view (`1 - v` for the mover). A
//! child's static value counts as one return, so selection starts from the
//! statics.

use std::collections::HashMap;

use crate::bfs::{Evaluator, Params, Tree, DEAD, LEAF, TERMINAL};
use crate::steps::StepState;

#[derive(Clone, Debug)]
pub struct TwoTierParams {
    /// UCT exploration constant (values are win probabilities in [0, 1]).
    pub c_uct: f32,
    /// Tier-2 descents in a row allowed to end on known turns before a
    /// node's widening is dropped.
    pub retries: u32,
    /// The widening's value is the newest turn's current value (chosen when
    /// that turn is still the best, ties included; it never raises the
    /// max), not its value when it was found (which floors the max at a
    /// static value the found turns' deeper values cannot pull down).
    pub widen_current: bool,
}

impl Default for TwoTierParams {
    fn default() -> Self {
        Self {
            c_uct: 0.1,
            retries: 20,
            widen_current: true,
        }
    }
}

/// A tier-1 node: a turn boundary (or the search's root, which may be mid-turn).
#[derive(Clone, Debug)]
pub struct Node1 {
    pub parent: u32,
    /// The tier-2 node (in the parent's turn) where this node's turn ended.
    pub end: u32,
    /// The value's point of view differs from the parent's (a handover; a
    /// step that ends the game may not hand over).
    pub flip: bool,
    pub terminal: bool,
    /// Value for this node's mover: its static value until it has children,
    /// then the max over them and the widening.
    pub value: f32,
    pub prior: f32,
    pub visits: u32,
    /// Nothing left to search below (a game end, a proven win, or no
    /// widening and every child resolved).
    pub resolved: bool,
    pub children: Vec<u32>,
    /// The widening pseudo-child's value for this node's mover: the value of
    /// the turn found last when it was found (see
    /// [`TwoTierParams::widen_current`]). `None` before the first turn is
    /// found and once dropped.
    pub widen: Option<f32>,
}

/// A tier-1 node's turn: its tier-2 tree, the returns through each tier-2
/// node (from the turn's mover's point of view), and the turns found.
struct Turn {
    tree: Tree,
    sum: Vec<f64>,
    cnt: Vec<u32>,
    found: HashMap<u32, u32>,
}

#[derive(Clone, Debug, Default)]
pub struct TwoTierStats {
    pub expansions: u64,
    pub evals: u64,
    pub descents: u64,
    pub t2_descents: u64,
    /// Tier-2 descents that ended on a turn already found.
    pub t2_repeats: u64,
    pub widen_dropped: u64,
}

pub struct TwoTier {
    pub nodes: Vec<Node1>,
    states: Vec<StepState>,
    turns: Vec<Option<Turn>>,
    pub params: TwoTierParams,
    /// Tier-2 trees: one leaf per expansion, built on the calling thread
    /// (their own backup is unused: tier 2 keeps mean returns).
    tparams: Params,
    pub stats: TwoTierStats,
}

impl TwoTier {
    pub fn new(
        root: StepState,
        params: TwoTierParams,
        eval: &dyn Evaluator,
    ) -> anyhow::Result<Self> {
        let (terminal, value) = if root.is_terminal() {
            (true, root.terminal_value())
        } else {
            (false, eval.eval(&[root.view()])?[0])
        };
        let mut s = Self {
            nodes: vec![Node1 {
                parent: u32::MAX,
                end: u32::MAX,
                flip: false,
                terminal,
                value,
                prior: value,
                visits: 0,
                resolved: terminal,
                children: Vec::new(),
                widen: None,
            }],
            states: vec![root],
            turns: vec![None],
            params,
            tparams: Params {
                heads: 1,
                power: 1.0,
                threads: 1,
            },
            stats: TwoTierStats::default(),
        };
        s.stats.evals += u64::from(!terminal);
        Ok(s)
    }

    pub fn root_value(&self) -> f32 {
        self.nodes[0].value
    }

    /// Child `c`'s value from its parent's mover's point of view.
    pub fn q_parent(&self, c: u32) -> f32 {
        let n = &self.nodes[c as usize];
        if n.flip {
            1.0 - n.value
        } else {
            n.value
        }
    }

    /// The root's best turn (by value), if any was found.
    pub fn best_child(&self) -> Option<u32> {
        self.nodes[0]
            .children
            .iter()
            .copied()
            .max_by(|&a, &b| self.q_parent(a).total_cmp(&self.q_parent(b)))
    }

    /// The state at the end of tier-1 node `c`'s turn (`c` not the root).
    pub fn end_state(&self, c: u32) -> StepState {
        let nd = &self.nodes[c as usize];
        let turn = self.turns[nd.parent as usize]
            .as_ref()
            .expect("a found turn's parent has a tier-2 tree");
        turn.tree.state_of(nd.end)
    }

    /// Search until `budget` tier-2 expansions are spent or nothing is left.
    pub fn run(&mut self, eval: &dyn Evaluator, budget: u64) -> anyhow::Result<()> {
        let mut idle = 0;
        while self.stats.expansions < budget && !self.nodes[0].resolved {
            let before = self.stats.expansions;
            self.descend(eval, budget)?;
            // Descents that expand nothing still resolve nodes, so this only
            // guards against a bug spinning forever.
            idle = if self.stats.expansions == before {
                idle + 1
            } else {
                0
            };
            if idle >= 1000 {
                break;
            }
        }
        Ok(())
    }

    /// One tier-1 descent: from the root along the best children (a new turn
    /// when the widening is best) until a game end, a resolved node, or the
    /// budget; then the max backup along the path.
    pub fn descend(&mut self, eval: &dyn Evaluator, budget: u64) -> anyhow::Result<()> {
        let mut path = vec![0u32];
        let mut n = 0u32;
        self.nodes[0].visits += 1;
        loop {
            let nd = &self.nodes[n as usize];
            if nd.terminal || nd.resolved || self.stats.expansions >= budget {
                break;
            }
            let next = if nd.children.is_empty() {
                let c = self.discover(n, eval)?;
                if c.is_none() {
                    // No turn at all from here: a dead end, kept at its
                    // static value.
                    self.nodes[n as usize].resolved = true;
                }
                c
            } else {
                let best = self.nodes[n as usize]
                    .children
                    .iter()
                    .copied()
                    .filter(|&c| !self.nodes[c as usize].resolved)
                    .max_by(|&a, &b| {
                        let ka = (self.q_parent(a), -(self.nodes[a as usize].visits as i64));
                        let kb = (self.q_parent(b), -(self.nodes[b as usize].visits as i64));
                        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                let bq = best.map_or(f32::NEG_INFINITY, |c| self.q_parent(c));
                let nd = &self.nodes[n as usize];
                let widen = match nd.widen {
                    None => false,
                    Some(_) if self.params.widen_current => {
                        // The newest turn's value now; ties go to the widening.
                        let last = *nd.children.last().expect("a widening follows a turn");
                        self.q_parent(last) >= bq
                    }
                    Some(w) => w > bq,
                };
                if widen {
                    self.discover(n, eval)?.or(best)
                } else {
                    best
                }
            };
            let Some(c) = next else { break };
            self.nodes[c as usize].visits += 1;
            path.push(c);
            n = c;
        }
        for &p in path.iter().rev() {
            self.recompute(p);
        }
        self.stats.descents += 1;
        Ok(())
    }

    /// `n`'s value and resolution from its children and widening.
    fn recompute(&mut self, n: u32) {
        let nd = &self.nodes[n as usize];
        if nd.terminal || nd.children.is_empty() {
            return;
        }
        let mut v = f32::NEG_INFINITY;
        let (mut all, mut win) = (true, false);
        for &c in &nd.children {
            let q = self.q_parent(c);
            let r = self.nodes[c as usize].resolved;
            v = v.max(q);
            all &= r;
            win |= r && q >= 1.0 - 1e-6;
        }
        if let Some(w) = nd.widen {
            if !self.params.widen_current {
                v = v.max(w);
            }
            all = false;
        }
        let nd = &mut self.nodes[n as usize];
        nd.value = v.clamp(0.0, 1.0);
        nd.resolved = win || all;
    }

    /// Tier-2 descents in `n`'s turn until one ends on a turn not found yet
    /// (the new child, which also sets the widening's value), at most
    /// `retries` of them; after that the widening is dropped.
    fn discover(&mut self, n: u32, eval: &dyn Evaluator) -> anyhow::Result<Option<u32>> {
        let mut turn = match self.turns[n as usize].take() {
            Some(t) => t,
            None => Turn {
                tree: Tree::with_root_value(
                    self.states[n as usize].clone(),
                    self.tparams.clone(),
                    self.nodes[n as usize].prior,
                ),
                sum: vec![0.0],
                cnt: vec![0],
                found: HashMap::new(),
            },
        };
        let mut made = None;
        for _ in 0..self.params.retries.max(1) {
            let (e0, v0) = (turn.tree.stats.expansions, turn.tree.stats.evals);
            let end = t2_descent(&mut turn, eval, self.params.c_uct);
            self.stats.expansions += turn.tree.stats.expansions - e0;
            self.stats.evals += turn.tree.stats.evals - v0;
            self.stats.t2_descents += 1;
            let end = match end {
                Ok(e) => e,
                Err(err) => {
                    self.turns[n as usize] = Some(turn);
                    return Err(err);
                }
            };
            match end {
                None if turn.tree.kind(0) == DEAD => break,
                None => {}
                Some(e) if turn.found.contains_key(&e) => self.stats.t2_repeats += 1,
                Some(e) => {
                    let end = turn.tree.nodes[e as usize];
                    let id = self.nodes.len() as u32;
                    let terminal = end.kind == TERMINAL;
                    self.nodes.push(Node1 {
                        parent: n,
                        end: e,
                        flip: end.flip,
                        terminal,
                        value: end.value,
                        prior: end.value,
                        visits: 0,
                        resolved: terminal,
                        children: Vec::new(),
                        widen: None,
                    });
                    self.states.push(turn.tree.state_of(e));
                    self.turns.push(None);
                    turn.found.insert(e, id);
                    self.nodes[n as usize].children.push(id);
                    made = Some(id);
                    break;
                }
            }
        }
        self.turns[n as usize] = Some(turn);
        match made {
            Some(c) => self.nodes[n as usize].widen = Some(self.q_parent(c)),
            None => {
                if self.nodes[n as usize].widen.take().is_some() {
                    self.stats.widen_dropped += 1;
                }
            }
        }
        Ok(made)
    }
}

/// One tier-2 descent: UCT from the turn's start to its end, expanding each
/// unexpanded node on the way, then the end's value (for the turn's mover)
/// added to every node on the path. Returns the end node, or `None` when the
/// descent ran into a dead end (no legal options).
fn t2_descent(turn: &mut Turn, eval: &dyn Evaluator, c_uct: f32) -> anyhow::Result<Option<u32>> {
    let mut path = vec![0u32];
    let mut n = 0u32;
    loop {
        let kind = turn.tree.kind(n);
        if n != 0 && (turn.tree.nodes[n as usize].flip || kind == TERMINAL) {
            break;
        }
        if kind == TERMINAL || kind == DEAD {
            return Ok(None);
        }
        if kind == LEAF {
            turn.tree.ensure_expanded(n, eval)?;
            if turn.tree.kind(n) == DEAD {
                return Ok(None);
            }
        }
        let len = turn.tree.nodes.len();
        turn.sum.resize(len, 0.0);
        turn.cnt.resize(len, 0);
        let ln = (1.0 + turn.cnt[n as usize] as f32).ln();
        let mut best = None;
        let mut bk = (f32::NEG_INFINITY, i64::MIN);
        for c in turn.tree.children(n) {
            let nd = &turn.tree.nodes[c as usize];
            if nd.kind == DEAD {
                continue;
            }
            // The static value (for this turn's mover) counts as one return.
            let q0 = if nd.flip { 1.0 - nd.prior } else { nd.prior };
            let m = turn.cnt[c as usize];
            let q = ((q0 as f64 + turn.sum[c as usize]) / (1.0 + m as f64)) as f32;
            let u = q + c_uct * (ln / (1.0 + m as f32)).sqrt();
            let k = (u, -(m as i64));
            if k > bk {
                bk = k;
                best = Some(c);
            }
        }
        let Some(c) = best else { return Ok(None) };
        path.push(c);
        n = c;
    }
    let nd = &turn.tree.nodes[n as usize];
    let v = if nd.flip { 1.0 - nd.value } else { nd.value };
    for &p in &path {
        turn.sum[p as usize] += v as f64;
        turn.cnt[p as usize] += 1;
    }
    Ok(Some(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::MinionsState;
    use crate::steps::View;

    /// A pseudo-random value per distinct view, so deeper turns disagree with
    /// shallower ones and the widening gets chosen.
    struct HashEval;

    impl Evaluator for HashEval {
        fn eval(&self, views: &[View]) -> anyhow::Result<Vec<f32>> {
            use std::hash::{Hash, Hasher};
            Ok(views
                .iter()
                .map(|v| {
                    let mut bytes = Vec::new();
                    crate::stepsp::encode_record(v, 0.0, 0.0, 0, &mut bytes);
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    bytes.hash(&mut h);
                    (h.finish() % 1_000_000) as f32 / 1e6
                })
                .collect())
        }
    }

    /// Tier-1 children are turn ends (handovers or game ends) found once
    /// each, values are probabilities, every non-root node's value from its
    /// parent's side is at most the parent's (the max backup), and the budget
    /// holds to within one tier-2 descent.
    #[test]
    fn two_tier_invariants() {
        for widen_current in [false, true] {
            invariants(TwoTierParams {
                widen_current,
                ..Default::default()
            });
        }
    }

    fn invariants(params: TwoTierParams) {
        let s = StepState::begin(&MinionsState::with_bid(0), 7);
        let mut t = TwoTier::new(s, params, &HashEval).unwrap();
        t.run(&HashEval, 3000).unwrap();
        assert!(t.stats.descents > 1, "{:?}", t.stats);
        assert!(
            t.stats.t2_repeats > 0,
            "the widening never retried: {:?}",
            t.stats
        );
        assert!(t.nodes[0].children.len() > 1, "{:?}", t.stats);
        assert!(
            t.nodes.len() > t.nodes[0].children.len(),
            "never went past the first turn"
        );
        for (i, nd) in t.nodes.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&nd.value),
                "node {i} value {}",
                nd.value
            );
            if i == 0 {
                continue;
            }
            assert!(nd.flip || nd.terminal, "node {i} is not a turn end");
            let p = &t.nodes[nd.parent as usize];
            assert!(
                t.q_parent(i as u32) <= p.value + 1e-6,
                "node {i}: {} above its parent's {}",
                t.q_parent(i as u32),
                p.value
            );
        }
        // Every found end is a distinct tier-2 node of its parent's turn.
        let mut ends: Vec<(u32, u32)> = t.nodes[1..].iter().map(|n| (n.parent, n.end)).collect();
        let len = ends.len();
        ends.sort_unstable();
        ends.dedup();
        assert_eq!(ends.len(), len, "a turn was found twice");
        assert!(t.stats.expansions < 3000 + 200, "{:?}", t.stats);
        // The best turn's end is a real position after the root's turn.
        let b = t.best_child().unwrap();
        let fin = t.end_state(b);
        assert!(fin.flipped != t.states[0].flipped || fin.is_terminal());
    }
}
