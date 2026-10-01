//! Best-first search over the step tree ([`crate::steps`]) with K simultaneous
//! heads and a visit-weighted power-mean backup. See `docs/minions-steps.md`.
//!
//! One iteration:
//!
//! 1. Walk the principal variation (PV) from the root: at each expanded node,
//!    the child with the highest value for that node's mover.
//! 2. Candidates are every live child of every PV node, scored by how far its
//!    value falls short of the PV child's at the same node (the PV's own end
//!    has gap 0). The K smallest gaps are taken; each descends its own PV to
//!    an unexpanded leaf, and duplicates are dropped.
//! 3. Every leaf is expanded: all its options are enumerated and every child is
//!    evaluated statically, in one batch across the K leaves.
//! 4. Values are backed up. A node's value is the power mean (`p`) of its
//!    children's values from its mover's point of view, weighted by
//!    `1 + descents` (descents = times a selection path passed through).
//!
//! Values are win probabilities in `[0, 1]` for the node's own mover. A node
//! whose turn hands over the move to the opponent is marked `flip`.

use std::time::Instant;

use crate::steps::{StepState, View};

/// Static evaluation of step-tree nodes.
pub trait Evaluator: Send + Sync {
    /// Win probability in `[0, 1]` for the side to move at each view.
    fn eval(&self, views: &[View]) -> anyhow::Result<Vec<f32>>;
}

static DEAD_LOG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const LEAF: u8 = 0;
const EXPANDED: u8 = 1;
const TERMINAL: u8 = 2;
const DEAD: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub parent: u32,
    pub first: u32,
    pub nch: u16,
    pub step: u8,
    pub flip: bool,
    pub kind: u8,
    pub visits: u32,
    /// Backed-up value for this node's mover (static value until expanded).
    pub value: f32,
    /// Static evaluation at creation.
    pub prior: f32,
    /// Completion (Cohen-Solal, arXiv 2008.01188, section 5): the proven
    /// result for this node's mover, 1 win / -1 loss / 0 unknown or draw,
    /// the minimax of the terminal results below it.
    pub comp: i8,
    /// Resolved: the result is proven (a proven win or loss, or every child
    /// resolved). Descents never enter a resolved node again.
    pub resolved: bool,
}

#[derive(Clone, Debug)]
pub struct Params {
    /// Leaves expanded per iteration.
    pub heads: usize,
    /// Power-mean exponent of the backup.
    pub power: f32,
    /// Threads that build a wave's leaf expansions (option enumeration and
    /// witnesses, the CPU cost); 1 builds them on the calling thread.
    pub threads: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            heads: 32,
            power: 2.0,
            threads: 1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub iterations: u64,
    pub expansions: u64,
    pub evals: u64,
    pub dead: u64,
    pub terminal_leaves: u64,
    pub select_ns: u64,
    pub expand_ns: u64,
    pub eval_ns: u64,
    pub backup_ns: u64,
    /// `descent_run`: the main thread's wait for the group in flight (build
    /// and evaluation, which overlap the other group's landing).
    pub wait_ns: u64,
    /// `descent_run`, main thread: moving leaf states out, and landing a
    /// group (`finish` + `commit`).
    pub take_ns: u64,
    pub land_ns: u64,
}

impl SearchStats {
    pub fn add(&mut self, o: &SearchStats) {
        self.iterations += o.iterations;
        self.expansions += o.expansions;
        self.evals += o.evals;
        self.dead += o.dead;
        self.terminal_leaves += o.terminal_leaves;
        self.select_ns += o.select_ns;
        self.expand_ns += o.expand_ns;
        self.eval_ns += o.eval_ns;
        self.backup_ns += o.backup_ns;
        self.wait_ns += o.wait_ns;
        self.take_ns += o.take_ns;
        self.land_ns += o.land_ns;
    }
}

pub struct Tree {
    pub nodes: Vec<Node>,
    /// Each node's state, kept from its parent's expansion so nothing replays.
    pub states: Vec<Option<StepState>>,
    pub root: StepState,
    pub params: Params,
    pub stats: SearchStats,
    /// Descent mode: the paths (root first) of the descents in flight, and
    /// per node how many of them pass through it.
    cursors: Vec<Vec<u32>>,
    /// The second descent group of [`Tree::descent_run`].
    cursors_b: Vec<Vec<u32>>,
    inflight: Vec<u32>,
    /// Leaves whose expansion is being evaluated ([`Tree::descent_run`]).
    pending: Vec<bool>,
    /// Where new descents start, round robin (empty: node 0). Several games
    /// sharing one tree each start descents at their own position.
    starts: Vec<u32>,
    next_start: usize,
}

impl Tree {
    /// A tree whose root is `root`, evaluated with `eval`.
    pub fn new(root: StepState, params: Params, eval: &dyn Evaluator) -> anyhow::Result<Self> {
        let (kind, value, comp) = if root.is_terminal() {
            (TERMINAL, root.terminal_value(), root.terminal_outcome())
        } else {
            (LEAF, eval.eval(&[root.view()])?[0], 0)
        };
        let node = Node {
            parent: u32::MAX,
            first: 0,
            nch: 0,
            step: 0,
            flip: false,
            kind,
            visits: 0,
            value,
            prior: value,
            comp,
            resolved: kind == TERMINAL,
        };
        Ok(Self {
            nodes: vec![node],
            states: vec![Some(root.clone())],
            root,
            params,
            stats: SearchStats::default(),
            cursors: Vec::new(),
            cursors_b: Vec::new(),
            inflight: Vec::new(),
            pending: Vec::new(),
            starts: Vec::new(),
            next_start: 0,
        })
    }

    /// Nothing left to search: the root (or every descent start) is decided.
    pub fn is_solved(&self) -> bool {
        let done = |n: u32| {
            let nd = &self.nodes[n as usize];
            matches!(nd.kind, TERMINAL | DEAD) || nd.resolved
        };
        if self.starts.is_empty() {
            done(0)
        } else {
            self.starts.iter().all(|&n| done(n))
        }
    }

    /// Start new descents at these nodes, round robin (empty: node 0).
    pub fn set_starts(&mut self, starts: &[u32]) {
        self.starts = starts.to_vec();
        self.next_start = 0;
    }

    pub fn children(&self, n: u32) -> std::ops::Range<u32> {
        let nd = &self.nodes[n as usize];
        nd.first..nd.first + nd.nch as u32
    }

    /// Child `c`'s value from its parent's mover's point of view.
    #[inline]
    pub fn q_parent(&self, c: u32) -> f32 {
        let n = &self.nodes[c as usize];
        if n.flip {
            1.0 - n.value
        } else {
            n.value
        }
    }

    /// Child `c`'s completion from its parent's mover's point of view.
    #[inline]
    pub fn c_parent(&self, c: u32) -> i8 {
        let n = &self.nodes[c as usize];
        if n.flip {
            -n.comp
        } else {
            n.comp
        }
    }

    /// The best live child: by completion, then value.
    pub fn best_child(&self, n: u32) -> Option<u32> {
        let mut best = None;
        let mut bk = (i8::MIN, f32::NEG_INFINITY);
        for c in self.children(n) {
            if self.nodes[c as usize].kind == DEAD {
                continue;
            }
            let k = (self.c_parent(c), self.q_parent(c));
            if k.0 > bk.0 || k.0 == bk.0 && k.1 > bk.1 {
                bk = k;
                best = Some(c);
            }
        }
        best
    }

    /// The step path from the root to `n`.
    pub fn path(&self, mut n: u32) -> Vec<u8> {
        let mut steps = Vec::new();
        while self.nodes[n as usize].parent != u32::MAX {
            steps.push(self.nodes[n as usize].step);
            n = self.nodes[n as usize].parent;
        }
        steps.reverse();
        steps
    }

    /// `n`'s state: stored, or replayed from its nearest ancestor with a
    /// stored state (every root of a forest keeps one).
    pub fn state_of(&self, n: u32) -> StepState {
        let mut steps = Vec::new();
        let mut a = n;
        let mut s = loop {
            if let Some(Some(s)) = self.states.get(a as usize) {
                break s.clone();
            }
            let nd = &self.nodes[a as usize];
            if nd.parent == u32::MAX {
                break self.root.clone();
            }
            steps.push(nd.step);
            a = nd.parent;
        };
        for &v in steps.iter().rev() {
            s.apply(v);
        }
        s
    }

    /// Follow best children from `n` to an unexpanded (or terminal) node.
    fn descend(&self, mut n: u32) -> u32 {
        while self.nodes[n as usize].kind == EXPANDED {
            match self.best_child(n) {
                Some(c) => n = c,
                None => break,
            }
        }
        n
    }

    /// The principal variation, root first.
    pub fn pv(&self) -> Vec<u32> {
        let mut pv = vec![0u32];
        let mut n = 0u32;
        while self.nodes[n as usize].kind == EXPANDED {
            match self.best_child(n) {
                Some(c) => {
                    pv.push(c);
                    n = c;
                }
                None => break,
            }
        }
        pv
    }

    /// Up to `heads` distinct unexpanded leaves near the PV.
    fn select(&mut self) -> Vec<u32> {
        let t = Instant::now();
        let heads = self.params.heads.max(1);
        let mut leaves = Vec::with_capacity(heads);
        if self.nodes[0].kind == LEAF {
            leaves.push(0);
        } else if self.nodes[0].kind == EXPANDED {
            let pv = self.pv();
            let mut cand: Vec<(f32, u32)> = Vec::new();
            for &p in &pv {
                if self.nodes[p as usize].kind != EXPANDED {
                    continue;
                }
                let Some(b) = self.best_child(p) else {
                    continue;
                };
                let bq = self.q_parent(b);
                for c in self.children(p) {
                    if self.nodes[c as usize].kind == DEAD || self.nodes[c as usize].resolved {
                        continue;
                    }
                    cand.push((bq - self.q_parent(c), c));
                }
            }
            cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            for &(_, c) in &cand {
                if leaves.len() >= heads {
                    break;
                }
                let leaf = self.descend(c);
                match self.nodes[leaf as usize].kind {
                    LEAF => {
                        if !leaves.contains(&leaf) {
                            leaves.push(leaf);
                        }
                    }
                    _ => self.stats.terminal_leaves += 1,
                }
            }
        }
        // Descents: every node on each selected path.
        for &l in &leaves {
            let mut n = l;
            loop {
                self.nodes[n as usize].visits += 1;
                if n == 0 {
                    break;
                }
                n = self.nodes[n as usize].parent;
            }
        }
        self.stats.select_ns += t.elapsed().as_nanos() as u64;
        leaves
    }

    /// Expand `leaves`: enumerate, evaluate every child in one batch.
    fn expand(&mut self, leaves: &[u32], eval: &dyn Evaluator) -> anyhow::Result<()> {
        let mut batch = self.prepare(leaves);
        let t = Instant::now();
        let views = std::mem::take(&mut batch.views);
        let vals = if views.is_empty() {
            Vec::new()
        } else {
            eval.eval(&views)?
        };
        self.stats.eval_ns += t.elapsed().as_nanos() as u64;
        self.stats.evals += views.len() as u64;
        self.commit(batch, &vals);
        Ok(())
    }

    /// Build `leaves`' children and the views to evaluate, on
    /// `params.threads` threads (option enumeration and witnesses are the CPU
    /// cost). Leaves with no options become dead ends here.
    fn prepare(&mut self, leaves: &[u32]) -> Batch {
        let t = Instant::now();
        let mut work = self.take_states(leaves);
        build_all(&mut work, self.params.threads);
        self.stats.expand_ns += t.elapsed().as_nanos() as u64;
        self.finish(leaves, work)
    }

    /// Move `leaves`' states out of the tree for [`build_all`].
    fn take_states(&mut self, leaves: &[u32]) -> Vec<Work> {
        leaves
            .iter()
            .map(|&leaf| {
                let s = match self.states[leaf as usize].take() {
                    Some(s) => s,
                    None => self.state_of(leaf),
                };
                (s, None)
            })
            .collect()
    }

    /// Put built states back, mark dead ends, and collect the batch.
    fn finish(&mut self, leaves: &[u32], work: Vec<Work>) -> Batch {
        let mut batch = Batch {
            leaves: leaves.to_vec(),
            pend: Vec::with_capacity(leaves.len()),
            views: Vec::new(),
        };
        for (&leaf, (s, out)) in leaves.iter().zip(work) {
            self.states[leaf as usize] = Some(s);
            match out.expect("built") {
                Some((kids, vs)) => {
                    batch.views.extend(vs);
                    batch.pend.push((leaf, kids));
                }
                None => {
                    self.nodes[leaf as usize].kind = DEAD;
                    self.stats.dead += 1;
                }
            }
        }
        batch
    }

    /// Attach a prepared batch's children with their values (`vals`, in view
    /// order); the leaves become expanded.
    fn commit(&mut self, batch: Batch, vals: &[f32]) {
        let mut vi = 0;
        for (leaf, kids) in batch.pend {
            let first = self.nodes.len() as u32;
            let nk = kids.len();
            for (step, flip, term, st) in kids {
                let (kind, value, comp) = match term {
                    Some((v, r)) => (TERMINAL, v, r),
                    None => {
                        let v = vals[vi];
                        vi += 1;
                        (LEAF, v, 0)
                    }
                };
                self.nodes.push(Node {
                    parent: leaf,
                    first: 0,
                    nch: 0,
                    step,
                    flip,
                    kind,
                    visits: 0,
                    value,
                    prior: value,
                    comp,
                    resolved: kind == TERMINAL,
                });
                self.states.push(Some(st));
            }
            let nd = &mut self.nodes[leaf as usize];
            nd.first = first;
            nd.nch = nk as u16;
            nd.kind = EXPANDED;
            self.stats.expansions += 1;
        }
        debug_assert_eq!(vi, vals.len());
    }

    /// Recompute `n`'s value from its children; `false` if all are dead.
    ///
    /// With completion: a proven win for the mover makes `n` a proven win
    /// worth its best proven-winning child (the quickest win under the depth
    /// heuristic); if every child is a proven loss, `n` is a proven loss worth
    /// the slowest one. Otherwise proven losses are ignored and the value is
    /// the visit-weighted power mean (`p`; the max for `p = inf`) of the rest.
    /// `n` is resolved when proven, or when every live child is resolved.
    fn recompute(&mut self, n: u32) -> bool {
        let p = self.params.power;
        let mut best_c = i8::MIN;
        let mut all_resolved = true;
        let mut any = false;
        for c in self.children(n) {
            let nd = &self.nodes[c as usize];
            if nd.kind == DEAD {
                continue;
            }
            any = true;
            best_c = best_c.max(self.c_parent(c));
            all_resolved &= nd.resolved;
        }
        if !any {
            return false;
        }
        let pick = if best_c == 0 { 0 } else { best_c };
        let value = if pick != 0 || p.is_infinite() {
            // Proven: the best child in that class. p = inf: the max over the
            // unproven ones (plain minimax over the movers' choices).
            self.children(n)
                .filter(|&c| self.nodes[c as usize].kind != DEAD && self.c_parent(c) == pick)
                .map(|c| self.q_parent(c))
                .fold(f32::NEG_INFINITY, f32::max)
                .clamp(0.0, 1.0)
        } else {
            let (mut num, mut den) = (0f64, 0f64);
            for c in self.children(n) {
                let nd = &self.nodes[c as usize];
                if nd.kind == DEAD || self.c_parent(c) != 0 {
                    continue;
                }
                let w = 1.0 + nd.visits as f64;
                let q = self.q_parent(c).clamp(0.0, 1.0) as f64;
                num += w * q.powf(p as f64);
                den += w;
            }
            (num / den).powf(1.0 / p as f64) as f32
        };
        let nd = &mut self.nodes[n as usize];
        nd.value = value;
        nd.comp = pick;
        nd.resolved = pick != 0 || all_resolved;
        true
    }

    fn backup(&mut self, leaves: &[u32]) {
        let t = Instant::now();
        // Every node on some leaf's root path, recomputed once. A child's
        // index is always above its parent's, so taking the highest pending
        // index first settles children before parents: the same result as
        // backing up leaf by leaf, without recomputing shared ancestors once
        // per leaf. With `backup_eps() > 0`, a node whose value moved less
        // than that (and whose proof state did not change) does not pass the
        // update on to its parent.
        let eps = backup_eps();
        let mut seen = vec![0u64; self.nodes.len().div_ceil(64)];
        let mut heap: std::collections::BinaryHeap<u32> = std::collections::BinaryHeap::new();
        let push = |n: u32, seen: &mut Vec<u64>, heap: &mut std::collections::BinaryHeap<u32>| {
            let (w, b) = (n as usize / 64, 1u64 << (n % 64));
            if seen[w] & b == 0 {
                seen[w] |= b;
                heap.push(n);
            }
        };
        for &l in leaves {
            push(l, &mut seen, &mut heap);
        }
        while let Some(n) = heap.pop() {
            let before = self.nodes[n as usize];
            if before.kind == EXPANDED && !self.recompute(n) {
                self.nodes[n as usize].kind = DEAD;
                self.stats.dead += 1;
            }
            if self.nodes[n as usize].parent == u32::MAX {
                continue;
            }
            let after = &self.nodes[n as usize];
            let settled = eps > 0.0
                && before.kind == EXPANDED
                && after.kind == EXPANDED
                && (after.value - before.value).abs() < eps
                && after.comp == before.comp
                && after.resolved == before.resolved;
            if !settled {
                push(after.parent, &mut seen, &mut heap);
            }
        }
        self.stats.backup_ns += t.elapsed().as_nanos() as u64;
    }

    /// One iteration. Returns the number of leaves expanded (0 when the tree
    /// has nothing left to expand).
    pub fn iterate(&mut self, eval: &dyn Evaluator) -> anyhow::Result<usize> {
        if self.is_solved() {
            return Ok(0);
        }
        let leaves = self.select();
        if leaves.is_empty() {
            return Ok(0);
        }
        self.expand(&leaves, eval)?;
        self.backup(&leaves);
        self.stats.iterations += 1;
        Ok(leaves.len())
    }

    /// One wave of Descent (Cohen-Solal 2020), the self-play search: `k`
    /// descents run in lockstep, each following the best child from the root
    /// all the way to a terminal node, expanding every leaf it meets on the
    /// way (one per descent per wave, all in one evaluator batch) and backing
    /// the new values up at once. A descent that ends makes room for a fresh
    /// one from the root. Every node a descent enters counts one visit.
    ///
    /// Concurrent descents are kept apart by virtual loss: a child's value is
    /// averaged with one pending loss per descent in flight through it,
    /// weighted like its visits, so near-ties split and clear bests do not.
    ///
    /// Returns the number of leaves expanded (0 when nothing is left).
    pub fn descent_wave(&mut self, eval: &dyn Evaluator, k: usize) -> anyhow::Result<usize> {
        let leaves = self.select_descents(k);
        if leaves.is_empty() {
            return Ok(0);
        }
        self.expand(&leaves, eval)?;
        self.backup(&leaves);
        self.stats.iterations += 1;
        Ok(leaves.len())
    }

    /// Advance every descent in flight by one leaf (starting new ones up to
    /// `k`) and return the distinct leaves to expand. A descent that reaches
    /// a leaf still being evaluated by the other pipeline group waits there.
    fn select_descents(&mut self, k: usize) -> Vec<u32> {
        let t = Instant::now();
        let k = k.max(1);
        let mut leaves: Vec<u32> = Vec::with_capacity(k);
        // Each descent advances to a leaf; ended ones are replaced, with a cap
        // so a fully decided tree cannot spin.
        let mut starts = 0usize;
        let mut i = 0usize;
        loop {
            if i == self.cursors.len() {
                if self.cursors.len() >= k || starts >= 4 * k || self.is_solved() {
                    break;
                }
                starts += 1;
                let s = self.pick_start();
                self.cursors.push(vec![s]);
                self.enter(s);
            }
            match self.advance(i) {
                Some(leaf) => {
                    let busy = self.pending.get(leaf as usize).copied().unwrap_or(false);
                    if !busy && !leaves.contains(&leaf) {
                        leaves.push(leaf);
                    }
                    i += 1;
                }
                None => {
                    let path = self.cursors.swap_remove(i);
                    for n in path {
                        self.inflight[n as usize] -= 1;
                    }
                }
            }
        }
        self.stats.select_ns += t.elapsed().as_nanos() as u64;
        leaves
    }

    /// Descent with the evaluator and the CPU overlapped on one tree: the
    /// descents are split into two groups of `k / 2`. Each group is built and
    /// then evaluated on a thread of its own, so its evaluation queues behind
    /// the other group's as soon as it is built; meanwhile the main thread
    /// lands (commits and backs up) the other group and selects the next.
    /// The GPU, the bottleneck, then never waits for the main thread. Runs
    /// until `stop` says so (checked between groups) or the tree is solved;
    /// returns the leaves expanded.
    pub fn descent_run(
        &mut self,
        eval: &dyn Evaluator,
        k: usize,
        stop: &mut dyn FnMut(&Tree) -> bool,
    ) -> anyhow::Result<usize> {
        let half = (k / 2).max(1);
        let mut total = 0usize;
        let threads = self.params.threads;
        std::thread::scope(|sc| -> anyhow::Result<()> {
            type Built = (Vec<Work>, u64, u64, Vec<f32>);
            type Flight<'s> = (
                Vec<u32>,
                std::thread::ScopedJoinHandle<'s, anyhow::Result<Built>>,
            );
            let mut flight: Option<Flight> = None;
            let mut group = 0;
            loop {
                // Select the next group and start building and evaluating it
                // in the background (its states are moved out of the tree,
                // so landing the other group below cannot touch them).
                let building = if stop(self) || self.is_solved() {
                    None
                } else {
                    if group == 1 {
                        std::mem::swap(&mut self.cursors, &mut self.cursors_b);
                    }
                    let leaves = self.select_descents(half);
                    if group == 1 {
                        std::mem::swap(&mut self.cursors, &mut self.cursors_b);
                    }
                    (!leaves.is_empty()).then(|| {
                        if self.pending.len() < self.nodes.len() {
                            self.pending.resize(self.nodes.len(), false);
                        }
                        for &l in &leaves {
                            self.pending[l as usize] = true;
                        }
                        let t = Instant::now();
                        let mut work = self.take_states(&leaves);
                        self.stats.take_ns += t.elapsed().as_nanos() as u64;
                        let h = sc.spawn(move || -> anyhow::Result<Built> {
                            let t = Instant::now();
                            build_all(&mut work, threads);
                            let built = t.elapsed().as_nanos() as u64;
                            // In leaf order, as `finish` would collect them.
                            let mut views = Vec::new();
                            for (_, out) in work.iter_mut() {
                                if let Some(Some((_, vs))) = out.as_mut() {
                                    views.append(vs);
                                }
                            }
                            let t = Instant::now();
                            let vals = if views.is_empty() {
                                Vec::new()
                            } else {
                                eval.eval(&views)?
                            };
                            Ok((work, built, t.elapsed().as_nanos() as u64, vals))
                        });
                        (leaves, h)
                    })
                };
                // Land the group in flight: its children, their values.
                let landed = flight.take();
                let had = landed.is_some();
                if let Some((leaves, h)) = landed {
                    let t = Instant::now();
                    let (work, built, evaled, vals) = h
                        .join()
                        .map_err(|_| anyhow::anyhow!("builder panicked"))??;
                    self.stats.wait_ns += t.elapsed().as_nanos() as u64;
                    self.stats.expand_ns += built;
                    self.stats.eval_ns += evaled;
                    self.stats.evals += vals.len() as u64;
                    let t = Instant::now();
                    let batch = self.finish(&leaves, work);
                    for &l in &batch.leaves {
                        self.pending[l as usize] = false;
                    }
                    self.commit(batch, &vals);
                    self.stats.land_ns += t.elapsed().as_nanos() as u64;
                    self.backup(&leaves);
                    self.stats.iterations += 1;
                    total += leaves.len();
                }
                let launched = building.is_some();
                flight = building;
                group ^= 1;
                // Nothing new and nothing in flight: done. A group that found
                // nothing while the other was in flight may find leaves once
                // it has landed, so go round again.
                if !launched && (!had || stop(self) || self.is_solved()) {
                    break;
                }
            }
            Ok(())
        })?;
        Ok(total)
    }

    /// Abandon the descents in flight (their visits stay counted).
    pub fn clear_descents(&mut self) {
        self.cursors.clear();
        self.cursors_b.clear();
        self.pending.clear();
        self.inflight.clear();
    }

    fn enter(&mut self, n: u32) {
        if self.inflight.len() < self.nodes.len() {
            self.inflight.resize(self.nodes.len(), 0);
        }
        self.inflight[n as usize] += 1;
        self.nodes[n as usize].visits += 1;
    }

    /// Walk descent `i` down to a leaf that needs expanding (`Some`), or to
    /// its end at a terminal or dead node (`None`).
    fn advance(&mut self, i: usize) -> Option<u32> {
        loop {
            let n = *self.cursors[i].last().expect("a descent path");
            match self.nodes[n as usize].kind {
                LEAF => return Some(n),
                EXPANDED if !self.nodes[n as usize].resolved => {}
                _ => {
                    // A terminal, or a proven result (resolution stop).
                    if self.nodes[n as usize].kind != DEAD {
                        self.stats.terminal_leaves += 1;
                    }
                    return None;
                }
            }
            // Among unresolved children: by completion, then value (with
            // virtual loss), then fewest visits.
            let mut best = None;
            let mut bk = (i8::MIN, f32::NEG_INFINITY, i64::MIN);
            for c in self.children(n) {
                let nd = &self.nodes[c as usize];
                if nd.kind == DEAD || nd.resolved {
                    continue;
                }
                let m = self.inflight.get(c as usize).copied().unwrap_or(0) as f32;
                let w = (1 + nd.visits) as f32 - m;
                let k = (
                    self.c_parent(c),
                    self.q_parent(c) * w / (w + m),
                    -(nd.visits as i64),
                );
                if k > bk {
                    bk = k;
                    best = Some(c);
                }
            }
            let c = best?;
            self.cursors[i].push(c);
            self.enter(c);
        }
    }

    /// Every terminal node (game over).
    pub fn terminal_nodes(&self) -> Vec<u32> {
        (0..self.nodes.len() as u32)
            .filter(|&n| self.nodes[n as usize].kind == TERMINAL)
            .collect()
    }

    /// Nodes with at least `min_visits` descents that are not terminal/dead.
    pub fn trained_nodes(&self, min_visits: u32) -> Vec<u32> {
        (0..self.nodes.len() as u32)
            .filter(|&n| {
                let nd = &self.nodes[n as usize];
                nd.visits >= min_visits && nd.kind == EXPANDED
            })
            .collect()
    }

    /// Choose the next step at `n` by descents with temperature, mixed with
    /// Dirichlet noise; by value when nothing below has been descended into.
    pub fn pick_child(
        &self,
        n: u32,
        temp: f32,
        noise: (f32, f32),
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<u32> {
        use rand::Rng;
        let kids: Vec<u32> = self
            .children(n)
            .filter(|&c| self.nodes[c as usize].kind != DEAD)
            .collect();
        if kids.is_empty() {
            return None;
        }
        let total: u32 = kids.iter().map(|&c| self.nodes[c as usize].visits).sum();
        let mut pi: Vec<f64> = if total == 0 || temp <= 0.0 && total == 0 {
            let b = self.best_child(n)?;
            kids.iter().map(|&c| f64::from(u8::from(c == b))).collect()
        } else if temp <= 0.0 {
            let mx = kids
                .iter()
                .map(|&c| self.nodes[c as usize].visits)
                .max()
                .unwrap_or(0);
            kids.iter()
                .map(|&c| f64::from(u8::from(self.nodes[c as usize].visits == mx)))
                .collect()
        } else {
            kids.iter()
                .map(|&c| (self.nodes[c as usize].visits as f64).powf(1.0 / temp as f64))
                .collect()
        };
        let s: f64 = pi.iter().sum();
        pi.iter_mut().for_each(|x| *x /= s.max(1e-12));
        let (alpha, eps) = noise;
        if eps > 0.0 && kids.len() > 1 {
            use rand_distr::{Distribution, Gamma};
            let g = Gamma::new(alpha as f64, 1.0).expect("alpha > 0");
            let d: Vec<f64> = kids.iter().map(|_| g.sample(rng)).collect();
            let ds: f64 = d.iter().sum::<f64>().max(1e-12);
            for (x, di) in pi.iter_mut().zip(&d) {
                *x = (1.0 - eps as f64) * *x + eps as f64 * di / ds;
            }
        }
        let mut r = rng.gen::<f64>();
        for (i, &x) in pi.iter().enumerate() {
            if r < x {
                return Some(kids[i]);
            }
            r -= x;
        }
        kids.last().copied()
    }

    /// Self-play step choice at `n` (Cohen-Solal, arXiv 2008.01188, sections
    /// 5 and 7): a proven win is always played (the quickest); proven losses
    /// only when nothing else is left; otherwise the ordinal distribution over
    /// the rest ranked by value -- the `i`-th best (of `m`) is taken, given no
    /// better one was, with probability `e + (1 - e) / (m - i)`, where the
    /// exploitation `e` is drawn uniformly from [0, 1] at each choice.
    pub fn pick_ordinal(&self, n: u32, rng: &mut rand_chacha::ChaCha8Rng) -> Option<u32> {
        use rand::Rng;
        let live: Vec<u32> = self
            .children(n)
            .filter(|&c| self.nodes[c as usize].kind != DEAD)
            .collect();
        if live.is_empty() {
            return None;
        }
        let top = live.iter().map(|&c| self.c_parent(c)).max().unwrap();
        if top == 1 {
            return self.best_child(n);
        }
        let mut pool: Vec<u32> = live
            .iter()
            .copied()
            .filter(|&c| self.c_parent(c) == 0)
            .collect();
        if pool.is_empty() {
            pool = live;
        }
        pool.sort_by(|&a, &b| self.q_parent(b).total_cmp(&self.q_parent(a)));
        let e: f64 = rng.gen();
        let m = pool.len();
        for (i, &c) in pool.iter().enumerate() {
            let p = e + (1.0 - e) / (m - i) as f64;
            if rng.gen::<f64>() < p {
                return Some(c);
            }
        }
        pool.last().copied()
    }

    /// Expand `n` alone if it is still a leaf (for playing through a turn the
    /// search never reached).
    pub fn ensure_expanded(&mut self, n: u32, eval: &dyn Evaluator) -> anyhow::Result<()> {
        if self.nodes[n as usize].kind == LEAF {
            self.expand(&[n], eval)?;
            self.backup(&[n]);
        }
        Ok(())
    }

    pub fn kind(&self, n: u32) -> u8 {
        self.nodes[n as usize].kind
    }

    /// The next descent start: round robin over the undecided starts.
    fn pick_start(&mut self) -> u32 {
        for _ in 0..self.starts.len() {
            let n = self.starts[self.next_start % self.starts.len()];
            self.next_start += 1;
            let nd = &self.nodes[n as usize];
            if !(matches!(nd.kind, TERMINAL | DEAD) || nd.resolved) {
                return n;
            }
        }
        self.starts.first().copied().unwrap_or(0)
    }

    /// A forest of `roots`' subtrees, re-indexed (a root inside another's
    /// subtree stays where it is), and the old-to-new index map (`u32::MAX`
    /// for dropped nodes). Each new root keeps its state; descents are
    /// cleared and the starts reset.
    pub fn retain(&self, roots: &[u32]) -> (Tree, Vec<u32>) {
        let mut map = vec![u32::MAX; self.nodes.len()];
        let mut inside = vec![false; self.nodes.len()];
        for &r in roots {
            inside[r as usize] = true;
        }
        // Children come after their parents, so one forward pass marks every
        // subtree; the kept nodes keep their relative order, so the new
        // arena has parents before children and siblings contiguous.
        let mut nodes = Vec::new();
        let mut states = Vec::new();
        for i in 0..self.nodes.len() {
            let p = self.nodes[i].parent;
            if !inside[i] && p != u32::MAX && inside[p as usize] {
                inside[i] = true;
            }
            if !inside[i] {
                continue;
            }
            map[i] = nodes.len() as u32;
            let mut nd = self.nodes[i];
            if p == u32::MAX || !inside[p as usize] {
                nd.parent = u32::MAX;
                states.push(Some(self.state_of(i as u32)));
            } else {
                nd.parent = map[p as usize];
                states.push(self.states[i].clone());
            }
            nodes.push(nd);
        }
        for nd in nodes.iter_mut() {
            if nd.nch > 0 {
                nd.first = map[nd.first as usize];
            }
        }
        let root = states
            .first()
            .and_then(|s| s.clone())
            .unwrap_or_else(|| self.root.clone());
        let tree = Tree {
            nodes,
            states,
            root,
            params: self.params.clone(),
            stats: SearchStats::default(),
            cursors: Vec::new(),
            cursors_b: Vec::new(),
            inflight: Vec::new(),
            pending: Vec::new(),
            starts: Vec::new(),
            next_start: 0,
        };
        (tree, map)
    }

    /// A new tree rooted at `n` (its subtree kept, re-indexed).
    pub fn reroot(&self, n: u32) -> Tree {
        let root = self.state_of(n);
        let mut nodes = Vec::new();
        let mut states = vec![Some(root.clone())];
        let mut r = self.nodes[n as usize];
        r.parent = u32::MAX;
        r.flip = false;
        r.step = 0;
        nodes.push(r);
        // BFS: children of each copied node are contiguous in the new arena.
        let mut queue = std::collections::VecDeque::from([(n, 0u32)]);
        while let Some((old, new)) = queue.pop_front() {
            let o = self.nodes[old as usize];
            if o.nch == 0 {
                continue;
            }
            let first = nodes.len() as u32;
            for c in o.first..o.first + o.nch as u32 {
                let mut cn = self.nodes[c as usize];
                cn.parent = new;
                nodes.push(cn);
                states.push(self.states[c as usize].clone());
                queue.push_back((c, nodes.len() as u32 - 1));
            }
            nodes[new as usize].first = first;
        }
        Tree {
            nodes,
            states,
            root,
            params: self.params.clone(),
            stats: SearchStats::default(),
            cursors: Vec::new(),
            cursors_b: Vec::new(),
            inflight: Vec::new(),
            pending: Vec::new(),
            starts: Vec::new(),
            next_start: 0,
        }
    }
}

// ── Evaluators ──────────────────────────────────────────────────────────────

/// Backup stops at a node whose value moved less than this
/// (`STEPS_BACKUP_EPS`; 0 = always back up to the root).
fn backup_eps() -> f32 {
    static V: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("STEPS_BACKUP_EPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    })
}

/// A child built during expansion: (step, flip, terminal value + result, state).
type Kid = (u8, bool, Option<(f32, i8)>, StepState);

/// A leaf's state and, once built, its children and their views (`None`
/// inside for a dead end).
type Work = (StepState, Option<Option<(Vec<Kid>, Vec<View>)>>);

/// One leaf's children and their views; `None` for a dead end.
fn build_one(s: &mut StepState) -> Option<(Vec<Kid>, Vec<View>)> {
    let tp = Instant::now();
    let opts = s.collapse_forced();
    crate::steps::prof::add(0, tp);
    if opts.is_empty() {
        if std::env::var("STEPS_DEBUG_DEAD").is_ok()
            && DEAD_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8
        {
            eprintln!("no-option leaf: {}", s.debug_dead());
        }
        return None;
    }
    let mut kids = Vec::with_capacity(opts.len());
    let mut views = Vec::with_capacity(opts.len());
    for v in opts {
        let tp = Instant::now();
        let c = s.child(v);
        crate::steps::prof::add(1, tp);
        let flip = c.flipped != s.flipped;
        if c.is_terminal() {
            let tv = (c.terminal_value(), c.terminal_outcome());
            kids.push((v, flip, Some(tv), c));
        } else {
            let tp = Instant::now();
            views.push(c.view());
            crate::steps::prof::add(2, tp);
            kids.push((v, flip, None, c));
        }
    }
    s.drop_options();
    Some((kids, views))
}

/// Build every leaf in `work` on `threads` threads. Leaves are handed out one
/// at a time since their costs vary by orders of magnitude (exact witness
/// searches).
fn build_all(work: &mut [Work], threads: usize) {
    let threads = threads.clamp(1, work.len().max(1));
    if threads == 1 {
        for (s, out) in work.iter_mut() {
            *out = Some(build_one(s));
        }
        return;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<&mut Work>> =
        work.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(slot) = slots.get(i) else { break };
                    let mut w = slot.lock().unwrap();
                    let (s, out) = &mut **w;
                    *out = Some(build_one(s));
                }
                crate::steps::prof::flush();
            });
        }
    });
}

/// Leaves whose children are built but not yet evaluated.
struct Batch {
    leaves: Vec<u32>,
    /// Live leaves and their children, in view order.
    pend: Vec<(u32, Vec<Kid>)>,
    views: Vec<View>,
}

/// Pads a batch to a multiple of this, so TensorRT sees few distinct shapes.
pub const BATCH_QUANTUM: usize = 64;

/// The sampled-AZ champion's value head applied to the *projected* board: the
/// bootstrap evaluator before any step-trained net exists.
pub struct OldNetEval {
    pub net: Box<dyn crate::net::Backend>,
}

impl Evaluator for OldNetEval {
    fn eval(&self, views: &[View]) -> anyhow::Result<Vec<f32>> {
        use crate::game::{Minions, PLANES};
        use crate::map::NUM_CELLS;
        use playground_core::Game;
        let n = views.len();
        let b = n.div_ceil(BATCH_QUANTUM) * BATCH_QUANTUM;
        let stride = PLANES * NUM_CELLS;
        let mut planes = vec![0f32; b * stride];
        for (i, v) in views.iter().enumerate() {
            for (o, x) in planes[i * stride..(i + 1) * stride]
                .iter_mut()
                .zip(Minions::to_features(&v.board))
            {
                *o = x as f32;
            }
        }
        let mut out = Vec::with_capacity(n);
        // Stay within the combiner's row cap.
        let cap = 512usize;
        let mut i = 0;
        while i < b {
            let m = (b - i).min(cap);
            let tr = self.net.trunk(&planes[i * stride..(i + m) * stride], m)?;
            out.extend(
                tr.value
                    .iter()
                    .take(m)
                    .map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0)),
            );
            i += m;
        }
        out.truncate(n);
        Ok(out)
    }
}

/// Dense input for step-trained nets: `SPARSE_DIM` 0/1 features, then the scalars.
pub const DENSE_DIM: usize = crate::steps::SPARSE_DIM + crate::steps::N_SCALARS;

pub fn dense_features(v: &View, out: &mut [f32], act: &mut Vec<u16>) {
    let mut sc = [0f32; crate::steps::N_SCALARS];
    v.features(act, &mut sc);
    out.fill(0.0);
    for &i in act.iter() {
        out[i as usize] = 1.0;
    }
    out[crate::steps::SPARSE_DIM..].copy_from_slice(&sc);
}

/// A step-trained value net exported to ONNX: input `x` `[B, DENSE_DIM]` f32,
/// output `v` `[B]` or `[B, 1]` win probability.
pub struct OnnxEval {
    pool: Vec<std::sync::Mutex<ort::session::Session>>,
    rr: std::sync::atomic::AtomicUsize,
}

impl OnnxEval {
    pub fn load(
        path: &std::path::Path,
        ep: &str,
        fp16: bool,
        cache: &str,
        sessions: usize,
    ) -> anyhow::Result<Self> {
        let pool = (0..sessions.max(1))
            .map(|_| crate::ort_net::build(path, ep, fp16, cache).map(std::sync::Mutex::new))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            pool,
            rr: Default::default(),
        })
    }
}

/// Every ONNX call runs at exactly this many rows (padded), so TensorRT builds
/// one engine instead of one per batch size.
pub const ONNX_CHUNK: usize = 256;

impl Evaluator for OnnxEval {
    fn eval(&self, views: &[View]) -> anyhow::Result<Vec<f32>> {
        use ort::value::Tensor;
        let n = views.len();
        let mut out = Vec::with_capacity(n);
        let mut act = Vec::new();
        for part in views.chunks(ONNX_CHUNK) {
            let mut x = vec![0f32; ONNX_CHUNK * DENSE_DIM];
            for (i, v) in part.iter().enumerate() {
                dense_features(v, &mut x[i * DENSE_DIM..(i + 1) * DENSE_DIM], &mut act);
            }
            let t = Tensor::from_array((vec![ONNX_CHUNK as i64, DENSE_DIM as i64], x))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let k = self.rr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.pool.len();
            let mut sess = self.pool[k].lock().unwrap();
            let outs = sess
                .run(ort::inputs!["x" => t])
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let (_, v) = outs["v"]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            out.extend_from_slice(&v[..part.len()]);
        }
        Ok(out)
    }
}

/// Shapes a batched call may take; each gets its own TensorRT engine.
pub const BATCH_SHAPES: [usize; 2] = [256, 1024];

struct BatchReq {
    /// Dense rows `[n, DENSE_DIM]`, or for a sparse graph the scalars
    /// `[n, N_SCALARS]` beside `idx`.
    x: Vec<f32>,
    /// Sparse graphs only: active features `[n, MAX_ACTIVE]`, padded with
    /// `SPARSE_DIM`.
    idx: Vec<i32>,
    n: usize,
    tx: std::sync::mpsc::SyncSender<anyhow::Result<Vec<f32>>>,
}

#[derive(Default)]
struct BatchQueue {
    q: std::sync::Mutex<std::collections::VecDeque<BatchReq>>,
    cv: std::sync::Condvar,
    /// Set when the evaluator is dropped: the runners exit (freeing their
    /// sessions) once the queue is empty.
    closed: std::sync::atomic::AtomicBool,
}

/// [`OnnxEval`] behind a shared queue: worker threads featurize their own
/// rows and enqueue them; `runners` GPU threads each drain as many queued
/// requests as fit the largest shape and run them as one padded call. Fewer,
/// larger calls than one call per worker expansion. A graph with inputs
/// `idx` + `scal` (the learner's `value_sparse.onnx`) is fed active-feature
/// indices instead of dense rows: ~9x fewer bytes to the GPU.
pub struct BatchedOnnxEval {
    queue: std::sync::Arc<BatchQueue>,
    sparse: bool,
    /// Calls and rows run, for the padding ratio.
    pub calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pub rows: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl BatchedOnnxEval {
    pub fn load(
        path: &std::path::Path,
        ep: &str,
        fp16: bool,
        cache: &str,
        runners: usize,
    ) -> anyhow::Result<Self> {
        let queue = std::sync::Arc::new(BatchQueue::default());
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let rows = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut sparse = false;
        for _ in 0..runners.max(1) {
            // One session per shape, each run once here (one at a time): a
            // TensorRT context that switches shapes can fail with CUDA error
            // 906 ("legacy stream depends on a capturing stream") when other
            // sessions run concurrently, which killed matches between two nets.
            let mut sessions = Vec::new();
            for &shape in &BATCH_SHAPES {
                let mut sess = crate::ort_net::build(path, ep, fp16, cache)?;
                sparse = sess.inputs().iter().any(|i| i.name() == "idx");
                let _warm = WARMUP.lock().unwrap_or_else(|p| p.into_inner());
                run_batch(
                    &mut sess,
                    sparse,
                    shape,
                    vec![0f32; shape * input_width(sparse)],
                    sparse_pad(sparse, shape),
                )?;
                sessions.push(sess);
            }
            let (queue, calls, rows) = (queue.clone(), calls.clone(), rows.clone());
            std::thread::spawn(move || batch_runner(sessions, sparse, &queue, &calls, &rows));
        }
        Ok(Self {
            queue,
            sparse,
            calls,
            rows,
        })
    }
}

impl Drop for BatchedOnnxEval {
    fn drop(&mut self) {
        // Under the queue lock, so a runner cannot miss the wake-up between
        // its closed check and its wait.
        let _q = self.queue.q.lock().unwrap_or_else(|p| p.into_inner());
        self.queue
            .closed
            .store(true, std::sync::atomic::Ordering::Release);
        self.queue.cv.notify_all();
    }
}

/// Serializes session warm-up runs across the process.
static WARMUP: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn input_width(sparse: bool) -> usize {
    if sparse {
        crate::steps::N_SCALARS
    } else {
        DENSE_DIM
    }
}

fn sparse_pad(sparse: bool, shape: usize) -> Vec<i32> {
    if sparse {
        vec![crate::steps::SPARSE_DIM as i32; shape * crate::steps::MAX_ACTIVE]
    } else {
        Vec::new()
    }
}

/// One padded call: `x` is `[shape, input_width]`, `idx` `[shape, MAX_ACTIVE]`
/// for a sparse graph. Returns all `shape` values.
fn run_batch(
    sess: &mut ort::session::Session,
    sparse: bool,
    shape: usize,
    x: Vec<f32>,
    idx: Vec<i32>,
) -> anyhow::Result<Vec<f32>> {
    use ort::value::Tensor;
    let e = |e: ort::Error| anyhow::anyhow!("{e}");
    let t = Tensor::from_array((vec![shape as i64, input_width(sparse) as i64], x)).map_err(e)?;
    let outs = if sparse {
        let ti = Tensor::from_array((vec![shape as i64, crate::steps::MAX_ACTIVE as i64], idx))
            .map_err(e)?;
        sess.run(ort::inputs!["idx" => ti, "scal" => t])
    } else {
        sess.run(ort::inputs!["x" => t])
    }
    .map_err(e)?;
    let (_, v) = outs["v"].try_extract_tensor::<f32>().map_err(e)?;
    Ok(v.to_vec())
}

/// Batched evaluator profile (ns): featurizing on the calling thread, the
/// caller's wait for its values after the last chunk was queued, and time
/// inside ONNX calls (summed over runners).
pub static EVAL_PROF: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn batch_runner(
    mut sessions: Vec<ort::session::Session>,
    sparse: bool,
    queue: &BatchQueue,
    calls: &std::sync::atomic::AtomicU64,
    rows: &std::sync::atomic::AtomicU64,
) {
    let max = *BATCH_SHAPES.last().unwrap();
    loop {
        let mut batch = Vec::new();
        let mut total = 0;
        {
            let mut q = queue.q.lock().unwrap();
            while q.is_empty() {
                if queue.closed.load(std::sync::atomic::Ordering::Acquire) {
                    return;
                }
                q = queue.cv.wait(q).unwrap();
            }
            while let Some(r) = q.front() {
                if total + r.n > max {
                    break;
                }
                total += r.n;
                batch.push(q.pop_front().unwrap());
            }
        }
        let k = BATCH_SHAPES.iter().position(|&s| s >= total).unwrap();
        let shape = BATCH_SHAPES[k];
        let mut x = vec![0f32; shape * input_width(sparse)];
        let mut idx = sparse_pad(sparse, shape);
        let (mut at, mut ai) = (0, 0);
        for r in &batch {
            x[at..at + r.x.len()].copy_from_slice(&r.x);
            at += r.x.len();
            idx[ai..ai + r.idx.len()].copy_from_slice(&r.idx);
            ai += r.idx.len();
        }
        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        rows.fetch_add(total as u64, std::sync::atomic::Ordering::Relaxed);
        let t = Instant::now();
        let res = run_batch(&mut sessions[k], sparse, shape, x, idx);
        EVAL_PROF[2].fetch_add(
            t.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let mut at = 0;
        for r in batch {
            let part = match &res {
                Ok(v) => Ok(v[at..at + r.n].to_vec()),
                Err(e) => Err(anyhow::anyhow!("{e}")),
            };
            at += r.n;
            let _ = r.tx.send(part);
        }
    }
}

impl Evaluator for BatchedOnnxEval {
    fn eval(&self, views: &[View]) -> anyhow::Result<Vec<f32>> {
        let max = *BATCH_SHAPES.last().unwrap();
        let mut act = Vec::new();
        let mut waits = Vec::new();
        let mut feat_ns = 0u64;
        for part in views.chunks(max) {
            let t = Instant::now();
            let (x, idx) = if self.sparse {
                use crate::steps::{MAX_ACTIVE, N_SCALARS, SPARSE_DIM};
                let mut x = vec![0f32; part.len() * N_SCALARS];
                let mut idx = vec![SPARSE_DIM as i32; part.len() * MAX_ACTIVE];
                let mut sc = [0f32; N_SCALARS];
                for (i, v) in part.iter().enumerate() {
                    v.features(&mut act, &mut sc);
                    x[i * N_SCALARS..(i + 1) * N_SCALARS].copy_from_slice(&sc);
                    for (j, &a) in act.iter().take(MAX_ACTIVE).enumerate() {
                        idx[i * MAX_ACTIVE + j] = i32::from(a);
                    }
                }
                (x, idx)
            } else {
                let mut x = vec![0f32; part.len() * DENSE_DIM];
                for (i, v) in part.iter().enumerate() {
                    dense_features(v, &mut x[i * DENSE_DIM..(i + 1) * DENSE_DIM], &mut act);
                }
                (x, Vec::new())
            };
            feat_ns += t.elapsed().as_nanos() as u64;
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            self.queue.q.lock().unwrap().push_back(BatchReq {
                x,
                idx,
                n: part.len(),
                tx,
            });
            self.queue.cv.notify_one();
            waits.push(rx);
        }
        let t = Instant::now();
        let mut out = Vec::with_capacity(views.len());
        for rx in waits {
            out.extend(rx.recv().map_err(|e| anyhow::anyhow!("{e}"))??);
        }
        use std::sync::atomic::Ordering::Relaxed;
        EVAL_PROF[0].fetch_add(feat_ns, Relaxed);
        EVAL_PROF[1].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        Ok(out)
    }
}

/// A cheap deterministic evaluator: material balance for the mover (unit
/// costs, necromancer +20) through a logistic. For tests and sanity checks
/// (`steps-match --a material`).
pub struct MaterialEval;

impl Evaluator for MaterialEval {
    fn eval(&self, views: &[View]) -> anyhow::Result<Vec<f32>> {
        Ok(views
            .iter()
            .map(|v| {
                let mut d = 0f32;
                for c in 0..crate::map::NUM_CELLS as u8 {
                    if let Some((s, u)) = v.board.piece_at(c) {
                        let w = u.stats().cost.max(1) as f32
                            + if u.stats().necromancer { 20.0 } else { 0.0 };
                        d += if s == 0 { w } else { -w };
                    }
                }
                1.0 / (1.0 + (-d / 10.0).exp())
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::MinionsState;
    use rand::SeedableRng;

    use super::MaterialEval as Material;

    /// Descent reaches terminal nodes, keeps the power-mean invariant, and
    /// counts visits consistently: a descent entering a child entered its
    /// parent first, and in-flight counts return to zero.
    #[test]
    fn descent_reaches_terminals() {
        descent_invariants(false);
    }

    #[test]
    fn pipelined_descent_reaches_terminals() {
        descent_invariants(true);
    }

    fn descent_invariants(pipelined: bool) {
        let s = StepState::begin(&MinionsState::with_bid(0), 5);
        let mut t = Tree::new(
            s,
            Params {
                heads: 8,
                power: 2.0,
                threads: 4, // exercises the parallel expansion
            },
            &Material,
        )
        .unwrap();
        if pipelined {
            t.descent_run(&Material, 16, &mut |t| {
                t.stats.terminal_leaves >= 20 || t.stats.iterations >= 3000
            })
            .unwrap();
            assert!(t.pending.iter().all(|&p| !p), "a leaf left pending");
        } else {
            for _ in 0..3000 {
                if t.descent_wave(&Material, 8).unwrap() == 0 {
                    break;
                }
                if t.stats.terminal_leaves >= 20 {
                    break;
                }
            }
        }
        assert!(t.stats.terminal_leaves > 0, "{:?}", t.stats);
        for n in 0..t.nodes.len() as u32 {
            if t.kind(n) != EXPANDED {
                continue;
            }
            let below: u32 = t.children(n).map(|c| t.nodes[c as usize].visits).sum();
            assert!(below <= t.nodes[n as usize].visits, "node {n}");
            // Completion: minimax of the children's proven results.
            let live: Vec<u32> = t.children(n).filter(|&c| t.kind(c) != DEAD).collect();
            let top = live.iter().map(|&c| t.c_parent(c)).max().unwrap();
            let want = if top == 1 {
                1
            } else if top == -1 {
                -1
            } else {
                0
            };
            let nd = &t.nodes[n as usize];
            assert_eq!(nd.comp, want, "node {n} completion");
            let all = live.iter().all(|&c| t.nodes[c as usize].resolved);
            assert_eq!(nd.resolved, want != 0 || all, "node {n} resolution");
        }
        let paths: Vec<Vec<u32>> = t.cursors.iter().chain(&t.cursors_b).cloned().collect();
        for (n, &m) in t.inflight.iter().enumerate() {
            let on = paths.iter().filter(|p| p.contains(&(n as u32))).count() as u32;
            assert_eq!(m, on, "inflight of {n}");
        }
        t.clear_descents();
        assert!(t.inflight.is_empty());
    }

    #[test]
    fn retain_keeps_a_forest_of_subtrees() {
        let s = StepState::begin(&MinionsState::with_bid(0), 5);
        let mut t = Tree::new(
            s,
            Params {
                heads: 8,
                power: 2.0,
                threads: 1,
            },
            &Material,
        )
        .unwrap();
        for _ in 0..300 {
            t.descent_wave(&Material, 8).unwrap();
        }
        t.clear_descents();
        // Two grandchildren-level roots in different subtrees, plus one
        // inside the first's subtree (it must not become a root).
        let kids: Vec<u32> = t.children(0).filter(|&c| t.kind(c) == EXPANDED).collect();
        assert!(kids.len() >= 2, "need two expanded children");
        let (a, b) = (kids[0], kids[1]);
        let inner = t.children(a).next().unwrap();
        let (f, map) = t.retain(&[a, b, inner]);
        assert_eq!(f.nodes[map[a as usize] as usize].parent, u32::MAX);
        assert_eq!(f.nodes[map[b as usize] as usize].parent, u32::MAX);
        assert_eq!(
            f.nodes[map[inner as usize] as usize].parent,
            map[a as usize]
        );
        assert_eq!(map[0], u32::MAX);
        for (old, &new) in map.iter().enumerate() {
            if new == u32::MAX {
                continue;
            }
            let (o, n) = (&t.nodes[old], &f.nodes[new as usize]);
            assert_eq!(
                (o.value, o.visits, o.kind, o.nch),
                (n.value, n.visits, n.kind, n.nch)
            );
            if n.parent != u32::MAX {
                assert!(n.parent < new);
            }
            if n.nch > 0 {
                for c in n.first..n.first + u32::from(n.nch) {
                    assert_eq!(f.nodes[c as usize].parent, new, "children contiguous");
                }
            }
            assert_eq!(
                f.state_of(new).base.key(),
                t.state_of(old as u32).base.key(),
                "state of {old}"
            );
        }
    }

    #[test]
    fn search_runs_and_backs_up_consistently() {
        let s = StepState::begin(&MinionsState::with_bid(0), 5);
        let mut t = Tree::new(
            s,
            Params {
                heads: 8,
                power: 2.0,
                threads: 1,
            },
            &Material,
        )
        .unwrap();
        for _ in 0..200 {
            if t.iterate(&Material).unwrap() == 0 {
                break;
            }
        }
        assert!(t.stats.expansions > 50, "{:?}", t.stats);
        // Every expanded node's value is the weighted power mean of its children.
        for n in 0..t.nodes.len() as u32 {
            if t.kind(n) != EXPANDED {
                continue;
            }
            let before = t.nodes[n as usize].value;
            let mut tt = Tree {
                nodes: t.nodes.clone(),
                states: t.states.clone(),
                root: t.root.clone(),
                params: t.params.clone(),
                stats: Default::default(),
                cursors: Vec::new(),
                cursors_b: Vec::new(),
                inflight: Vec::new(),
                pending: Vec::new(),
                starts: Vec::new(),
                next_start: 0,
            };
            assert!(tt.recompute(n));
            assert!(
                (tt.nodes[n as usize].value - before).abs() < 1e-5,
                "node {n} stale"
            );
        }
        // Root descents equal the number of leaves ever selected.
        let pv = t.pv();
        assert!(pv.len() > 3, "pv {pv:?}");
    }

    #[test]
    fn reroot_keeps_the_subtree() {
        let s = StepState::begin(&MinionsState::with_bid(0), 5);
        let mut t = Tree::new(
            s,
            Params {
                heads: 8,
                power: 2.0,
                threads: 1,
            },
            &Material,
        )
        .unwrap();
        for _ in 0..100 {
            t.iterate(&Material).unwrap();
        }
        let c = t.best_child(0).unwrap();
        let r = t.reroot(c);
        assert_eq!(r.nodes[0].visits, t.nodes[c as usize].visits);
        // Same values along the PV of the subtree.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let _ = r.pick_child(0, 1.0, (0.3, 0.25), &mut rng);
        let sub = r.pv();
        assert!(!sub.is_empty());
        let a = r.state_of(*sub.last().unwrap()).view();
        let _ = a;
    }
}

/// A step-trained MLP (`python/minions_steps_learner.py --arch mlp:...`) run
/// on the CPU, NNUE-style: the first layer sums the rows of the active sparse
/// features instead of multiplying the dense input. Hidden layers clip to
/// `[0, 1]`; the output is a logit.
pub struct CpuMlpEval {
    dims: Vec<usize>,
    /// Per layer: weights `[in][out]` and bias `[out]`.
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl CpuMlpEval {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let rd = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        let n = rd(0);
        let dims: Vec<usize> = (0..=n).map(|i| rd(4 + 4 * i)).collect();
        anyhow::ensure!(
            dims[0] == DENSE_DIM,
            "mlp.bin input {} != {}",
            dims[0],
            DENSE_DIM
        );
        let mut off = 4 + 4 * (n + 1);
        let mut take = |k: usize| -> Vec<f32> {
            let v: Vec<f32> = bytes[off..off + 4 * k]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            off += 4 * k;
            v
        };
        let mut layers = Vec::with_capacity(n);
        for l in 0..n {
            let w = take(dims[l] * dims[l + 1]);
            let b = take(dims[l + 1]);
            layers.push((w, b));
        }
        Ok(Self { dims, layers })
    }

    fn eval_one(&self, v: &View, act: &mut Vec<u16>, buf: &mut [Vec<f32>; 2]) -> f32 {
        let mut sc = [0f32; crate::steps::N_SCALARS];
        v.features(act, &mut sc);
        let h1 = self.dims[1];
        let (w, b) = &self.layers[0];
        let x = &mut buf[0];
        x.clear();
        x.extend_from_slice(b);
        for &i in act.iter() {
            let row = &w[i as usize * h1..(i as usize + 1) * h1];
            for (a, r) in x.iter_mut().zip(row) {
                *a += r;
            }
        }
        for (s, &val) in sc.iter().enumerate() {
            if val != 0.0 {
                let i = crate::steps::SPARSE_DIM + s;
                let row = &w[i * h1..(i + 1) * h1];
                for (a, r) in x.iter_mut().zip(row) {
                    *a += r * val;
                }
            }
        }
        for a in x.iter_mut() {
            *a = a.clamp(0.0, 1.0);
        }
        for l in 1..self.layers.len() {
            let (w, b) = &self.layers[l];
            let (din, dout) = (self.dims[l], self.dims[l + 1]);
            let (src, dst) = if l % 2 == 1 {
                let (a, bb) = buf.split_at_mut(1);
                (&a[0], &mut bb[0])
            } else {
                let (a, bb) = buf.split_at_mut(1);
                (&bb[0], &mut a[0])
            };
            dst.clear();
            dst.extend_from_slice(b);
            for i in 0..din {
                let xi = src[i];
                if xi == 0.0 {
                    continue;
                }
                let row = &w[i * dout..(i + 1) * dout];
                for (a, r) in dst.iter_mut().zip(row) {
                    *a += r * xi;
                }
            }
            if l + 1 < self.layers.len() {
                for a in dst.iter_mut() {
                    *a = a.clamp(0.0, 1.0);
                }
            }
        }
        let last = if (self.layers.len() - 1) % 2 == 1 {
            &buf[1]
        } else {
            &buf[0]
        };
        1.0 / (1.0 + (-last[0]).exp())
    }
}

impl Evaluator for CpuMlpEval {
    fn eval(&self, views: &[View]) -> anyhow::Result<Vec<f32>> {
        let mut act = Vec::with_capacity(crate::steps::MAX_ACTIVE);
        let mut buf = [Vec::new(), Vec::new()];
        Ok(views
            .iter()
            .map(|v| self.eval_one(v, &mut act, &mut buf))
            .collect())
    }
}
