//! Turn-lattice search, tier 1: AlphaZero-style MCTS over whole turns with
//! progressive widening. See `docs/minions-lattice.md`.
//!
//! A node is a turn's start; its children are the full turns its lattice
//! ([`crate::lattice`]) has found. While a node has fewer than
//! `ceil(pw_c * n^pw_alpha)` children, a visit to it widens: one tier-2
//! descent of its lattice, retried while it ends on known turns, up to
//! `retries` times -- after which the node stops widening and the visit
//! carries on into its children. A visit ends at a new child (backing up
//! that child's value) or at a terminal node. A child's prior is the
//! exponential of its mean log prior along the lattice paths that reached
//! it, normalized over the node's children. Values alternate perspective at
//! every edge (each edge is a turn); backups are means.

use anyhow::{bail, Result};

use crate::lattice::{Cursor, ExpandReq, Lattice, LatticeEval, Need, ValueOut, NONE};
use crate::steps::StepState;

#[derive(Clone, Debug)]
pub struct T1Params {
    pub c_puct: f32,
    pub pw_c: f32,
    pub pw_alpha: f32,
    /// Tier-2 descents in a row ending on known turns before a node stops
    /// widening.
    pub retries: u32,
    /// Tier-2 PUCT constant.
    pub c2: f32,
    /// Expansions one widening may request before it is abandoned (the node
    /// then stops widening). A descent in a lattice of lazily-refuted
    /// options can otherwise wander dead ends without end (2026-10-07: one
    /// grew a lattice past 88k nodes inside a single descent; match hangs).
    pub widen_budget: u32,
    /// Children's priors from their energies (softmax of energy / temp)
    /// instead of their lattice paths' mean log prior. Needs a lattice-view
    /// net with an energy output; falls back while any child lacks one.
    pub energy_prior: bool,
    pub energy_temp: f32,
}

impl Default for T1Params {
    fn default() -> Self {
        T1Params {
            c_puct: 1.5,
            pw_c: 1.0,
            pw_alpha: 0.5,
            retries: 20,
            c2: 1.5,
            widen_budget: 1000,
            energy_prior: false,
            energy_temp: 1.0,
        }
    }
}

pub struct T1Node {
    pub state: StepState,
    pub parent: u32,
    /// The sink in the parent's lattice this turn is.
    pub sink: u32,
    pub lattice: Option<Lattice>,
    /// Known value for this node's mover: the end of the game, or a turn
    /// whose mover kills the enemy necromancer.
    pub terminal: Option<f32>,
    /// Visits and value sum for this node's mover, its creation included.
    pub n: u32,
    pub w: f64,
    pub children: Vec<u32>,
    pub widen_done: bool,
    /// Widenings in flight.
    widening: u32,
    /// The trunk image of this turn start (from its sink's value), handed
    /// to its lattice when that is made.
    trunk: Option<crate::lattice::Trunk>,
}

impl T1Node {
    pub fn q(&self) -> f64 {
        self.w / f64::from(self.n.max(1))
    }
}

#[derive(Clone, Debug, Default)]
pub struct T1Stats {
    pub visits: u64,
    pub widenings: u64,
    pub t2_descents: u64,
    pub t2_repeats: u64,
    pub widen_dropped: u64,
    pub widen_abandoned: u64,
    pub widen_followed: u64,
}

/// One tier-1 visit in progress: the tier-1 path, and the widening under
/// way (node, tier-2 descent, retries used).
pub struct Visit {
    path: Vec<u32>,
    widen: Option<(u32, Cursor, u32)>,
    /// Expansions the current widening has requested.
    work: u32,
}

pub enum Need1 {
    /// Expand lattice node `n` of tier-1 node `t`.
    Expand {
        t: u32,
        n: u32,
    },
    /// Value sink `n` of tier-1 node `t`'s lattice (for the sink's mover).
    SinkValue {
        t: u32,
        n: u32,
    },
    /// Value tier-1 node `t` itself (only the root needs it).
    NodeValue {
        t: u32,
    },
    /// Waiting on another visit's request in flight.
    Wait,
    Done,
}

pub struct Tier1 {
    pub nodes: Vec<T1Node>,
    pub params: T1Params,
    pub stats: T1Stats,
    /// The root's value request is in flight.
    root_pending: bool,
}

impl Tier1 {
    /// A search from turn start `root` (a fresh [`StepState::begin`]); the
    /// root's value comes through the first [`Need1::NodeValue`].
    pub fn new(root: StepState, params: T1Params) -> Self {
        Tier1 {
            nodes: vec![T1Node {
                state: root,
                parent: NONE,
                sink: NONE,
                lattice: None,
                terminal: None,
                n: 0,
                w: 0.0,
                children: Vec::new(),
                widen_done: false,
                widening: 0,
                trunk: None,
            }],
            params,
            stats: T1Stats::default(),
            root_pending: false,
        }
    }

    pub fn root_value(&self) -> f64 {
        self.nodes[0].q()
    }

    pub fn visit(&self) -> Visit {
        Visit {
            path: vec![0],
            widen: None,
            work: 0,
        }
    }

    pub fn feed_node_value(&mut self, t: u32, out: &ValueOut) {
        self.root_pending = false;
        let nd = &mut self.nodes[t as usize];
        nd.n += 1;
        nd.w += f64::from(out.v);
        if let Some(l) = nd.lattice.as_mut() {
            if l.trunk.is_none() {
                l.trunk.clone_from(&out.trunk);
            }
        }
    }

    pub fn feed_expand(&mut self, t: u32, n: u32, out: &crate::lattice::ExpandOut) {
        self.lattice_mut(t).feed_expand(n, out);
    }

    pub fn feed_sink_value(&mut self, t: u32, n: u32, out: &ValueOut) {
        self.lattice_mut(t).feed_value(n, out);
    }

    pub fn lattice(&self, t: u32) -> &Lattice {
        self.nodes[t as usize].lattice.as_ref().expect("lattice")
    }

    pub fn lattice_mut(&mut self, t: u32) -> &mut Lattice {
        self.nodes[t as usize].lattice.as_mut().expect("lattice")
    }

    /// Normalized priors of `t`'s children.
    pub fn priors(&self, t: u32) -> Vec<f64> {
        let nd = &self.nodes[t as usize];
        let Some(l) = nd.lattice.as_ref() else {
            return Vec::new();
        };
        let en: Option<Vec<f64>> = self
            .params
            .energy_prior
            .then(|| {
                nd.children
                    .iter()
                    .map(|&c| l.nodes[self.nodes[c as usize].sink as usize].energy)
                    .collect::<Vec<Option<f32>>>()
            })
            .and_then(|e| {
                // Children without one (terminal turns) get the known ones' mean.
                let known: Vec<f32> = e.iter().flatten().copied().collect();
                if known.is_empty() {
                    return None;
                }
                let m = known.iter().sum::<f32>() / known.len() as f32;
                Some(
                    e.into_iter()
                        .map(|x| f64::from(x.unwrap_or(m) / self.params.energy_temp))
                        .collect(),
                )
            });
        let lp: Vec<f64> = en.unwrap_or_else(|| {
            nd.children
                .iter()
                .map(|&c| l.sink_logp(self.nodes[c as usize].sink))
                .collect()
        });
        let m = lp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = lp.iter().map(|&x| (x - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.into_iter().map(|x| x / s).collect()
    }

    fn select(&self, t: u32) -> Option<u32> {
        let nd = &self.nodes[t as usize];
        let pri = self.priors(t);
        let sq = f64::from(nd.n.max(1)).sqrt();
        let c = f64::from(self.params.c_puct);
        let mut best = None;
        let mut best_u = f64::NEG_INFINITY;
        for (i, &ch) in nd.children.iter().enumerate() {
            let cn = &self.nodes[ch as usize];
            let u = (1.0 - cn.q()) + c * pri[i] * sq / (1.0 + f64::from(cn.n));
            if u > best_u {
                best_u = u;
                best = Some(ch);
            }
        }
        best
    }

    /// Add `v` (for the last node's mover) to every node on `path`.
    fn backup(&mut self, path: &[u32], mut v: f64) {
        for &t in path.iter().rev() {
            let nd = &mut self.nodes[t as usize];
            nd.n += 1;
            nd.w += v;
            v = 1.0 - v;
        }
    }

    fn wants_widen(&self, t: u32) -> bool {
        let nd = &self.nodes[t as usize];
        let p = &self.params;
        let k = (f64::from(p.pw_c) * f64::from(nd.n.max(1)).powf(f64::from(p.pw_alpha))).ceil();
        !nd.widen_done && ((nd.children.len() + nd.widening as usize) as f64) < k
    }

    /// Visits in flight weigh on their path: each node below the root
    /// counts a pending visit worth a win to its own mover (a loss to the
    /// parent choosing it).
    fn virtual_loss(&mut self, t: u32, sign: f64) {
        let nd = &mut self.nodes[t as usize];
        if sign > 0.0 {
            nd.n += 1;
        } else {
            nd.n -= 1;
        }
        nd.w += sign;
    }

    fn end_visit(&mut self, vis: &Visit) {
        for &t in &vis.path[1..] {
            self.virtual_loss(t, -1.0);
        }
    }

    /// A new child of `t` for its lattice's sink `sink`, worth `value` to
    /// `t`'s mover; its creation is its first visit.
    fn add_child(&mut self, t: u32, sink: u32, value: f32) -> u32 {
        let l = self.lattice(t);
        let sn = &l.nodes[sink as usize];
        let v = 1.0 - value;
        let id = self.nodes.len() as u32;
        self.nodes.push(T1Node {
            state: sn.state.clone(),
            parent: t,
            sink,
            lattice: None,
            terminal: sn.terminal.then_some(v),
            n: 1,
            w: f64::from(v),
            children: Vec::new(),
            widen_done: false,
            widening: 0,
            trunk: sn.trunk.clone(),
        });
        self.nodes[t as usize].children.push(id);
        id
    }

    /// Run visit `vis` until it needs the net, waits on another visit's
    /// request, or is done. Visits may run concurrently (virtual loss keeps
    /// them apart).
    pub fn advance(&mut self, vis: &mut Visit) -> Result<Need1> {
        if self.nodes[0].lattice.is_none() {
            let l = Lattice::new(self.nodes[0].state.clone())?;
            let root = &mut self.nodes[0];
            if let Some(v) = l.terminal {
                root.terminal = Some(v);
                root.n = 1;
                root.w = f64::from(v);
            }
            root.lattice = Some(l);
        }
        if self.nodes[0].n == 0 {
            if self.root_pending {
                return Ok(Need1::Wait);
            }
            self.root_pending = true;
            return Ok(Need1::NodeValue { t: 0 });
        }
        loop {
            if let Some((wt, cur, _)) = vis.widen.as_mut() {
                // Over budget: first follow witnesses (a guaranteed turn
                // unless one is refuted), with a fresh budget.
                if vis.work > self.params.widen_budget && !cur.follow {
                    cur.follow = true;
                    vis.work = 0;
                    self.stats.widen_followed += 1;
                    if std::env::var_os("STEPS_SLOW_TURN_SECS").is_some() {
                        eprintln!(
                            "widening over budget at tier-1 node {}: following witnesses",
                            *wt
                        );
                    }
                }
                // Still over: give up -- but not at a root without turns
                // yet (a search must return one).
                if vis.work > self.params.widen_budget
                    && (*wt != 0 || !self.nodes[0].children.is_empty())
                {
                    let wt = *wt;
                    self.nodes[wt as usize]
                        .lattice
                        .as_mut()
                        .expect("lattice")
                        .abandon(cur);
                    vis.widen = None;
                    let nd = &mut self.nodes[wt as usize];
                    nd.widening -= 1;
                    nd.widen_done = true;
                    self.stats.widen_abandoned += 1;
                    if std::env::var_os("STEPS_SLOW_TURN_SECS").is_some() {
                        eprintln!("widening still over budget at tier-1 node {wt}: abandoned");
                    }
                }
            }
            if let Some((wt, cur, tries)) = vis.widen.as_mut() {
                let wt = *wt;
                let c2 = self.params.c2;
                let l = self.nodes[wt as usize].lattice.as_mut().expect("lattice");
                match l.advance(cur, c2)? {
                    Need::Expand(n) => {
                        vis.work += 1;
                        return Ok(Need1::Expand { t: wt, n });
                    }
                    Need::Value(n) => return Ok(Need1::SinkValue { t: wt, n }),
                    Need::Wait => return Ok(Need1::Wait),
                    Need::Dead => bail!("tier-1 node {wt}'s lattice root is a dead end"),
                    Need::Done { sink, value, .. } => {
                        self.stats.t2_descents += 1;
                        // New to this tier-1 node, not to its lattice: a
                        // lattice adopted from the previous turn's search
                        // already knows the sinks it reached there (with few
                        // distinct turns, all of them -- and the root then
                        // never got a child).
                        let fresh = !self.nodes[wt as usize]
                            .children
                            .iter()
                            .any(|&c| self.nodes[c as usize].sink == sink);
                        if fresh {
                            vis.widen = None;
                            self.nodes[wt as usize].widening -= 1;
                            self.add_child(wt, sink, value);
                            self.end_visit(vis);
                            let p = vis.path.clone();
                            self.backup(&p, f64::from(value));
                            self.stats.visits += 1;
                            return Ok(Need1::Done);
                        }
                        self.stats.t2_repeats += 1;
                        *tries += 1;
                        if *tries < self.params.retries {
                            *cur = Cursor::at(0, 0.0);
                            continue;
                        }
                        vis.widen = None;
                        let nd = &mut self.nodes[wt as usize];
                        nd.widening -= 1;
                        if !nd.widen_done {
                            nd.widen_done = true;
                            self.stats.widen_dropped += 1;
                        }
                    }
                }
            }
            let t = *vis.path.last().expect("path");
            let nd = &self.nodes[t as usize];
            if let Some(v) = nd.terminal {
                self.end_visit(vis);
                let p = vis.path.clone();
                self.backup(&p, f64::from(v));
                self.stats.visits += 1;
                return Ok(Need1::Done);
            }
            if nd.lattice.is_none() {
                let mut l = Lattice::new(nd.state.clone())?;
                if l.terminal.is_some() {
                    // The parent's lattice scores such turns itself, so a
                    // child is never a necromancer kill it missed.
                    bail!("tier-1 node {t} turned out terminal");
                }
                l.trunk = self.nodes[t as usize].trunk.take();
                self.nodes[t as usize].lattice = Some(l);
            }
            if self.wants_widen(t) {
                self.stats.widenings += 1;
                self.nodes[t as usize].widening += 1;
                vis.widen = Some((t, Cursor::at(0, 0.0), 0));
                vis.work = 0;
                continue;
            }
            match self.select(t) {
                Some(c) => {
                    self.virtual_loss(c, 1.0);
                    vis.path.push(c);
                }
                // Its first turns are still being found by other visits.
                None if self.nodes[t as usize].widening > 0 => return Ok(Need1::Wait),
                // Its widening was abandoned before any turn was found: a
                // leaf, worth its own value again.
                None if self.nodes[t as usize].n > 0 => {
                    let v = self.nodes[t as usize].q();
                    self.end_visit(vis);
                    let p = vis.path.clone();
                    self.backup(&p, v);
                    self.stats.visits += 1;
                    return Ok(Need1::Done);
                }
                None => bail!("tier-1 node {t} has no children and stopped widening"),
            }
        }
    }

    /// `visits` tier-1 visits, calling `eval` for every net request.
    pub fn run(&mut self, eval: &dyn LatticeEval, visits: u32) -> Result<()> {
        for _ in 0..visits {
            let mut vis = self.visit();
            loop {
                match self.advance(&mut vis)? {
                    Need1::Expand { t, n } => {
                        let l = self.lattice(t);
                        let (slots, node) = l.request(n);
                        let req = ExpandReq {
                            root: &l.start,
                            node,
                            slots: &slots,
                            trunk: l.trunk.clone(),
                        };
                        let out = eval.expand(&[req])?.pop().expect("one output");
                        self.feed_expand(t, n, &out);
                    }
                    Need1::SinkValue { t, n } => {
                        let o = eval.value1(&self.lattice(t).nodes[n as usize].state)?;
                        self.feed_sink_value(t, n, &o);
                    }
                    Need1::NodeValue { t } => {
                        let o = eval.value1(&self.nodes[t as usize].state)?;
                        self.feed_node_value(t, &o);
                    }
                    Need1::Done => break,
                    Need1::Wait => bail!("a lone visit waited on itself"),
                }
            }
            if self.nodes[0].terminal.is_some() {
                break;
            }
        }
        Ok(())
    }

    /// `visits` tier-1 visits with up to `inflight` of them at a time, each
    /// round's requests in one `eval.both` call (the match player).
    pub fn run_inflight(
        &mut self,
        eval: &dyn LatticeEval,
        visits: u32,
        inflight: usize,
    ) -> Result<()> {
        enum R {
            Expand(u32, u32),
            Sink(u32, u32),
            Node(u32),
        }
        let mut vis: Vec<Visit> = Vec::new();
        let mut started = 0;
        // STEPS_SLOW_TURN_SECS: progress reports while a search runs long.
        let slow = std::env::var("STEPS_SLOW_TURN_SECS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok());
        let t0 = std::time::Instant::now();
        let mut next = slow.unwrap_or(f64::MAX);
        loop {
            if self.nodes[0].terminal.is_some() || (started >= visits && vis.is_empty()) {
                return Ok(());
            }
            if t0.elapsed().as_secs_f64() > next {
                next += slow.unwrap_or(f64::MAX);
                let sizes: Vec<usize> = self
                    .nodes
                    .iter()
                    .filter_map(|n| n.lattice.as_ref().map(|l| l.nodes.len()))
                    .collect();
                eprintln!(
                    "slow search {:.0}s: started {started} in flight {} root n {} children {}; stats {:?}; {} lattices, largest {} nodes, root lattice {} nodes",
                    t0.elapsed().as_secs_f64(),
                    vis.len(),
                    self.nodes[0].n,
                    self.nodes[0].children.len(),
                    self.stats,
                    sizes.len(),
                    sizes.iter().max().unwrap_or(&0),
                    sizes.first().unwrap_or(&0)
                );
            }
            // Advance every visit until none finishes; the rest wait on a
            // request (handed in once: a busy node makes the next pass Wait).
            let mut reqs = Vec::new();
            loop {
                while started < visits && vis.len() < inflight.max(1) {
                    vis.push(self.visit());
                    started += 1;
                }
                let mut progressed = false;
                let mut i = 0;
                while i < vis.len() {
                    match self.advance(&mut vis[i])? {
                        Need1::Expand { t, n } => reqs.push(R::Expand(t, n)),
                        Need1::SinkValue { t, n } => reqs.push(R::Sink(t, n)),
                        Need1::NodeValue { t } => reqs.push(R::Node(t)),
                        Need1::Wait => {}
                        Need1::Done => {
                            vis.swap_remove(i);
                            progressed = true;
                            continue;
                        }
                    }
                    i += 1;
                }
                if !progressed || self.nodes[0].terminal.is_some() {
                    break;
                }
            }
            if reqs.is_empty() {
                if vis.is_empty() {
                    continue;
                }
                bail!("{} visits wait on nothing", vis.len());
            }
            let slots: Vec<Vec<(u8, u8)>> = reqs
                .iter()
                .filter_map(|r| match *r {
                    R::Expand(t, n) => Some(self.lattice(t).request(n).0),
                    _ => None,
                })
                .collect();
            // With energy priors, each sink value request also asks for its
            // declaration's energy (a kid-less tier-2 row).
            let decls: Vec<(u32, u32, StepState)> = if self.params.energy_prior {
                reqs.iter()
                    .filter_map(|r| match *r {
                        R::Sink(t, n) => {
                            let l = self.lattice(t);
                            let s = &l.nodes[n as usize].state;
                            Some((t, n, crate::lview::decl_state(&l.start, &s.done)))
                        }
                        _ => None,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let (outs, vals) = {
                let mut ex = Vec::with_capacity(slots.len());
                let mut states = Vec::new();
                for r in &reqs {
                    match *r {
                        R::Expand(t, n) => {
                            let l = self.lattice(t);
                            ex.push(ExpandReq {
                                root: &l.start,
                                node: &l.nodes[n as usize].state,
                                slots: &slots[ex.len()],
                                trunk: l.trunk.clone(),
                            });
                        }
                        R::Sink(t, n) => states.push(&self.lattice(t).nodes[n as usize].state),
                        R::Node(t) => states.push(&self.nodes[t as usize].state),
                    }
                }
                for (t, _, d) in &decls {
                    let l = self.lattice(*t);
                    ex.push(ExpandReq {
                        root: &l.start,
                        node: d,
                        slots: &[],
                        trunk: l.trunk.clone(),
                    });
                }
                eval.both(&ex, &states)?
            };
            let n_ex = outs.len() - decls.len();
            for ((t, n, _), o) in decls.iter().zip(&outs[n_ex..]) {
                self.lattice_mut(*t).feed_energy(*n, o.energy);
            }
            let (mut e, mut v) = (outs.iter(), vals.iter());
            for r in reqs {
                match r {
                    R::Expand(t, n) => self.feed_expand(t, n, e.next().expect("an output")),
                    R::Sink(t, n) => self.feed_sink_value(t, n, v.next().expect("a value")),
                    R::Node(t) => self.feed_node_value(t, v.next().expect("a value")),
                }
            }
        }
    }

    /// The most visited root child (ties: higher mean value).
    pub fn best_child(&self) -> Option<u32> {
        let root = &self.nodes[0];
        root.children.iter().copied().max_by(|&a, &b| {
            let (x, y) = (&self.nodes[a as usize], &self.nodes[b as usize]);
            x.n.cmp(&y.n).then(y.q().total_cmp(&x.q()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::MinionsState;
    use crate::lattice::tests::HashLatticeEval;

    /// Visits add up, every node's visit count is its creation plus its
    /// children's, values are probabilities, and the root widens roughly
    /// as sqrt(n) and goes deeper than one turn.
    #[test]
    fn tier1_invariants() {
        let s = StepState::begin(&MinionsState::with_bid(0), 7);
        let mut t = Tier1::new(s, T1Params::default());
        t.run(&HashLatticeEval, 200).unwrap();
        assert_eq!(t.stats.visits, 200);
        let root = &t.nodes[0];
        assert_eq!(root.n, 201, "root visits");
        assert!(
            root.children.len() >= 10,
            "{} root children",
            root.children.len()
        );
        assert!(
            t.nodes.len() > root.children.len() + 1,
            "never went past the first turn"
        );
        for (i, nd) in t.nodes.iter().enumerate() {
            let kids: u32 = nd.children.iter().map(|&c| t.nodes[c as usize].n).sum();
            let own = if nd.terminal.is_some() && i != 0 {
                nd.n
            } else {
                1
            };
            if nd.terminal.is_none() {
                assert_eq!(nd.n, own + kids, "node {i}");
            }
            assert!((0.0..=1.0).contains(&nd.q()), "node {i} q {}", nd.q());
            if i > 0 {
                let p = &t.nodes[nd.parent as usize];
                let st = &p.lattice.as_ref().unwrap().nodes[nd.sink as usize].state;
                assert!(st.is_terminal() || st.flipped != p.state.flipped);
            }
        }
        assert!(t.nodes.iter().all(|n| n.widening == 0));
        let pri = t.priors(0);
        assert!((pri.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        eprintln!("{:?} root q {:.3}", t.stats, t.root_value());
    }

    /// Several visits in flight, fed in rounds: the same invariants hold,
    /// and every virtual loss and widening count is gone at the end.
    /// A search that adopts the lattice of the turn just played (as
    /// `latgen` does) widens its root, and its children are distinct turns.
    /// (Widening once asked the lattice whether a sink was new; an adopted
    /// lattice's known sinks then never became children, and gen 27 of the
    /// lattice cycle died on a position with few turns.)
    #[test]
    fn tier1_adopted_lattice() {
        let s = StepState::begin(&MinionsState::with_bid(0), 5);
        let mut t = Tier1::new(s, T1Params::default());
        t.run(&HashLatticeEval, 200).unwrap();
        let best = t.best_child().unwrap() as usize;
        let lat = t.nodes[best]
            .lattice
            .take()
            .expect("the best child was searched");
        let mut t2 = Tier1::new(t.nodes[best].state.clone(), T1Params::default());
        t2.nodes[0].lattice = Some(lat);
        t2.run(&HashLatticeEval, 100).unwrap();
        let kids = &t2.nodes[0].children;
        assert!(!kids.is_empty());
        let sinks: std::collections::HashSet<u32> =
            kids.iter().map(|&c| t2.nodes[c as usize].sink).collect();
        assert_eq!(sinks.len(), kids.len(), "two children are the same turn");
    }

    /// Tier-2 records from the tier-1 search: along the path, each cell's
    /// kids split the node's tier-1 visits exactly, and the path ends at the
    /// target turn.
    #[test]
    fn tier1_records_consistent() {
        use rand::SeedableRng;
        let s = StepState::begin(&MinionsState::with_bid(0), 9);
        let mut t = Tier1::new(s, T1Params::default());
        t.run(&HashLatticeEval, 200).unwrap();
        let turns: Vec<(u32, f64, f64)> = t.nodes[0]
            .children
            .iter()
            .map(|&c| {
                let ch = &t.nodes[c as usize];
                (ch.sink, f64::from(ch.n), f64::from(ch.n) - ch.w)
            })
            .collect();
        let target = t.nodes[t.best_child().unwrap() as usize].sink;
        let mut lat = t.nodes[0].lattice.take().unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let recs = lat.tier1_records(&turns, target, &mut rng).unwrap();
        assert!(recs.len() > 3, "{} records", recs.len());
        let total: f64 = turns.iter().map(|t| t.1).sum();
        assert!((f64::from(recs[0].n) - total).abs() < 1e-3);
        for r in &recs {
            let mut cells: Vec<u8> = r.kids.iter().map(|k| k.0).collect();
            cells.sort();
            cells.dedup();
            for c in cells {
                let n: f32 = r.kids.iter().filter(|k| k.0 == c).map(|k| k.2).sum();
                assert!((n - r.n).abs() < 1e-3, "cell {c}: kids {n} vs node {}", r.n);
            }
            assert!(r.q >= 0.0 && r.q <= 1.0);
        }
    }

    #[test]
    fn tier1_inflight() {
        let s = StepState::begin(&MinionsState::with_bid(0), 9);
        let mut t = Tier1::new(s, T1Params::default());
        let (total, k) = (160u32, 8usize);
        let mut started = 0;
        let mut vis: Vec<Visit> = Vec::new();
        let mut rounds = 0;
        loop {
            while started < total && vis.len() < k {
                vis.push(t.visit());
                started += 1;
            }
            if vis.is_empty() {
                break;
            }
            rounds += 1;
            let mut needs = Vec::new();
            let mut i = 0;
            while i < vis.len() {
                match t.advance(&mut vis[i]).unwrap() {
                    Need1::Done => {
                        vis.swap_remove(i);
                        continue;
                    }
                    Need1::Wait => {}
                    n => needs.push(n),
                }
                i += 1;
            }
            for n in needs {
                match n {
                    Need1::Expand { t: tt, n } => {
                        let l = t.lattice(tt);
                        let (slots, node) = l.request(n);
                        let req = ExpandReq {
                            root: &l.start,
                            node,
                            slots: &slots,
                            trunk: None,
                        };
                        let out = HashLatticeEval.expand(&[req]).unwrap().pop().unwrap();
                        t.feed_expand(tt, n, &out);
                    }
                    Need1::SinkValue { t: tt, n } => {
                        let o = HashLatticeEval
                            .value1(&t.lattice(tt).nodes[n as usize].state)
                            .unwrap();
                        t.feed_sink_value(tt, n, &o);
                    }
                    Need1::NodeValue { t: tt } => {
                        let o = HashLatticeEval.value1(&t.nodes[tt as usize].state).unwrap();
                        t.feed_node_value(tt, &o);
                    }
                    _ => unreachable!(),
                }
            }
        }
        assert_eq!(t.stats.visits, u64::from(total));
        assert_eq!(t.nodes[0].n, total + 1);
        for (i, nd) in t.nodes.iter().enumerate() {
            assert_eq!(nd.widening, 0, "node {i}");
            if nd.terminal.is_none() {
                let kids: u32 = nd.children.iter().map(|&c| t.nodes[c as usize].n).sum();
                assert_eq!(nd.n, 1 + kids, "node {i}");
            }
            assert!((0.0..=1.0).contains(&nd.q()), "node {i} q {}", nd.q());
            if let Some(l) = nd.lattice.as_ref() {
                let direct: u32 = l.nodes.iter().map(|n| n.n).sum();
                assert_eq!(
                    direct as u64,
                    l.stats.expansions + l.stats.descents,
                    "node {i}: virtual visits left"
                );
            }
        }
        eprintln!("{rounds} rounds for {total} visits, {:?}", t.stats);
    }
}
