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

use crate::lattice::{Cursor, ExpandReq, Lattice, LatticeEval, Need, NONE};
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
}

impl Default for T1Params {
    fn default() -> Self {
        T1Params {
            c_puct: 1.5,
            pw_c: 1.0,
            pw_alpha: 0.5,
            retries: 20,
            c2: 1.5,
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
}

/// One tier-1 visit in progress: the tier-1 path, and the widening under
/// way (node, tier-2 descent, retries used).
pub struct Visit {
    path: Vec<u32>,
    widen: Option<(u32, Cursor, u32)>,
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
    Done,
}

pub struct Tier1 {
    pub nodes: Vec<T1Node>,
    pub params: T1Params,
    pub stats: T1Stats,
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
            }],
            params,
            stats: T1Stats::default(),
        }
    }

    pub fn root_value(&self) -> f64 {
        self.nodes[0].q()
    }

    pub fn visit(&self) -> Visit {
        Visit {
            path: vec![0],
            widen: None,
        }
    }

    pub fn feed_node_value(&mut self, t: u32, v: f32) {
        let nd = &mut self.nodes[t as usize];
        nd.n += 1;
        nd.w += f64::from(v);
    }

    pub fn feed_expand(&mut self, t: u32, n: u32, out: &crate::lattice::ExpandOut) {
        self.lattice_mut(t).feed_expand(n, out);
    }

    pub fn feed_sink_value(&mut self, t: u32, n: u32, v: f32) {
        self.lattice_mut(t).feed_value(n, v);
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
        let lp: Vec<f64> = nd
            .children
            .iter()
            .map(|&c| l.sink_logp(self.nodes[c as usize].sink))
            .collect();
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
        !nd.widen_done && (nd.children.len() as f64) < k
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
        });
        self.nodes[t as usize].children.push(id);
        id
    }

    /// Run visit `vis` until it needs the net or is done.
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
            return Ok(Need1::NodeValue { t: 0 });
        }
        loop {
            if let Some((wt, cur, tries)) = vis.widen.as_mut() {
                let wt = *wt;
                let c2 = self.params.c2;
                let l = self.nodes[wt as usize].lattice.as_mut().expect("lattice");
                match l.advance(cur, c2)? {
                    Need::Expand(n) => return Ok(Need1::Expand { t: wt, n }),
                    Need::Value(n) => return Ok(Need1::SinkValue { t: wt, n }),
                    Need::Done { sink, value, fresh } => {
                        self.stats.t2_descents += 1;
                        if fresh {
                            vis.widen = None;
                            self.add_child(wt, sink, value);
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
                        self.nodes[wt as usize].widen_done = true;
                        self.stats.widen_dropped += 1;
                    }
                }
            }
            let t = *vis.path.last().expect("path");
            let nd = &self.nodes[t as usize];
            if let Some(v) = nd.terminal {
                let p = vis.path.clone();
                self.backup(&p, f64::from(v));
                self.stats.visits += 1;
                return Ok(Need1::Done);
            }
            if nd.lattice.is_none() {
                let l = Lattice::new(nd.state.clone())?;
                if l.terminal.is_some() {
                    // The parent's lattice scores such turns itself, so a
                    // child is never a necromancer kill it missed.
                    bail!("tier-1 node {t} turned out terminal");
                }
                self.nodes[t as usize].lattice = Some(l);
            }
            if self.wants_widen(t) {
                self.stats.widenings += 1;
                vis.widen = Some((t, Cursor::at(0, 0.0), 0));
                continue;
            }
            match self.select(t) {
                Some(c) => vis.path.push(c),
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
                        };
                        let out = eval.expand(&[req])?.pop().expect("one output");
                        self.feed_expand(t, n, &out);
                    }
                    Need1::SinkValue { t, n } => {
                        let v = eval.value(&[&self.lattice(t).nodes[n as usize].state])?[0];
                        self.feed_sink_value(t, n, v);
                    }
                    Need1::NodeValue { t } => {
                        let v = eval.value(&[&self.nodes[t as usize].state])?[0];
                        self.feed_node_value(t, v);
                    }
                    Need1::Done => break,
                }
            }
            if self.nodes[0].terminal.is_some() {
                break;
            }
        }
        Ok(())
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
        let pri = t.priors(0);
        assert!((pri.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        eprintln!("{:?} root q {:.3}", t.stats, t.root_value());
    }
}
