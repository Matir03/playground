//! Turn-lattice search, tier 2: prior-based MCTS over the partial
//! commitments of one turn. See `docs/minions-lattice.md`.
//!
//! A lattice node is a set of commitments (an enemy's fate, a unit's
//! destination, a spawn cell's spawn) that can still be completed to a legal
//! turn. Nodes are ordered by inclusion; an edge adds one commitment, any
//! undecided variable of the current stage (fates, then moves, then spawns).
//! The sinks are full turns.
//!
//! A visit to a node counts as a visit to every node below it (every subset
//! of its commitments), so nothing is backed up along a path: the statistics
//! of a node are sums over the *direct* visits of the nodes above it, and
//! selecting at a node computes all its children's statistics in one scan
//! of the direct-visit list (see [`Lattice::select`]). Expanding a node is a
//! direct visit to it (at its own predicted value) and to each of its
//! children (at theirs); a sink is a direct visit at its tier-1 value or its
//! terminal score.

use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::map::NUM_CELLS;
use crate::steps::{Stage, StepState};
use crate::turn::{TurnSpec, TARGET_STAY};

/// Time profile (ns) of the lattice's CPU work: prepare (children and
/// witnesses), stats scans, child creation, the bridge's child views; and
/// the touched nodes scanned.
pub static PROF: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Live lattices, nodes and kids (for memory accounting).
pub static LIVE: [std::sync::atomic::AtomicI64; 3] =
    [const { std::sync::atomic::AtomicI64::new(0) }; 3];

pub static SCANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn prof(i: usize, t0: std::time::Instant) {
    PROF[i].fetch_add(
        t0.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Variables per turn the key can hold.
pub const MAX_VARS: usize = 64;
const PER_WORD: usize = 10;
const WORDS: usize = 7;
pub const NONE: u32 = u32::MAX;

/// A node's commitments: one bit per decided variable and its choice in a
/// 6-bit field. Variables are numbered per turn: the non-necromancer enemies
/// (sorted by cell), then the friendly units, then the spawn cells (sorted
/// by cell; only ever compared between nodes whose fates and moves agree).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Key {
    pub mask: u64,
    pub packed: [u64; WORDS],
}

/// For each 10-bit group of a variable mask, the 6-bit fields it covers.
const FIELD: [u64; 1024] = build_field();

const fn build_field() -> [u64; 1024] {
    let mut t = [0u64; 1024];
    let mut m = 0;
    while m < 1024 {
        let mut f = 0u64;
        let mut i = 0;
        while i < PER_WORD {
            if (m >> i) & 1 == 1 {
                f |= 63 << (6 * i);
            }
            i += 1;
        }
        t[m] = f;
        m += 1;
    }
    t
}

impl Key {
    pub fn get(&self, v: usize) -> u8 {
        ((self.packed[v / PER_WORD] >> (6 * (v % PER_WORD))) & 63) as u8
    }

    pub fn with(mut self, v: usize, c: u8) -> Key {
        debug_assert!(v < MAX_VARS && c < 64 && self.mask & (1 << v) == 0);
        self.mask |= 1 << v;
        self.packed[v / PER_WORD] |= u64::from(c) << (6 * (v % PER_WORD));
        self
    }

    /// Some variable decided in both, differently.
    pub fn conflicts(&self, o: &Key) -> bool {
        let common = self.mask & o.mask;
        (0..WORDS).any(|i| {
            (self.packed[i] ^ o.packed[i]) & FIELD[((common >> (PER_WORD * i)) & 1023) as usize]
                != 0
        })
    }

    /// Every commitment of `o` is one of ours.
    pub fn includes(&self, o: &Key) -> bool {
        o.mask & !self.mask == 0 && !self.conflicts(o)
    }
}

/// A node to expand: the turn's start (whose tier-1 trunk the tier-2 net
/// reads) and the node's state, plus the `(cell, slot)` of each child. The
/// slot is the choice: a fate code, a move offset, or a spawn's unit code.
pub struct ExpandReq<'a> {
    pub root: &'a StepState,
    pub node: &'a StepState,
    pub slots: &'a [(u8, u8)],
}

/// Per child (parallel to the request's slots) a prior logit and a value,
/// and the node's own value. Values are for the turn's mover.
pub struct ExpandOut {
    pub logits: Vec<f32>,
    pub values: Vec<f32>,
    pub value: f32,
}

pub trait LatticeEval: Send + Sync {
    fn expand(&self, reqs: &[ExpandReq]) -> Result<Vec<ExpandOut>>;
    /// Tier-1 values of turn starts, each for its own mover.
    fn value(&self, states: &[&StepState]) -> Result<Vec<f32>>;
    /// Both kinds at once (one net call where the evaluator can).
    fn both(
        &self,
        reqs: &[ExpandReq],
        states: &[&StepState],
    ) -> Result<(Vec<ExpandOut>, Vec<f32>)> {
        let e = if reqs.is_empty() {
            Vec::new()
        } else {
            self.expand(reqs)?
        };
        let v = if states.is_empty() {
            Vec::new()
        } else {
            self.value(states)?
        };
        Ok((e, v))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Inner,
    /// A full turn: the next turn's start, or the end of the game.
    Sink,
    /// Its commitments cannot be completed after all (a lazily offered
    /// option the exact search refuted), or none of its children can.
    Dead,
}

/// A child of an expanded node: commit `choice` to variable `var` (cell
/// `cell`). `logp` is the log prior within its variable's choices.
#[derive(Clone, Copy, Debug)]
pub struct Kid {
    pub var: u8,
    pub choice: u8,
    pub cell: u8,
    pub dead: bool,
    pub logp: f32,
    pub v: f32,
    pub node: u32,
}

pub struct LNode {
    pub key: Key,
    pub state: StepState,
    pub kind: Kind,
    pub prepared: bool,
    pub expanded: bool,
    /// Sorted by `(var, choice)`.
    pub kids: Vec<Kid>,
    /// The kids whose witness is not this node's own with the kid's choice
    /// swapped in (see [`swapped`]), by kid index.
    wit_ex: Vec<(u16, TurnSpec)>,
    /// The live kids' direct visits from this node's expansion.
    kid_n: u32,
    kid_w: f64,
    /// Direct visits: the expansion's own value, and a sink's returns.
    pub n: u32,
    pub w: f64,
    /// Sink: its value for the lattice's mover once known, whether that is
    /// a terminal score, and the summed log prior of the descents that
    /// reached it (and how many).
    pub value: Option<f32>,
    pub terminal: bool,
    pub lp_sum: f64,
    pub lp_n: u32,
    parents: Vec<(u32, u16)>,
}

/// `own` with the choice `v` for the variable on `cell` swapped in: the
/// witness most options get (spawns leave witnesses alone).
fn swapped(own: &TurnSpec, stage: Stage, cell: u8, v: u8) -> TurnSpec {
    let mut w = *own;
    match stage {
        Stage::Fates => w.targets[cell as usize] = v,
        Stage::Moves => w.moves[cell as usize] = v,
        _ => {}
    }
    w
}

impl LNode {
    fn kid_witness(&self, k: usize) -> TurnSpec {
        match self.wit_ex.binary_search_by_key(&(k as u16), |x| x.0) {
            Ok(i) => self.wit_ex[i].1,
            Err(_) => {
                let kid = &self.kids[k];
                swapped(self.state.witness(), self.state.stage, kid.cell, kid.choice)
            }
        }
    }

    fn new(key: Key, state: StepState, kind: Kind) -> Self {
        LNode {
            key,
            state,
            kind,
            prepared: false,
            expanded: false,
            kids: Vec::new(),
            wit_ex: Vec::new(),
            kid_n: 0,
            kid_w: 0.0,
            n: 0,
            w: 0.0,
            value: None,
            terminal: false,
            lp_sum: 0.0,
            lp_n: 0,
            parents: Vec::new(),
        }
    }

    pub fn q(&self) -> f64 {
        self.w / f64::from(self.n.max(1))
    }
}

#[derive(Clone, Debug, Default)]
pub struct LatticeStats {
    pub expansions: u64,
    pub kids: u64,
    pub sinks: u64,
    pub descents: u64,
    pub dead: u64,
}

/// One descent in progress: where it is, the `(node, kid)` path from where
/// it started, and the summed log prior from the lattice's root.
#[derive(Clone, Debug)]
pub struct Cursor {
    pub at: u32,
    pub path: Vec<(u32, u16)>,
    pub lp: f64,
}

impl Cursor {
    pub fn at(node: u32, lp: f64) -> Self {
        Cursor {
            at: node,
            path: Vec::new(),
            lp,
        }
    }
}

pub enum Need {
    /// Expand this node ([`Lattice::request`], then [`Lattice::feed_expand`]).
    Expand(u32),
    /// Value this sink's position for its mover ([`Lattice::feed_value`]).
    Value(u32),
    /// The descent reached a sink. `fresh`: no descent had before.
    Done { sink: u32, value: f32, fresh: bool },
    /// The node the descent started at turned out to be a dead end.
    Dead,
}

/// Children statistics at a node, from [`Lattice::stats`].
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub n: f64,
    pub w: f64,
    /// Per kid (parallel to the node's kids): visits and value sum.
    pub kids: Vec<(f64, f64)>,
}

pub struct Lattice {
    /// The turn's start as it was begun (the tier-1 position).
    pub start: StepState,
    pub nodes: Vec<LNode>,
    index: HashMap<Key, u32>,
    /// Nodes with direct visits.
    touched: Vec<u32>,
    var_of: [u8; NUM_CELLS],
    n_fm: usize,
    /// The mover can kill the enemy necromancer: the turn is terminal with
    /// this value and has no lattice.
    pub terminal: Option<f32>,
    pub stats: LatticeStats,
}

thread_local! {
    /// Kid index by `var * 64 + choice` during a stats scan.
    static SLOT: std::cell::RefCell<Vec<u16>> = std::cell::RefCell::new(vec![u16::MAX; MAX_VARS * 64]);
}

fn var_index(var_of: &[u8; NUM_CELLS], n_fm: usize, s: &StepState, cell: u8) -> usize {
    match s.stage {
        Stage::Spawns => n_fm + s.spawn_cells.iter().filter(|&&c| c < cell).count(),
        _ => var_of[cell as usize] as usize,
    }
}

/// `v` for `s`'s mover, as a value for `root`'s.
fn for_root(root: &StepState, s: &StepState, v: f32) -> f32 {
    if s.flipped != root.flipped {
        1.0 - v
    } else {
        v
    }
}

/// At a turn's start: the end of the game if the mover can kill the enemy
/// necromancer, else the state with the necromancer spared (its fate is
/// never a choice). `start` must be a fresh [`StepState::begin`].
pub fn necro_kill(start: &StepState) -> std::result::Result<StepState, StepState> {
    let s = start;
    if s.stage != Stage::Fates
        || s.idx != 0
        || !s.enemies.first().is_some_and(|&e| {
            s.base
                .piece_at(e)
                .is_some_and(|(_, u)| u.stats().necromancer)
        })
    {
        return Err(s.clone());
    }
    let (opts, wits) = s.lattice_options();
    let mut spare = None;
    for (&o, &w) in opts.iter().zip(&wits) {
        if o == TARGET_STAY {
            spare = Some(w);
            continue;
        }
        let c = s.lattice_child(o, w);
        if c.is_terminal() {
            return Ok(c);
        }
    }
    // "Spare" is always offered: the empty declaration's witness keeps it.
    Err(s.lattice_child(TARGET_STAY, spare.unwrap_or_default()))
}

impl Drop for Lattice {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        LIVE[0].fetch_sub(1, Relaxed);
        LIVE[1].fetch_sub(self.nodes.len() as i64, Relaxed);
        LIVE[2].fetch_sub(
            self.nodes.iter().map(|n| n.kids.len() as i64).sum(),
            Relaxed,
        );
    }
}

impl Lattice {
    /// The lattice of the turn starting at `start` (a fresh
    /// [`StepState::begin`]).
    pub fn new(start: StepState) -> Result<Self> {
        LIVE[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut l = Lattice {
            start: start.clone(),
            nodes: Vec::new(),
            index: HashMap::new(),
            touched: Vec::new(),
            var_of: [u8::MAX; NUM_CELLS],
            n_fm: 0,
            terminal: None,
            stats: LatticeStats::default(),
        };
        if start.is_terminal() {
            l.terminal = Some(start.terminal_value());
            return Ok(l);
        }
        let root = match necro_kill(&start) {
            Ok(end) => {
                l.terminal = Some(for_root(&start, &end, end.terminal_value()));
                return Ok(l);
            }
            Err(r) => r,
        };
        let mut fates: Vec<u8> = if root.stage == Stage::Fates {
            root.undecided().to_vec()
        } else {
            Vec::new()
        };
        fates.sort_unstable();
        let mut friends = root.friends.clone();
        friends.sort_unstable();
        for (i, &c) in fates.iter().chain(&friends).enumerate() {
            l.var_of[c as usize] = i as u8;
        }
        l.n_fm = fates.len() + friends.len();
        if l.n_fm >= MAX_VARS {
            bail!("{} fate and move variables", l.n_fm);
        }
        let kind = l.classify(&root);
        let mut node = LNode::new(Key::default(), root, kind);
        if kind == Kind::Sink {
            l.sink_value(&mut node);
        }
        l.nodes.push(node);
        LIVE[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        l.index.insert(Key::default(), 0);
        Ok(l)
    }

    fn classify(&self, s: &StepState) -> Kind {
        if s.is_terminal() || s.flipped != self.start.flipped {
            Kind::Sink
        } else {
            Kind::Inner
        }
    }

    /// A sink's value if it needs no net: the end of the game, or a turn
    /// after which the opponent can kill our necromancer.
    fn sink_value(&self, node: &mut LNode) {
        let s = &node.state;
        if s.is_terminal() {
            node.value = Some(for_root(&self.start, s, s.terminal_value()));
            node.terminal = true;
        } else if let Ok(end) = necro_kill(s) {
            node.value = Some(for_root(&self.start, &end, end.terminal_value()));
            node.terminal = true;
        }
    }

    fn var(&self, s: &StepState, cell: u8) -> usize {
        var_index(&self.var_of, self.n_fm, s, cell)
    }

    /// The node for the root plus kid `k` of `p`, created on first use.
    pub fn child(&mut self, p: u32, k: usize) -> Result<u32> {
        let kid = self.nodes[p as usize].kids[k];
        if kid.node != NONE {
            return Ok(kid.node);
        }
        let t0 = std::time::Instant::now();
        let r = self.make_child(p, k, kid);
        prof(2, t0);
        r
    }

    fn make_child(&mut self, p: u32, k: usize, kid: Kid) -> Result<u32> {
        let pn = &self.nodes[p as usize];
        let mut key = pn.key.with(kid.var as usize, kid.choice);
        let mut st = pn
            .state
            .with_next(kid.cell)
            .lattice_child(kid.choice, pn.kid_witness(k));
        // Nothing left to buy: the remaining spawn cells stay empty.
        while st.stage == Stage::Spawns && st.flipped == self.start.flipped {
            let (o, w) = st.lattice_options();
            if o != [0] {
                break;
            }
            let c = st.undecided()[0];
            let v = self.var(&st, c);
            if v >= MAX_VARS {
                bail!("spawn variable {v}");
            }
            key = key.with(v, 0);
            st = st.with_next(c).lattice_child(0, w[0]);
        }
        let id = match self.index.get(&key) {
            Some(&id) => id,
            None => {
                let kind = self.classify(&st);
                let mut node = LNode::new(key, st, kind);
                if kind == Kind::Sink {
                    self.sink_value(&mut node);
                    self.stats.sinks += 1;
                }
                let id = self.nodes.len() as u32;
                self.nodes.push(node);
                LIVE[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.index.insert(key, id);
                id
            }
        };
        self.nodes[id as usize].parents.push((p, k as u16));
        self.nodes[p as usize].kids[k].node = id;
        if self.nodes[id as usize].kind == Kind::Dead {
            // Reached another way and refuted already: this kid is dead too.
            let mut stack = Vec::new();
            self.drop_kid(p, k, &mut stack);
            self.kill_all(stack)?;
        }
        Ok(id)
    }

    /// List a node's children with their witnesses. False: none (dead).
    fn prepare(&mut self, n: u32) -> Result<bool> {
        let t0 = std::time::Instant::now();
        let r = self.prepare_inner(n);
        prof(0, t0);
        r
    }

    fn prepare_inner(&mut self, n: u32) -> Result<bool> {
        let (var_of, n_fm) = (self.var_of, self.n_fm);
        let node = &mut self.nodes[n as usize];
        node.prepared = true;
        if !node.state.resolve_pending() {
            return Ok(false);
        }
        let st = &node.state;
        let mut kids: Vec<(Kid, TurnSpec)> = Vec::new();
        let kid = |var: usize, choice: u8, cell: u8| Kid {
            var: var as u8,
            choice,
            cell,
            dead: false,
            logp: 0.0,
            v: 0.0,
            node: NONE,
        };
        match st.stage {
            Stage::Fates | Stage::Moves => {
                for &c in st.undecided() {
                    let (o, w) = st.with_next(c).lattice_options();
                    let v = var_of[c as usize] as usize;
                    kids.extend(o.into_iter().zip(w).map(|(x, w)| (kid(v, x, c), w)));
                }
            }
            Stage::Spawns => {
                // The same options at every cell: they depend on money only.
                let (o, w) = st.lattice_options();
                for &c in st.undecided() {
                    let v = var_index(&var_of, n_fm, st, c);
                    if v >= MAX_VARS {
                        bail!("spawn variable {v}");
                    }
                    kids.extend(o.iter().zip(&w).map(|(&x, &w)| (kid(v, x, c), w)));
                }
            }
            Stage::Terminal => {}
        }
        kids.sort_unstable_by_key(|(k, _)| (k.var, k.choice));
        let node = &mut self.nodes[n as usize];
        LIVE[2].fetch_add(kids.len() as i64, std::sync::atomic::Ordering::Relaxed);
        let own = *node.state.witness();
        let stage = node.state.stage;
        node.wit_ex = kids
            .iter()
            .enumerate()
            .filter(|(_, (k, w))| *w != swapped(&own, stage, k.cell, k.choice))
            .map(|(i, (_, w))| (i as u16, *w))
            .collect();
        node.kids = kids.into_iter().map(|x| x.0).collect();
        Ok(!node.kids.is_empty())
    }

    /// Make sure node `n` is expanded: `Some(Expand)` if it needs the net,
    /// `None` if it already is (or is not an inner node). A dead end is
    /// marked dead and reported as an error-free `None`.
    pub fn ensure_expanded(&mut self, n: u32) -> Result<Option<Need>> {
        let node = &self.nodes[n as usize];
        if node.kind != Kind::Inner || node.expanded {
            return Ok(None);
        }
        if node.prepared || self.prepare(n)? {
            return Ok(Some(Need::Expand(n)));
        }
        self.kill(n)?;
        Ok(None)
    }

    /// The request to expand prepared node `n`.
    pub fn request(&self, n: u32) -> (Vec<(u8, u8)>, &StepState) {
        let node = &self.nodes[n as usize];
        (
            node.kids.iter().map(|k| (k.cell, k.choice)).collect(),
            &node.state,
        )
    }

    pub fn feed_expand(&mut self, n: u32, out: &ExpandOut) {
        let node = &mut self.nodes[n as usize];
        debug_assert!(node.prepared && !node.expanded);
        debug_assert_eq!(out.logits.len(), node.kids.len());
        // Log-softmax within each variable's offered choices.
        let mut i = 0;
        while i < node.kids.len() {
            let var = node.kids[i].var;
            let j = i + node.kids[i..].iter().take_while(|k| k.var == var).count();
            let m = out.logits[i..j]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let lse = m + out.logits[i..j]
                .iter()
                .map(|&l| (l - m).exp())
                .sum::<f32>()
                .ln();
            for t in i..j {
                node.kids[t].logp = out.logits[t] - lse;
            }
            i = j;
        }
        let mut kw = 0.0;
        for (k, &v) in node.kids.iter_mut().zip(&out.values) {
            k.v = v;
            kw += f64::from(v);
        }
        node.kid_n = node.kids.len() as u32;
        node.kid_w = kw;
        node.expanded = true;
        self.stats.expansions += 1;
        self.stats.kids += node.kids.len() as u64;
        self.visit(n, out.value);
    }

    /// Record sink `n`'s value (`v` for the sink's own mover).
    pub fn feed_value(&mut self, n: u32, v: f32) {
        let s = &self.nodes[n as usize].state;
        let v = for_root(&self.start, s, v);
        self.nodes[n as usize].value = Some(v);
    }

    fn visit(&mut self, n: u32, v: f32) {
        let node = &mut self.nodes[n as usize];
        if node.n == 0 {
            self.touched.push(n);
        }
        node.n += 1;
        node.w += f64::from(v);
    }

    /// Mark `n` dead, and every parent all of whose kids are.
    fn kill(&mut self, n: u32) -> Result<()> {
        self.kill_all(vec![n])
    }

    /// Mark kid `k` of `p` dead; `p` goes on `stack` once all its kids are.
    fn drop_kid(&mut self, p: u32, k: usize, stack: &mut Vec<u32>) {
        let pn = &mut self.nodes[p as usize];
        let kid = &mut pn.kids[k];
        if kid.dead {
            return;
        }
        kid.dead = true;
        if pn.expanded {
            pn.kid_n -= 1;
            pn.kid_w -= f64::from(kid.v);
        }
        if pn.kids.iter().all(|k| k.dead) {
            stack.push(p);
        }
    }

    fn kill_all(&mut self, mut stack: Vec<u32>) -> Result<()> {
        while let Some(x) = stack.pop() {
            if self.nodes[x as usize].kind == Kind::Dead {
                continue;
            }
            if x == 0 {
                bail!("the lattice's root is a dead end");
            }
            self.stats.dead += 1;
            self.nodes[x as usize].kind = Kind::Dead;
            for (p, k) in self.nodes[x as usize].parents.clone() {
                self.drop_kid(p, k as usize, &mut stack);
            }
        }
        Ok(())
    }

    /// Visits and value sums of expanded node `p` and of each of its kids,
    /// over the direct visits of every node above them, in one scan.
    ///
    /// A direct-visit atom at commitments X counts for the node P iff X ⊇ P,
    /// and for P's kid (v, c) iff also X\[v\] = c. The atoms are each
    /// touched node's own visits, plus one per live kid of each expanded
    /// node (its expansion's visit to that child, at commitments X + kid).
    pub fn stats(&mut self, p: u32) -> Stats {
        let t0 = std::time::Instant::now();
        PROF[4].fetch_add(
            self.touched.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        SCANS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let r = self.stats_inner(p);
        prof(1, t0);
        r
    }

    fn stats_inner(&mut self, p: u32) -> Stats {
        SLOT.with(|slot| self.stats_with(p, &mut slot.borrow_mut()))
    }

    fn stats_with(&self, p: u32, slot_m: &mut [u16]) -> Stats {
        let pn = &self.nodes[p as usize];
        let pk = pn.key;
        let mut allowed = 0u64;
        for (i, k) in pn.kids.iter().enumerate() {
            if !k.dead {
                allowed |= 1 << k.var;
                slot_m[k.var as usize * 64 + k.choice as usize] = i as u16;
            }
        }
        let mut st = Stats {
            n: 0.0,
            w: 0.0,
            kids: vec![(0.0, 0.0); pn.kids.len()],
        };
        let slot = &*slot_m;
        let credit = |st: &mut Stats, x: &Key, mut extra: u64, cn: f64, cw: f64| {
            st.n += cn;
            st.w += cw;
            while extra != 0 {
                let v = extra.trailing_zeros() as usize;
                extra &= extra - 1;
                let s = slot[v * 64 + x.get(v) as usize];
                if s != u16::MAX {
                    st.kids[s as usize].0 += cn;
                    st.kids[s as usize].1 += cw;
                }
            }
        };
        for &x in &self.touched {
            let xn = &self.nodes[x as usize];
            if xn.kind == Kind::Dead || xn.key.conflicts(&pk) {
                continue;
            }
            let xk = &xn.key;
            let missing = pk.mask & !xk.mask;
            let extra = xk.mask & !pk.mask & allowed;
            if missing == 0 {
                credit(&mut st, xk, extra, f64::from(xn.n), xn.w);
                if xn.expanded && xn.kid_n > 0 {
                    credit(&mut st, xk, extra, f64::from(xn.kid_n), xn.kid_w);
                    for kid in &xn.kids {
                        if kid.dead || allowed & (1 << kid.var) == 0 {
                            continue;
                        }
                        let s = slot[kid.var as usize * 64 + kid.choice as usize];
                        if s != u16::MAX {
                            st.kids[s as usize].0 += 1.0;
                            st.kids[s as usize].1 += f64::from(kid.v);
                        }
                    }
                }
            } else if missing.count_ones() == 1 && xn.expanded {
                let v = missing.trailing_zeros() as u8;
                let c = pk.get(v as usize);
                if let Ok(i) = xn.kids.binary_search_by_key(&(v, c), |k| (k.var, k.choice)) {
                    let kid = xn.kids[i];
                    if !kid.dead {
                        credit(&mut st, xk, extra, 1.0, f64::from(kid.v));
                    }
                }
            }
        }
        for k in &pn.kids {
            slot_m[k.var as usize * 64 + k.choice as usize] = u16::MAX;
        }
        st
    }

    /// PUCT over the live kids of expanded node `p`: the prior is the kid's
    /// within its variable over the number of variables, so the priors sum
    /// to 1 over all kids.
    pub fn select(&mut self, p: u32, c_puct: f32) -> Option<usize> {
        let st = self.stats(p);
        let pn = &self.nodes[p as usize];
        let vars = pn
            .kids
            .iter()
            .filter(|k| !k.dead)
            .fold(0u64, |m, k| m | 1 << k.var)
            .count_ones()
            .max(1);
        let sq = st.n.max(1.0).sqrt();
        let mut best = None;
        let mut best_u = f64::NEG_INFINITY;
        for (i, k) in pn.kids.iter().enumerate() {
            if k.dead {
                continue;
            }
            let (n, w) = st.kids[i];
            let q = if n > 0.0 { w / n } else { f64::from(k.v) };
            let prior = f64::from(k.logp).exp() / f64::from(vars);
            let u = q + f64::from(c_puct) * prior * sq / (1.0 + n);
            if u > best_u {
                best_u = u;
                best = Some(i);
            }
        }
        best
    }

    /// Run descent `cur` until it needs the net or reaches a sink.
    pub fn advance(&mut self, cur: &mut Cursor, c_puct: f32) -> Result<Need> {
        for _ in 0..1_000_000 {
            let n = cur.at;
            let node = &self.nodes[n as usize];
            match node.kind {
                Kind::Dead => {
                    let Some((p, k)) = cur.path.pop() else {
                        return Ok(Need::Dead);
                    };
                    cur.lp -= f64::from(self.nodes[p as usize].kids[k as usize].logp);
                    cur.at = p;
                }
                Kind::Sink => {
                    let Some(v) = node.value else {
                        return Ok(Need::Value(n));
                    };
                    self.visit(n, v);
                    let node = &mut self.nodes[n as usize];
                    let fresh = node.lp_n == 0;
                    node.lp_sum += cur.lp;
                    node.lp_n += 1;
                    self.stats.descents += 1;
                    return Ok(Need::Done {
                        sink: n,
                        value: v,
                        fresh,
                    });
                }
                Kind::Inner if !node.expanded => {
                    if node.prepared || self.prepare(n)? {
                        return Ok(Need::Expand(n));
                    }
                    self.kill(n)?;
                }
                Kind::Inner => match self.select(n, c_puct) {
                    None => self.kill(n)?,
                    Some(k) => {
                        let c = self.child(n, k)?;
                        cur.lp += f64::from(self.nodes[n as usize].kids[k].logp);
                        cur.path.push((n, k as u16));
                        cur.at = c;
                    }
                },
            }
        }
        bail!("lattice descent made no progress in 1e6 steps")
    }

    /// One whole descent from `cur`, calling `eval` for every expansion and
    /// sink value on the way.
    pub fn descend(
        &mut self,
        eval: &dyn LatticeEval,
        mut cur: Cursor,
        c_puct: f32,
    ) -> Result<(u32, f32, bool)> {
        loop {
            match self.advance(&mut cur, c_puct)? {
                Need::Expand(n) => {
                    let (slots, node) = self.request(n);
                    let req = ExpandReq {
                        root: &self.start,
                        node,
                        slots: &slots,
                    };
                    let out = eval.expand(&[req])?.pop().expect("one output");
                    self.feed_expand(n, &out);
                }
                Need::Value(n) => {
                    let v = eval.value(&[&self.nodes[n as usize].state])?[0];
                    self.feed_value(n, v);
                }
                Need::Done { sink, value, fresh } => return Ok((sink, value, fresh)),
                Need::Dead => bail!("descent started at a dead node"),
            }
        }
    }

    /// The tier-1 prior's log-odds for sink `n`: the mean over the descents
    /// that reached it of the summed log priors along their paths.
    pub fn sink_logp(&self, n: u32) -> f64 {
        let s = &self.nodes[n as usize];
        s.lp_sum / f64::from(s.lp_n.max(1))
    }
}

/// Bootstrap evaluator from a steps-mode value net: a child's value is the
/// net's value of the child state, priors are uniform, and tier-1 values
/// are the net's values of turn starts.
pub struct StepsBridge {
    pub eval: std::sync::Arc<dyn crate::bfs::Evaluator>,
}

impl LatticeEval for StepsBridge {
    fn expand(&self, reqs: &[ExpandReq]) -> Result<Vec<ExpandOut>> {
        Ok(self.both(reqs, &[])?.0)
    }

    fn value(&self, states: &[&StepState]) -> Result<Vec<f32>> {
        Ok(self.both(&[], states)?.1)
    }

    fn both(
        &self,
        reqs: &[ExpandReq],
        states: &[&StepState],
    ) -> Result<(Vec<ExpandOut>, Vec<f32>)> {
        let t0 = std::time::Instant::now();
        let mut views = Vec::new();
        let mut flip = Vec::new();
        let mut term = Vec::new();
        for r in reqs {
            views.push(r.node.view());
            flip.push(false);
            term.push(None);
            for &(cell, slot) in r.slots {
                let c = r.node.with_next(cell).child_unchecked(slot);
                flip.push(c.flipped != r.node.flipped);
                term.push(c.is_terminal().then(|| c.terminal_value()));
                views.push(c.view());
            }
        }
        for s in states {
            views.push(s.view());
            flip.push(false);
            term.push(s.is_terminal().then(|| s.terminal_value()));
        }
        prof(3, t0);
        let vals = if views.is_empty() {
            Vec::new()
        } else {
            self.eval.eval(&views)?
        };
        let v = |i: usize| {
            let x = term[i].unwrap_or(vals[i]);
            if flip[i] {
                1.0 - x
            } else {
                x
            }
        };
        let mut out = Vec::with_capacity(reqs.len());
        let mut i = 0;
        for r in reqs {
            let value = v(i);
            let values = (0..r.slots.len()).map(|j| v(i + 1 + j)).collect();
            out.push(ExpandOut {
                logits: vec![0.0; r.slots.len()],
                values,
                value,
            });
            i += 1 + r.slots.len();
        }
        let sv = (i..views.len()).map(v).collect();
        Ok((out, sv))
    }
}

/// Sample from `weights^(1/temp)` mixed with Dirichlet(`alpha`) noise at
/// weight `frac` (temp 0 = argmax of the weights before noise).
pub fn noisy_pick(
    weights: &[f64],
    temp: f64,
    alpha: f64,
    frac: f64,
    rng: &mut impl rand::Rng,
) -> usize {
    use rand_distr::{Distribution, Gamma};
    let n = weights.len();
    if n == 1 {
        return 0;
    }
    let mut p: Vec<f64> = if temp <= 0.0 {
        let m = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        weights
            .iter()
            .map(|&w| f64::from(u8::from(w == m)))
            .collect()
    } else {
        weights
            .iter()
            .map(|&w| w.max(0.0).powf(1.0 / temp))
            .collect()
    };
    let s: f64 = p.iter().sum();
    if s > 0.0 {
        p.iter_mut().for_each(|x| *x /= s);
    } else {
        p.iter_mut().for_each(|x| *x = 1.0 / n as f64);
    }
    if frac > 0.0 && alpha > 0.0 {
        let g = Gamma::new(alpha, 1.0).expect("alpha > 0");
        let d: Vec<f64> = (0..n).map(|_| g.sample(rng)).collect();
        let ds: f64 = d.iter().sum::<f64>().max(1e-300);
        for (x, y) in p.iter_mut().zip(d) {
            *x = (1.0 - frac) * *x + frac * y / ds;
        }
    }
    let mut r = rng.gen::<f64>();
    for (i, &x) in p.iter().enumerate() {
        r -= x;
        if r <= 0.0 {
            return i;
        }
    }
    n - 1
}

/// A tier-2 training record: a node the walk locked a ply at, with its
/// children's statistics (indirect visits included) at that time.
pub struct T2Record {
    pub node: StepState,
    /// Per kid: cell, slot (the choice), visits, mean value.
    pub kids: Vec<(u8, u8, f32, f32)>,
    pub n: f32,
    pub q: f32,
}

#[derive(Clone, Debug)]
pub struct WalkParams {
    /// Descents per locked ply.
    pub n2: u32,
    pub c2: f32,
    pub temp: f64,
    pub dir_alpha: f64,
    pub dir_frac: f64,
}

impl Default for WalkParams {
    fn default() -> Self {
        WalkParams {
            n2: 20,
            c2: 1.5,
            temp: 1.0,
            dir_alpha: 0.3,
            dir_frac: 0.25,
        }
    }
}

/// The generation's tier-2 search over a turn, as a resumable walk: from
/// the root, `n2` descents, then lock one ply (a uniformly random variable,
/// its choice by tempered visits plus Dirichlet noise) and re-root there,
/// until the locked node is a full turn. Each locked-from node is a record.
pub struct Walk {
    pub at: u32,
    lp: f64,
    left: u32,
    cur: Option<Cursor>,
    pub records: Vec<T2Record>,
    pub done: bool,
}

impl Walk {
    pub fn new(p: &WalkParams) -> Self {
        Walk {
            at: 0,
            lp: 0.0,
            left: p.n2,
            cur: None,
            records: Vec::new(),
            done: false,
        }
    }
}

impl Lattice {
    /// Run walk `w` until it needs the net (`Some`) or ends (`None`).
    pub fn advance_walk(
        &mut self,
        w: &mut Walk,
        p: &WalkParams,
        rng: &mut impl rand::Rng,
    ) -> Result<Option<Need>> {
        loop {
            if w.done {
                return Ok(None);
            }
            if self.nodes[w.at as usize].kind != Kind::Inner {
                w.done = true;
                return Ok(None);
            }
            if w.left > 0 {
                let cur = w.cur.get_or_insert_with(|| Cursor::at(w.at, w.lp));
                match self.advance(cur, p.c2)? {
                    Need::Done { .. } => {
                        w.cur = None;
                        w.left -= 1;
                    }
                    Need::Dead => {
                        w.done = true;
                        return Ok(None);
                    }
                    need => return Ok(Some(need)),
                }
                continue;
            }
            let at = w.at;
            let st = self.stats(at);
            let nd = &self.nodes[at as usize];
            w.records.push(T2Record {
                node: nd.state.clone(),
                kids: nd
                    .kids
                    .iter()
                    .zip(&st.kids)
                    .filter(|(k, _)| !k.dead)
                    .map(|(k, &(n, sw))| (k.cell, k.choice, n as f32, (sw / n.max(1.0)) as f32))
                    .collect(),
                n: st.n as f32,
                q: (st.w / st.n.max(1.0)) as f32,
            });
            let mut vars: Vec<u8> = nd.kids.iter().filter(|k| !k.dead).map(|k| k.var).collect();
            vars.dedup();
            let var = vars[rng.gen_range(0..vars.len())];
            let idx: Vec<usize> = (0..nd.kids.len())
                .filter(|&i| nd.kids[i].var == var && !nd.kids[i].dead)
                .collect();
            let wts: Vec<f64> = idx.iter().map(|&i| st.kids[i].0).collect();
            let k = idx[noisy_pick(&wts, p.temp, p.dir_alpha, p.dir_frac, rng)];
            w.lp += f64::from(nd.kids[k].logp);
            w.at = self.child(at, k)?;
            w.left = p.n2;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::game::MinionsState;
    use rand::{Rng, SeedableRng};

    /// Deterministic pseudo-random logits and values from the node's view.
    pub(crate) struct HashLatticeEval;

    fn h(v: &crate::steps::View, salt: u64) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut bytes = Vec::new();
        crate::stepsp::encode_record(v, 0.0, 0.0, 0, &mut bytes);
        let mut s = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut s);
        salt.hash(&mut s);
        s.finish()
    }

    fn unit(x: u64) -> f32 {
        (x % 1_000_000) as f32 / 1e6
    }

    impl LatticeEval for HashLatticeEval {
        fn expand(&self, reqs: &[ExpandReq]) -> Result<Vec<ExpandOut>> {
            Ok(reqs
                .iter()
                .map(|r| {
                    let base = h(&r.node.view(), 0);
                    let f = |s: &(u8, u8), t: u64| {
                        unit(base ^ (u64::from(s.0) << 8 | u64::from(s.1)).wrapping_mul(t))
                    };
                    ExpandOut {
                        logits: r.slots.iter().map(|s| 4.0 * f(s, 0x9E37_79B9)).collect(),
                        values: r.slots.iter().map(|s| f(s, 0x85EB_CA6B)).collect(),
                        value: unit(base),
                    }
                })
                .collect())
        }

        fn value(&self, states: &[&StepState]) -> Result<Vec<f32>> {
            Ok(states.iter().map(|s| unit(h(&s.view(), 1))).collect())
        }
    }

    /// The scan's statistics against a brute-force sum over every atom.
    fn brute(l: &Lattice, p: u32) -> Stats {
        let mut atoms: Vec<(Key, f64, f64)> = Vec::new();
        for x in &l.nodes {
            if x.kind == Kind::Dead {
                continue;
            }
            if x.n > 0 {
                atoms.push((x.key, f64::from(x.n), x.w));
            }
            if x.expanded {
                for k in x.kids.iter().filter(|k| !k.dead) {
                    atoms.push((x.key.with(k.var as usize, k.choice), 1.0, f64::from(k.v)));
                }
            }
        }
        let pn = &l.nodes[p as usize];
        let sum = |key: &Key| {
            atoms
                .iter()
                .filter(|a| a.0.includes(key))
                .fold((0.0, 0.0), |s, a| (s.0 + a.1, s.1 + a.2))
        };
        let (n, w) = sum(&pn.key);
        Stats {
            n,
            w,
            kids: pn
                .kids
                .iter()
                .map(|k| {
                    if k.dead {
                        (0.0, 0.0)
                    } else {
                        sum(&pn.key.with(k.var as usize, k.choice))
                    }
                })
                .collect(),
        }
    }

    fn check_stats(l: &mut Lattice) {
        let ids: Vec<u32> = (0..l.nodes.len() as u32)
            .filter(|&i| l.nodes[i as usize].expanded && l.nodes[i as usize].kind != Kind::Dead)
            .collect();
        for p in ids {
            let a = l.stats(p);
            let b = brute(l, p);
            assert!(
                (a.n - b.n).abs() < 1e-6 && (a.w - b.w).abs() < 1e-6,
                "node {p}: {a:?} vs {b:?}"
            );
            for (i, (x, y)) in a.kids.iter().zip(&b.kids).enumerate() {
                assert!(
                    (x.0 - y.0).abs() < 1e-6 && (x.1 - y.1).abs() < 1e-6,
                    "node {p} kid {i}: {x:?} vs {y:?}"
                );
            }
        }
    }

    /// Positions some turns into a game, by playing each turn's best-valued
    /// sink of a short lattice search.
    fn positions(n_turns: usize, seed: u64) -> Vec<StepState> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut s = StepState::begin(&MinionsState::with_bid(0), seed);
        let mut out = vec![s.clone()];
        for _ in 0..n_turns {
            let mut l = Lattice::new(s.clone()).unwrap();
            if l.terminal.is_some() {
                break;
            }
            let mut sinks = Vec::new();
            for _ in 0..rng.gen_range(3..12) {
                sinks.push(
                    l.descend(&HashLatticeEval, Cursor::at(0, 0.0), 1.0)
                        .unwrap()
                        .0,
                );
            }
            let pick = sinks[rng.gen_range(0..sinks.len())];
            s = l.nodes[pick as usize].state.clone();
            if s.is_terminal() {
                break;
            }
            out.push(s.clone());
        }
        out
    }

    #[test]
    fn keys_are_order_free() {
        let a = Key::default().with(3, 5).with(12, 1);
        let b = Key::default().with(12, 1).with(3, 5);
        assert_eq!(a, b);
        assert!(a.includes(&Key::default().with(12, 1)));
        assert!(!a.includes(&Key::default().with(12, 2)));
        assert!(a.conflicts(&Key::default().with(3, 4)));
        assert!(!a.conflicts(&Key::default().with(40, 4)));
        // Two orders of the same two commitments reach the same node.
        let mut compared = 0;
        for s in positions(4, 3) {
            let mut l = Lattice::new(s).unwrap();
            if l.terminal.is_some() || l.nodes[0].kind != Kind::Inner {
                continue;
            }
            assert!(l.prepare(0).unwrap());
            let kids = l.nodes[0].kids.clone();
            let (Some(i), Some(j)) = (
                kids.iter().position(|k| k.var == kids[0].var),
                kids.iter().position(|k| k.var != kids[0].var),
            ) else {
                continue;
            };
            let a = l.child(0, i).unwrap();
            let b = l.child(0, j).unwrap();
            for (x, y) in [(a, kids[j]), (b, kids[i])] {
                if l.nodes[x as usize].kind != Kind::Inner || !l.prepare(x).unwrap() {
                    continue;
                }
                if let Some(k) = l.nodes[x as usize]
                    .kids
                    .iter()
                    .position(|k| k.var == y.var && k.choice == y.choice)
                {
                    l.child(x, k).unwrap();
                }
            }
            let ab = l.nodes[a as usize]
                .kids
                .iter()
                .find(|k| k.var == kids[j].var && k.choice == kids[j].choice)
                .map(|k| k.node);
            let ba = l.nodes[b as usize]
                .kids
                .iter()
                .find(|k| k.var == kids[i].var && k.choice == kids[i].choice)
                .map(|k| k.node);
            if let (Some(x), Some(y)) = (ab, ba) {
                assert_eq!(x, y, "the two orders reached different nodes");
                compared += 1;
            }
        }
        assert!(compared > 0, "no pair of orders compared");
    }

    #[test]
    #[ignore]
    fn sizes() {
        eprintln!(
            "StepState {} MinionsState {} TurnSpec {} LNode {} Kid {}",
            std::mem::size_of::<StepState>(),
            std::mem::size_of::<crate::game::MinionsState>(),
            std::mem::size_of::<TurnSpec>(),
            std::mem::size_of::<LNode>(),
            std::mem::size_of::<Kid>()
        );
    }

    /// Cost profile: kids per expansion and CPU time per expansion.
    #[test]
    #[ignore]
    fn lattice_profile() {
        for (t, s) in positions(10, 5).into_iter().enumerate() {
            let mut l = Lattice::new(s).unwrap();
            if l.terminal.is_some() {
                continue;
            }
            let t0 = std::time::Instant::now();
            let mut sinks = std::collections::HashSet::new();
            for _ in 0..100 {
                sinks.insert(
                    l.descend(&HashLatticeEval, Cursor::at(0, 0.0), 1.0)
                        .unwrap()
                        .0,
                );
            }
            let dt = t0.elapsed().as_secs_f64();
            let st = &l.stats;
            eprintln!(
                "turn {t} vars {}: {} expansions, {:.0} kids/exp, {} sinks distinct {}, dead {}, {:.2} ms/exp, {} nodes",
                l.n_fm, st.expansions, st.kids as f64 / st.expansions as f64, st.sinks, sinks.len(), st.dead,
                1e3 * dt / st.expansions as f64, l.nodes.len()
            );
        }
    }

    /// Descents end on full turns, the lazy statistics equal the brute-force
    /// sums, sinks are distinct, and the search spreads over several turns.
    #[test]
    fn lattice_descents() {
        let mut spawn_nodes = 0;
        for (t, s) in positions(6, 11).into_iter().enumerate() {
            let mut l = Lattice::new(s).unwrap();
            if l.terminal.is_some() {
                continue;
            }
            let mut sinks = std::collections::HashSet::new();
            for i in 0..60 {
                let (sink, v, fresh) = l
                    .descend(&HashLatticeEval, Cursor::at(0, 0.0), 1.0)
                    .unwrap();
                let sn = &l.nodes[sink as usize];
                assert_eq!(sn.kind, Kind::Sink);
                assert!(sn.state.is_terminal() || sn.state.flipped != l.start.flipped);
                assert!((0.0..=1.0).contains(&v));
                assert_eq!(fresh, sinks.insert(sink), "fresh flag wrong at descent {i}");
                if i % 20 == 19 {
                    check_stats(&mut l);
                }
            }
            spawn_nodes += l
                .nodes
                .iter()
                .filter(|n| n.state.stage == Stage::Spawns)
                .count();
            let keys: std::collections::HashSet<Key> = l.nodes.iter().map(|n| n.key).collect();
            assert_eq!(keys.len(), l.nodes.len());
            assert!(
                sinks.len() > 1 || l.nodes[0].kids.len() <= 1,
                "turn {t}: one sink only"
            );
        }
        assert!(spawn_nodes > 0, "never reached a spawn phase");
    }

    /// The walk ends on a full turn, records one node per locked ply, and
    /// each record's kid visits are at least one (the expansion's).
    #[test]
    fn walk_locks_down_to_a_turn() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        let p = WalkParams::default();
        for s in positions(5, 21) {
            let mut l = Lattice::new(s).unwrap();
            if l.terminal.is_some() {
                continue;
            }
            for _ in 0..10 {
                l.descend(&HashLatticeEval, Cursor::at(0, 0.0), 1.5)
                    .unwrap();
            }
            let mut w = Walk::new(&p);
            while let Some(need) = l.advance_walk(&mut w, &p, &mut rng).unwrap() {
                match need {
                    Need::Expand(n) => {
                        let (slots, node) = l.request(n);
                        let req = ExpandReq {
                            root: &l.start,
                            node,
                            slots: &slots,
                        };
                        let out = HashLatticeEval.expand(&[req]).unwrap().pop().unwrap();
                        l.feed_expand(n, &out);
                    }
                    Need::Value(n) => {
                        let v = HashLatticeEval
                            .value(&[&l.nodes[n as usize].state])
                            .unwrap()[0];
                        l.feed_value(n, v);
                    }
                    Need::Done { .. } | Need::Dead => unreachable!(),
                }
            }
            assert_eq!(l.nodes[w.at as usize].kind, Kind::Sink);
            assert!(!w.records.is_empty());
            for r in &w.records {
                assert!(r.kids.iter().all(|k| k.2 >= 1.0));
                assert!(r.n >= p.n2 as f32);
            }
        }
    }

    /// Under lazy witnesses some offered kids are refuted at their own
    /// expansion: descents still end on full turns and the statistics stay
    /// exact with dead nodes around.
    #[test]
    fn lattice_lazy_witness() {
        crate::steps::LAZY_WITNESS.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut dead = 0;
        for s in positions(8, 17) {
            let mut l = Lattice::new(s).unwrap();
            if l.terminal.is_some() {
                continue;
            }
            for _ in 0..80 {
                let (sink, _, _) = l
                    .descend(&HashLatticeEval, Cursor::at(0, 0.0), 1.5)
                    .unwrap();
                assert_eq!(l.nodes[sink as usize].kind, Kind::Sink);
            }
            check_stats(&mut l);
            dead += l.stats.dead;
        }
        eprintln!("dead nodes: {dead}");
    }
}
