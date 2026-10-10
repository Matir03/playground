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
//!
//! The intermediate visits (expansions' predicted values) only steer the
//! search. Training targets average the sinks' returns alone: tier-1 values
//! of complete turns ([`Stats::sn`], [`Stats::skids`]).

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
pub static LIVE: [std::sync::atomic::AtomicI64; 4] =
    [const { std::sync::atomic::AtomicI64::new(0) }; 4];

pub static SCANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Option-check cost breakdown while preparing nodes: per
/// [`crate::steps::cand_log`] class, candidates and nanoseconds.
pub static WIT_CLASS: [[std::sync::atomic::AtomicU64; 2]; crate::steps::cand_log::N] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; crate::steps::cand_log::N];

/// Deferred witnesses resolved at preparation: found, dead, nanoseconds;
/// nodes found dead by a nogood before any search, and deferred witnesses
/// taken from the pool.
pub static WIT_RESOLVE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// What inheriting candidates from the creating parent would change, over
/// nodes in the same stage as that parent: candidates the parent did not
/// offer live (count, nanoseconds spent checking them again), and how many
/// of those this node offers anyway (choices inheritance would drop).
pub static WIT_INHERIT: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

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

    /// With variable `v` (decided or not) set to `c`.
    pub fn set(mut self, v: usize, c: u8) -> Key {
        let sh = 6 * (v % PER_WORD);
        self.mask |= 1 << v;
        self.packed[v / PER_WORD] =
            (self.packed[v / PER_WORD] & !(63 << sh)) | (u64::from(c) << sh);
        self
    }
}

/// Known witnesses kept per lattice (complete fates+moves declarations as
/// keys), and known infeasible keys (`LATTICE_POOL`, `LATTICE_NOGOODS`: caps,
/// 0 = off).
fn pool_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LATTICE_POOL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048)
    })
}

fn nogood_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LATTICE_NOGOODS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096)
    })
}

/// A node to expand: the turn's start (whose tier-1 trunk the tier-2 net
/// reads) and the node's state, plus the `(cell, slot)` of each child. The
/// slot is the choice: a fate code, a move offset, or a spawn's unit code.
pub struct ExpandReq<'a> {
    pub root: &'a StepState,
    pub node: &'a StepState,
    pub slots: &'a [(u8, u8)],
    /// The turn start's tier-1 trunk image if known (else the evaluator
    /// computes it and returns it in [`ExpandOut::trunk`]).
    pub trunk: Option<Trunk>,
}

/// A turn start's tier-1 trunk image (fp16, C x 7 x 7), computed once per
/// turn start and reused by every tier-2 call in its lattice.
pub type Trunk = std::sync::Arc<[half::f16]>;

/// A tier-1 value for the state's mover, and the trunk image it came with.
#[derive(Clone)]
pub struct ValueOut {
    pub v: f32,
    pub trunk: Option<Trunk>,
}

/// Per child (parallel to the request's slots) a prior logit and a value,
/// and the node's own value. Values are for the turn's mover.
pub struct ExpandOut {
    pub logits: Vec<f32>,
    pub values: Vec<f32>,
    pub value: f32,
    /// The node's energy (lattice-view nets; NaN otherwise): a complete
    /// declaration's tier-1 prior logit.
    pub energy: f32,
    /// The turn start's trunk image, when the request had none.
    pub trunk: Option<Trunk>,
}

/// The results of a submitted net call, to collect later.
pub struct Pending(Box<dyn FnOnce() -> Result<(Vec<ExpandOut>, Vec<ValueOut>)> + Send>);

impl Pending {
    pub fn ready(r: Result<(Vec<ExpandOut>, Vec<ValueOut>)>) -> Self {
        Pending(Box::new(move || r))
    }

    pub fn new(
        f: impl FnOnce() -> Result<(Vec<ExpandOut>, Vec<ValueOut>)> + Send + 'static,
    ) -> Self {
        Pending(Box::new(f))
    }

    /// Block until the results are in.
    pub fn wait(self) -> Result<(Vec<ExpandOut>, Vec<ValueOut>)> {
        (self.0)()
    }
}

pub trait LatticeEval: Send + Sync {
    /// Expansions and tier-1 values in one go (one net call where the
    /// evaluator can).
    fn both(
        &self,
        reqs: &[ExpandReq],
        states: &[&StepState],
    ) -> Result<(Vec<ExpandOut>, Vec<ValueOut>)>;

    /// [`Self::both`] without waiting: the requests are read now, and the
    /// caller collects the results with [`Pending::wait`] (synchronous unless
    /// the evaluator overrides it).
    fn submit(&self, reqs: &[ExpandReq], states: &[&StepState]) -> Result<Pending> {
        Ok(Pending::ready(self.both(reqs, states)))
    }

    fn expand(&self, reqs: &[ExpandReq]) -> Result<Vec<ExpandOut>> {
        Ok(self.both(reqs, &[])?.0)
    }

    /// Tier-1 values of turn starts, each for its own mover.
    fn value(&self, states: &[&StepState]) -> Result<Vec<f32>> {
        Ok(self.both(&[], states)?.1.into_iter().map(|o| o.v).collect())
    }

    /// One tier-1 value with its trunk.
    fn value1(&self, s: &StepState) -> Result<ValueOut> {
        Ok(self.both(&[], &[s])?.1.pop().expect("one value"))
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
    /// A net request for this node (expansion or sink value) is in flight.
    pub busy: bool,
    /// Listed in the lattice's touched nodes.
    ever: bool,
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
    /// A sink's returns (descents that ended here) and their value sum: the
    /// tier-1 values the training targets average.
    pub ret_n: u32,
    pub ret_w: f64,
    /// A sink's energy (its complete declaration's tier-1 prior logit),
    /// once known.
    pub energy: Option<f32>,
    parents: Vec<(u32, u16)>,
    /// A sink's trunk image (from its tier-1 value), for its own lattice.
    pub trunk: Option<Trunk>,
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
            busy: false,
            ever: false,
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
            ret_n: 0,
            ret_w: 0.0,
            energy: None,
            parents: Vec::new(),
            trunk: None,
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
    /// Follow-mode SAT checks and how many refuted their node.
    pub sat_calls: u64,
    pub sat_refuted: u64,
}

/// One descent in progress: where it is, the `(node, kid)` path from where
/// it started, and the summed log prior from the lattice's root.
#[derive(Clone, Debug)]
pub struct Cursor {
    pub at: u32,
    pub path: Vec<(u32, u16)>,
    pub lp: f64,
    /// Follow each node's own (verified) witness instead of PUCT: the kid
    /// agreeing with it keeps that witness, so the descent cannot dead-end.
    /// The fallback for a descent lost among lazily refuted options.
    pub follow: bool,
    /// The node holding this descent's virtual loss (`NONE`: none yet).
    vl: u32,
}

impl Cursor {
    pub fn at(node: u32, lp: f64) -> Self {
        Cursor {
            at: node,
            path: Vec::new(),
            lp,
            follow: false,
            vl: NONE,
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
    /// Another descent's request for this node is in flight: try again
    /// after it is fed.
    Wait,
}

/// Children statistics at a node, from [`Lattice::stats`].
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub n: f64,
    pub w: f64,
    /// Per kid (parallel to the node's kids): visits and value sum.
    pub kids: Vec<(f64, f64)>,
    /// The same over the sinks' returns only (complete turns at their
    /// tier-1 values): the training targets.
    pub sn: f64,
    pub sw: f64,
    pub skids: Vec<(f64, f64)>,
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
    /// Fate variables (the first `n_f`).
    n_f: usize,
    /// A complete declaration of the turn (the root's witness): what a
    /// pool key's declaration is filled in from.
    wit0: TurnSpec,
    /// Known witnesses (keys over every fate and move variable) and their
    /// hashes, to keep them distinct.
    pool: Vec<Key>,
    pool_seen: std::collections::HashSet<u64>,
    /// Known infeasible commitment sets: every node including one is dead.
    nogoods: Vec<Key>,
    /// The mover can kill the enemy necromancer: the turn is terminal with
    /// this value and has no lattice.
    pub terminal: Option<f32>,
    /// The exact SAT check of this turn ([`crate::sat`]), built on first
    /// use: the follow mode's fallback.
    sat: Option<Box<crate::sat::TurnSat>>,
    /// Follow mode always asks SAT, even past a verified own witness
    /// (tests).
    pub follow_sat_only: bool,
    pub stats: LatticeStats,
    /// The turn start's tier-1 trunk image, once known.
    pub trunk: Option<Trunk>,
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
        LIVE[3].fetch_sub(
            self.nodes.iter().map(|n| n.wit_ex.len() as i64).sum(),
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
            n_f: 0,
            wit0: TurnSpec::default(),
            pool: Vec::new(),
            pool_seen: Default::default(),
            nogoods: Vec::new(),
            terminal: None,
            sat: None,
            follow_sat_only: false,
            stats: LatticeStats::default(),
            trunk: None,
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
        l.n_f = fates.len();
        l.wit0 = *root.witness();
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

    /// A complete declaration as a key over every fate and move variable.
    fn witness_key(&self, w: &TurnSpec) -> Key {
        let mut k = Key::default();
        for (c, &v) in self.var_of.iter().enumerate() {
            let v = v as usize;
            if v < self.n_fm {
                let ch = if v < self.n_f {
                    w.targets[c]
                } else {
                    match w.moves[c] {
                        crate::turn::NO_MOVE => 0,
                        m => m,
                    }
                };
                k = k.with(v, ch);
            }
        }
        k
    }

    /// The declaration a pool key stands for.
    fn key_witness(&self, k: &Key) -> TurnSpec {
        let mut w = self.wit0;
        for (c, &v) in self.var_of.iter().enumerate() {
            let v = v as usize;
            if v < self.n_f {
                w.targets[c] = k.get(v);
            } else if v < self.n_fm {
                let m = k.get(v);
                if !(m == 0 && w.moves[c] == crate::turn::NO_MOVE) {
                    w.moves[c] = m;
                }
            }
        }
        w
    }

    fn pool_add(&mut self, k: Key) {
        if self.pool.len() >= pool_cap() {
            return;
        }
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        k.hash(&mut h);
        if self.pool_seen.insert(h.finish()) {
            self.pool.push(k);
        }
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
        use std::sync::atomic::Ordering::Relaxed;
        self.nodes[n as usize].prepared = true;
        let xk = self.nodes[n as usize].key;
        let fm = matches!(
            self.nodes[n as usize].state.stage,
            Stage::Fates | Stage::Moves
        );
        // What the lattice knows about this node's commitments: a nogood
        // inside them kills it; one short of them by a single commitment
        // refutes that choice; a known witness including them certifies it.
        let mut refuted: Vec<(u8, u8)> = Vec::new();
        let mut hits: Vec<Key> = Vec::new();
        if fm {
            for g in &self.nogoods {
                let extra = g.mask & !xk.mask;
                if extra.count_ones() <= 1 && !g.conflicts(&xk) {
                    if extra == 0 {
                        WIT_RESOLVE[3].fetch_add(1, Relaxed);
                        return Ok(false);
                    }
                    let v = extra.trailing_zeros() as usize;
                    refuted.push((v as u8, g.get(v)));
                }
            }
            hits = self
                .pool
                .iter()
                .filter(|w| w.includes(&xk))
                .copied()
                .collect();
        }
        let pool_wit = hits.first().map(|w| self.key_witness(w));
        let node = &mut self.nodes[n as usize];
        let t_res = std::time::Instant::now();
        let was_pending = !node.state.resolve_pending_is_noop();
        if was_pending {
            if let Some(w) = pool_wit {
                node.state.set_witness(w);
                WIT_RESOLVE[4].fetch_add(1, Relaxed);
            }
        }
        let was_pending = !node.state.resolve_pending_is_noop();
        let ok = node.state.resolve_pending();
        if was_pending {
            WIT_RESOLVE[usize::from(!ok)].fetch_add(1, Relaxed);
            WIT_RESOLVE[2].fetch_add(t_res.elapsed().as_nanos() as u64, Relaxed);
        }
        if !ok {
            return Ok(false);
        }
        // The creating parent, when it decides the same stage's variables.
        let parent = self.nodes[n as usize]
            .parents
            .first()
            .map(|&(p, _)| p)
            .filter(|&p| self.nodes[p as usize].state.stage == self.nodes[n as usize].state.stage);
        let node = &self.nodes[n as usize];
        let st = &node.state;
        let mut kids: Vec<(Kid, TurnSpec)> = Vec::new();
        let mut new_nogoods: Vec<Key> = Vec::new();
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
                crate::steps::cand_log::on(true);
                let mut cls = [[0u64; 2]; crate::steps::cand_log::N];
                let mut inh = [0u64; 3];
                for &c in st.undecided() {
                    let v = var_of[c as usize] as usize;
                    let oracle = |choice: u8| -> crate::steps::Known {
                        if refuted.contains(&(v as u8, choice)) {
                            crate::steps::Known::Refuted
                        } else if let Some(w) = hits.iter().find(|w| w.get(v) == choice) {
                            crate::steps::Known::Witness(self.key_witness(w))
                        } else {
                            crate::steps::Known::Nothing
                        }
                    };
                    let (o, w) = st.with_next(c).lattice_options_with(&oracle);
                    let log = crate::steps::cand_log::take();
                    for &(choice, class, _) in &log {
                        use crate::steps::cand_log::{EXACT_NO, RELAXED};
                        if class == RELAXED || class == EXACT_NO {
                            new_nogoods.push(xk.with(v, choice));
                        }
                    }
                    let pk: Option<&[Kid]> = parent.map(|p| {
                        let ks = &self.nodes[p as usize].kids;
                        let a = ks.partition_point(|k| (k.var as usize) < v);
                        let b = ks.partition_point(|k| (k.var as usize) <= v);
                        &ks[a..b]
                    });
                    for &(choice, class, ns) in &log {
                        cls[class as usize][0] += 1;
                        cls[class as usize][1] += u64::from(ns);
                        if let Some(pk) = pk {
                            let live = pk.iter().any(|k| k.choice == choice && !k.dead);
                            if !live {
                                inh[0] += 1;
                                inh[1] += u64::from(ns);
                                inh[2] += u64::from(crate::steps::cand_log::offers(class));
                            }
                        }
                    }
                    kids.extend(o.into_iter().zip(w).map(|(x, w)| (kid(v, x, c), w)));
                }
                crate::steps::cand_log::on(false);
                for (a, b) in WIT_CLASS.iter().zip(cls) {
                    a[0].fetch_add(b[0], Relaxed);
                    a[1].fetch_add(b[1], Relaxed);
                }
                for (a, b) in WIT_INHERIT.iter().zip(inh) {
                    a.fetch_add(b, Relaxed);
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
        LIVE[3].fetch_add(
            node.wit_ex.len() as i64,
            std::sync::atomic::Ordering::Relaxed,
        );
        // Not `into_iter().collect()`: that reuses the (Kid, TurnSpec)
        // allocation in place, ~10x the bytes the kids need.
        node.kids = kids.iter().map(|x| x.0).collect();
        let alive = !node.kids.is_empty();
        if fm {
            let cap = nogood_cap();
            for g in new_nogoods {
                if self.nogoods.len() < cap {
                    self.nogoods.push(g);
                }
            }
            // Every verified witness here joins the pool: the node's own, and
            // its kids' (a kid's is the own one with its choice swapped in,
            // unless listed as an exception).
            if pool_cap() > 0 && !crate::steps::witness_is_pending(&own) {
                let ok = self.witness_key(&own);
                self.pool_add(ok);
                for (k, w) in &kids {
                    if crate::steps::witness_is_pending(w) {
                        continue;
                    }
                    let key = if *w == swapped(&own, stage, k.cell, k.choice) {
                        ok.set(k.var as usize, k.choice)
                    } else {
                        self.witness_key(w)
                    };
                    self.pool_add(key);
                }
            }
        }
        Ok(alive)
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
        if self.trunk.is_none() {
            self.trunk.clone_from(&out.trunk);
        }
        let node = &mut self.nodes[n as usize];
        node.busy = false;
        if node.expanded || node.kind != Kind::Inner {
            return;
        }
        debug_assert!(node.prepared);
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

    /// Record sink `n`'s value (`v` for the sink's own mover) and trunk.
    pub fn feed_value(&mut self, n: u32, out: &ValueOut) {
        let v = out.v;
        self.nodes[n as usize].trunk.clone_from(&out.trunk);
        let s = &self.nodes[n as usize].state;
        let v = for_root(&self.start, s, v);
        let node = &mut self.nodes[n as usize];
        node.busy = false;
        if node.value.is_none() {
            node.value = Some(v);
        }
    }

    fn visit(&mut self, n: u32, v: f32) {
        let node = &mut self.nodes[n as usize];
        if !node.ever {
            node.ever = true;
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
            let nd = &self.nodes[x as usize];
            // No completion: every node including these commitments is dead.
            if nd.kind == Kind::Inner
                && matches!(nd.state.stage, Stage::Fates | Stage::Moves)
                && self.nogoods.len() < nogood_cap()
            {
                self.nogoods.push(nd.key);
            }
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
            sn: 0.0,
            sw: 0.0,
            skids: vec![(0.0, 0.0); pn.kids.len()],
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
                if xn.ret_n > 0 {
                    let (cn, cw) = (f64::from(xn.ret_n), xn.ret_w);
                    st.sn += cn;
                    st.sw += cw;
                    let mut e = extra;
                    while e != 0 {
                        let v = e.trailing_zeros() as usize;
                        e &= e - 1;
                        let s = slot[v * 64 + xk.get(v) as usize];
                        if s != u16::MAX {
                            st.skids[s as usize].0 += cn;
                            st.skids[s as usize].1 += cw;
                        }
                    }
                }
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

    /// PUCT over the live kids of expanded node `p`. Each variable's kids
    /// are their own set, with priors summing to 1 within it (not
    /// normalized over all kids): one comparison over every kid just picks
    /// a variable and its choice at once, and how the variable is picked
    /// may change later.
    pub fn select(&mut self, p: u32, c_puct: f32) -> Option<usize> {
        let st = self.stats(p);
        let pn = &self.nodes[p as usize];
        let sq = st.n.max(1.0).sqrt();
        let mut best = None;
        let mut best_u = f64::NEG_INFINITY;
        for (i, k) in pn.kids.iter().enumerate() {
            if k.dead {
                continue;
            }
            let (n, w) = st.kids[i];
            let q = if n > 0.0 { w / n } else { f64::from(k.v) };
            let prior = f64::from(k.logp).exp();
            let u = q + f64::from(c_puct) * prior * sq / (1.0 + n);
            if u > best_u {
                best_u = u;
                best = Some(i);
            }
        }
        if best.is_none() && pn.kids.iter().any(|k| !k.dead) {
            let live: Vec<_> = pn
                .kids
                .iter()
                .zip(&st.kids)
                .filter(|(k, _)| !k.dead)
                .map(|(k, s)| (k.logp, k.v, s.0, s.1))
                .collect();
            panic!(
                "no PUCT candidate at node {p} (n {} w {}): live kids (logp, v, n, w) {live:?}",
                st.n, st.w
            );
        }
        best
    }

    /// Run descent `cur` until it needs the net or reaches a sink.
    pub fn advance(&mut self, cur: &mut Cursor, c_puct: f32) -> Result<Need> {
        for _ in 0..1_000_000 {
            let n = cur.at;
            // Virtual loss: a pending zero-value visit where the descent is,
            // which counts against every node below it as well.
            if cur.vl != n {
                self.move_vl(cur, n);
            }
            let node = &self.nodes[n as usize];
            match node.kind {
                Kind::Dead => {
                    let Some((p, k)) = cur.path.pop() else {
                        self.move_vl(cur, NONE);
                        return Ok(Need::Dead);
                    };
                    cur.lp -= f64::from(self.nodes[p as usize].kids[k as usize].logp);
                    cur.at = p;
                }
                Kind::Sink => {
                    let Some(v) = node.value else {
                        if node.busy {
                            return Ok(Need::Wait);
                        }
                        self.nodes[n as usize].busy = true;
                        return Ok(Need::Value(n));
                    };
                    self.move_vl(cur, NONE);
                    self.visit(n, v);
                    let node = &mut self.nodes[n as usize];
                    node.ret_n += 1;
                    node.ret_w += f64::from(v);
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
                    if node.busy {
                        return Ok(Need::Wait);
                    }
                    if node.prepared || self.prepare(n)? {
                        self.nodes[n as usize].busy = true;
                        return Ok(Need::Expand(n));
                    }
                    self.kill(n)?;
                }
                Kind::Inner => match if cur.follow {
                    self.follow_kid(n, c_puct)
                } else {
                    self.select(n, c_puct)
                } {
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

    /// The live kid of expanded node `n` agreeing with declaration `w`.
    fn agreeing_kid(&self, n: u32, w: &TurnSpec) -> Option<usize> {
        let node = &self.nodes[n as usize];
        node.kids.iter().position(|k| {
            !k.dead
                && match node.state.stage {
                    Stage::Fates => w.targets[k.cell as usize] == k.choice,
                    Stage::Moves => w.moves[k.cell as usize] == k.choice,
                    _ => w.spawns[k.cell as usize] == k.choice,
                }
        })
    }

    /// The follow mode's step from expanded node `n`: the kid agreeing with
    /// the node's own witness if verified, else with an exact SAT witness
    /// of its decisions (which the kid then inherits); None: SAT refuted
    /// the node. In the spawn phase (moves all decided), any kid by PUCT.
    fn follow_kid(&mut self, n: u32, c_puct: f32) -> Option<usize> {
        let st = &self.nodes[n as usize].state;
        if matches!(st.stage, Stage::Spawns | Stage::Terminal) {
            return self.select(n, c_puct);
        }
        let own = *st.witness();
        if !self.follow_sat_only && !crate::steps::witness_is_pending(&own) {
            if let Some(k) = self.agreeing_kid(n, &own) {
                return Some(k);
            }
        }
        let fixes = crate::sat::fixes_of(st);
        let base = &self.start.base;
        let sat = self
            .sat
            .get_or_insert_with(|| Box::new(crate::sat::TurnSat::new(base)));
        self.stats.sat_calls += 1;
        match sat.complete(&fixes) {
            crate::sat::SatResult::Witness(w, _) => {
                let Some(k) = self.agreeing_kid(n, &w) else {
                    // The witness's choice is not among the kids (an option
                    // refuted inexactly): any live kid.
                    return self.select(n, c_puct);
                };
                let node = &mut self.nodes[n as usize];
                match node.wit_ex.binary_search_by_key(&(k as u16), |x| x.0) {
                    Ok(i) => node.wit_ex[i].1 = w,
                    Err(i) => node.wit_ex.insert(i, (k as u16, w)),
                }
                Some(k)
            }
            crate::sat::SatResult::Refuted(_) => {
                self.stats.sat_refuted += 1;
                None
            }
        }
    }

    /// Record sink `n`'s energy (ignored unless finite).
    pub fn feed_energy(&mut self, n: u32, e: f32) {
        if e.is_finite() {
            self.nodes[n as usize].energy = Some(e);
        }
    }

    /// Give up descent `cur`: drop its virtual loss.
    pub fn abandon(&mut self, cur: &mut Cursor) {
        self.move_vl(cur, NONE);
    }

    /// Move `cur`'s virtual loss to node `to` (`NONE`: drop it).
    fn move_vl(&mut self, cur: &mut Cursor, to: u32) {
        if cur.vl != NONE {
            self.nodes[cur.vl as usize].n -= 1;
        }
        if to != NONE {
            let node = &mut self.nodes[to as usize];
            if !node.ever {
                node.ever = true;
                self.touched.push(to);
            }
            node.n += 1;
        }
        cur.vl = to;
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
                        trunk: self.trunk.clone(),
                    };
                    let out = eval.expand(&[req])?.pop().expect("one output");
                    self.feed_expand(n, &out);
                }
                Need::Value(n) => {
                    let o = eval
                        .both(&[], &[&self.nodes[n as usize].state])?
                        .1
                        .pop()
                        .expect("one value");
                    self.feed_value(n, &o);
                }
                Need::Done { sink, value, fresh } => return Ok((sink, value, fresh)),
                Need::Dead => bail!("descent started at a dead node"),
                Need::Wait => bail!("a lone descent waited on itself"),
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

/// Expansions from one evaluator, turn-start (tier-1) values from another:
/// e.g. a lattice net's tier 2 with complete turns valued by a steps value
/// net. The values carry no trunk image, so the expansions compute their own.
pub struct HybridEval {
    pub expand: std::sync::Arc<dyn LatticeEval>,
    pub values: std::sync::Arc<dyn LatticeEval>,
}

impl LatticeEval for HybridEval {
    fn both(
        &self,
        reqs: &[ExpandReq],
        states: &[&StepState],
    ) -> Result<(Vec<ExpandOut>, Vec<ValueOut>)> {
        let e = if reqs.is_empty() {
            Vec::new()
        } else {
            self.expand.both(reqs, &[])?.0
        };
        let v = if states.is_empty() {
            Vec::new()
        } else {
            self.values.both(&[], states)?.1
        };
        Ok((e, v))
    }

    fn submit(&self, reqs: &[ExpandReq], states: &[&StepState]) -> Result<Pending> {
        let p = if reqs.is_empty() {
            None
        } else {
            Some(self.expand.submit(reqs, &[])?)
        };
        let v = if states.is_empty() {
            Vec::new()
        } else {
            self.values.both(&[], states)?.1
        };
        Ok(Pending::new(move || {
            let e = match p {
                Some(p) => p.wait()?.0,
                None => Vec::new(),
            };
            Ok((e, v))
        }))
    }
}

/// Bootstrap evaluator from a steps-mode value net: a child's value is the
/// net's value of the child state, priors are uniform, and tier-1 values
/// are the net's values of turn starts.
pub struct StepsBridge {
    pub eval: std::sync::Arc<dyn crate::bfs::Evaluator>,
}

impl LatticeEval for StepsBridge {
    fn both(
        &self,
        reqs: &[ExpandReq],
        states: &[&StepState],
    ) -> Result<(Vec<ExpandOut>, Vec<ValueOut>)> {
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
                energy: f32::NAN,
                trunk: None,
            });
            i += 1 + r.slots.len();
        }
        let sv = (i..views.len())
            .map(|j| ValueOut {
                v: v(j),
                trunk: None,
            })
            .collect();
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
    /// Per kid: cell, slot (the choice), visits, and the mean of the sink
    /// returns below it (-1 if none) with their count.
    pub kids: Vec<(u8, u8, f32, f32, f32)>,
    /// Visits; the mean of the sink returns below the node (-1 if none) and
    /// their count.
    pub n: f32,
    pub q: f32,
    pub ns: f32,
}

/// A sink-return mean, or -1 (no target) without returns.
fn ret_mean(n: f64, w: f64) -> f32 {
    if n > 0.0 {
        (w / n) as f32
    } else {
        -1.0
    }
}

#[derive(Clone, Debug)]
pub struct WalkParams {
    /// Descents per locked ply.
    pub n2: u32,
    pub c2: f32,
    pub temp: f64,
    pub dir_alpha: f64,
    pub dir_frac: f64,
    /// Descents in flight at a time.
    pub inflight: usize,
    /// Self-distillation (see [`Lattice::advance_sd`]): descents per kid,
    /// and the temperature of the per-cell softmax prior targets.
    pub sd_descents: u32,
    pub sd_tau: f64,
    /// The walk's descents select on their own ply's statistics (a fresh
    /// n2-visit search per ply, see [`Lattice::advance_fresh`]) instead of
    /// the lattice's accumulated ones.
    pub fresh: bool,
    /// The records count each descent's first step from the recorded node
    /// (the commitment the search chose to make there: AlphaZero's edge
    /// visits) instead of the complete turns containing each kid (the
    /// marginals, which also count choices left over once other variables
    /// were committed).
    pub first_step: bool,
    /// Completed-Q records (Gumbel MuZero's policy improvement): at the
    /// locked node the first step selects on edge statistics (the descents
    /// that started with each kid), and the prior target is
    /// softmax(log prior + sigma * q), q a kid's searched mean (its own
    /// intermediate value if unvisited), sigma = (cq_visit + max visits) *
    /// cq_scale. Value targets: visited kids' searched means.
    pub cq: bool,
    pub cq_visit: f64,
    pub cq_scale: f64,
}

impl Default for WalkParams {
    fn default() -> Self {
        WalkParams {
            n2: 20,
            c2: 1.5,
            temp: 1.0,
            dir_alpha: 0.3,
            dir_frac: 0.25,
            inflight: 1,
            sd_descents: 1,
            sd_tau: 0.02,
            fresh: true,
            first_step: true,
            cq: false,
            cq_visit: 50.0,
            cq_scale: 0.1,
        }
    }
}

/// A from-scratch tier-2 search rooted at one node, for its tier-2 record
/// (the generation's separate tier-2 phase, see [`Lattice::advance_rooted`]).
#[derive(Clone, Debug)]
pub struct RootedParams {
    /// Descents in all, `min_per_kid` of them from each live kid first.
    pub n: u32,
    pub min_per_kid: u32,
    pub c2: f32,
    pub inflight: usize,
    /// Prior targets: a per-cell softmax of the kids' best complete-turn
    /// values at this temperature; 0: the kids' shares of the complete
    /// turns found (visit counts).
    pub tau: f64,
    /// Expansions after which the descents follow witnesses (exact SAT
    /// fallback), and after twice which they are abandoned.
    pub budget: u32,
}

impl Default for RootedParams {
    fn default() -> Self {
        RootedParams {
            n: 128,
            min_per_kid: 1,
            c2: 1.5,
            inflight: 8,
            tau: 0.02,
            budget: 4000,
        }
    }
}

pub struct Rooted {
    pub at: u32,
    init: bool,
    /// Forced first kids still to start, PUCT descents still to start.
    todo: Vec<usize>,
    left: u32,
    curs: Vec<Cursor>,
    /// The complete turns reached and their values for the lattice's mover.
    sinks: Vec<(u32, f32)>,
    expansions: u32,
    pub abandoned: bool,
    pub record: Option<T2Record>,
}

impl Rooted {
    pub fn new(at: u32) -> Self {
        Rooted {
            at,
            init: false,
            todo: Vec::new(),
            left: 0,
            curs: Vec::new(),
            sinks: Vec::new(),
            expansions: 0,
            abandoned: false,
            record: None,
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
    started: u32,
    finished: u32,
    curs: Vec<Cursor>,
    pub records: Vec<T2Record>,
    pub done: bool,
    /// The complete turns (sink, value for the lattice's mover) reached by
    /// this ply's descents: the record's targets count these alone.
    ply_sinks: Vec<(u32, f32)>,
    /// Per finished descent of this ply: its first step's kid index at the
    /// locked node, and its complete turn's value.
    ply_first: Vec<(u16, f32)>,
    /// Self-distillation: the nodes still to record, the one being recorded
    /// (`NONE`: none), its kids' best complete-turn values so far, the
    /// descents still to start (kid index), and those in flight.
    pub sd: Vec<u32>,
    sd_at: u32,
    sd_best: Vec<f32>,
    sd_todo: Vec<usize>,
    sd_curs: Vec<(Cursor, usize)>,
}

impl Walk {
    pub fn new(_p: &WalkParams) -> Self {
        Walk {
            at: 0,
            lp: 0.0,
            started: 0,
            finished: 0,
            curs: Vec::new(),
            records: Vec::new(),
            done: false,
            ply_sinks: Vec::new(),
            ply_first: Vec::new(),
            sd: Vec::new(),
            sd_at: NONE,
            sd_best: Vec::new(),
            sd_todo: Vec::new(),
            sd_curs: Vec::new(),
        }
    }

    /// A self-distillation walk recording `nodes` (see
    /// [`Lattice::advance_sd`]).
    pub fn self_distill(p: &WalkParams, mut nodes: Vec<u32>) -> Self {
        nodes.reverse();
        let mut w = Walk::new(p);
        w.sd = nodes;
        w
    }
}

impl Lattice {
    /// Tier-2 records whose targets come from the tier-1 search alone:
    /// `turns` are the tier-1 children (sink, visits, value sum for this
    /// lattice's mover). Along a path to `target` (one undecided variable at
    /// a time, in random order), each node records, per kid, the visits of
    /// the turns that contain it and their mean value (-1: none), and the
    /// same for the node. No net calls: nodes are only prepared.
    pub fn tier1_records(
        &mut self,
        turns: &[(u32, f64, f64)],
        target: u32,
        rng: &mut impl rand::Rng,
    ) -> Result<Vec<T2Record>> {
        let keys: Vec<(Key, f64, f64)> = turns
            .iter()
            .map(|&(s, n, w)| (self.nodes[s as usize].key, n, w))
            .collect();
        let tk = self.nodes[target as usize].key;
        let sum = |k: &Key| {
            keys.iter()
                .filter(|t| t.0.includes(k))
                .fold((0.0, 0.0), |a, t| (a.0 + t.1, a.1 + t.2))
        };
        let mut out = Vec::new();
        let mut at = 0u32;
        while self.nodes[at as usize].kind == Kind::Inner {
            if !self.nodes[at as usize].prepared && !self.prepare(at)? {
                break;
            }
            let nd = &self.nodes[at as usize];
            let (n, w) = sum(&nd.key);
            if n <= 0.0 {
                break;
            }
            let kids: Vec<(u8, u8, f32, f32, f32)> = nd
                .kids
                .iter()
                .filter(|k| !k.dead)
                .map(|k| {
                    let (kn, kw) = sum(&nd.key.with(k.var as usize, k.choice));
                    (k.cell, k.choice, kn as f32, ret_mean(kn, kw), kn as f32)
                })
                .collect();
            out.push(T2Record {
                node: nd.state.clone(),
                kids,
                n: n as f32,
                q: ret_mean(n, w),
                ns: n as f32,
            });
            let mut vars: Vec<u8> = nd.kids.iter().filter(|k| !k.dead).map(|k| k.var).collect();
            vars.dedup();
            vars.retain(|&v| tk.mask & (1 << v) != 0);
            if vars.is_empty() {
                break;
            }
            let var = vars[rng.gen_range(0..vars.len())];
            let c = tk.get(var as usize);
            let Some(k) = nd
                .kids
                .iter()
                .position(|k| k.var == var && k.choice == c && !k.dead)
            else {
                break;
            };
            at = self.child(at, k)?;
        }
        Ok(out)
    }

    /// Inner node `at`'s tier-2 record from tier-1 turns `turns` (sink,
    /// visits, summed value for the turn's mover, as in
    /// [`Self::tier1_records`]): each live kid's visits and value summed over
    /// the turns including its commitment. None: `at` is a dead end.
    pub fn marginal_record(&mut self, at: u32, turns: &[(u32, f64, f64)]) -> Result<Option<T2Record>> {
        if self.nodes[at as usize].kind != Kind::Inner {
            return Ok(None);
        }
        if !self.nodes[at as usize].prepared && !self.prepare(at)? {
            self.kill(at)?;
            return Ok(None);
        }
        let keys: Vec<(Key, f64, f64)> = turns
            .iter()
            .map(|&(s, n, w)| (self.nodes[s as usize].key, n, w))
            .collect();
        let sum = |k: &Key| {
            keys.iter()
                .filter(|t| t.0.includes(k))
                .fold((0.0, 0.0), |a, t| (a.0 + t.1, a.1 + t.2))
        };
        let nd = &self.nodes[at as usize];
        if nd.kind != Kind::Inner || nd.kids.iter().all(|k| k.dead) {
            return Ok(None);
        }
        let (n, w) = sum(&nd.key);
        let kids: Vec<(u8, u8, f32, f32, f32)> = nd
            .kids
            .iter()
            .filter(|k| !k.dead)
            .map(|k| {
                let (kn, kw) = sum(&nd.key.with(k.var as usize, k.choice));
                (k.cell, k.choice, kn as f32, ret_mean(kn, kw), kn as f32)
            })
            .collect();
        Ok(Some(T2Record {
            node: nd.state.clone(),
            kids,
            n: n as f32,
            q: ret_mean(n, w),
            ns: n as f32,
        }))
    }

    /// Commit kid `k` of `at`: the child node, prepared. None: it is a dead
    /// end (then killed, and the kid with it).
    pub fn step_to(&mut self, at: u32, k: usize) -> Result<Option<u32>> {
        let c = self.child(at, k)?;
        let nd = &self.nodes[c as usize];
        let ok = match nd.kind {
            Kind::Dead => false,
            Kind::Sink => true,
            Kind::Inner => nd.prepared || self.prepare(c)?,
        };
        if !ok {
            if self.nodes[c as usize].kind != Kind::Dead {
                self.kill(c)?;
            }
            return Ok(None);
        }
        Ok(Some(c))
    }

    /// The nodes along a path from the root to `target` (each step commits
    /// one of the target's variables, picked uniformly at random), the root
    /// included and the target not. Nodes are prepared, not expanded.
    pub fn path_to(&mut self, target: u32, rng: &mut impl rand::Rng) -> Result<Vec<u32>> {
        let tk = self.nodes[target as usize].key;
        let mut out = Vec::new();
        let mut at = 0u32;
        while self.nodes[at as usize].kind == Kind::Inner {
            if !self.nodes[at as usize].prepared && !self.prepare(at)? {
                break;
            }
            out.push(at);
            let nd = &self.nodes[at as usize];
            let mut vars: Vec<u8> = nd.kids.iter().filter(|k| !k.dead).map(|k| k.var).collect();
            vars.dedup();
            vars.retain(|&v| tk.mask & (1 << v) != 0);
            if vars.is_empty() {
                break;
            }
            let var = vars[rng.gen_range(0..vars.len())];
            let c = tk.get(var as usize);
            let Some(k) = nd
                .kids
                .iter()
                .position(|k| k.var == var && k.choice == c && !k.dead)
            else {
                break;
            };
            at = self.child(at, k)?;
        }
        Ok(out)
    }

    /// Run descent `curs[i]` of a fresh per-ply search until it needs the
    /// net or reaches a complete turn. Selection is PUCT over this ply's own
    /// statistics only: a kid's visits are the ply's finished descents whose
    /// turn contains it (`done`: sink, value) plus the other descents in
    /// flight that passed through it (virtual losses, value 0); its value is
    /// the mean of those turns' tier-1 values, or its intermediate value
    /// while unvisited. The lattice's own statistics are left alone.
    pub fn advance_fresh(
        &mut self,
        curs: &mut [Cursor],
        i: usize,
        done: &[(u32, f32)],
        root: Option<(u32, &[(u16, f32)])>,
        c_puct: f32,
    ) -> Result<Need> {
        // Edge statistics at the root (completed-Q walks): finished first
        // steps, plus the other descents in flight as virtual losses.
        let mut edges: Vec<(u16, f32)> = Vec::new();
        if let Some((r, first)) = root {
            edges.extend_from_slice(first);
            for (j, c) in curs.iter().enumerate() {
                if j != i {
                    if let Some(&(n0, k0)) = c.path.first() {
                        if n0 == r {
                            edges.push((k0, 0.0));
                        }
                    }
                }
            }
        }
        let mut atoms: Vec<(Key, f32)> = done
            .iter()
            .map(|&(s, v)| (self.nodes[s as usize].key, v))
            .collect();
        for (j, c) in curs.iter().enumerate() {
            if j != i {
                atoms.push((self.nodes[c.at as usize].key, 0.0));
            }
        }
        let cur = &mut curs[i];
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
                        if node.busy {
                            return Ok(Need::Wait);
                        }
                        self.nodes[n as usize].busy = true;
                        return Ok(Need::Value(n));
                    };
                    // No direct visit: the lattice's statistics stay as the
                    // tier-1 search left them.
                    return Ok(Need::Done {
                        sink: n,
                        value: v,
                        fresh: false,
                    });
                }
                Kind::Inner if !node.expanded => {
                    if node.busy {
                        return Ok(Need::Wait);
                    }
                    if node.prepared || self.prepare(n)? {
                        self.nodes[n as usize].busy = true;
                        return Ok(Need::Expand(n));
                    }
                    self.kill(n)?;
                }
                Kind::Inner if root.is_some_and(|r| r.0 == n) && cur.path.is_empty() => {
                    let sq = (edges.len() as f64).max(1.0).sqrt();
                    let mut best = None;
                    let mut best_u = f64::NEG_INFINITY;
                    for (k, kid) in node.kids.iter().enumerate() {
                        if kid.dead {
                            continue;
                        }
                        let (kn, kw) = edges
                            .iter()
                            .filter(|e| e.0 as usize == k)
                            .fold((0.0f64, 0.0f64), |s, e| (s.0 + 1.0, s.1 + f64::from(e.1)));
                        let q = if kn > 0.0 { kw / kn } else { f64::from(kid.v) };
                        let u = q + f64::from(c_puct) * f64::from(kid.logp).exp() * sq / (1.0 + kn);
                        if u > best_u {
                            best_u = u;
                            best = Some(k);
                        }
                    }
                    match best {
                        None => self.kill(n)?,
                        Some(k) => {
                            let c = self.child(n, k)?;
                            cur.lp += f64::from(self.nodes[n as usize].kids[k].logp);
                            cur.path.push((n, k as u16));
                            cur.at = c;
                        }
                    }
                }
                Kind::Inner => {
                    let pk = node.key;
                    let tot = atoms.iter().filter(|a| a.0.includes(&pk)).count() as f64;
                    let sq = tot.max(1.0).sqrt();
                    let mut best = None;
                    let mut best_u = f64::NEG_INFINITY;
                    for (k, kid) in node.kids.iter().enumerate() {
                        if kid.dead {
                            continue;
                        }
                        let kk = pk.with(kid.var as usize, kid.choice);
                        let (kn, kw) = atoms
                            .iter()
                            .filter(|a| a.0.includes(&kk))
                            .fold((0.0f64, 0.0f64), |s, a| (s.0 + 1.0, s.1 + f64::from(a.1)));
                        let q = if kn > 0.0 { kw / kn } else { f64::from(kid.v) };
                        let u = q + f64::from(c_puct) * f64::from(kid.logp).exp() * sq / (1.0 + kn);
                        if u > best_u {
                            best_u = u;
                            best = Some(k);
                        }
                    }
                    match best {
                        None => self.kill(n)?,
                        Some(k) => {
                            let c = self.child(n, k)?;
                            cur.lp += f64::from(self.nodes[n as usize].kids[k].logp);
                            cur.path.push((n, k as u16));
                            cur.at = c;
                        }
                    }
                }
            }
        }
        bail!("fresh lattice descent made no progress in 1e6 steps")
    }

    /// Self-distillation: for each of `w.sd`'s nodes, `p.sd_descents`
    /// descents from each live kid, each ending at a complete turn valued by
    /// tier 1. A kid's targets are the best of those values (value) and a
    /// per-cell softmax of them at `p.sd_tau` (prior); the node's value is
    /// its best kid's. Intermediate values only steer the descents. Requests
    /// go to `out`; true once every node is recorded.
    pub fn advance_sd(
        &mut self,
        w: &mut Walk,
        p: &WalkParams,
        out: &mut Vec<Need>,
    ) -> Result<bool> {
        loop {
            if w.sd_at == NONE {
                let Some(n) = w.sd.pop() else {
                    w.done = true;
                    return Ok(true);
                };
                if self.nodes[n as usize].kind != Kind::Inner || !self.nodes[n as usize].prepared {
                    continue;
                }
                let nk = self.nodes[n as usize].kids.len();
                w.sd_at = n;
                w.sd_best = vec![f32::NEG_INFINITY; nk];
                w.sd_todo = (0..nk)
                    .filter(|&k| !self.nodes[n as usize].kids[k].dead)
                    .flat_map(|k| std::iter::repeat(k).take(p.sd_descents.max(1) as usize))
                    .collect();
            }
            let at = w.sd_at;
            while w.sd_curs.len() < p.inflight.max(1) {
                let Some(k) = w.sd_todo.pop() else { break };
                if self.nodes[at as usize].kids[k].dead {
                    continue;
                }
                let c = self.child(at, k)?;
                w.sd_curs.push((Cursor::at(c, 0.0), k));
            }
            let mut progressed = false;
            let mut i = 0;
            while i < w.sd_curs.len() {
                match self.advance(&mut w.sd_curs[i].0, p.c2)? {
                    Need::Done { value, .. } => {
                        let k = w.sd_curs[i].1;
                        w.sd_best[k] = w.sd_best[k].max(value);
                        w.sd_curs.swap_remove(i);
                        progressed = true;
                    }
                    Need::Dead => {
                        w.sd_curs.swap_remove(i);
                        progressed = true;
                    }
                    Need::Wait => i += 1,
                    need => {
                        out.push(need);
                        i += 1;
                    }
                }
            }
            if !w.sd_curs.is_empty() || !w.sd_todo.is_empty() {
                if !progressed {
                    return Ok(false);
                }
                continue;
            }
            // Every descent from this node is in: record it.
            let nd = &self.nodes[at as usize];
            let live: Vec<usize> = (0..nd.kids.len())
                .filter(|&k| !nd.kids[k].dead && w.sd_best[k].is_finite())
                .collect();
            if !live.is_empty() {
                let mut cell_best: HashMap<u8, f32> = HashMap::new();
                for &k in &live {
                    let b = cell_best
                        .entry(nd.kids[k].cell)
                        .or_insert(f32::NEG_INFINITY);
                    *b = b.max(w.sd_best[k]);
                }
                let kids = live
                    .iter()
                    .map(|&k| {
                        let kid = &nd.kids[k];
                        let v = w.sd_best[k];
                        let wt = (f64::from(v - cell_best[&kid.cell]) / p.sd_tau).exp() as f32;
                        (kid.cell, kid.choice, wt, v, 1.0)
                    })
                    .collect();
                let q = live
                    .iter()
                    .map(|&k| w.sd_best[k])
                    .fold(f32::NEG_INFINITY, f32::max);
                w.records.push(T2Record {
                    node: nd.state.clone(),
                    kids,
                    n: 1.0,
                    q,
                    ns: 1.0,
                });
            }
            w.sd_at = NONE;
        }
    }

    /// The node with commitments `key`, reached from the root one
    /// commitment at a time (the nodes on the way are prepared, not
    /// expanded). None: not reachable here (refuted on the way).
    pub fn reach(&mut self, key: Key) -> Result<Option<u32>> {
        let mut at = 0u32;
        loop {
            let nd = &self.nodes[at as usize];
            if nd.key == key {
                return Ok(Some(at));
            }
            if nd.kind != Kind::Inner || (!nd.prepared && !self.prepare(at)?) {
                return Ok(None);
            }
            let Some(k) = self.nodes[at as usize].kids.iter().position(|k| {
                !k.dead && key.mask & (1 << k.var) != 0 && key.get(k.var as usize) == k.choice
            }) else {
                return Ok(None);
            };
            at = self.child(at, k)?;
        }
    }

    /// A from-scratch search rooted at `r.at` (in a lattice of its own, so
    /// nothing but this search's descents is in its statistics):
    /// `p.min_per_kid` descents from each live kid, then PUCT descents from
    /// the node up to `p.n`, each ending at a complete turn valued by tier 1.
    /// The record: per kid, the best complete turn found containing it
    /// (value) and a per-cell softmax of those (prior; or its share of the
    /// turns found, `p.tau` 0); the node's value is the best turn found.
    /// Requests go to `out`; true once done (`r.record` is then set, unless
    /// the node is dead or nothing was found).
    pub fn advance_rooted(
        &mut self,
        r: &mut Rooted,
        p: &RootedParams,
        out: &mut Vec<Need>,
    ) -> Result<bool> {
        let at = r.at;
        if !r.init {
            r.init = true;
            let nd = &self.nodes[at as usize];
            if nd.kind != Kind::Inner || (!nd.prepared && !self.prepare(at)?) {
                return Ok(true);
            }
            let nd = &self.nodes[at as usize];
            r.todo = (0..nd.kids.len())
                .filter(|&k| !nd.kids[k].dead)
                .flat_map(|k| std::iter::repeat(k).take(p.min_per_kid as usize))
                .collect();
            r.todo.reverse();
            r.left = p.n.saturating_sub(r.todo.len() as u32);
        }
        loop {
            let over = r.expansions >= p.budget;
            if r.expansions >= 2 * p.budget && !r.abandoned {
                // Runaway descents (dead ends refuted one by one): keep what
                // was found.
                r.abandoned = true;
                for c in &mut r.curs {
                    self.abandon(c);
                }
                r.curs.clear();
                r.todo.clear();
                r.left = 0;
            }
            while r.curs.len() < p.inflight.max(1) {
                let mut c = if let Some(k) = r.todo.pop() {
                    if self.nodes[at as usize].kind != Kind::Inner
                        || self.nodes[at as usize].kids[k].dead
                    {
                        continue;
                    }
                    Cursor::at(self.child(at, k)?, 0.0)
                } else if r.left > 0 {
                    r.left -= 1;
                    Cursor::at(at, 0.0)
                } else {
                    break;
                };
                c.follow = over;
                r.curs.push(c);
            }
            let mut progressed = false;
            let mut i = 0;
            while i < r.curs.len() {
                if over {
                    r.curs[i].follow = true;
                }
                match self.advance(&mut r.curs[i], p.c2)? {
                    Need::Done { sink, value, .. } => {
                        r.sinks.push((sink, value));
                        r.curs.swap_remove(i);
                        progressed = true;
                    }
                    Need::Dead => {
                        r.curs.swap_remove(i);
                        progressed = true;
                    }
                    Need::Wait => i += 1,
                    need => {
                        if matches!(need, Need::Expand(_)) {
                            r.expansions += 1;
                        }
                        out.push(need);
                        i += 1;
                    }
                }
            }
            if !r.curs.is_empty() || !r.todo.is_empty() || r.left > 0 {
                if !progressed {
                    return Ok(false);
                }
                continue;
            }
            r.record = self.rooted_record(r, p);
            return Ok(true);
        }
    }

    fn rooted_record(&self, r: &Rooted, p: &RootedParams) -> Option<T2Record> {
        let nd = &self.nodes[r.at as usize];
        if nd.kind != Kind::Inner || r.sinks.is_empty() {
            return None;
        }
        let nk = nd.kids.len();
        let (mut best, mut cnt) = (vec![f32::NEG_INFINITY; nk], vec![0f32; nk]);
        for &(s, v) in &r.sinks {
            let key = self.nodes[s as usize].key;
            for (k, kid) in nd.kids.iter().enumerate() {
                if key.mask & (1 << kid.var) != 0 && key.get(kid.var as usize) == kid.choice {
                    best[k] = best[k].max(v);
                    cnt[k] += 1.0;
                }
            }
        }
        let live: Vec<usize> = (0..nk)
            .filter(|&k| !nd.kids[k].dead && cnt[k] > 0.0)
            .collect();
        if live.is_empty() {
            return None;
        }
        let mut cell_best: HashMap<u8, f32> = HashMap::new();
        for &k in &live {
            let b = cell_best
                .entry(nd.kids[k].cell)
                .or_insert(f32::NEG_INFINITY);
            *b = b.max(best[k]);
        }
        let kids = live
            .iter()
            .map(|&k| {
                let kid = &nd.kids[k];
                let wt = if p.tau > 0.0 {
                    (f64::from(best[k] - cell_best[&kid.cell]) / p.tau).exp() as f32
                } else {
                    cnt[k]
                };
                (kid.cell, kid.choice, wt, best[k], cnt[k])
            })
            .collect();
        let q = r
            .sinks
            .iter()
            .map(|x| x.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let n = r.sinks.len() as f32;
        Some(T2Record {
            node: nd.state.clone(),
            kids,
            n,
            q,
            ns: n,
        })
    }

    /// Run walk `w` until every descent in flight (up to `p.inflight` at a
    /// time) waits on the net; their requests go to `out`. True once the walk
    /// is over.
    pub fn advance_walk(
        &mut self,
        w: &mut Walk,
        p: &WalkParams,
        rng: &mut impl rand::Rng,
        out: &mut Vec<Need>,
    ) -> Result<bool> {
        loop {
            if w.done {
                return Ok(true);
            }
            if self.nodes[w.at as usize].kind != Kind::Inner {
                w.done = true;
                return Ok(true);
            }
            if w.finished < p.n2 {
                while w.started < p.n2 && w.curs.len() < p.inflight.max(1) {
                    w.curs.push(Cursor::at(w.at, w.lp));
                    w.started += 1;
                }
                let mut progressed = false;
                let mut i = 0;
                while i < w.curs.len() {
                    let need = if p.fresh {
                        let root = p.cq.then_some((w.at, &w.ply_first[..]));
                        self.advance_fresh(&mut w.curs, i, &w.ply_sinks, root, p.c2)?
                    } else {
                        self.advance(&mut w.curs[i], p.c2)?
                    };
                    match need {
                        Need::Done { sink, value, .. } => {
                            w.ply_sinks.push((sink, value));
                            if let Some(&(n0, k0)) = w.curs[i].path.first() {
                                if n0 == w.at {
                                    w.ply_first.push((k0, value));
                                }
                            }
                            w.curs.swap_remove(i);
                            w.finished += 1;
                            progressed = true;
                        }
                        Need::Dead => {
                            for mut c in std::mem::take(&mut w.curs) {
                                self.move_vl(&mut c, NONE);
                            }
                            w.done = true;
                            return Ok(true);
                        }
                        Need::Wait => i += 1,
                        need => {
                            out.push(need);
                            i += 1;
                        }
                    }
                }
                if !progressed {
                    return Ok(false);
                }
                continue;
            }
            // The ply's targets: this ply's own descents only (not the
            // lattice's accumulated statistics, which hold every earlier
            // search's descents and one atom per child of every expansion).
            // A descent ends at a complete turn, which takes exactly one
            // choice per variable, so each variable's kid visits sum to n2.
            let at = w.at;
            let sinks: Vec<(Key, f32)> = std::mem::take(&mut w.ply_sinks)
                .into_iter()
                .map(|(s, v)| (self.nodes[s as usize].key, v))
                .collect();
            let nd = &self.nodes[at as usize];
            let count = |k: &Key| {
                sinks
                    .iter()
                    .filter(|x| x.0.includes(k))
                    .fold((0.0f64, 0.0f64), |a, x| (a.0 + 1.0, a.1 + f64::from(x.1)))
            };
            let first = std::mem::take(&mut w.ply_first);
            let kid_n: Vec<(f64, f64)> = if p.first_step {
                let mut v = vec![(0.0f64, 0.0f64); nd.kids.len()];
                for &(k, val) in &first {
                    v[k as usize].0 += 1.0;
                    v[k as usize].1 += f64::from(val);
                }
                v
            } else {
                nd.kids
                    .iter()
                    .map(|k| count(&nd.key.with(k.var as usize, k.choice)))
                    .collect()
            };
            let (n, wsum) = (
                sinks.len() as f64,
                sinks.iter().map(|x| f64::from(x.1)).sum::<f64>(),
            );
            let kids: Vec<(u8, u8, f32, f32, f32)> = if p.cq {
                // Completed Q over the edge statistics.
                let mut e = vec![(0.0f64, 0.0f64); nd.kids.len()];
                for &(k, val) in &first {
                    e[k as usize].0 += 1.0;
                    e[k as usize].1 += f64::from(val);
                }
                let maxn = e.iter().map(|x| x.0).fold(0.0, f64::max);
                let sigma = (p.cq_visit + maxn) * p.cq_scale;
                // Per variable: softmax(logp + sigma * q), q the completed
                // values min-max normalized over the variable's live kids
                // (as Gumbel MuZero normalizes Q to [0, 1]; raw values here
                // differ by ~0.02, which moved the logits by ~0.1).
                let cq: Vec<f64> = nd
                    .kids
                    .iter()
                    .zip(&e)
                    .map(|(k, &(kn, kw))| if kn > 0.0 { kw / kn } else { f64::from(k.v) })
                    .collect();
                let mut range: HashMap<u8, (f64, f64)> = HashMap::new();
                for (k, &q) in nd.kids.iter().zip(&cq) {
                    if !k.dead {
                        let r = range
                            .entry(k.var)
                            .or_insert((f64::INFINITY, f64::NEG_INFINITY));
                        r.0 = r.0.min(q);
                        r.1 = r.1.max(q);
                    }
                }
                let score: Vec<f64> = nd
                    .kids
                    .iter()
                    .zip(&cq)
                    .map(|(k, &q)| {
                        let (lo, hi) = range.get(&k.var).copied().unwrap_or((0.0, 0.0));
                        let qn = if hi > lo { (q - lo) / (hi - lo) } else { 0.5 };
                        f64::from(k.logp) + sigma * qn
                    })
                    .collect();
                let mut top: HashMap<u8, f64> = HashMap::new();
                for (k, &sc) in nd.kids.iter().zip(&score) {
                    if !k.dead {
                        let t = top.entry(k.var).or_insert(f64::NEG_INFINITY);
                        *t = t.max(sc);
                    }
                }
                nd.kids
                    .iter()
                    .zip(e.iter().zip(&score))
                    .filter(|(k, _)| !k.dead)
                    .map(|(k, (&(kn, kw), &sc))| {
                        let wt = (sc - top[&k.var]).exp() as f32;
                        (k.cell, k.choice, wt, ret_mean(kn, kw), kn as f32)
                    })
                    .collect()
            } else {
                nd.kids
                    .iter()
                    .zip(&kid_n)
                    .filter(|(k, _)| !k.dead)
                    .map(|(k, &(kn, kw))| {
                        (k.cell, k.choice, kn as f32, ret_mean(kn, kw), kn as f32)
                    })
                    .collect()
            };
            w.records.push(T2Record {
                node: nd.state.clone(),
                kids,
                n: n as f32,
                q: ret_mean(n, wsum),
                ns: n as f32,
            });
            let mut vars: Vec<u8> = nd.kids.iter().filter(|k| !k.dead).map(|k| k.var).collect();
            vars.dedup();
            let var = vars[rng.gen_range(0..vars.len())];
            let idx: Vec<usize> = (0..nd.kids.len())
                .filter(|&i| nd.kids[i].var == var && !nd.kids[i].dead)
                .collect();
            // Lock by the complete turns found (the marginals): the walk
            // follows one of them down, whichever targets were recorded.
            let wts: Vec<f64> = idx
                .iter()
                .map(|&i| {
                    let k = &nd.kids[i];
                    count(&nd.key.with(k.var as usize, k.choice)).0
                })
                .collect();
            let k = idx[noisy_pick(&wts, p.temp, p.dir_alpha, p.dir_frac, rng)];
            w.lp += f64::from(nd.kids[k].logp);
            w.at = self.child(at, k)?;
            w.started = 0;
            w.finished = 0;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::game::MinionsState;
    use rand::{Rng, SeedableRng};

    /// Deterministic pseudo-random logits and values from the node's view.
    /// Follow-mode descents (SAT at every step) always reach a complete
    /// turn, and the turn realizes.
    #[test]
    fn follow_descents_reach_sinks() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(23);
        let mut sinks = 0;
        for g in 0..30u64 {
            let real = crate::selfplay::random_opening(22, 1.0884, &mut rng).0;
            let start = StepState::begin(&real, g);
            let mut l = Lattice::new(start).unwrap();
            if l.terminal.is_some() {
                continue;
            }
            l.follow_sat_only = true;
            for _ in 0..8 {
                let mut cur = Cursor::at(0, 0.0);
                cur.follow = true;
                let (sink, _, _) = l.descend(&HashLatticeEval, cur, 1.5).unwrap();
                let s = &l.nodes[sink as usize];
                assert_eq!(s.kind, Kind::Sink);
                let base = &l.start.base;
                let full = crate::steps::exact_realize(
                    base,
                    &s.state.done,
                    base.side_bb(0),
                    base.side_bb(1),
                    5_000_000,
                );
                assert!(
                    !matches!(full, crate::steps::Exact::Impossible),
                    "a followed turn does not realize"
                );
                sinks += 1;
            }
            assert!(l.stats.sat_calls > 0);
        }
        assert!(sinks > 100);
    }

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
        fn both(
            &self,
            reqs: &[ExpandReq],
            states: &[&StepState],
        ) -> Result<(Vec<ExpandOut>, Vec<ValueOut>)> {
            let e = reqs
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
                        energy: f32::NAN,
                        trunk: None,
                    }
                })
                .collect();
            let v = states
                .iter()
                .map(|s| ValueOut {
                    v: unit(h(&s.view(), 1)),
                    trunk: None,
                })
                .collect();
            Ok((e, v))
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
        let rets: Vec<(Key, f64, f64)> = l
            .nodes
            .iter()
            .filter(|x| x.kind != Kind::Dead && x.ret_n > 0)
            .map(|x| (x.key, f64::from(x.ret_n), x.ret_w))
            .collect();
        let ssum = |key: &Key| {
            rets.iter()
                .filter(|a| a.0.includes(key))
                .fold((0.0, 0.0), |s, a| (s.0 + a.1, s.1 + a.2))
        };
        let (sn, sw) = ssum(&pn.key);
        let skids = pn
            .kids
            .iter()
            .map(|k| {
                if k.dead {
                    (0.0, 0.0)
                } else {
                    ssum(&pn.key.with(k.var as usize, k.choice))
                }
            })
            .collect();
        Stats {
            n,
            w,
            sn,
            sw,
            skids,
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
            assert!(
                (a.sn - b.sn).abs() < 1e-6 && (a.sw - b.sw).abs() < 1e-6,
                "node {p} sinks: {a:?} vs {b:?}"
            );
            for (i, (x, y)) in a.skids.iter().zip(&b.skids).enumerate() {
                assert!(
                    (x.0 - y.0).abs() < 1e-6 && (x.1 - y.1).abs() < 1e-6,
                    "node {p} kid {i} sinks: {x:?} vs {y:?}"
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
    /// The pool and the nogoods change no answer: every offered kid's
    /// witness includes the kid's commitments and realizes, and every
    /// choice the plain checks certify with a witness is still offered.
    #[test]
    fn pool_and_nogoods_sound() {
        let (mut witnesses, mut certified) = (0, 0);
        for s in positions(8, 3).into_iter().chain(positions(8, 17)) {
            let mut l = Lattice::new(s).unwrap();
            if l.terminal.is_some() {
                continue;
            }
            for _ in 0..80 {
                l.descend(&HashLatticeEval, Cursor::at(0, 0.0), 1.0)
                    .unwrap();
            }
            for nd in &l.nodes {
                if !nd.prepared
                    || nd.kind != Kind::Inner
                    || !matches!(nd.state.stage, Stage::Fates | Stage::Moves)
                {
                    continue;
                }
                for (i, k) in nd.kids.iter().enumerate() {
                    let w = nd.kid_witness(i);
                    if crate::steps::witness_is_pending(&w) {
                        continue;
                    }
                    let kk = nd.key.with(k.var as usize, k.choice);
                    assert!(
                        l.witness_key(&w).includes(&kk),
                        "a witness misses its kid's commitments"
                    );
                    assert!(
                        crate::steps::attack_part(&nd.state.base, &w).is_some(),
                        "a witness does not realize"
                    );
                    witnesses += 1;
                }
                for &c in nd.state.undecided() {
                    let v = l.var_of[c as usize] as usize;
                    let (o, ws) = nd.state.with_next(c).lattice_options();
                    for (&x, w) in o.iter().zip(&ws) {
                        if crate::steps::witness_is_pending(w) {
                            continue;
                        }
                        assert!(
                            nd.kids.iter().any(|k| k.var as usize == v && k.choice == x),
                            "a certified choice (var {v}, choice {x}) is not offered"
                        );
                        certified += 1;
                    }
                }
            }
            assert!(l.pool.len() > 1, "the pool stayed empty");
        }
        assert!(
            witnesses > 100 && certified > 100,
            "{witnesses} witnesses, {certified} certified"
        );
    }

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
    /// Self-distillation records every path node with a live kid; each
    /// kid's value is a complete turn's (in [0, 1]), the node's is its best
    /// kid's, and the best kid of each cell has prior weight 1.
    #[test]
    fn self_distill_records() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        for (inflight, descents) in [(1, 1), (4, 2)] {
            let p = WalkParams {
                inflight,
                sd_descents: descents,
                ..WalkParams::default()
            };
            let mut total = 0;
            for s in positions(4, 31) {
                let mut l = Lattice::new(s).unwrap();
                if l.terminal.is_some() {
                    continue;
                }
                let (sink, _, _) = l
                    .descend(&HashLatticeEval, Cursor::at(0, 0.0), 1.5)
                    .unwrap();
                let path = l.path_to(sink, &mut rng).unwrap();
                assert!(!path.is_empty() && path[0] == 0);
                let mut w = Walk::self_distill(&p, path.clone());
                let mut needs = Vec::new();
                while !l.advance_sd(&mut w, &p, &mut needs).unwrap() {
                    assert!(!needs.is_empty(), "stalled without requests");
                    for need in needs.drain(..) {
                        match need {
                            Need::Expand(n) => {
                                let (slots, node) = l.request(n);
                                let req = ExpandReq {
                                    root: &l.start,
                                    node,
                                    slots: &slots,
                                    trunk: None,
                                };
                                let out = HashLatticeEval.expand(&[req]).unwrap().pop().unwrap();
                                l.feed_expand(n, &out);
                            }
                            Need::Value(n) => {
                                let o = HashLatticeEval
                                    .both(&[], &[&l.nodes[n as usize].state])
                                    .unwrap()
                                    .1
                                    .pop()
                                    .unwrap();
                                l.feed_value(n, &o);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                assert_eq!(w.records.len(), path.len());
                for r in &w.records {
                    let best = r.kids.iter().map(|k| k.3).fold(f32::NEG_INFINITY, f32::max);
                    assert_eq!(r.q, best);
                    for k in &r.kids {
                        assert!((0.0..=1.0).contains(&k.3) && k.2 > 0.0 && k.2 <= 1.0);
                    }
                    let mut cells: Vec<u8> = r.kids.iter().map(|k| k.0).collect();
                    cells.dedup();
                    for c in cells {
                        let m = r
                            .kids
                            .iter()
                            .filter(|k| k.0 == c)
                            .map(|k| k.2)
                            .fold(0.0, f32::max);
                        assert_eq!(m, 1.0);
                    }
                }
                total += w.records.len();
            }
            assert!(total > 10, "{total} records");
        }
    }

    /// Serve one search's requests with the test evaluator.
    fn serve(l: &mut Lattice, needs: &mut Vec<Need>) {
        for need in needs.drain(..) {
            match need {
                Need::Expand(n) => {
                    let (slots, node) = l.request(n);
                    let req = ExpandReq {
                        root: &l.start,
                        node,
                        slots: &slots,
                        trunk: None,
                    };
                    let out = HashLatticeEval.expand(&[req]).unwrap().pop().unwrap();
                    l.feed_expand(n, &out);
                }
                Need::Value(n) => {
                    let o = HashLatticeEval
                        .both(&[], &[&l.nodes[n as usize].state])
                        .unwrap()
                        .1
                        .pop()
                        .unwrap();
                    l.feed_value(n, &o);
                }
                _ => unreachable!(),
            }
        }
    }

    /// The tier-2 phase: nodes from one lattice are reached by key in a
    /// fresh one of the same turn, and a search rooted there records every
    /// kid it found a complete turn for, at the best such turn's value
    /// (checked against the sinks below the kid), with a per-cell softmax
    /// prior (or visit shares at tau 0).
    #[test]
    fn rooted_records() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        for (inflight, tau) in [(1, 0.02), (8, 0.02), (4, 0.0)] {
            let p = RootedParams {
                n: 40,
                inflight,
                tau,
                ..RootedParams::default()
            };
            let (mut total, mut reached) = (0, 0);
            for s in positions(4, 41) {
                let mut l = Lattice::new(s.clone()).unwrap();
                if l.terminal.is_some() {
                    continue;
                }
                let (sink, _, _) = l
                    .descend(&HashLatticeEval, Cursor::at(0, 0.0), 1.5)
                    .unwrap();
                for n in l.path_to(sink, &mut rng).unwrap() {
                    let key = l.nodes[n as usize].key;
                    let mut f = Lattice::new(s.clone()).unwrap();
                    let at = f.reach(key).unwrap().expect("reachable in a fresh lattice");
                    assert_eq!(f.nodes[at as usize].key, key);
                    assert_eq!(
                        f.nodes[at as usize].state.spec,
                        l.nodes[n as usize].state.spec
                    );
                    reached += 1;
                    let mut r = Rooted::new(at);
                    let mut needs = Vec::new();
                    while !f.advance_rooted(&mut r, &p, &mut needs).unwrap() {
                        assert!(!needs.is_empty(), "stalled without requests");
                        serve(&mut f, &mut needs);
                    }
                    let Some(rec) = r.record else { continue };
                    assert!(r.sinks.len() as u32 >= p.n.min(1));
                    let best = rec
                        .kids
                        .iter()
                        .map(|k| k.3)
                        .fold(f32::NEG_INFINITY, f32::max);
                    assert_eq!(rec.q, best);
                    let nd = &f.nodes[at as usize];
                    for k in &rec.kids {
                        assert!((0.0..=1.0).contains(&k.3) && k.2 > 0.0 && k.4 >= 1.0);
                        let kid = nd
                            .kids
                            .iter()
                            .find(|x| x.cell == k.0 && x.choice == k.1)
                            .unwrap();
                        // The best sink containing the kid.
                        let b = r
                            .sinks
                            .iter()
                            .filter(|&&(sn, _)| {
                                let sk = f.nodes[sn as usize].key;
                                sk.mask & (1 << kid.var) != 0
                                    && sk.get(kid.var as usize) == kid.choice
                            })
                            .map(|x| x.1)
                            .fold(f32::NEG_INFINITY, f32::max);
                        assert_eq!(k.3, b);
                        if tau > 0.0 {
                            assert!(k.2 <= 1.0);
                        } else {
                            assert_eq!(k.2, k.4);
                        }
                    }
                    // Every sink completes the node.
                    for &(sn, _) in &r.sinks {
                        assert!(f.nodes[sn as usize].key.includes(&key));
                    }
                    total += 1;
                }
            }
            assert!(
                total > 10 && reached >= total,
                "{total} records, {reached} reached"
            );
        }
    }

    #[test]
    fn walk_locks_down_to_a_turn() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        for (inflight, fresh, first_step, cq) in [
            (1, false, false, false),
            (4, false, false, false),
            (1, true, false, false),
            (4, true, false, false),
            (1, true, true, false),
            (4, true, true, false),
            (1, true, false, true),
            (4, true, false, true),
        ] {
            let p = WalkParams {
                inflight,
                fresh,
                first_step,
                cq,
                ..WalkParams::default()
            };
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
                let mut needs = Vec::new();
                while !l.advance_walk(&mut w, &p, &mut rng, &mut needs).unwrap() {
                    for need in needs.drain(..) {
                        match need {
                            Need::Expand(n) => {
                                let (slots, node) = l.request(n);
                                let req = ExpandReq {
                                    root: &l.start,
                                    node,
                                    slots: &slots,
                                    trunk: None,
                                };
                                let out = HashLatticeEval.expand(&[req]).unwrap().pop().unwrap();
                                l.feed_expand(n, &out);
                            }
                            Need::Value(n) => {
                                let o = HashLatticeEval
                                    .both(&[], &[&l.nodes[n as usize].state])
                                    .unwrap()
                                    .1
                                    .pop()
                                    .unwrap();
                                l.feed_value(n, &o);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                assert_eq!(l.nodes[w.at as usize].kind, Kind::Sink);
                assert!(!w.records.is_empty());
                // Each record counts its ply's n2 descents alone, and every
                // variable's kids split them (a complete turn takes one
                // choice per variable).
                for r in &w.records {
                    assert_eq!(r.n, p.n2 as f32);
                    if cq {
                        // Prior weights: each variable's best kid has 1.
                        let mut cells: Vec<u8> = r.kids.iter().map(|k| k.0).collect();
                        cells.dedup();
                        for c in cells {
                            let m = r
                                .kids
                                .iter()
                                .filter(|k| k.0 == c)
                                .map(|k| k.2)
                                .fold(0.0, f32::max);
                            assert!((m - 1.0).abs() < 1e-6, "cell {c}: top weight {m}");
                        }
                        // One first step per descent: the visit counts sum to n2.
                        let n: f32 = r.kids.iter().map(|k| k.4).sum();
                        assert_eq!(n, r.n);
                        continue;
                    }
                    if first_step {
                        // One first step per descent, over all the kids.
                        let n: f32 = r.kids.iter().map(|k| k.2).sum();
                        assert_eq!(n, r.n);
                        continue;
                    }
                    let mut cells: Vec<u8> = r.kids.iter().map(|k| k.0).collect();
                    cells.sort();
                    cells.dedup();
                    for c in cells {
                        let n: f32 = r.kids.iter().filter(|k| k.0 == c).map(|k| k.2).sum();
                        assert_eq!(n, r.n, "cell {c}");
                    }
                    assert!(r.q >= 0.0 && r.q <= 1.0);
                }
                // No virtual loss left behind.
                let direct: u32 = l.nodes.iter().map(|n| n.n).sum();
                let real = l.stats.expansions as u32 + l.stats.descents as u32;
                assert_eq!(direct, real, "virtual visits left over");
            }
        }
    }

    /// Under lazy witnesses some offered kids are refuted at their own
    /// expansion: descents still end on full turns and the statistics stay
    /// exact with dead nodes around.
    /// Ignored: `LAZY_WITNESS` is process-wide, and the steps tests running
    /// in parallel assume it off. Run alone: `cargo test lattice_lazy -- --ignored`.
    #[test]
    #[ignore]
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
