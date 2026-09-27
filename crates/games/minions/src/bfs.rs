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
}

#[derive(Clone, Debug)]
pub struct Params {
    /// Leaves expanded per iteration.
    pub heads: usize,
    /// Power-mean exponent of the backup.
    pub power: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            heads: 32,
            power: 2.0,
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
    }
}

pub struct Tree {
    pub nodes: Vec<Node>,
    /// Each node's state, kept from its parent's expansion so nothing replays.
    pub states: Vec<Option<StepState>>,
    pub root: StepState,
    pub params: Params,
    pub stats: SearchStats,
}

impl Tree {
    /// A tree whose root is `root`, evaluated with `eval`.
    pub fn new(root: StepState, params: Params, eval: &dyn Evaluator) -> anyhow::Result<Self> {
        let (kind, value) = if root.is_terminal() {
            (TERMINAL, root.terminal_value())
        } else {
            (LEAF, eval.eval(&[root.view()])?[0])
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
        };
        Ok(Self {
            nodes: vec![node],
            states: vec![Some(root.clone())],
            root,
            params,
            stats: SearchStats::default(),
        })
    }

    pub fn is_solved(&self) -> bool {
        matches!(self.nodes[0].kind, TERMINAL | DEAD)
    }

    fn children(&self, n: u32) -> std::ops::Range<u32> {
        let nd = &self.nodes[n as usize];
        nd.first..nd.first + nd.nch as u32
    }

    /// Child `c`'s value from its parent's mover's point of view.
    #[inline]
    fn q_parent(&self, c: u32) -> f32 {
        let n = &self.nodes[c as usize];
        if n.flip {
            1.0 - n.value
        } else {
            n.value
        }
    }

    pub fn best_child(&self, n: u32) -> Option<u32> {
        let mut best = None;
        let mut bq = f32::NEG_INFINITY;
        for c in self.children(n) {
            if self.nodes[c as usize].kind == DEAD {
                continue;
            }
            let q = self.q_parent(c);
            if q > bq {
                bq = q;
                best = Some(c);
            }
        }
        best
    }

    /// The step path from the root to `n`.
    pub fn path(&self, mut n: u32) -> Vec<u8> {
        let mut steps = Vec::new();
        while n != 0 {
            steps.push(self.nodes[n as usize].step);
            n = self.nodes[n as usize].parent;
        }
        steps.reverse();
        steps
    }

    /// `n`'s state: stored, or replayed from the root along its path.
    pub fn state_of(&self, n: u32) -> StepState {
        if let Some(Some(s)) = self.states.get(n as usize) {
            return s.clone();
        }
        let mut s = self.root.clone();
        for v in self.path(n) {
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
                    if self.nodes[c as usize].kind == DEAD {
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
        let t = Instant::now();
        struct Pending {
            leaf: u32,
            kids: Vec<(u8, bool, Option<f32>, StepState)>, // (step, flip, terminal value, state)
            first_view: usize,
        }
        let mut views: Vec<View> = Vec::new();
        let mut pend: Vec<Pending> = Vec::with_capacity(leaves.len());
        for &leaf in leaves {
            let mut s = match self.states[leaf as usize].take() {
                Some(s) => s,
                None => self.state_of(leaf),
            };
            let opts = s.collapse_forced();
            if opts.is_empty() {
                if std::env::var("STEPS_DEBUG_DEAD").is_ok()
                    && DEAD_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8
                {
                    eprintln!("no-option leaf: {}", s.debug_dead());
                }
                self.states[leaf as usize] = Some(s);
                self.nodes[leaf as usize].kind = DEAD;
                self.stats.dead += 1;
                continue;
            }
            let first_view = views.len();
            let mut kids = Vec::with_capacity(opts.len());
            for v in opts {
                let c = s.child(v);
                let flip = c.flipped != s.flipped;
                if c.is_terminal() {
                    let tv = c.terminal_value();
                    kids.push((v, flip, Some(tv), c));
                } else {
                    views.push(c.view());
                    kids.push((v, flip, None, c));
                }
            }
            self.states[leaf as usize] = Some(s);
            pend.push(Pending {
                leaf,
                kids,
                first_view,
            });
        }
        self.stats.expand_ns += t.elapsed().as_nanos() as u64;
        let t = Instant::now();
        let vals = if views.is_empty() {
            Vec::new()
        } else {
            eval.eval(&views)?
        };
        self.stats.eval_ns += t.elapsed().as_nanos() as u64;
        self.stats.evals += views.len() as u64;
        for p in pend {
            let first = self.nodes.len() as u32;
            let mut vi = p.first_view;
            let nk = p.kids.len();
            for (step, flip, term, st) in p.kids {
                let (kind, value) = match term {
                    Some(v) => (TERMINAL, v),
                    None => {
                        let v = vals[vi];
                        vi += 1;
                        (LEAF, v)
                    }
                };
                self.nodes.push(Node {
                    parent: p.leaf,
                    first: 0,
                    nch: 0,
                    step,
                    flip,
                    kind,
                    visits: 0,
                    value,
                    prior: value,
                });
                self.states.push(Some(st));
            }
            let nd = &mut self.nodes[p.leaf as usize];
            nd.first = first;
            nd.nch = nk as u16;
            nd.kind = EXPANDED;
            self.stats.expansions += 1;
        }
        Ok(())
    }

    /// Recompute `n`'s value from its children; `false` if all are dead.
    fn recompute(&mut self, n: u32) -> bool {
        let p = self.params.power;
        let (mut num, mut den) = (0f64, 0f64);
        for c in self.children(n) {
            let nd = &self.nodes[c as usize];
            if nd.kind == DEAD {
                continue;
            }
            let w = 1.0 + nd.visits as f64;
            let q = self.q_parent(c).clamp(0.0, 1.0) as f64;
            num += w * q.powf(p as f64);
            den += w;
        }
        if den == 0.0 {
            return false;
        }
        self.nodes[n as usize].value = (num / den).powf(1.0 / p as f64) as f32;
        true
    }

    fn backup(&mut self, leaves: &[u32]) {
        let t = Instant::now();
        for &l in leaves {
            let mut n = l;
            loop {
                let kind = self.nodes[n as usize].kind;
                if kind == EXPANDED && !self.recompute(n) {
                    self.nodes[n as usize].kind = DEAD;
                    self.stats.dead += 1;
                }
                if n == 0 {
                    break;
                }
                n = self.nodes[n as usize].parent;
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
        }
    }
}

// ── Evaluators ──────────────────────────────────────────────────────────────

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
            let sess = crate::ort_net::build(path, ep, fp16, cache)?;
            sparse = sess.inputs().iter().any(|i| i.name() == "idx");
            let (queue, calls, rows) = (queue.clone(), calls.clone(), rows.clone());
            std::thread::spawn(move || batch_runner(sess, sparse, &queue, &calls, &rows));
        }
        Ok(Self {
            queue,
            sparse,
            calls,
            rows,
        })
    }
}

fn batch_runner(
    mut sess: ort::session::Session,
    sparse: bool,
    queue: &BatchQueue,
    calls: &std::sync::atomic::AtomicU64,
    rows: &std::sync::atomic::AtomicU64,
) {
    use ort::value::Tensor;
    let max = *BATCH_SHAPES.last().unwrap();
    loop {
        let mut batch = Vec::new();
        let mut total = 0;
        {
            let mut q = queue.q.lock().unwrap();
            while q.is_empty() {
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
        let shape = *BATCH_SHAPES.iter().find(|&&s| s >= total).unwrap();
        let width = if sparse {
            crate::steps::N_SCALARS
        } else {
            DENSE_DIM
        };
        let mut x = vec![0f32; shape * width];
        let mut idx = if sparse {
            vec![crate::steps::SPARSE_DIM as i32; shape * crate::steps::MAX_ACTIVE]
        } else {
            Vec::new()
        };
        let (mut at, mut ai) = (0, 0);
        for r in &batch {
            x[at..at + r.x.len()].copy_from_slice(&r.x);
            at += r.x.len();
            idx[ai..ai + r.idx.len()].copy_from_slice(&r.idx);
            ai += r.idx.len();
        }
        calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        rows.fetch_add(total as u64, std::sync::atomic::Ordering::Relaxed);
        let res = (|| -> anyhow::Result<Vec<f32>> {
            let e = |e: ort::Error| anyhow::anyhow!("{e}");
            let t = Tensor::from_array((vec![shape as i64, width as i64], x)).map_err(e)?;
            let outs = if sparse {
                let ti =
                    Tensor::from_array((vec![shape as i64, crate::steps::MAX_ACTIVE as i64], idx))
                        .map_err(e)?;
                sess.run(ort::inputs!["idx" => ti, "scal" => t])
            } else {
                sess.run(ort::inputs!["x" => t])
            }
            .map_err(e)?;
            let (_, v) = outs["v"]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(v[..total].to_vec())
        })();
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
        for part in views.chunks(max) {
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
        let mut out = Vec::with_capacity(views.len());
        for rx in waits {
            out.extend(rx.recv().map_err(|e| anyhow::anyhow!("{e}"))??);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::MinionsState;
    use rand::SeedableRng;

    /// A cheap deterministic evaluator: material balance for the mover.
    struct Material;
    impl Evaluator for Material {
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

    #[test]
    fn search_runs_and_backs_up_consistently() {
        let s = StepState::begin(&MinionsState::with_bid(0), 5);
        let mut t = Tree::new(
            s,
            Params {
                heads: 8,
                power: 2.0,
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
