//! Turn-lattice generation: games whose every turn is a tier-1 search, plus
//! a tier-2 walk per turn for the tier-2 net's data. See
//! `docs/minions-lattice.md` ("Generation").
//!
//! Each worker thread runs many games at once as resumable tasks: a game's
//! opening, its tier-1 searches, and the tier-2 walk of each turn already
//! searched (which runs alongside the game's next tier-1 search). A round
//! advances every task to its next net request, evaluates all of them in one
//! batched call per kind, and feeds the results back; several workers keep
//! the batcher's GPU runners busy.
//!
//! Output under `out`:
//! - `t1/chunk_W_NNNNN.bin`: steps-format value records
//!   ([`crate::stepsp::encode_record`]) of each searched turn start, target =
//!   the tier-1 root value after the search, prior = the root's own value,
//!   visits = the root's visits.
//! - `t2/chunk_W_NNNNN.bin` + `.kids`: tier-2 records, header
//!   `b"MLT2", version, count, T2_REC_BYTES`, then per record:
//!
//! | off | type | field |
//! |---|---|---|
//! | 0 | u8 | root active count |
//! | 1 | u8 | node active count |
//! | 2 | u8 | node stage (0 fates, 1 moves, 2 spawns) |
//! | 4 | u16 | kids |
//! | 6 | u16 | turn |
//! | 8 | f32 | node visits (indirect included) |
//! | 12 | f32 | node mean value |
//! | 16 | u16 × 96 | root (turn start) active indices |
//! | 208 | f32 × 16 | root scalars |
//! | 272 | u16 × 96 | node active indices (`next` plane left out) |
//! | 464 | f32 × 16 | node scalars |
//!
//!   The `.kids` file (header `b"MLTK", version, count, 12`) holds each
//!   record's kids in order: `u8 cell, u8 slot, u16 pad, f32 visits, f32
//!   mean value`.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use rand::{Rng, SeedableRng};

use crate::lattice::{
    noisy_pick, ExpandReq, Key, Kind, Lattice, LatticeEval, Need, Rooted, RootedParams, T2Record,
    Walk, WalkParams, NONE,
};
use crate::steps::{Stage, StepState, View, MAX_ACTIVE, N_SCALARS};
use crate::tier1::{Need1, T1Params, Tier1, Visit};

pub const T2_VERSION: u32 = 1;
pub const T2_REC_BYTES: usize = 528;
pub const T2_KID_BYTES: usize = 12;

#[derive(Clone, Debug)]
pub struct GenConfig {
    pub out: PathBuf,
    pub workers: usize,
    /// Games per worker.
    pub games: usize,
    /// Tier-1 visits per turn.
    pub n1: u32,
    /// Tier-1 visits in flight per search.
    pub inflight1: usize,
    pub t1: T1Params,
    pub walk: WalkParams,
    /// The next turn: tempered root visits plus Dirichlet noise.
    pub temp1: f64,
    pub dir1_alpha: f64,
    pub dir1_frac: f64,
    /// Games start `MAX_TURNS - G` turns in, G geometric on 1, 2, ... with
    /// this mean (the opening is played ply by ply from the net's values).
    pub from_end_mean: f64,
    /// If positive, G's median instead (overrides the mean): turn
    /// `MAX_TURNS - median` is the median start.
    pub from_end_median: f64,
    /// If positive, overrides both: the start turn itself is geometric on
    /// 0, 1, 2, ... (below `MAX_TURNS`) with this mean.
    pub start_turn_mean: f64,
    pub secs: f64,
    /// Stop after this many tier-2 records (0 = no limit).
    pub target_t2: u64,
    pub seed: u64,
    pub chunk_records: usize,
    /// Also record every non-root tier-1 node with at least this many
    /// visits as a tier-1 value target (0 = the root only).
    pub t1_min_visits: u32,
    /// Also record each tier-2 record's node, and each of its kids, with at
    /// least this many sink returns below it as tier-1 value targets on
    /// their (partial turn) views, at the returns' mean: the targets a
    /// bridge evaluator needs (0 = none).
    pub view_min_visits: u32,
    /// Tier-2 records from the tier-1 search (see
    /// [`Lattice::tier1_records`]) instead of the walk.
    pub t2_from_t1: bool,
    /// Distillation (if positive): the tier-2 records are the expansions
    /// themselves, a `distill_frac` sample of them, with the evaluator's
    /// child and node values as value targets and a softmax of the child
    /// values within each cell at this temperature as prior targets. Run
    /// with a steps value net (the bridge) to distill it into tier 2; no
    /// other tier-2 records are written.
    pub distill_tau: f64,
    pub distill_frac: f64,
    /// Self-distillation (if positive): tier-2 records from
    /// [`Lattice::advance_sd`] at each node on the path to the chosen turn
    /// with this probability, instead of any other tier-2 records.
    pub sd_frac: f64,
    /// Records in the lattice view (`crate::lview`: original position plus
    /// decisions, per-kind output slots), and energy records (`t1e/`: each
    /// turn start, then its tier-1 children's complete declarations with
    /// their visits).
    pub lview: bool,
    /// Interior tier-1 turns with at least this many visits also write
    /// energy groups (0: the root's only).
    pub energy_min_visits: u32,
    /// Tier-2 records also carry the old (steps) view of root and node in a
    /// `.old` side file: targets from an old-view teacher net.
    pub teacher_view: bool,
    /// Groups of games per worker taking turns, so that the worker advances
    /// one group while another's net call is in flight (1 = no overlap).
    pub pipeline: usize,
    /// Openings played up front, in a phase of their own, before the search
    /// workers start: all at once across `open_workers` threads, so their
    /// net calls batch (an opening ply is one row; played inline, an opening
    /// held a game slot for ~20 sequential calls a turn). Searches start
    /// from this pool and play their openings inline only once it is empty.
    pub openings: usize,
    pub open_workers: usize,
    /// A saved opening pool: searches start from it (oldest first) and the
    /// rest is written back at the end, so openings are played in large,
    /// infrequent batches (and may be a few generations stale). With a pool,
    /// `openings` is ignored.
    pub pool: Option<PathBuf>,
    /// Refill mode: play openings until the pool holds this many, save it,
    /// and stop (no searches).
    pub refill_to: usize,
    /// The separate tier-2 phase (if positive): self-play writes no tier-2
    /// records; each searched turn hands the nodes on a path to its chosen
    /// turn, each with this probability, to a second phase run after the
    /// self-play one (`secs`), which records each node from a search of its
    /// own rooted there ([`Lattice::advance_rooted`]) for up to
    /// `rooted_secs` (nodes in random order: the ones left are dropped).
    pub rooted_frac: f64,
    pub rooted: RootedParams,
    pub rooted_secs: f64,
    /// Searches per worker at a time in that phase.
    pub rooted_tasks: usize,
}

/// The continuation probability q of a geometric on 0..n (P(k) ∝ q^k)
/// whose mean is `mean`, by bisection.
fn truncated_geometric_q(mean: f64, n: u32) -> f64 {
    let m = |q: f64| {
        let (mut num, mut den, mut w) = (0.0, 0.0, 1.0);
        for k in 0..n {
            num += k as f64 * w;
            den += w;
            w *= q;
        }
        num / den
    };
    let (mut lo, mut hi) = (0.0, 1.0 - 1e-9);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if m(mid) < mean {
            lo = mid
        } else {
            hi = mid
        }
    }
    0.5 * (lo + hi)
}

const POOL_MAGIC: &[u8; 4] = b"MOPN";

/// The pool file: magic, record size, then each opening's real position
/// (`Game::encode_state`), oldest first. Read back with fresh salts.
fn load_pool(path: &Path) -> Result<Vec<StepState>> {
    use playground_core::Game;
    let b = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    anyhow::ensure!(
        b.len() >= 8 && &b[..4] == POOL_MAGIC,
        "{}: not an opening pool",
        path.display()
    );
    let rec = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    if b.len() == 8 {
        return Ok(Vec::new());
    }
    anyhow::ensure!(
        rec > 0 && (b.len() - 8) % rec == 0,
        "{}: truncated",
        path.display()
    );
    let salt = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let mut out = Vec::with_capacity((b.len() - 8) / rec);
    for (i, r) in b[8..].chunks(rec).enumerate() {
        let real = crate::game::Minions::decode_state(r)?;
        out.push(StepState::begin(
            &real,
            salt ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        ));
    }
    Ok(out)
}

/// Write `pool` (oldest first), atomically.
fn save_pool(path: &Path, pool: &[StepState]) -> Result<()> {
    use playground_core::Game;
    let recs: Vec<Vec<u8>> = pool
        .iter()
        .map(|s| crate::game::Minions::encode_state(&s.real_base()))
        .collect();
    let mut b = POOL_MAGIC.to_vec();
    b.extend_from_slice(&(recs.first().map_or(0, |r| r.len()) as u32).to_le_bytes());
    for r in &recs {
        b.extend_from_slice(r);
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &b)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Default)]
pub struct GenCounters {
    pub games: AtomicU64,
    pub turns: AtomicU64,
    pub t1_records: AtomicU64,
    pub t2_records: AtomicU64,
    pub energy_groups: AtomicU64,
    pub expansions: AtomicU64,
    pub kids: AtomicU64,
    pub values: AtomicU64,
    pub rounds: AtomicU64,
    pub eval_ns: AtomicU64,
    pub cpu_ns: AtomicU64,
    pub opening_turns: AtomicU64,
    /// Games started from the opening pool (the rest played theirs inline).
    pub pooled: AtomicU64,
    /// Tier-1 records on partial-turn views (`view_min_visits`).
    pub view_records: AtomicU64,
    /// The tier-2 phase: nodes queued, searched, not reachable in a fresh
    /// lattice, and searches abandoned over budget.
    pub rooted_queued: AtomicU64,
    pub rooted_done: AtomicU64,
    pub rooted_unreached: AtomicU64,
    pub rooted_abandoned: AtomicU64,
}

enum Task {
    Opening {
        lat: Lattice,
        at: u32,
        left: u32,
    },
    /// `started` of `n1` visits begun; `vis` those in flight.
    Search {
        t1: Tier1,
        started: u32,
        vis: Vec<Visit>,
    },
    Walk {
        lat: Lattice,
        walk: Walk,
    },
    /// The tier-2 phase: a search rooted at one node, in a fresh lattice.
    Rooted {
        lat: Lattice,
        r: Rooted,
    },
}

#[derive(Clone, Copy)]
enum Req {
    /// Expand node `n` of tier-1 node `t1`'s lattice (`NONE`: the task's own).
    Expand { t1: u32, n: u32 },
    /// A tier-1 search's sink value (`n`) or node value (`n` = `NONE`).
    T1Value { t1: u32, n: u32 },
    /// A sink of the task's own lattice.
    LatValue { n: u32 },
}

enum Next {
    /// Requests to evaluate this round (possibly none: everything in flight
    /// waits on requests already queued).
    Wait(Vec<Req>),
    /// The task is finished; run these instead.
    Replace(Vec<Task>),
}

struct ChunkWriter {
    dir: PathBuf,
    tag: usize,
    seq: usize,
    magic: &'static [u8; 4],
    version: u32,
    rec_bytes: usize,
    buf: Vec<u8>,
    n: usize,
    side: Option<(&'static [u8; 4], Vec<u8>, usize)>,
    /// A second side file (`.old`), one record per main record.
    side2: Option<(&'static [u8; 4], Vec<u8>, usize)>,
    per_chunk: usize,
}

impl ChunkWriter {
    fn new(
        dir: PathBuf,
        tag: usize,
        magic: &'static [u8; 4],
        version: u32,
        rec_bytes: usize,
        per_chunk: usize,
    ) -> Self {
        ChunkWriter {
            dir,
            tag,
            seq: 0,
            magic,
            version,
            rec_bytes,
            buf: Vec::new(),
            n: 0,
            side: None,
            side2: None,
            per_chunk,
        }
    }

    fn added(&mut self) -> Result<()> {
        self.n += 1;
        if self.n >= self.per_chunk {
            self.flush()?;
        }
        Ok(())
    }

    fn write(
        path: &Path,
        magic: &[u8; 4],
        version: u32,
        count: usize,
        rec: usize,
        body: &[u8],
    ) -> Result<()> {
        let tmp = path.with_extension("tmp");
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(magic)?;
        f.write_all(&version.to_le_bytes())?;
        f.write_all(&(count as u32).to_le_bytes())?;
        f.write_all(&(rec as u32).to_le_bytes())?;
        f.write_all(body)?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.n == 0 {
            return Ok(());
        }
        let stem = self.dir.join(format!("chunk_{}_{:05}", self.tag, self.seq));
        // The side file first: a reader keys on the main file.
        if let Some((magic, body, rec)) = self.side.as_mut() {
            Self::write(
                &stem.with_extension("kids"),
                magic,
                self.version,
                body.len() / *rec,
                *rec,
                body,
            )?;
            body.clear();
        }
        if let Some((magic, body, rec)) = self.side2.as_mut() {
            Self::write(
                &stem.with_extension("old"),
                magic,
                self.version,
                body.len() / *rec,
                *rec,
                body,
            )?;
            body.clear();
        }
        Self::write(
            &stem.with_extension("bin"),
            self.magic,
            self.version,
            self.n,
            self.rec_bytes,
            &self.buf,
        )?;
        self.seq += 1;
        self.buf.clear();
        self.n = 0;
        Ok(())
    }
}

fn push_view(v: &View, out: &mut Vec<u8>) -> u8 {
    let mut act = Vec::with_capacity(MAX_ACTIVE);
    let mut sc = [0f32; N_SCALARS];
    v.features(&mut act, &mut sc);
    for i in 0..MAX_ACTIVE {
        out.extend_from_slice(&act.get(i).copied().unwrap_or(0).to_le_bytes());
    }
    for x in sc {
        out.extend_from_slice(&x.to_le_bytes());
    }
    act.len().min(MAX_ACTIVE) as u8
}

/// A turn record (the steps record layout) with lattice-view features.
fn encode_turn_lv(s: &StepState, target: f32, visits: u32, out: &mut Vec<u8>) {
    let (mut act, mut sc) = (Vec::with_capacity(MAX_ACTIVE), [0f32; N_SCALARS]);
    crate::lview::node_features(s, &mut act, &mut sc);
    let start = out.len();
    out.push(act.len().min(MAX_ACTIVE) as u8);
    out.push(stage_code(s.stage));
    out.extend_from_slice(&(visits.min(u16::MAX as u32) as u16).to_le_bytes());
    out.extend_from_slice(&target.to_le_bytes());
    out.extend_from_slice(&0f32.to_le_bytes());
    out.extend_from_slice(&s.base.turn.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    crate::lview::push_features(&act, &sc, out);
    debug_assert_eq!(out.len() - start, crate::stepsp::REC_BYTES);
}

/// An energy record: kind (0 = turn start, 1 = one of its tier-1 children,
/// as its complete declaration), active count, 2 pad bytes, visits, then
/// (version 2) its searched value and static value for the turn start's
/// mover (a turn start: its own, NaN) and its tier-1 prior (a turn start:
/// 1), all f32, then the features.
pub const E_REC_BYTES: usize = 20 + MAX_ACTIVE * 2 + N_SCALARS * 4;
pub const E_VERSION: u32 = 2;

/// One energy record's statistics: visits, searched value, static value,
/// tier-1 prior.
type EStats = [f32; 4];

/// Append one energy record's bytes (the caller counts the records).
fn encode_energy(kind: u8, st: EStats, act: &[u16], sc: &[f32; N_SCALARS], buf: &mut Vec<u8>) {
    let start = buf.len();
    buf.extend_from_slice(&[kind, 0, 0, 0]);
    for x in st {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    let n = crate::lview::push_features(act, sc, buf);
    buf[start + 1] = n;
    debug_assert_eq!(buf.len() - start, E_REC_BYTES);
}

fn stage_code(s: Stage) -> u8 {
    match s {
        Stage::Fates => 0,
        Stage::Moves => 1,
        Stage::Spawns => 2,
        Stage::Terminal => 3,
    }
}

/// The tier-2 net's view of a lattice node: the steps view without the
/// `next` plane (the lattice has no variable order).
pub fn t2_view(s: &StepState) -> View {
    let mut v = s.view();
    v.next = 0;
    v
}

/// Header byte 3 of a tier-2 record: its values average sink returns only
/// (-1 = no target). Older records (0) averaged every visit.
const T2_SINK_TARGETS: u8 = 1;

/// A `.old` side record (`--teacher-view`): the root and node in the old
/// (steps) view, for a teacher net reading it: active counts, 2 pad bytes,
/// then root and node features.
pub const OLD_REC_BYTES: usize = 4 + 2 * (MAX_ACTIVE * 2 + N_SCALARS * 4);

fn encode_t2(root: &StepState, r: &T2Record, w: &mut ChunkWriter, lview: bool) -> Result<()> {
    if let Some((_, body, _)) = w.side2.as_mut() {
        let at = body.len();
        body.extend_from_slice(&[0, 0, 0, 0]);
        let mut tmp = Vec::with_capacity(512);
        let ra = push_view(&root.view(), &mut tmp);
        let na = push_view(&t2_view(&r.node), &mut tmp);
        body.extend_from_slice(&tmp);
        body[at] = ra;
        body[at + 1] = na;
        debug_assert_eq!(body.len() - at, OLD_REC_BYTES);
    }
    let start = w.buf.len();
    w.buf
        .extend_from_slice(&[0, 0, stage_code(r.node.stage), T2_SINK_TARGETS]);
    w.buf
        .extend_from_slice(&(r.kids.len() as u16).to_le_bytes());
    w.buf.extend_from_slice(&r.node.base.turn.to_le_bytes());
    w.buf.extend_from_slice(&r.n.to_le_bytes());
    w.buf.extend_from_slice(&r.q.to_le_bytes());
    let mut tmp = Vec::with_capacity(512);
    let (ra, na) = if lview {
        let (mut act, mut sc) = (Vec::with_capacity(MAX_ACTIVE), [0f32; N_SCALARS]);
        crate::lview::node_features(root, &mut act, &mut sc);
        let ra = crate::lview::push_features(&act, &sc, &mut tmp);
        crate::lview::node_features(&r.node, &mut act, &mut sc);
        (ra, crate::lview::push_features(&act, &sc, &mut tmp))
    } else {
        (
            push_view(&root.view(), &mut tmp),
            push_view(&t2_view(&r.node), &mut tmp),
        )
    };
    w.buf.extend_from_slice(&tmp);
    w.buf[start] = ra;
    w.buf[start + 1] = na;
    debug_assert_eq!(w.buf.len() - start, T2_REC_BYTES);
    let side = &mut w.side.as_mut().expect("kids file").1;
    for &(cell, slot, n, q, _) in &r.kids {
        let slot = if lview {
            crate::lview::net_slot(r.node.stage, slot)
        } else {
            slot
        };
        side.extend_from_slice(&[cell, slot, 0, 0]);
        side.extend_from_slice(&n.to_le_bytes());
        side.extend_from_slice(&q.to_le_bytes());
    }
    w.added()
}

/// The ordinal distribution over `vals` ranked best first: the i-th best of
/// m, given no better one was taken, with probability `e + (1 - e) / (m -
/// i)`, `e` uniform per choice.
fn pick_ordinal(vals: &[f32], rng: &mut impl Rng) -> usize {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
    let e: f64 = rng.gen();
    let m = idx.len();
    for (i, &k) in idx.iter().enumerate() {
        if rng.gen::<f64>() < e + (1.0 - e) / (m - i) as f64 {
            return k;
        }
    }
    *idx.last().expect("nonempty")
}

fn lat_of(task: &Task, t1: u32) -> &Lattice {
    match task {
        Task::Search { t1: s, .. } => s.lattice(t1),
        Task::Opening { lat, .. } | Task::Walk { lat, .. } | Task::Rooted { lat, .. } => lat,
    }
}

struct Worker<'a> {
    cfg: &'a GenConfig,
    c: &'a GenCounters,
    rng: rand_chacha::ChaCha8Rng,
    bid: i16,
    t1w: ChunkWriter,
    t2w: ChunkWriter,
    t1e: ChunkWriter,
    /// The opening phase: finished openings go here, not into searches.
    collect: Option<Vec<StepState>>,
    /// The search phase: games start from these.
    pool: Option<&'a Mutex<Vec<StepState>>>,
    /// The tier-2 phase's queue (turn start, node), and when it stops
    /// starting searches.
    rooted_q: Option<&'a Mutex<Vec<(StepState, Key)>>>,
    deadline: Option<Instant>,
}

impl Worker<'_> {
    fn new_game(&mut self) -> Result<Task> {
        if self.collect.is_none() {
            self.c.games.fetch_add(1, Relaxed);
            if let Some(s) = self.pool.and_then(|p| p.lock().unwrap().pop()) {
                self.c.pooled.fetch_add(1, Relaxed);
                return Ok(self.search(s, None));
            }
        }
        let salt: u64 = self.rng.gen();
        let s = StepState::begin(&crate::game::MinionsState::with_bid(self.bid), salt);
        if self.cfg.start_turn_mean > 0.0 {
            let q = truncated_geometric_q(self.cfg.start_turn_mean, crate::game::MAX_TURNS.into());
            let k = loop {
                let mut k = 0u32;
                while self.rng.gen::<f64>() < q {
                    k += 1;
                }
                if k < u32::from(crate::game::MAX_TURNS) {
                    break k;
                }
            };
            return self.start(s, k);
        }
        // P(G <= g) = 1 - (1 - p)^g: a median of m needs (1 - p)^m = 1/2.
        let p = if self.cfg.from_end_median > 0.0 {
            1.0 - 0.5f64.powf(1.0 / self.cfg.from_end_median.max(1.0))
        } else {
            1.0 / self.cfg.from_end_mean.max(1.0)
        };
        let mut g = 1u32;
        while g < 1000 && self.rng.gen::<f64>() >= p {
            g += 1;
        }
        let k = u32::from(crate::game::MAX_TURNS).saturating_sub(g);
        self.start(s, k)
    }

    /// A game at the start position `s` after `k` opening turns.
    fn start(&mut self, s: StepState, k: u32) -> Result<Task> {
        if k == 0 {
            if let Some(out) = self.collect.as_mut() {
                out.push(s);
                return self.new_game();
            }
            return Ok(self.search(s, None));
        }
        Ok(Task::Opening {
            lat: Lattice::new(s)?,
            at: 0,
            left: k,
        })
    }

    fn search(&self, s: StepState, lat: Option<Lattice>) -> Task {
        let mut t1 = Tier1::new(s, self.cfg.t1.clone());
        t1.nodes[0].lattice = lat;
        Task::Search {
            t1,
            started: 0,
            vis: Vec::new(),
        }
    }

    fn advance(&mut self, task: &mut Task) -> Result<Next> {
        match task {
            Task::Opening { lat, at, left } => loop {
                if lat.terminal.is_some() {
                    return Ok(Next::Replace(vec![self.new_game()?]));
                }
                let n = *at;
                match lat.nodes[n as usize].kind {
                    Kind::Sink => {
                        let s = lat.nodes[n as usize].state.clone();
                        self.c.opening_turns.fetch_add(1, Relaxed);
                        if s.is_terminal() {
                            return Ok(Next::Replace(vec![self.new_game()?]));
                        }
                        *left -= 1;
                        if *left == 0 {
                            if let Some(out) = self.collect.as_mut() {
                                out.push(s);
                                return Ok(Next::Replace(Vec::new()));
                            }
                            return Ok(Next::Replace(vec![self.search(s, None)]));
                        }
                        *lat = Lattice::new(s)?;
                        *at = 0;
                    }
                    Kind::Dead => *at = 0,
                    Kind::Inner => {
                        if let Some(Need::Expand(e)) = lat.ensure_expanded(n)? {
                            return Ok(Next::Wait(vec![Req::Expand { t1: NONE, n: e }]));
                        }
                        let nd = &lat.nodes[n as usize];
                        if nd.kind != Kind::Inner {
                            continue;
                        }
                        let mut vars: Vec<u8> =
                            nd.kids.iter().filter(|k| !k.dead).map(|k| k.var).collect();
                        vars.dedup();
                        let var = vars[self.rng.gen_range(0..vars.len())];
                        let idx: Vec<usize> = (0..nd.kids.len())
                            .filter(|&i| nd.kids[i].var == var && !nd.kids[i].dead)
                            .collect();
                        let vals: Vec<f32> = idx.iter().map(|&i| nd.kids[i].v).collect();
                        let k = idx[pick_ordinal(&vals, &mut self.rng)];
                        *at = lat.child(n, k)?;
                    }
                }
            },
            Task::Search { t1, started, vis } => {
                // Visits that already handed in a request wait on it (the
                // node is busy) when advanced again, so passes repeat until
                // none finishes and every request is collected once.
                let mut reqs = Vec::new();
                loop {
                    if t1.nodes[0].terminal.is_some() || (*started >= self.cfg.n1 && vis.is_empty())
                    {
                        return self.finish_turn(t1);
                    }
                    while *started < self.cfg.n1 && vis.len() < self.cfg.inflight1.max(1) {
                        vis.push(t1.visit());
                        *started += 1;
                    }
                    let mut progressed = false;
                    let mut i = 0;
                    while i < vis.len() {
                        match t1.advance(&mut vis[i])? {
                            Need1::Expand { t, n } => reqs.push(Req::Expand { t1: t, n }),
                            Need1::SinkValue { t, n } => reqs.push(Req::T1Value { t1: t, n }),
                            Need1::NodeValue { t } => reqs.push(Req::T1Value { t1: t, n: NONE }),
                            Need1::Wait => {}
                            Need1::Done => {
                                vis.swap_remove(i);
                                progressed = true;
                                continue;
                            }
                        }
                        i += 1;
                    }
                    if !progressed {
                        return Ok(Next::Wait(reqs));
                    }
                }
            }
            Task::Walk { lat, walk } => {
                let mut needs = Vec::new();
                let over = if self.cfg.sd_frac > 0.0 {
                    lat.advance_sd(walk, &self.cfg.walk, &mut needs)?
                } else {
                    lat.advance_walk(walk, &self.cfg.walk, &mut self.rng, &mut needs)?
                };
                if !over {
                    return Ok(Next::Wait(
                        needs
                            .into_iter()
                            .map(|n| match n {
                                Need::Expand(n) => Req::Expand { t1: NONE, n },
                                Need::Value(n) => Req::LatValue { n },
                                _ => unreachable!("walks only hand out requests"),
                            })
                            .collect(),
                    ));
                }
                for r in &walk.records {
                    encode_t2(&lat.start, r, &mut self.t2w, self.cfg.lview)?;
                    self.view_targets(r)?;
                }
                self.c
                    .t2_records
                    .fetch_add(walk.records.len() as u64, Relaxed);
                Ok(Next::Replace(Vec::new()))
            }
            Task::Rooted { lat, r } => {
                let mut needs = Vec::new();
                if !lat.advance_rooted(r, &self.cfg.rooted, &mut needs)? {
                    return Ok(Next::Wait(
                        needs
                            .into_iter()
                            .map(|n| match n {
                                Need::Expand(n) => Req::Expand { t1: NONE, n },
                                Need::Value(n) => Req::LatValue { n },
                                _ => unreachable!("searches only hand out requests"),
                            })
                            .collect(),
                    ));
                }
                self.c.rooted_done.fetch_add(1, Relaxed);
                if r.abandoned {
                    self.c.rooted_abandoned.fetch_add(1, Relaxed);
                }
                if let Some(rec) = r.record.take() {
                    encode_t2(&lat.start, &rec, &mut self.t2w, self.cfg.lview)?;
                    self.view_targets(&rec)?;
                    self.c.t2_records.fetch_add(1, Relaxed);
                }
                Ok(Next::Replace(self.next_rooted()?.into_iter().collect()))
            }
        }
    }

    /// The tier-2 phase's next search: the next queued node reachable in a
    /// fresh lattice of its turn (None: the queue is empty or time is up).
    fn next_rooted(&mut self) -> Result<Option<Task>> {
        let Some(q) = self.rooted_q else {
            return Ok(None);
        };
        loop {
            if self.deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(None);
            }
            let Some((s, key)) = q.lock().unwrap().pop() else {
                return Ok(None);
            };
            let mut lat = Lattice::new(s)?;
            if lat.terminal.is_some() {
                continue;
            }
            match lat.reach(key)? {
                Some(at) => {
                    return Ok(Some(Task::Rooted {
                        lat,
                        r: Rooted::new(at),
                    }))
                }
                None => {
                    self.c.rooted_unreached.fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// The tier-2 phase: searches from the queue until it is empty or the
    /// deadline passes (searches already started finish).
    fn rooted_phase(&mut self, eval: &dyn LatticeEval) -> Result<()> {
        let p = self.cfg.pipeline.max(1);
        let mut groups: Vec<Group> = (0..p).map(|_| Group::default()).collect();
        for i in 0..self.cfg.rooted_tasks.max(1) {
            match self.next_rooted()? {
                Some(t) => groups[i % p].tasks.push(t),
                None => break,
            }
        }
        let mut g = 0;
        while groups
            .iter()
            .any(|x| !x.tasks.is_empty() || x.inflight.is_some())
        {
            let grp = &mut groups[g];
            if let Some(f) = grp.inflight.take() {
                self.feed(&mut grp.tasks, f)?;
            }
            if !grp.tasks.is_empty() {
                grp.inflight = Some(self.submit_round(&mut grp.tasks, eval)?);
            }
            g = (g + 1) % p;
        }
        self.t1w.flush()?;
        self.t2w.flush()
    }

    /// Partial-turn value targets from a tier-2 record (see
    /// `GenConfig::view_min_visits`). Values are the turn mover's, as the
    /// bridge reads them: a kid that starts the next turn is flipped.
    fn view_targets(&mut self, r: &T2Record) -> Result<()> {
        let min = self.cfg.view_min_visits;
        if min == 0 {
            return Ok(());
        }
        let mut rec = Vec::new();
        let mut put = |w: &mut ChunkWriter, c: &StepState, v: f32, n: f32| -> Result<()> {
            rec.clear();
            crate::stepsp::encode_record(&c.view(), v, 0.0, n as u32, &mut rec);
            w.buf.extend_from_slice(&rec);
            w.added()
        };
        let mut k = 0;
        // Sink-return means only, from at least `min` returns.
        if r.ns >= min as f32 && !r.node.is_terminal() {
            put(&mut self.t1w, &r.node, r.q, r.ns)?;
            k += 1;
        }
        for &(cell, slot, _, q, ns) in &r.kids {
            if ns < min as f32 {
                continue;
            }
            let c = r.node.with_next(cell).child_unchecked(slot);
            if c.is_terminal() {
                continue;
            }
            let v = if c.flipped != r.node.flipped {
                1.0 - q
            } else {
                q
            };
            put(&mut self.t1w, &c, v, ns)?;
            k += 1;
        }
        self.c.t1_records.fetch_add(k, Relaxed);
        self.c.view_records.fetch_add(k, Relaxed);
        Ok(())
    }

    /// Record the searched turn, pick the next one, and hand the root's
    /// lattice to a walk.
    fn finish_turn(&mut self, t1: &mut Tier1) -> Result<Next> {
        let root = &t1.nodes[0];
        if root.terminal.is_some() {
            // The mover kills the necromancer: the game ends here.
            return Ok(Next::Replace(vec![self.new_game()?]));
        }
        self.c.turns.fetch_add(1, Relaxed);
        let mut rec = Vec::new();
        if self.cfg.lview {
            encode_turn_lv(&root.state, t1.root_value() as f32, root.n, &mut rec);
        } else {
            crate::stepsp::encode_record(
                &root.state.view(),
                t1.root_value() as f32,
                0.0,
                root.n,
                &mut rec,
            );
        }
        self.t1w.buf.extend_from_slice(&rec);
        self.t1w.added()?;
        // Energy targets: a turn start's children (complete turns) and their
        // visits -- the root's, and interior turns' searched at least
        // `energy_min_visits` deep (0: the root only).
        for (i, nd) in t1.nodes.iter().enumerate() {
            let min = if i == 0 {
                1
            } else {
                self.cfg.energy_min_visits
            };
            if !self.cfg.lview
                || min == 0
                || nd.n < min
                || nd.children.len() < 2
                || nd.terminal.is_some()
                || nd.state.is_terminal()
            {
                continue;
            }
            let (mut act, mut sc) = (Vec::with_capacity(MAX_ACTIVE), [0f32; N_SCALARS]);
            crate::lview::node_features(&nd.state, &mut act, &mut sc);
            encode_energy(
                0,
                [nd.n as f32, nd.q() as f32, f32::NAN, 1.0],
                &act,
                &sc,
                &mut self.t1e.buf,
            );
            let pri = t1.priors(i as u32);
            let lat = t1.lattice(i as u32);
            for (k, &c) in nd.children.iter().enumerate() {
                let ch = &t1.nodes[c as usize];
                crate::lview::decl_features(&nd.state, &ch.state.done, &mut act, &mut sc);
                let sv = lat.nodes[ch.sink as usize].value.unwrap_or(f32::NAN);
                encode_energy(
                    1,
                    [ch.n as f32, (1.0 - ch.q()) as f32, sv, pri[k] as f32],
                    &act,
                    &sc,
                    &mut self.t1e.buf,
                );
            }
            // Whole groups per chunk: flush only between groups.
            self.t1e.n += 1 + nd.children.len();
            if self.t1e.n >= self.t1e.per_chunk {
                self.t1e.flush()?;
            }
            self.c.energy_groups.fetch_add(1, Relaxed);
        }
        self.c.t1_records.fetch_add(1, Relaxed);
        // Interior turns searched deeply enough are value targets too: each
        // turn's tree is fresh, so they repeat nothing.
        if self.cfg.t1_min_visits > 0 {
            for node in &t1.nodes[1..] {
                if node.n < self.cfg.t1_min_visits
                    || node.terminal.is_some()
                    || node.state.is_terminal()
                {
                    continue;
                }
                rec.clear();
                if self.cfg.lview {
                    encode_turn_lv(&node.state, node.q() as f32, node.n, &mut rec);
                } else {
                    crate::stepsp::encode_record(
                        &node.state.view(),
                        node.q() as f32,
                        0.0,
                        node.n,
                        &mut rec,
                    );
                }
                self.t1w.buf.extend_from_slice(&rec);
                self.t1w.added()?;
                self.c.t1_records.fetch_add(1, Relaxed);
            }
        }
        let visits: Vec<f64> = root
            .children
            .iter()
            .map(|&c| f64::from(t1.nodes[c as usize].n))
            .collect();
        let pick = root.children[noisy_pick(
            &visits,
            self.cfg.temp1,
            self.cfg.dir1_alpha,
            self.cfg.dir1_frac,
            &mut self.rng,
        )];
        let mut out = Vec::with_capacity(2);
        if self.cfg.distill_tau > 0.0 {
            // Distilling: the expansions are the tier-2 records.
        } else if self.cfg.rooted_frac > 0.0 {
            if let (Some(mut lat), Some(q)) = (t1.nodes[0].lattice.take(), self.rooted_q) {
                let target = t1.nodes[pick as usize].sink;
                let mut nodes = lat.path_to(target, &mut self.rng)?;
                let f = self.cfg.rooted_frac;
                nodes.retain(|_| self.rng.gen::<f64>() < f);
                self.c.rooted_queued.fetch_add(nodes.len() as u64, Relaxed);
                let mut q = q.lock().unwrap();
                for n in nodes {
                    q.push((lat.start.clone(), lat.nodes[n as usize].key));
                }
            }
        } else if self.cfg.sd_frac > 0.0 {
            if let Some(mut lat) = t1.nodes[0].lattice.take() {
                let target = t1.nodes[pick as usize].sink;
                let mut nodes = lat.path_to(target, &mut self.rng)?;
                let f = self.cfg.sd_frac;
                nodes.retain(|_| self.rng.gen::<f64>() < f);
                if !nodes.is_empty() {
                    out.push(Task::Walk {
                        lat,
                        walk: Walk::self_distill(&self.cfg.walk, nodes),
                    });
                }
            }
        } else if self.cfg.t2_from_t1 {
            if let Some(mut lat) = t1.nodes[0].lattice.take() {
                let turns: Vec<(u32, f64, f64)> = t1.nodes[0]
                    .children
                    .iter()
                    .map(|&c| {
                        let ch = &t1.nodes[c as usize];
                        // The child's value is its own mover's.
                        (ch.sink, f64::from(ch.n), f64::from(ch.n) - ch.w)
                    })
                    .collect();
                let target = t1.nodes[pick as usize].sink;
                let recs = lat.tier1_records(&turns, target, &mut self.rng)?;
                for r in &recs {
                    encode_t2(&lat.start, r, &mut self.t2w, self.cfg.lview)?;
                }
                self.c.t2_records.fetch_add(recs.len() as u64, Relaxed);
            }
        } else if let Some(lat) = t1.nodes[0].lattice.take() {
            out.push(Task::Walk {
                lat,
                walk: Walk::new(&self.cfg.walk),
            });
        }
        let child = &mut t1.nodes[pick as usize];
        if child.terminal.is_some() || child.state.is_terminal() {
            out.push(self.new_game()?);
        } else {
            let s = child.state.clone();
            let lat = child.lattice.take();
            out.push(self.search(s, lat));
        }
        Ok(Next::Replace(out))
    }

    /// Play `n` openings at once (a game that ends inside its opening is
    /// replaced) and return their final positions.
    fn open_phase(&mut self, eval: &dyn LatticeEval, n: usize) -> Result<Vec<StepState>> {
        self.collect = Some(Vec::with_capacity(n));
        let p = self.cfg.pipeline.max(1);
        let mut groups: Vec<Group> = (0..p).map(|_| Group::default()).collect();
        for i in 0..n {
            groups[i % p].tasks.push(self.new_game()?);
        }
        let mut g = 0;
        while groups
            .iter()
            .any(|x| !x.tasks.is_empty() || x.inflight.is_some())
        {
            let grp = &mut groups[g];
            if let Some(f) = grp.inflight.take() {
                self.feed(&mut grp.tasks, f)?;
            }
            if !grp.tasks.is_empty() {
                grp.inflight = Some(self.submit_round(&mut grp.tasks, eval)?);
            }
            g = (g + 1) % p;
        }
        Ok(self.collect.take().unwrap_or_default())
    }

    fn run(&mut self, eval: &dyn LatticeEval, stop: &AtomicBool) -> Result<()> {
        // The games in groups that take turns: while one group's net call is
        // in flight, the worker advances the next group.
        let n = self.cfg.pipeline.max(1);
        let mut groups: Vec<Group> = (0..n).map(|_| Group::default()).collect();
        for i in 0..self.cfg.games {
            groups[i % n].tasks.push(self.new_game()?);
        }
        let mut g = 0;
        while !stop.load(Relaxed) {
            let grp = &mut groups[g];
            if let Some(f) = grp.inflight.take() {
                self.feed(&mut grp.tasks, f)?;
            }
            grp.inflight = Some(self.submit_round(&mut grp.tasks, eval)?);
            g = (g + 1) % n;
        }
        for grp in &mut groups {
            if let Some(f) = grp.inflight.take() {
                f.pending.wait()?;
            }
        }
        self.t1w.flush()?;
        self.t1e.flush()?;
        self.t2w.flush()
    }

    /// Advance `tasks` until each waits on the net, and submit their
    /// requests as one call.
    fn submit_round(&mut self, tasks: &mut Vec<Task>, eval: &dyn LatticeEval) -> Result<InFlight> {
        let t0 = Instant::now();
        let mut reqs: Vec<(usize, Req)> = Vec::with_capacity(tasks.len());
        let mut i = 0;
        while i < tasks.len() {
            match self.advance(&mut tasks[i])? {
                Next::Wait(r) => {
                    reqs.extend(r.into_iter().map(|r| (i, r)));
                    i += 1;
                }
                Next::Replace(v) => {
                    // Queued requests are all for tasks before `i`; the
                    // one moved into `i` and the new ones come next.
                    tasks.swap_remove(i);
                    tasks.extend(v);
                }
            }
        }
        // Expansions and values, one call.
        let mut slots: Vec<Vec<(u8, u8)>> = Vec::new();
        let mut ex_idx = Vec::new();
        let mut val_idx = Vec::new();
        for (k, &(ti, r)) in reqs.iter().enumerate() {
            match r {
                Req::Expand { t1, n } => {
                    let l = lat_of(&tasks[ti], t1);
                    slots.push(l.request(n).0);
                    ex_idx.push(k);
                }
                _ => val_idx.push(k),
            }
        }
        let mut ex_reqs = Vec::with_capacity(ex_idx.len());
        for (j, &k) in ex_idx.iter().enumerate() {
            let (ti, Req::Expand { t1, n }) = reqs[k] else {
                unreachable!()
            };
            let l = lat_of(&tasks[ti], t1);
            ex_reqs.push(ExpandReq {
                root: &l.start,
                node: &l.nodes[n as usize].state,
                slots: &slots[j],
                trunk: l.trunk.clone(),
            });
        }
        // Energy priors: each tier-1 sink value request also asks for its
        // declaration's energy (a kid-less tier-2 row).
        let mut en_idx = Vec::new();
        let mut decls = Vec::new();
        if self.cfg.t1.energy_prior {
            for &k in &val_idx {
                let (ti, r) = reqs[k];
                if let (Req::T1Value { t1, n }, Task::Search { t1: s, .. }) = (r, &tasks[ti]) {
                    if n != NONE {
                        let l = s.lattice(t1);
                        decls.push(crate::lview::decl_state(
                            &l.start,
                            &l.nodes[n as usize].state.done,
                        ));
                        en_idx.push(k);
                    }
                }
            }
        }
        for (&k, d) in en_idx.iter().zip(&decls) {
            let (ti, Req::T1Value { t1, .. }) = reqs[k] else {
                unreachable!()
            };
            let l = lat_of(&tasks[ti], t1);
            ex_reqs.push(ExpandReq {
                root: &l.start,
                node: d,
                slots: &[],
                trunk: l.trunk.clone(),
            });
        }
        let mut states: Vec<&StepState> = Vec::with_capacity(val_idx.len());
        for &k in &val_idx {
            let (ti, r) = reqs[k];
            let s = match (r, &tasks[ti]) {
                (Req::T1Value { t1, n }, Task::Search { t1: s, .. }) => {
                    if n == NONE {
                        &s.nodes[t1 as usize].state
                    } else {
                        &s.lattice(t1).nodes[n as usize].state
                    }
                }
                (Req::LatValue { n }, Task::Walk { lat, .. } | Task::Rooted { lat, .. }) => {
                    &lat.nodes[n as usize].state
                }
                _ => unreachable!("request and task disagree"),
            };
            states.push(s);
        }
        let pending = eval.submit(&ex_reqs, &states)?;
        let kids = slots.iter().map(|s| s.len()).sum();
        self.c
            .cpu_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        Ok(InFlight {
            pending,
            reqs,
            ex_idx,
            val_idx,
            en_idx,
            kids,
        })
    }

    /// Wait for a submitted call and hand its results to the tasks.
    fn feed(&mut self, tasks: &mut [Task], f: InFlight) -> Result<()> {
        let t1 = Instant::now();
        let (outs, vals) = f.pending.wait()?;
        let t2 = Instant::now();
        let reqs = &f.reqs;
        for (j, &k) in f.ex_idx.iter().enumerate() {
            let (ti, Req::Expand { t1, n }) = reqs[k] else {
                unreachable!()
            };
            if self.cfg.distill_tau > 0.0 && self.rng.gen::<f64>() < self.cfg.distill_frac {
                let l = lat_of(&tasks[ti], t1);
                let o = &outs[j];
                let slots = l.request(n).0;
                let tau = self.cfg.distill_tau;
                let mut best: HashMap<u8, f32> = HashMap::new();
                for (&(cell, _), &v) in slots.iter().zip(&o.values) {
                    let b = best.entry(cell).or_insert(v);
                    *b = b.max(v);
                }
                let kids = slots
                    .iter()
                    .zip(&o.values)
                    .map(|(&(cell, slot), &v)| {
                        let w = (f64::from(v - best[&cell]) / tau).exp() as f32;
                        (cell, slot, w, v, 1.0)
                    })
                    .collect();
                let rec = T2Record {
                    node: l.nodes[n as usize].state.clone(),
                    kids,
                    n: 1.0,
                    q: o.value,
                    ns: 1.0,
                };
                encode_t2(&l.start, &rec, &mut self.t2w, self.cfg.lview)?;
                self.c.t2_records.fetch_add(1, Relaxed);
            }
            match &mut tasks[ti] {
                Task::Search { t1: s, .. } => s.feed_expand(t1, n, &outs[j]),
                Task::Opening { lat, .. } | Task::Walk { lat, .. } | Task::Rooted { lat, .. } => {
                    lat.feed_expand(n, &outs[j])
                }
            }
        }
        for (j, &k) in f.en_idx.iter().enumerate() {
            let (ti, Req::T1Value { t1, n }) = reqs[k] else {
                unreachable!()
            };
            if let Task::Search { t1: s, .. } = &mut tasks[ti] {
                s.lattice_mut(t1)
                    .feed_energy(n, outs[f.ex_idx.len() + j].energy);
            }
        }
        for (j, &k) in f.val_idx.iter().enumerate() {
            let (ti, r) = reqs[k];
            match (r, &mut tasks[ti]) {
                (Req::T1Value { t1, n }, Task::Search { t1: s, .. }) => {
                    if n == NONE {
                        s.feed_node_value(t1, &vals[j]);
                    } else {
                        s.feed_sink_value(t1, n, &vals[j]);
                    }
                }
                (Req::LatValue { n }, Task::Walk { lat, .. } | Task::Rooted { lat, .. }) => {
                    lat.feed_value(n, &vals[j])
                }
                _ => unreachable!(),
            }
        }
        let c = self.c;
        c.rounds.fetch_add(1, Relaxed);
        c.expansions.fetch_add(f.ex_idx.len() as u64, Relaxed);
        c.kids.fetch_add(f.kids as u64, Relaxed);
        c.values.fetch_add(f.val_idx.len() as u64, Relaxed);
        c.eval_ns.fetch_add((t2 - t1).as_nanos() as u64, Relaxed);
        c.cpu_ns.fetch_add(t2.elapsed().as_nanos() as u64, Relaxed);
        Ok(())
    }
}

/// A worker's share of games that submit their net calls together.
#[derive(Default)]
struct Group {
    tasks: Vec<Task>,
    inflight: Option<InFlight>,
}

/// A submitted round: the call, and which task asked for each result.
struct InFlight {
    pending: crate::lattice::Pending,
    reqs: Vec<(usize, Req)>,
    ex_idx: Vec<usize>,
    val_idx: Vec<usize>,
    /// Sink value requests that also asked for their declaration's energy
    /// (outputs after the expansions').
    en_idx: Vec<usize>,
    kids: usize,
}

/// The bid at which the tier-1 values rate the opening even (as
/// [`crate::stepsp::bid`]).
pub fn lattice_bid(eval: &dyn LatticeEval) -> Result<i16> {
    use crate::game::{MinionsState, BID_MAX, BID_MIN};
    let v = |b: i16| -> Result<f32> {
        let s = StepState::begin(&MinionsState::with_bid(b), 0);
        Ok(eval.value(&[&s])?[0])
    };
    let (mut lo, mut hi) = (BID_MIN, BID_MAX);
    if v(lo)? < 0.5 {
        return Ok(lo);
    }
    if v(hi)? >= 0.5 {
        return Ok(hi);
    }
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if v(mid)? >= 0.5 {
            lo = mid
        } else {
            hi = mid
        }
    }
    Ok(lo)
}

fn make_worker<'a>(
    cfg: &'a GenConfig,
    bid: i16,
    w: usize,
    t1_dir: &Path,
    t2_dir: &Path,
    c: &'a GenCounters,
) -> Worker<'a> {
    Worker {
        cfg,
        c,
        rng: rand_chacha::ChaCha8Rng::seed_from_u64(
            cfg.seed ^ (w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        ),
        bid,
        t1w: ChunkWriter::new(
            t1_dir.to_path_buf(),
            w,
            b"MSTP",
            crate::stepsp::VERSION,
            crate::stepsp::REC_BYTES,
            cfg.chunk_records,
        ),
        t2w: {
            let mut x = ChunkWriter::new(
                t2_dir.to_path_buf(),
                w,
                b"MLT2",
                T2_VERSION,
                T2_REC_BYTES,
                cfg.chunk_records,
            );
            x.side = Some((b"MLTK", Vec::new(), T2_KID_BYTES));
            if cfg.teacher_view {
                x.side2 = Some((b"MLTO", Vec::new(), OLD_REC_BYTES));
            }
            x
        },
        t1e: ChunkWriter::new(
            t1_dir.with_file_name("t1e"),
            w,
            b"MLTE",
            E_VERSION,
            E_REC_BYTES,
            cfg.chunk_records * 16,
        ),
        collect: None,
        pool: None,
        rooted_q: None,
        deadline: None,
    }
}

pub fn generate(cfg: &GenConfig, eval: Arc<dyn LatticeEval>) -> Result<()> {
    let t1_dir = cfg.out.join("t1");
    let t2_dir = cfg.out.join("t2");
    std::fs::create_dir_all(&t1_dir)?;
    std::fs::create_dir_all(&t2_dir)?;
    if cfg.lview {
        std::fs::create_dir_all(cfg.out.join("t1e"))?;
    }
    let bid = lattice_bid(eval.as_ref())?;
    eprintln!(
        "lattice-gen: bid {bid}, {} workers x {} games, n1 {}, n2 {}",
        cfg.workers, cfg.games, cfg.n1, cfg.walk.n2
    );
    let c = Arc::new(GenCounters::default());
    // The opening phase. Its workers write no records; their seeds (and
    // writer tags) follow the search workers'.
    let loaded = match &cfg.pool {
        Some(p) => load_pool(p)?,
        None => Vec::new(),
    };
    let n_open = if cfg.refill_to > 0 {
        cfg.refill_to.saturating_sub(loaded.len())
    } else if cfg.pool.is_some() {
        0
    } else {
        cfg.openings
    };
    if cfg.pool.is_some() {
        eprintln!("  opening pool: {} loaded", loaded.len());
    }
    let pool = Mutex::new(Vec::new());
    if n_open > 0 {
        let t = Instant::now();
        let ow = cfg.open_workers.max(1);
        let res: Vec<Result<Vec<StepState>>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..ow)
                .map(|w| {
                    let (c, eval) = (c.clone(), eval.clone());
                    let (t1_dir, t2_dir) = (t1_dir.clone(), t2_dir.clone());
                    let n = n_open / ow + usize::from(w < n_open % ow);
                    sc.spawn(move || {
                        let mut wk = make_worker(cfg, bid, cfg.workers + w, &t1_dir, &t2_dir, &c);
                        wk.open_phase(eval.as_ref(), n)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("opening worker"))
                .collect()
        });
        let mut p = pool.lock().unwrap();
        for r in res {
            p.extend(r?);
        }
        eprintln!(
            "  openings: {} in {:.1}s ({} turns, {} rounds, {} expansions) on {ow} threads",
            p.len(),
            t.elapsed().as_secs_f64(),
            c.opening_turns.load(Relaxed),
            c.rounds.load(Relaxed),
            c.expansions.load(Relaxed),
        );
        for x in [
            &c.rounds,
            &c.expansions,
            &c.kids,
            &c.values,
            &c.eval_ns,
            &c.cpu_ns,
        ] {
            x.store(0, Relaxed);
        }
    }
    // Searches pop from the end, so the vector is newest first: fresh
    // openings, then the loaded ones reversed (the oldest pops first).
    {
        let mut p = pool.lock().unwrap();
        p.extend(loaded.into_iter().rev());
    }
    let save = |p: &[StepState]| -> Result<()> {
        if let Some(path) = &cfg.pool {
            let v: Vec<StepState> = p.iter().rev().cloned().collect();
            save_pool(path, &v)?;
            eprintln!("  opening pool: {} saved", v.len());
        }
        Ok(())
    };
    if cfg.refill_to > 0 {
        return save(&pool.lock().unwrap());
    }
    let pool = &pool;
    let rooted_q: Mutex<Vec<(StepState, Key)>> = Mutex::new(Vec::new());
    let rooted_q = &rooted_q;
    let stop = Arc::new(AtomicBool::new(false));
    let t0 = Instant::now();
    let res: Vec<Result<()>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..cfg.workers)
            .map(|w| {
                let (c, stop, eval) = (c.clone(), stop.clone(), eval.clone());
                let (t1_dir, t2_dir) = (t1_dir.clone(), t2_dir.clone());
                sc.spawn(move || {
                    let mut wk = make_worker(cfg, bid, w, &t1_dir, &t2_dir, &c);
                    wk.pool = Some(pool);
                    wk.rooted_q = Some(rooted_q);
                    let r = wk.run(eval.as_ref(), &stop);
                    if r.is_err() {
                        stop.store(true, Relaxed);
                    }
                    r
                })
            })
            .collect();
        let mut last = (Instant::now(), 0u64, 0u64, 0u64);
        loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let el = t0.elapsed().as_secs_f64();
            let t2r = c.t2_records.load(Relaxed);
            if el >= cfg.secs || (cfg.target_t2 > 0 && t2r >= cfg.target_t2) {
                stop.store(true, Relaxed);
            }
            let done = hs.iter().all(|h| h.is_finished());
            if last.0.elapsed().as_secs_f64() >= 10.0 || stop.load(Relaxed) && done {
                let dt = last.0.elapsed().as_secs_f64();
                let ex = c.expansions.load(Relaxed);
                let va = c.values.load(Relaxed);
                let kids = c.kids.load(Relaxed);
                let ev = c.eval_ns.load(Relaxed) as f64;
                let cpu = c.cpu_ns.load(Relaxed) as f64;
                eprintln!(
                    "  {el:.0}s: games {} ({} pooled) turns {} (+{} opening), t1 {} ({} views) t2 {} ({:.0}/s), expansions {:.0}/s ({:.1} kids), values {:.0}/s, eval share {:.0}%, rounds {}",
                    c.games.load(Relaxed),
                    c.pooled.load(Relaxed),
                    c.turns.load(Relaxed),
                    c.opening_turns.load(Relaxed),
                    c.t1_records.load(Relaxed),
                    c.view_records.load(Relaxed),
                    t2r,
                    (t2r - last.3) as f64 / dt,
                    (ex - last.1) as f64 / dt,
                    kids as f64 / ex.max(1) as f64,
                    (va - last.2) as f64 / dt,
                    100.0 * ev / (ev + cpu).max(1.0),
                    c.rounds.load(Relaxed),
                );
                let pr: Vec<f64> = crate::lattice::PROF
                    .iter()
                    .map(|x| x.load(Relaxed) as f64)
                    .collect();
                let tot = (ev + cpu).max(1.0);
                eprintln!(
                    "    thread time: prepare {:.0}% stats {:.0}% child {:.0}% bridge-views {:.0}%; {:.0} nodes per scan; live lattices {} nodes {} kids {} witness exceptions {}",
                    100.0 * pr[0] / tot,
                    100.0 * pr[1] / tot,
                    100.0 * pr[2] / tot,
                    100.0 * pr[3] / tot,
                    pr[4] / (crate::lattice::SCANS.load(Relaxed).max(1) as f64),
                    crate::lattice::LIVE[0].load(Relaxed),
                    crate::lattice::LIVE[1].load(Relaxed),
                    crate::lattice::LIVE[2].load(Relaxed),
                    crate::lattice::LIVE[3].load(Relaxed),
                );
                {
                    use crate::lattice::{WIT_CLASS, WIT_INHERIT, WIT_RESOLVE};
                    use crate::steps::cand_log::NAMES;
                    let cls: Vec<(u64, u64)> = WIT_CLASS
                        .iter()
                        .map(|x| (x[0].load(Relaxed), x[1].load(Relaxed)))
                        .collect();
                    let ns_all: u64 = cls.iter().map(|x| x.1).sum();
                    let n_all: u64 = cls.iter().map(|x| x.0).sum();
                    let mut line = String::new();
                    for (name, (n, ns)) in NAMES.iter().zip(&cls) {
                        if *n > 0 {
                            line += &format!(
                                " {name} {:.0}%/{:.0}%t",
                                100.0 * *n as f64 / n_all.max(1) as f64,
                                100.0 * *ns as f64 / ns_all.max(1) as f64
                            );
                        }
                    }
                    let r: Vec<u64> = WIT_RESOLVE.iter().map(|x| x.load(Relaxed)).collect();
                    let ih: Vec<u64> = WIT_INHERIT.iter().map(|x| x.load(Relaxed)).collect();
                    eprintln!(
                        "    witnesses: {:.0}% of thread time in option checks, {:.0}% in resolutions ({} found, {} dead, {} from the pool; {} nodes refuted by nogoods); candidates (share of count / of check time):{line}; not live at the parent: {:.0}% of candidates, {:.0}% of check time, {} of them offered here",
                        100.0 * ns_all as f64 / tot,
                        100.0 * r[2] as f64 / tot,
                        r[0],
                        r[1],
                        r[4],
                        r[3],
                        100.0 * ih[0] as f64 / n_all.max(1) as f64,
                        100.0 * ih[1] as f64 / ns_all.max(1) as f64,
                        ih[2]
                    );
                }
                let rp: Vec<f64> = crate::lattice_ort::RUN_PROF
                    .iter()
                    .map(|x| x.load(Relaxed) as f64)
                    .collect();
                if rp[1] > 0.0 {
                    eprintln!(
                        "    lattice runner (of wall): merge {:.0}% run {:.0}% (tier 1 {:.0}%, tier-2 inputs {:.0}%, tier-2 session {:.0}%) split {:.0}%; {:.0} calls/s, {:.0} rows/call, {:.2} ms/call",
                        100.0 * rp[0] / (el * 1e9),
                        100.0 * rp[1] / (el * 1e9),
                        100.0 * rp[3] / (el * 1e9),
                        100.0 * rp[6] / (el * 1e9),
                        100.0 * rp[7] / (el * 1e9),
                        100.0 * rp[2] / (el * 1e9),
                        rp[4] / el,
                        rp[5] / rp[4].max(1.0),
                        1e-6 * rp[1] / rp[4].max(1.0)
                    );
                }
                last = (Instant::now(), ex, va, t2r);
            }
            if done {
                break;
            }
        }
        hs.into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });
    for r in res {
        r?;
    }
    let left = std::mem::take(&mut *pool.lock().unwrap());
    save(&left)?;
    if cfg.rooted_frac > 0.0 {
        rooted_phase(cfg, bid, eval, &c, rooted_q, &t1_dir, &t2_dir)?;
    }
    Ok(())
}

/// The separate tier-2 phase: every queued node (in random order) gets a
/// search of its own rooted there, until the queue is empty or
/// `cfg.rooted_secs` pass.
fn rooted_phase(
    cfg: &GenConfig,
    bid: i16,
    eval: Arc<dyn LatticeEval>,
    c: &GenCounters,
    q: &Mutex<Vec<(StepState, Key)>>,
    t1_dir: &Path,
    t2_dir: &Path,
) -> Result<()> {
    use rand::seq::SliceRandom;
    let n = {
        let mut v = q.lock().unwrap();
        v.shuffle(&mut rand_chacha::ChaCha8Rng::seed_from_u64(
            cfg.seed ^ 0x7007,
        ));
        v.len()
    };
    let t0 = Instant::now();
    let deadline = t0 + std::time::Duration::from_secs_f64(cfg.rooted_secs);
    let (ex0, va0, t2_0) = (
        c.expansions.load(Relaxed),
        c.values.load(Relaxed),
        c.t2_records.load(Relaxed),
    );
    eprintln!(
        "  tier-2 phase: {n} nodes queued, {} searches per worker, n {} (min {} per kid), budget {:.0}s",
        cfg.rooted_tasks, cfg.rooted.n, cfg.rooted.min_per_kid, cfg.rooted_secs
    );
    let res: Vec<Result<()>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..cfg.workers)
            .map(|w| {
                let eval = eval.clone();
                sc.spawn(move || {
                    // Writer tags after the self-play workers' and openers'.
                    let mut wk = make_worker(cfg, bid, 1000 + w, t1_dir, t2_dir, c);
                    wk.rooted_q = Some(q);
                    wk.deadline = Some(deadline);
                    wk.rooted_phase(eval.as_ref())
                })
            })
            .collect();
        let mut last = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let done = hs.iter().all(|h| h.is_finished());
            if last.elapsed().as_secs_f64() >= 10.0 || done {
                last = Instant::now();
                let el = t0.elapsed().as_secs_f64();
                eprintln!(
                    "  tier-2 phase {el:.0}s: {} searched ({:.1}/s), {} records, {} left, {} unreachable, {} abandoned; expansions {:.0}/s, values {:.0}/s",
                    c.rooted_done.load(Relaxed),
                    c.rooted_done.load(Relaxed) as f64 / el.max(1e-9),
                    c.t2_records.load(Relaxed) - t2_0,
                    q.lock().unwrap().len(),
                    c.rooted_unreached.load(Relaxed),
                    c.rooted_abandoned.load(Relaxed),
                    (c.expansions.load(Relaxed) - ex0) as f64 / el.max(1e-9),
                    (c.values.load(Relaxed) - va0) as f64 / el.max(1e-9),
                );
            }
            if done {
                break;
            }
        }
        hs.into_iter()
            .map(|h| h.join().expect("tier-2 phase worker panicked"))
            .collect()
    });
    for r in res {
        r?;
    }
    let dropped = q.lock().unwrap().len();
    if dropped > 0 {
        eprintln!("  tier-2 phase: time up, {dropped} nodes dropped");
    }
    Ok(())
}
