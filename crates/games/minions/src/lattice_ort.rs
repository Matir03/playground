//! The turn-lattice nets under ONNX Runtime (TensorRT), from
//! `python/minions_lattice_learner.py`: `tier1.onnx` maps a turn start to
//! its tier-1 value and trunk image (fp16), `tier2.onnx` maps a trunk image,
//! a lattice node and the node's children (`cell * 37 + slot`) to the
//! children's prior logits and values and the node's value. The trunk runs
//! once per turn start: callers cache its image ([`crate::lattice::Trunk`])
//! and hand it back with every expansion of that turn's lattice; a request
//! without one gets its turn start run through tier 1 first, in the same
//! call.
//!
//! Workers queue their rows; `runners` threads, each with its own pair of
//! sessions, merge everything queued into one tier-1 and one tier-2 call.
//! A call costs ~1 ms of host overhead however small, so a runner can wait
//! for more rows (`LATTICE_MIN_ROWS`, `LATTICE_WAIT_US`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use half::f16;
use ort::session::Session;

use crate::latgen::t2_view;
use crate::lattice::{ExpandOut, ExpandReq, LatticeEval, Trunk, ValueOut};
use crate::steps::{StepState, View, MAX_ACTIVE, N_SCALARS, SPARSE_DIM};

pub const SLOTS: usize = 37;
/// The TensorRT profiles' limits: rows per call and children per row.
pub const MAX_ROWS: usize = 4096;
pub const MAX_KIDS: usize = 512;
/// Children over a whole tier-2 call (flattened).
pub const MAX_TOTAL_KIDS: usize = 1 << 19;
const CELLS: usize = crate::map::NUM_CELLS;

/// Runner profile (ns, of all runners): merging rows, inside the two
/// `Session::run`s, splitting the outputs; the tier-1 part of the runs;
/// calls; rows.
pub static RUN_PROF: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn push_view(v: &View, idx: &mut Vec<i32>, scal: &mut Vec<f32>) {
    let mut act = Vec::with_capacity(MAX_ACTIVE);
    let mut sc = [0f32; N_SCALARS];
    v.features(&mut act, &mut sc);
    for i in 0..MAX_ACTIVE {
        idx.push(act.get(i).map_or(SPARSE_DIM as i32, |&x| i32::from(x)));
    }
    scal.extend_from_slice(&sc);
}

/// Where a tier-2 row's trunk image comes from.
enum TrunkSrc {
    Given(Trunk),
    /// Tier-1 row `i` of the same batch.
    Row(usize),
}

#[derive(Default)]
struct Batch {
    /// Tier-1 rows: turn starts.
    ridx: Vec<i32>,
    rscal: Vec<f32>,
    n1: usize,
    /// Tier-2 rows: trunk, node, kids.
    trunk: Vec<TrunkSrc>,
    nidx: Vec<i32>,
    nscal: Vec<f32>,
    kids: Vec<Vec<i32>>,
}

impl Batch {
    fn rows(&self) -> usize {
        self.n1 + self.kids.len()
    }

    fn append(&mut self, o: Batch) {
        let off = self.n1;
        self.ridx.extend(o.ridx);
        self.rscal.extend(o.rscal);
        self.n1 += o.n1;
        self.trunk.extend(o.trunk.into_iter().map(|t| match t {
            TrunkSrc::Row(i) => TrunkSrc::Row(i + off),
            g => g,
        }));
        self.nidx.extend(o.nidx);
        self.nscal.extend(o.nscal);
        self.kids.extend(o.kids);
    }
}

/// Per tier-1 row (value, trunk image); per tier-2 row (logits, values,
/// node value).
#[derive(Default)]
struct Outs {
    v1: Vec<f32>,
    h: Vec<Trunk>,
    logit: Vec<Vec<f32>>,
    value: Vec<Vec<f32>>,
    v2: Vec<f32>,
}

struct Job {
    batch: Batch,
    tx: std::sync::mpsc::SyncSender<Result<Outs>>,
}

#[derive(Default)]
struct Queue {
    q: Mutex<std::collections::VecDeque<Job>>,
    cv: std::sync::Condvar,
    closed: std::sync::atomic::AtomicBool,
}

pub struct OrtLatticeEval {
    queue: Arc<Queue>,
    runners: Vec<std::thread::JoinHandle<()>>,
}

impl Drop for OrtLatticeEval {
    fn drop(&mut self) {
        self.queue
            .closed
            .store(true, std::sync::atomic::Ordering::Release);
        self.queue.cv.notify_all();
        for h in self.runners.drain(..) {
            let _ = h.join();
        }
    }
}

fn run_batch(s1: &mut Session, s2: &mut Session, c: usize, b: Batch) -> Result<Outs> {
    use ort::value::Tensor;
    let e = |e: ort::Error| anyhow::anyhow!("{e}");
    let hw = c * CELLS;
    let mut out = Outs::default();
    let t0 = std::time::Instant::now();
    if b.n1 > 0 {
        anyhow::ensure!(b.n1 <= MAX_ROWS, "tier-1 batch {} over the profile", b.n1);
        let n1 = b.n1 as i64;
        let outs = s1
            .run(ort::inputs![
                "ridx" => Tensor::from_array((vec![n1, MAX_ACTIVE as i64], b.ridx)).map_err(e)?,
                "rscal" => Tensor::from_array((vec![n1, N_SCALARS as i64], b.rscal)).map_err(e)?,
            ])
            .map_err(e)?;
        let (_, v1) = outs["v1"].try_extract_tensor::<f32>().map_err(e)?;
        let (_, h) = outs["h"].try_extract_tensor::<f16>().map_err(e)?;
        anyhow::ensure!(
            h.len() == b.n1 * hw,
            "tier1 h has {} values, expected {}",
            h.len(),
            b.n1 * hw
        );
        out.v1 = v1.to_vec();
        out.h = h.chunks(hw).map(Trunk::from).collect();
    }
    RUN_PROF[3].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    let n2 = b.kids.len();
    if n2 > 0 {
        // Children flattened over the batch: row * 49*37 + cell * 37 + slot.
        let per_row = (CELLS * SLOTS) as i32;
        let kid: Vec<i32> = b
            .kids
            .iter()
            .enumerate()
            .flat_map(|(i, ks)| ks.iter().map(move |&x| i as i32 * per_row + x))
            .collect();
        anyhow::ensure!(
            n2 <= MAX_ROWS && kid.len() <= MAX_TOTAL_KIDS,
            "tier-2 batch {n2} rows, {} kids over the profile",
            kid.len()
        );
        let t = kid.len().max(1);
        let kid = if kid.is_empty() { vec![0] } else { kid };
        let mut h = Vec::with_capacity(n2 * hw);
        for tr in &b.trunk {
            match tr {
                TrunkSrc::Given(x) => h.extend_from_slice(x),
                TrunkSrc::Row(i) => h.extend_from_slice(&out.h[*i]),
            }
        }
        let (n, a, sc) = (n2 as i64, MAX_ACTIVE as i64, N_SCALARS as i64);
        let outs = s2
            .run(ort::inputs![
                "h" => Tensor::from_array((vec![n, c as i64, 7, 7], h)).map_err(e)?,
                "nidx" => Tensor::from_array((vec![n, a], b.nidx)).map_err(e)?,
                "nscal" => Tensor::from_array((vec![n, sc], b.nscal)).map_err(e)?,
                "kid" => Tensor::from_array((vec![t as i64], kid)).map_err(e)?,
            ])
            .map_err(e)?;
        let get = |name: &str| -> Result<Vec<f32>> {
            let (_, v) = outs[name].try_extract_tensor::<f32>().map_err(e)?;
            Ok(v.to_vec())
        };
        let (logit, value, v2) = (get("logit")?, get("value")?, get("v2")?);
        let per = |x: &[f32]| -> Vec<Vec<f32>> {
            let mut at = 0;
            b.kids
                .iter()
                .map(|ks| {
                    let r = x[at..at + ks.len()].to_vec();
                    at += ks.len();
                    r
                })
                .collect()
        };
        out.logit = per(&logit);
        out.value = per(&value);
        out.v2 = v2;
    }
    RUN_PROF[1].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    Ok(out)
}

/// Rows a call waits for (`LATTICE_MIN_ROWS`, default 1024) and the
/// longest it waits (`LATTICE_WAIT_US`, default 1500).
fn min_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("LATTICE_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
    })
}

fn batch_wait() -> std::time::Duration {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    std::time::Duration::from_micros(*V.get_or_init(|| {
        std::env::var("LATTICE_WAIT_US")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1500)
    }))
}

fn runner(mut s1: Session, mut s2: Session, c: usize, q: &Queue) {
    loop {
        let mut jobs = Vec::new();
        {
            let mut g = q.q.lock().unwrap_or_else(|p| p.into_inner());
            while g.is_empty() {
                if q.closed.load(std::sync::atomic::Ordering::Acquire) {
                    return;
                }
                g = q.cv.wait(g).unwrap_or_else(|p| p.into_inner());
            }
            let deadline = std::time::Instant::now() + batch_wait();
            loop {
                let queued: usize = g.iter().map(|j| j.batch.rows()).sum();
                let now = std::time::Instant::now();
                if queued >= min_rows() || now >= deadline {
                    break;
                }
                g =
                    q.cv.wait_timeout(g, deadline - now)
                        .unwrap_or_else(|p| p.into_inner())
                        .0;
            }
            let (mut t1, mut t2) = (0, 0);
            while let Some(j) = g.front() {
                if !jobs.is_empty()
                    && (t1 + j.batch.n1 > MAX_ROWS || t2 + j.batch.kids.len() > MAX_ROWS)
                {
                    break;
                }
                t1 += j.batch.n1;
                t2 += j.batch.kids.len();
                jobs.push(g.pop_front().expect("front"));
            }
        }
        let t0 = std::time::Instant::now();
        let sizes: Vec<(usize, usize)> = jobs
            .iter()
            .map(|j| (j.batch.n1, j.batch.kids.len()))
            .collect();
        let mut txs = Vec::with_capacity(jobs.len());
        let mut all = Batch::default();
        for j in jobs {
            txs.push(j.tx);
            all.append(j.batch);
        }
        let rows = all.rows();
        RUN_PROF[0].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        match run_batch(&mut s1, &mut s2, c, all) {
            Ok(o) => {
                let t2 = std::time::Instant::now();
                RUN_PROF[4].fetch_add(1, Relaxed);
                RUN_PROF[5].fetch_add(rows as u64, Relaxed);
                let mut v1 = o.v1.into_iter();
                let mut h = o.h.into_iter();
                let mut lg = o.logit.into_iter();
                let mut va = o.value.into_iter();
                let mut v2 = o.v2.into_iter();
                for (tx, &(m1, m2)) in txs.iter().zip(&sizes) {
                    let part = Outs {
                        v1: v1.by_ref().take(m1).collect(),
                        h: h.by_ref().take(m1).collect(),
                        logit: lg.by_ref().take(m2).collect(),
                        value: va.by_ref().take(m2).collect(),
                        v2: v2.by_ref().take(m2).collect(),
                    };
                    let _ = tx.send(Ok(part));
                }
                RUN_PROF[2].fetch_add(t2.elapsed().as_nanos() as u64, Relaxed);
            }
            Err(err) => {
                for tx in txs {
                    let _ = tx.send(Err(anyhow::anyhow!("{err}")));
                }
            }
        }
    }
}

impl OrtLatticeEval {
    /// `dir` holds `tier1.onnx`, `tier2.onnx` and `lattice.json`.
    pub fn load(dir: &Path, ep: &str, fp16: bool, runners: usize) -> Result<Self> {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("lattice.json"))?)?;
        let c = meta["c_trunk"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("lattice.json has no c_trunk"))?
            as usize;
        let p1 = |b: usize| format!("ridx:{b}x{MAX_ACTIVE},rscal:{b}x{N_SCALARS}");
        let p2 = |b: usize, t: usize| {
            format!("h:{b}x{c}x7x7,nidx:{b}x{MAX_ACTIVE},nscal:{b}x{N_SCALARS},kid:{t}")
        };
        let build = |name: &str, min: String, opt: String, max: String| -> Result<Session> {
            let p = dir.join(name);
            let cache = p.with_extension("trt_cache").to_string_lossy().into_owned();
            if ep == "trt" {
                // One engine for every shape. The CUDA EP returned wrong
                // values (NaN, or another row's) for some shapes of the
                // one-graph export whose inputs were fine on the CPU EP.
                crate::ort_net::build_trt_profile(&p, fp16, &cache, &min, &opt, &max)
            } else {
                crate::ort_net::build(&p, ep, fp16, &cache)
            }
        };
        let queue = Arc::new(Queue::default());
        let mut hs = Vec::new();
        for _ in 0..runners.max(1) {
            let s1 = build("tier1.onnx", p1(1), p1(128), p1(MAX_ROWS))?;
            let s2 = build(
                "tier2.onnx",
                p2(1, 1),
                p2(1024, 32 * 1024),
                p2(MAX_ROWS, MAX_TOTAL_KIDS),
            )?;
            let q = queue.clone();
            hs.push(std::thread::spawn(move || runner(s1, s2, c, &q)));
        }
        Ok(OrtLatticeEval { queue, runners: hs })
    }

    fn run(&self, batch: Batch) -> Result<Outs> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        {
            let mut g = self.queue.q.lock().unwrap_or_else(|p| p.into_inner());
            g.push_back(Job { batch, tx });
        }
        self.queue.cv.notify_all();
        rx.recv()
            .map_err(|_| anyhow::anyhow!("lattice runner gone"))?
    }
}

impl LatticeEval for OrtLatticeEval {
    fn both(
        &self,
        reqs: &[ExpandReq],
        states: &[&StepState],
    ) -> Result<(Vec<ExpandOut>, Vec<ValueOut>)> {
        let mut b = Batch::default();
        // Tier-1 rows: the values asked for, then each turn start an
        // expansion needs a trunk for (once per turn start).
        let live: Vec<usize> = (0..states.len())
            .filter(|&i| !states[i].is_terminal())
            .collect();
        for &i in &live {
            push_view(&states[i].view(), &mut b.ridx, &mut b.rscal);
            b.n1 += 1;
        }
        let mut root_row: HashMap<*const StepState, usize> = HashMap::new();
        let mut computed = Vec::with_capacity(reqs.len());
        for r in reqs {
            let src = match &r.trunk {
                Some(t) => {
                    computed.push(None);
                    TrunkSrc::Given(t.clone())
                }
                None => {
                    let row = *root_row
                        .entry(r.root as *const StepState)
                        .or_insert_with(|| {
                            push_view(&r.root.view(), &mut b.ridx, &mut b.rscal);
                            b.n1 += 1;
                            b.n1 - 1
                        });
                    computed.push(Some(row));
                    TrunkSrc::Row(row)
                }
            };
            b.trunk.push(src);
            push_view(&t2_view(r.node), &mut b.nidx, &mut b.nscal);
            b.kids.push(
                r.slots
                    .iter()
                    .map(|&(c, s)| c as i32 * SLOTS as i32 + i32::from(s))
                    .collect(),
            );
        }
        let o = if b.rows() == 0 {
            Outs::default()
        } else {
            self.run(b)?
        };
        let mut lg = o.logit.into_iter();
        let mut va = o.value.into_iter();
        let out = (0..reqs.len())
            .map(|i| ExpandOut {
                logits: lg.next().expect("row"),
                values: va.next().expect("row"),
                value: o.v2[i],
                trunk: computed[i].map(|r| o.h[r].clone()),
            })
            .collect();
        let mut vals: Vec<ValueOut> = states
            .iter()
            .map(|s| ValueOut {
                v: s.terminal_value(),
                trunk: None,
            })
            .collect();
        for (j, &i) in live.iter().enumerate() {
            vals[i] = ValueOut {
                v: o.v1[j],
                trunk: Some(o.h[j].clone()),
            };
        }
        Ok((out, vals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LATTICE_DIR`'s tier-1 value matches `STEPS_ONNX` (the dense steps
    /// net its trunk came from) on turn starts; expansions are finite and
    /// the same with a cached trunk as with one computed in the call, alone
    /// or batched.
    /// Tier-1 search on random openings next to the greedy steps turn:
    /// root children's visits, search values and static values (all for the
    /// root's mover), and the greedy turn's static value.
    #[test]
    #[ignore]
    fn lattice_search_diag() {
        let (Ok(lat), Ok(steps)) = (std::env::var("LATTICE_DIR"), std::env::var("STEPS_ONNX"))
        else {
            eprintln!("set LATTICE_DIR and STEPS_ONNX");
            return;
        };
        use crate::bfs::Evaluator;
        use rand::SeedableRng;
        let le = OrtLatticeEval::load(Path::new(&lat), "trt", true, 1).unwrap();
        let se = crate::bfs::OnnxEval::load(Path::new(&steps), "trt", true, "/tmp/steps_trt", 1)
            .unwrap();
        let sv = |s: &StepState| -> f32 {
            if s.is_terminal() {
                s.terminal_value()
            } else {
                se.eval(&[s.view()]).unwrap()[0]
            }
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        for g in 0..4 {
            let real = crate::selfplay::random_opening(22, 1.0884, &mut rng).0;
            let start = StepState::begin(&real, g);
            if start.is_terminal() || crate::lattice::necro_kill(&start).is_ok() {
                continue;
            }
            let v0 = sv(&start);
            let mut t = crate::tier1::Tier1::new(start.clone(), Default::default());
            t.run_inflight(&le, 400, 16).unwrap();
            let gr = crate::stepsp::greedy_turn(&real, &se, g).unwrap();
            let gs = StepState::begin(&gr, g);
            let gv = 1.0 - sv(&gs);
            eprintln!(
                "position {g}: static {v0:.3}, search {:.3}, {} root children, {} nodes; greedy turn static {gv:.3}",
                t.root_value(),
                t.nodes[0].children.len(),
                t.nodes.len()
            );
            let mut kids = t.nodes[0].children.clone();
            kids.sort_by_key(|&c| std::cmp::Reverse(t.nodes[c as usize].n));
            for c in kids.iter().take(12) {
                let nd = &t.nodes[*c as usize];
                eprintln!(
                    "  n {:4} q {:.3} static {:.3} kids {}",
                    nd.n,
                    1.0 - nd.q(),
                    1.0 - sv(&nd.state),
                    nd.children.len()
                );
            }
            let best = kids
                .iter()
                .map(|&c| 1.0 - sv(&t.nodes[c as usize].state))
                .fold(f32::MIN, f32::max);
            eprintln!("  best static among root children {best:.3}");
        }
    }

    #[test]
    #[ignore]
    fn lattice_onnx_parity() {
        let (Ok(lat), Ok(steps)) = (std::env::var("LATTICE_DIR"), std::env::var("STEPS_ONNX"))
        else {
            eprintln!("set LATTICE_DIR and STEPS_ONNX");
            return;
        };
        let le = OrtLatticeEval::load(Path::new(&lat), "trt", true, 1).unwrap();
        let se = crate::bfs::OnnxEval::load(Path::new(&steps), "cuda", false, "/tmp/steps_trt", 1)
            .unwrap();
        let mut s = StepState::begin(&crate::game::MinionsState::with_bid(3), 1);
        let mut states = Vec::new();
        for _ in 0..6 {
            states.push(s.clone());
            let mut l = crate::lattice::Lattice::new(s.clone()).unwrap();
            if l.terminal.is_some() {
                break;
            }
            let (sink, _, _) = l
                .descend(&le, crate::lattice::Cursor::at(0, 0.0), 1.5)
                .unwrap();
            assert!(l.trunk.is_some(), "the descent never got a trunk");
            s = l.nodes[sink as usize].state.clone();
            if s.is_terminal() {
                break;
            }
        }
        let refs: Vec<&StepState> = states.iter().collect();
        let a = le.value(&refs).unwrap();
        let views: Vec<View> = states.iter().map(|s| s.view()).collect();
        use crate::bfs::Evaluator;
        let b = se.eval(&views).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 2e-3, "{a:?} vs {b:?}");
        }
        eprintln!("tier-1 values {a:?} match");
        let mut l = crate::lattice::Lattice::new(states[0].clone()).unwrap();
        let mut cur = crate::lattice::Cursor::at(0, 0.0);
        if let crate::lattice::Need::Expand(n) = l.advance(&mut cur, 1.5).unwrap() {
            let (slots, node) = l.request(n);
            let req = |trunk| ExpandReq {
                root: &l.start,
                node,
                slots: &slots,
                trunk,
            };
            let fresh = le.expand(&[req(None)]).unwrap().pop().unwrap();
            let trunk = fresh.trunk.clone().expect("computed trunk");
            let cached = le
                .expand(&[req(Some(trunk.clone()))])
                .unwrap()
                .pop()
                .unwrap();
            assert!(fresh
                .logits
                .iter()
                .chain(&fresh.values)
                .all(|x| x.is_finite()));
            for (x, y) in fresh.logits.iter().zip(&cached.logits) {
                assert!(
                    (x - y).abs() < 1e-3,
                    "cached trunk differs: {:?} vs {:?}",
                    fresh.logits,
                    cached.logits
                );
            }
            let reqs: Vec<ExpandReq> = (0..5)
                .map(|i| {
                    req(if i % 2 == 0 {
                        None
                    } else {
                        Some(trunk.clone())
                    })
                })
                .collect();
            for bsz in [1usize, 2, 5] {
                let (outs, vals) = le.both(&reqs[..bsz], &refs).unwrap();
                for o in &outs {
                    for (x, y) in o.logits.iter().zip(&fresh.logits) {
                        assert!((x - y).abs() < 1e-3, "batch {bsz} differs");
                    }
                }
                assert!((vals[0].v - a[0]).abs() < 2e-3);
            }
            eprintln!(
                "logits {:?} values {:?} v {}",
                fresh.logits, fresh.values, fresh.value
            );
        }
    }
}
