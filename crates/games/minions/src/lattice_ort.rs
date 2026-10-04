//! The turn-lattice nets under ONNX Runtime: `lattice.onnx` from
//! `python/minions_lattice_learner.py`, one graph answering both tiers'
//! requests. A row is a turn start (tier-1 input), a lattice node and the
//! node's children as `cell * 37 + slot`; the graph returns the turn start's
//! tier-1 value, the children's prior logits and values, and the node's
//! value. Tier-1 value requests are rows whose node is the turn start itself
//! with one dummy child. Rows sharing a turn start recompute its trunk.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use anyhow::Result;
use ort::session::Session;

use crate::latgen::t2_view;
use crate::lattice::{ExpandOut, ExpandReq, LatticeEval};
use crate::steps::{StepState, View, MAX_ACTIVE, N_SCALARS, SPARSE_DIM};

pub const SLOTS: usize = 37;
/// The TensorRT profile's limits: rows per call and children per row.
pub const MAX_ROWS: usize = 4096;
pub const MAX_KIDS: usize = 512;

/// Runner profile (ns): merging rows, inside `Session::run` (inputs and
/// outputs included), splitting the outputs; and the kids padded in.
pub static RUN_PROF: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Per row: tier-1 value, children's logits and values, node value.
type Outs = (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>);

struct Job {
    rows: Rows,
    tx: std::sync::mpsc::SyncSender<Result<Outs>>,
}

#[derive(Default)]
struct Queue {
    q: Mutex<std::collections::VecDeque<Job>>,
    cv: std::sync::Condvar,
    closed: std::sync::atomic::AtomicBool,
}

/// Workers queue their rows; `runners` threads, each with its own session,
/// merge everything queued (up to the profile's limits) into one call.
/// Fewer, larger calls: concurrent sessions contend on the GPU (2 sessions
/// beat 8 on direct calls).
pub struct OrtLatticeEval {
    queue: std::sync::Arc<Queue>,
    pub calls: std::sync::Arc<AtomicU64>,
    pub rows: std::sync::Arc<AtomicU64>,
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

struct Rows {
    ridx: Vec<i32>,
    rscal: Vec<f32>,
    nidx: Vec<i32>,
    nscal: Vec<f32>,
    kids: Vec<Vec<i32>>,
}

impl Rows {
    fn new(n: usize) -> Self {
        Rows {
            ridx: Vec::with_capacity(n * MAX_ACTIVE),
            rscal: Vec::with_capacity(n * N_SCALARS),
            nidx: Vec::with_capacity(n * MAX_ACTIVE),
            nscal: Vec::with_capacity(n * N_SCALARS),
            kids: Vec::with_capacity(n),
        }
    }

    fn push_view(v: &View, idx: &mut Vec<i32>, scal: &mut Vec<f32>) {
        let mut act = Vec::with_capacity(MAX_ACTIVE);
        let mut sc = [0f32; N_SCALARS];
        v.features(&mut act, &mut sc);
        for i in 0..MAX_ACTIVE {
            idx.push(act.get(i).map_or(SPARSE_DIM as i32, |&x| i32::from(x)));
        }
        scal.extend_from_slice(&sc);
    }

    fn push(&mut self, root: &View, node: &View, kids: Vec<i32>) {
        Self::push_view(root, &mut self.ridx, &mut self.rscal);
        Self::push_view(node, &mut self.nidx, &mut self.nscal);
        self.kids.push(kids);
    }

    fn append(&mut self, o: Rows) {
        self.ridx.extend(o.ridx);
        self.rscal.extend(o.rscal);
        self.nidx.extend(o.nidx);
        self.nscal.extend(o.nscal);
        self.kids.extend(o.kids);
    }
}

fn run_session(sess: &mut Session, rows: Rows) -> Result<Outs> {
    use ort::value::Tensor;
    let e = |e: ort::Error| anyhow::anyhow!("{e}");
    let b = rows.kids.len();
    let k = rows.kids.iter().map(|x| x.len()).max().unwrap_or(1).max(1);
    anyhow::ensure!(
        b <= MAX_ROWS && k <= MAX_KIDS,
        "lattice batch {b} x {k} over the profile"
    );
    let mut kid = vec![0i32; b * k];
    for (i, ks) in rows.kids.iter().enumerate() {
        kid[i * k..i * k + ks.len()].copy_from_slice(ks);
    }
    let (bi, ai, si) = (b as i64, MAX_ACTIVE as i64, N_SCALARS as i64);
    let inputs = ort::inputs![
        "ridx" => Tensor::from_array((vec![bi, ai], rows.ridx)).map_err(e)?,
        "rscal" => Tensor::from_array((vec![bi, si], rows.rscal)).map_err(e)?,
        "nidx" => Tensor::from_array((vec![bi, ai], rows.nidx)).map_err(e)?,
        "nscal" => Tensor::from_array((vec![bi, si], rows.nscal)).map_err(e)?,
        "kid" => Tensor::from_array((vec![bi, k as i64], kid)).map_err(e)?,
    ];
    let outs = sess.run(inputs).map_err(e)?;
    let get = |name: &str| -> Result<Vec<f32>> {
        let (_, v) = outs[name].try_extract_tensor::<f32>().map_err(e)?;
        Ok(v.to_vec())
    };
    let (v1, logit, value, v2) = (get("v1")?, get("logit")?, get("value")?, get("v2")?);
    let per = |x: &[f32]| -> Vec<Vec<f32>> {
        rows.kids
            .iter()
            .enumerate()
            .map(|(i, ks)| x[i * k..i * k + ks.len()].to_vec())
            .collect()
    };
    Ok((v1, per(&logit), per(&value), v2))
}

/// Rows a call waits for (`LATTICE_MIN_ROWS`, default 1024) and the
/// longest it waits (`LATTICE_WAIT_US`, default 2000).
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
            .unwrap_or(2000)
    }))
}

fn runner(mut sess: Session, q: &Queue, calls: &AtomicU64, rows_n: &AtomicU64) {
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
            // Dynamic batching: a call costs ~1 ms however small (570k rows/s
            // at 1024 rows, 60k/s at 64), so wait a little for more rows.
            let deadline = std::time::Instant::now() + batch_wait();
            loop {
                let queued: usize = g.iter().map(|j| j.rows.kids.len()).sum();
                let now = std::time::Instant::now();
                if queued >= min_rows() || now >= deadline {
                    break;
                }
                g =
                    q.cv.wait_timeout(g, deadline - now)
                        .unwrap_or_else(|p| p.into_inner())
                        .0;
            }
            let mut total = 0;
            while let Some(j) = g.front() {
                let n = j.rows.kids.len();
                if !jobs.is_empty() && total + n > MAX_ROWS {
                    break;
                }
                total += n;
                jobs.push(g.pop_front().expect("front"));
            }
        }
        let t0 = std::time::Instant::now();
        let sizes: Vec<usize> = jobs.iter().map(|j| j.rows.kids.len()).collect();
        let mut txs = Vec::with_capacity(jobs.len());
        let mut all = Rows::new(sizes.iter().sum());
        for j in jobs {
            txs.push(j.tx);
            all.append(j.rows);
        }
        let n = all.kids.len();
        let kmax = all.kids.iter().map(|k| k.len()).max().unwrap_or(1);
        let ksum: usize = all.kids.iter().map(|k| k.len()).sum();
        RUN_PROF[3].fetch_add((n * kmax - ksum) as u64, Relaxed);
        RUN_PROF[4].fetch_add(1, Relaxed);
        RUN_PROF[5].fetch_add(n as u64, Relaxed);
        let t1 = std::time::Instant::now();
        RUN_PROF[0].fetch_add((t1 - t0).as_nanos() as u64, Relaxed);
        let r = run_session(&mut sess, all);
        let t2 = std::time::Instant::now();
        RUN_PROF[1].fetch_add((t2 - t1).as_nanos() as u64, Relaxed);
        match r {
            Ok((v1, lg, va, v2)) => {
                calls.fetch_add(1, Relaxed);
                rows_n.fetch_add(n as u64, Relaxed);
                let (mut v1, mut lg, mut va, mut v2) = (
                    v1.into_iter(),
                    lg.into_iter(),
                    va.into_iter(),
                    v2.into_iter(),
                );
                for (tx, &m) in txs.iter().zip(&sizes) {
                    let part = (
                        v1.by_ref().take(m).collect(),
                        lg.by_ref().take(m).collect(),
                        va.by_ref().take(m).collect(),
                        v2.by_ref().take(m).collect(),
                    );
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
    pub fn load(path: &Path, ep: &str, fp16: bool, cache: &str, runners: usize) -> Result<Self> {
        // TensorRT: one engine for every (rows, children) shape. The CUDA EP
        // returned wrong values (NaN, or another row's) for some shapes of
        // this graph whose inputs ran fine on the CPU EP and under TensorRT.
        let prof = |b: usize, k: usize| {
            format!("ridx:{b}x{MAX_ACTIVE},rscal:{b}x{N_SCALARS},nidx:{b}x{MAX_ACTIVE},nscal:{b}x{N_SCALARS},kid:{b}x{k}")
        };
        let queue = std::sync::Arc::new(Queue::default());
        let calls = std::sync::Arc::new(AtomicU64::new(0));
        let rows = std::sync::Arc::new(AtomicU64::new(0));
        let mut hs = Vec::new();
        for _ in 0..runners.max(1) {
            let sess = if ep == "trt" {
                crate::ort_net::build_trt_profile(
                    path,
                    fp16,
                    cache,
                    &prof(1, 1),
                    &prof(512, 32),
                    &prof(MAX_ROWS, MAX_KIDS),
                )?
            } else {
                crate::ort_net::build(path, ep, fp16, cache)?
            };
            let (q, c, r) = (queue.clone(), calls.clone(), rows.clone());
            hs.push(std::thread::spawn(move || runner(sess, &q, &c, &r)));
        }
        Ok(OrtLatticeEval {
            queue,
            calls,
            rows,
            runners: hs,
        })
    }

    fn run(&self, rows: Rows) -> Result<Outs> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        {
            let mut g = self.queue.q.lock().unwrap_or_else(|p| p.into_inner());
            g.push_back(Job { rows, tx });
        }
        self.queue.cv.notify_all();
        rx.recv()
            .map_err(|_| anyhow::anyhow!("lattice runner gone"))?
    }
}

impl LatticeEval for OrtLatticeEval {
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
        let mut rows = Rows::new(reqs.len() + states.len());
        for r in reqs {
            let kids = r
                .slots
                .iter()
                .map(|&(c, s)| c as i32 * SLOTS as i32 + i32::from(s))
                .collect();
            rows.push(&r.root.view(), &t2_view(r.node), kids);
        }
        let live: Vec<usize> = (0..states.len())
            .filter(|&i| !states[i].is_terminal())
            .collect();
        for &i in &live {
            let v = states[i].view();
            rows.push(&v, &t2_view(states[i]), vec![0]);
        }
        if rows.kids.is_empty() {
            return Ok((
                Vec::new(),
                states.iter().map(|s| s.terminal_value()).collect(),
            ));
        }
        let (v1, logit, value, v2) = self.run(rows)?;
        let mut logit = logit.into_iter();
        let mut value = value.into_iter();
        let out = (0..reqs.len())
            .map(|i| ExpandOut {
                logits: logit.next().expect("row"),
                values: value.next().expect("row"),
                value: v2[i],
            })
            .collect();
        let mut vals: Vec<f32> = states.iter().map(|s| s.terminal_value()).collect();
        for (j, &i) in live.iter().enumerate() {
            vals[i] = v1[reqs.len() + j];
        }
        Ok((out, vals))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LATTICE_ONNX`'s tier-1 value matches `STEPS_ONNX` (the steps net its
    /// trunk came from) on turn starts, and its outputs are well formed.
    #[test]
    #[ignore]
    fn lattice_onnx_parity() {
        let (Ok(lat), Ok(steps)) = (std::env::var("LATTICE_ONNX"), std::env::var("STEPS_ONNX"))
        else {
            eprintln!("set LATTICE_ONNX and STEPS_ONNX");
            return;
        };
        let le = OrtLatticeEval::load(
            Path::new(&lat),
            &std::env::var("LATTICE_EP").unwrap_or("cuda".into()),
            true,
            "/tmp/lat_trt",
            1,
        )
        .unwrap();
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
            assert!((x - y).abs() < 1e-3, "{a:?} vs {b:?}");
        }
        eprintln!("tier-1 values {a:?} match");
        // Expansion outputs are finite.
        let mut l = crate::lattice::Lattice::new(states[0].clone()).unwrap();
        let mut cur = crate::lattice::Cursor::at(0, 0.0);
        if let crate::lattice::Need::Expand(n) = l.advance(&mut cur, 1.5).unwrap() {
            let (slots, node) = l.request(n);
            let req = ExpandReq {
                root: &l.start,
                node,
                slots: &slots,
            };
            let out = le.expand(&[req]).unwrap().pop().unwrap();
            eprintln!(
                "slots {:?}\nlogits {:?}\nvalues {:?}\nv {}",
                slots, out.logits, out.values, out.value
            );
            assert!(out.logits.iter().chain(&out.values).all(|x| x.is_finite()));
            // A mixed batch: several expansions and values in one call.
            let reqs: Vec<ExpandReq> = (0..5)
                .map(|_| ExpandReq {
                    root: &l.start,
                    node,
                    slots: &slots,
                })
                .collect();
            for b in [1usize, 2, 3, 5] {
                let (outs, vals) = le.both(&reqs[..b], &refs).unwrap();
                eprintln!(
                    "batch {b}: v2 {:?} vals {vals:?} logits0 {:?}",
                    outs.iter().map(|o| o.value).collect::<Vec<_>>(),
                    outs[0].logits
                );
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// Session time per call against rows (K = 32 kids): fixed overhead or
    /// per-row cost? `LATTICE_ONNX` must be set.
    #[test]
    #[ignore]
    fn lattice_call_scaling() {
        let Ok(lat) = std::env::var("LATTICE_ONNX") else {
            return;
        };
        let p = Path::new(&lat);
        let cache = p.with_extension("trt_cache").to_string_lossy().into_owned();
        let prof = |b: usize, k: usize| {
            format!("ridx:{b}x{MAX_ACTIVE},rscal:{b}x{N_SCALARS},nidx:{b}x{MAX_ACTIVE},nscal:{b}x{N_SCALARS},kid:{b}x{k}")
        };
        let mut sess = crate::ort_net::build_trt_profile(
            p,
            true,
            &cache,
            &prof(1, 1),
            &prof(512, 32),
            &prof(MAX_ROWS, MAX_KIDS),
        )
        .unwrap();
        let s = StepState::begin(&crate::game::MinionsState::with_bid(3), 1);
        let v = s.view();
        for &(b, k) in &[
            (64, 32),
            (256, 32),
            (1024, 32),
            (4096, 32),
            (1024, 8),
            (1024, 128),
        ] {
            let mk = || {
                let mut rows = Rows::new(b);
                for _ in 0..b {
                    rows.push(&v, &v, (0..k as i32).collect());
                }
                rows
            };
            run_session(&mut sess, mk()).unwrap();
            let t0 = std::time::Instant::now();
            let reps = 20;
            for _ in 0..reps {
                run_session(&mut sess, mk()).unwrap();
            }
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            eprintln!(
                "rows {b:5} kids {k:3}: {:.2} ms/call, {:.0} rows/s",
                1e3 * dt,
                b as f64 / dt
            );
        }
        // Shapes varying call to call, as in generation.
        let mut rng = 12345u64;
        let mut next = |m: usize| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as usize % m
        };
        let t0 = std::time::Instant::now();
        let mut total = 0;
        for _ in 0..40 {
            let b = 400 + next(120);
            let k = 60 + next(120);
            total += b;
            let mut rows = Rows::new(b);
            for _ in 0..b {
                rows.push(&v, &v, (0..k as i32).collect());
            }
            run_session(&mut sess, rows).unwrap();
        }
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "varying shapes: {:.2} ms/call, {:.0} rows/s",
            1e3 * dt / 40.0,
            total as f64 / dt
        );
    }
}
