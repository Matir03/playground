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
use crate::lattice::{ExpandOut, ExpandReq, LatticeEval, Pending, Trunk, ValueOut};
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
/// calls; rows; the tier-2 inputs' assembly (kid indices, trunks) and
/// the tier-2 `Session::run` alone.
pub static RUN_PROF: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

fn push_view(v: &View, idx: &mut Vec<i32>, scal: &mut Vec<f32>) {
    let mut act = Vec::with_capacity(MAX_ACTIVE);
    let mut sc = [0f32; N_SCALARS];
    v.features(&mut act, &mut sc);
    push_act(&act, &sc, idx, scal);
}

fn push_act(act: &[u16], sc: &[f32; N_SCALARS], idx: &mut Vec<i32>, scal: &mut Vec<f32>) {
    for i in 0..MAX_ACTIVE {
        idx.push(act.get(i).map_or(SPARSE_DIM as i32, |&x| i32::from(x)));
    }
    scal.extend_from_slice(sc);
}

/// A lattice node's (or turn start's) features in the lattice view.
fn push_lview(s: &StepState, idx: &mut Vec<i32>, scal: &mut Vec<f32>) {
    let mut act = Vec::with_capacity(MAX_ACTIVE);
    let mut sc = [0f32; N_SCALARS];
    crate::lview::node_features(s, &mut act, &mut sc);
    push_act(&act, &sc, idx, scal);
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
    /// Output slots per cell (37 for the steps-view nets, `lview::NET_SLOTS`
    /// for lattice-view ones).
    slots: usize,
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
        if self.slots == 0 {
            self.slots = o.slots;
        }
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
    /// Per tier-2 row (NaN when the net has no energy output).
    energy: Vec<f32>,
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
    /// If positive (lattice.json `value_prior_tau`), the children's prior
    /// logits are their values over this temperature: the prior head is not
    /// used.
    value_prior_tau: f32,
    /// lattice.json `prior_temp` (default 1): the prior logits are divided
    /// by it (> 1 flattens the search's priors).
    prior_temp: f32,
    /// lattice.json `lview`: the net reads [`crate::lview`] features and has
    /// `lview::NET_SLOTS` output slots per cell.
    lview: bool,
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

/// `gather`: `tier2.onnx` takes the call's distinct trunk images (`hs`) and
/// each row's index into them (`hidx`) rather than an image per row (`h`;
/// nets exported before 2026-10-05).
fn run_batch(s1: &mut Session, s2: &mut Session, c: usize, gather: bool, b: Batch) -> Result<Outs> {
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
        let t_prep = std::time::Instant::now();
        // Children flattened over the batch: row * 49*37 + cell * 37 + slot.
        let per_row = (CELLS * if b.slots > 0 { b.slots } else { SLOTS }) as i32;
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
        let (n, a, sc) = (n2 as i64, MAX_ACTIVE as i64, N_SCALARS as i64);
        let mut h = Vec::with_capacity(if gather { 64 * hw } else { n2 * hw });
        let mut hidx: Vec<i32> = Vec::with_capacity(n2);
        if gather {
            // One image per distinct trunk: a given trunk by its allocation
            // (a lattice hands the same one to all its expansions), a
            // computed one by its tier-1 row.
            let mut seen: HashMap<(bool, usize), i32> = HashMap::new();
            for tr in &b.trunk {
                let (key, img): ((bool, usize), &[f16]) = match tr {
                    TrunkSrc::Given(x) => ((false, x.as_ptr() as usize), x),
                    TrunkSrc::Row(i) => ((true, *i), &out.h[*i]),
                };
                let u = *seen.entry(key).or_insert_with(|| {
                    h.extend_from_slice(img);
                    (h.len() / hw - 1) as i32
                });
                hidx.push(u);
            }
        } else {
            for tr in &b.trunk {
                match tr {
                    TrunkSrc::Given(x) => h.extend_from_slice(x),
                    TrunkSrc::Row(i) => h.extend_from_slice(&out.h[*i]),
                }
            }
        }
        let nh = (h.len() / hw) as i64;
        RUN_PROF[6].fetch_add(t_prep.elapsed().as_nanos() as u64, Relaxed);
        let t_run = std::time::Instant::now();
        let outs = if gather {
            s2.run(ort::inputs![
                "hs" => Tensor::from_array((vec![nh, c as i64, 7, 7], h)).map_err(e)?,
                "hidx" => Tensor::from_array((vec![n], hidx)).map_err(e)?,
                "nidx" => Tensor::from_array((vec![n, a], b.nidx)).map_err(e)?,
                "nscal" => Tensor::from_array((vec![n, sc], b.nscal)).map_err(e)?,
                "kid" => Tensor::from_array((vec![t as i64], kid)).map_err(e)?,
            ])
        } else {
            s2.run(ort::inputs![
                "h" => Tensor::from_array((vec![n, c as i64, 7, 7], h)).map_err(e)?,
                "nidx" => Tensor::from_array((vec![n, a], b.nidx)).map_err(e)?,
                "nscal" => Tensor::from_array((vec![n, sc], b.nscal)).map_err(e)?,
                "kid" => Tensor::from_array((vec![t as i64], kid)).map_err(e)?,
            ])
        }
        .map_err(e)?;
        RUN_PROF[7].fetch_add(t_run.elapsed().as_nanos() as u64, Relaxed);
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
        out.energy = if outs.contains_key("energy") {
            get("energy")?
        } else {
            vec![f32::NAN; v2.len()]
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

fn runner(mut s1: Session, mut s2: Session, c: usize, gather: bool, q: &Queue) {
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
        match run_batch(&mut s1, &mut s2, c, gather, all) {
            Ok(o) => {
                let t2 = std::time::Instant::now();
                RUN_PROF[4].fetch_add(1, Relaxed);
                RUN_PROF[5].fetch_add(rows as u64, Relaxed);
                let mut v1 = o.v1.into_iter();
                let mut h = o.h.into_iter();
                let mut lg = o.logit.into_iter();
                let mut va = o.value.into_iter();
                let mut v2 = o.v2.into_iter();
                let mut en = o.energy.into_iter();
                for (tx, &(m1, m2)) in txs.iter().zip(&sizes) {
                    let part = Outs {
                        v1: v1.by_ref().take(m1).collect(),
                        h: h.by_ref().take(m1).collect(),
                        logit: lg.by_ref().take(m2).collect(),
                        value: va.by_ref().take(m2).collect(),
                        v2: v2.by_ref().take(m2).collect(),
                        energy: en.by_ref().take(m2).collect(),
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
    /// Override lattice.json's tier-2 prior settings: `temp` divides the
    /// prior head's logits (infinite: uniform priors within each variable),
    /// `value_tau` > 0 takes the kids' values over it instead.
    pub fn set_t2_prior(&mut self, temp: Option<f32>, value_tau: Option<f32>) {
        if let Some(t) = temp {
            self.prior_temp = t;
        }
        if let Some(v) = value_tau {
            self.value_prior_tau = v;
        }
    }

    /// `dir` holds `tier1.onnx`, `tier2.onnx` and `lattice.json`.
    pub fn load(dir: &Path, ep: &str, fp16: bool, runners: usize) -> Result<Self> {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("lattice.json"))?)?;
        let c = meta["c_trunk"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("lattice.json has no c_trunk"))?
            as usize;
        let gather = meta["trunk_gather"].as_bool().unwrap_or(false);
        let value_prior_tau = meta["value_prior_tau"].as_f64().unwrap_or(0.0) as f32;
        let prior_temp = meta["prior_temp"].as_f64().unwrap_or(1.0) as f32;
        let lview = meta["lview"].as_bool().unwrap_or(false);
        let slots = if lview {
            crate::lview::NET_SLOTS
        } else {
            SLOTS
        };
        let p1 = |b: usize| format!("ridx:{b}x{MAX_ACTIVE},rscal:{b}x{N_SCALARS}");
        // `u`: distinct trunk images (gather nets).
        let p2 = |b: usize, t: usize, u: usize| {
            let h = if gather {
                format!("hs:{u}x{c}x7x7,hidx:{b}")
            } else {
                format!("h:{b}x{c}x7x7")
            };
            let _ = slots;
            format!("{h},nidx:{b}x{MAX_ACTIVE},nscal:{b}x{N_SCALARS},kid:{t}")
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
                p2(1, 1, 1),
                p2(1024, 32 * 1024, 64),
                p2(MAX_ROWS, MAX_TOTAL_KIDS, MAX_ROWS),
            )?;
            let q = queue.clone();
            hs.push(std::thread::spawn(move || runner(s1, s2, c, gather, &q)));
        }
        Ok(OrtLatticeEval {
            queue,
            runners: hs,
            value_prior_tau,
            prior_temp,
            lview,
        })
    }

    /// Queue `batch` for a runner; its results arrive on the receiver.
    fn enqueue(&self, batch: Batch) -> std::sync::mpsc::Receiver<Result<Outs>> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        {
            let mut g = self.queue.q.lock().unwrap_or_else(|p| p.into_inner());
            g.push_back(Job { batch, tx });
        }
        self.queue.cv.notify_all();
        rx
    }
}

impl LatticeEval for OrtLatticeEval {
    fn both(
        &self,
        reqs: &[ExpandReq],
        states: &[&StepState],
    ) -> Result<(Vec<ExpandOut>, Vec<ValueOut>)> {
        self.submit(reqs, states)?.wait()
    }

    fn submit(&self, reqs: &[ExpandReq], states: &[&StepState]) -> Result<Pending> {
        let mut b = Batch::default();
        let lv = self.lview;
        b.slots = if lv { crate::lview::NET_SLOTS } else { SLOTS };
        // Tier-1 rows: the values asked for, then each turn start an
        // expansion needs a trunk for (once per turn start).
        let live: Vec<usize> = (0..states.len())
            .filter(|&i| !states[i].is_terminal())
            .collect();
        for &i in &live {
            if lv {
                push_lview(states[i], &mut b.ridx, &mut b.rscal);
            } else {
                push_view(&states[i].view(), &mut b.ridx, &mut b.rscal);
            }
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
                            if lv {
                                push_lview(r.root, &mut b.ridx, &mut b.rscal);
                            } else {
                                push_view(&r.root.view(), &mut b.ridx, &mut b.rscal);
                            }
                            b.n1 += 1;
                            b.n1 - 1
                        });
                    computed.push(Some(row));
                    TrunkSrc::Row(row)
                }
            };
            b.trunk.push(src);
            if lv {
                push_lview(r.node, &mut b.nidx, &mut b.nscal);
                let st = r.node.stage;
                let ns = crate::lview::NET_SLOTS as i32;
                b.kids.push(
                    r.slots
                        .iter()
                        .map(|&(c, s)| c as i32 * ns + i32::from(crate::lview::net_slot(st, s)))
                        .collect(),
                );
            } else {
                push_view(&t2_view(r.node), &mut b.nidx, &mut b.nscal);
                b.kids.push(
                    r.slots
                        .iter()
                        .map(|&(c, s)| c as i32 * SLOTS as i32 + i32::from(s))
                        .collect(),
                );
            }
        }
        let rx = (b.rows() > 0).then(|| self.enqueue(b));
        let n_reqs = reqs.len();
        let tau = self.value_prior_tau;
        let temp = self.prior_temp;
        let terminal: Vec<f32> = states.iter().map(|s| s.terminal_value()).collect();
        Ok(Pending::new(move || {
            let o = match rx {
                Some(rx) => rx
                    .recv()
                    .map_err(|_| anyhow::anyhow!("lattice runner gone"))??,
                None => Outs::default(),
            };
            Ok(unpack(o, n_reqs, &computed, &terminal, &live, tau, temp))
        }))
    }
}

/// A call's outputs as per-request results.
fn unpack(
    o: Outs,
    n_reqs: usize,
    computed: &[Option<usize>],
    terminal: &[f32],
    live: &[usize],
    value_prior_tau: f32,
    prior_temp: f32,
) -> (Vec<ExpandOut>, Vec<ValueOut>) {
    {
        let reqs_len = n_reqs;
        let mut lg = o.logit.into_iter();
        let mut va = o.value.into_iter();
        let out = (0..reqs_len)
            .map(|i| {
                let logits = lg.next().expect("row");
                let values: Vec<f32> = va.next().expect("row");
                ExpandOut {
                    logits: if value_prior_tau > 0.0 {
                        values.iter().map(|v| v / value_prior_tau).collect()
                    } else if prior_temp != 1.0 {
                        logits.iter().map(|l| l / prior_temp).collect()
                    } else {
                        logits
                    },
                    values,
                    value: o.v2[i],
                    energy: o.energy.get(i).copied().unwrap_or(f32::NAN),
                    trunk: computed[i].map(|r| o.h[r].clone()),
                }
            })
            .collect();
        let mut vals: Vec<ValueOut> = terminal
            .iter()
            .map(|&v| ValueOut { v, trunk: None })
            .collect();
        for (j, &i) in live.iter().enumerate() {
            vals[i] = ValueOut {
                v: o.v1[j],
                trunk: Some(o.h[j].clone()),
            };
        }
        (out, vals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LATTICE_DIR`'s tier-1 value matches `STEPS_ONNX` (the dense steps
    /// net its trunk came from) on turn starts; expansions are finite and
    /// the same with a cached trunk as with one computed in the call, alone
    /// or batched.
    /// Runner call cost by shape, interleaved as in self-play (synthetic
    /// inputs, trunks given). `LATTICE_DIR` names the nets.
    ///
    /// Measured 2026-10-04 on gen-6 nets (res:64x6 trunk), with the cycle
    /// running beside it: ~1.3-1.9 ms per graph call whatever the size up to
    /// ~256 rows; the same shapes back to back, or each on its own session,
    /// 0.8-1.2 ms -- a shape change costs ~0.6 ms. Padding to x2 size
    /// buckets, one session each, lost that back on mid-size calls (250
    /// tier-1 rows padded to 512: 2.5 ms vs 1.8 ms), so it was dropped.
    /// The GPU clock does not matter here: holding P0 (1.3-1.9 GHz) instead
    /// of P8 (315 MHz) left calls at ~5 ms in lattice-gen.
    #[test]
    #[ignore]
    fn runner_call_cost() {
        let Ok(dir) = std::env::var("LATTICE_DIR") else {
            eprintln!("set LATTICE_DIR");
            return;
        };
        let dir = Path::new(&dir);
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("lattice.json")).unwrap())
                .unwrap();
        let c = meta["c_trunk"].as_u64().unwrap() as usize;
        let p1 = |b: usize| format!("ridx:{b}x{MAX_ACTIVE},rscal:{b}x{N_SCALARS}");
        let p2 = |b: usize, t: usize| {
            format!("h:{b}x{c}x7x7,nidx:{b}x{MAX_ACTIVE},nscal:{b}x{N_SCALARS},kid:{t}")
        };
        let build = |name: &str, min: String, opt: String, max: String| {
            let p = dir.join(name);
            let cache = p.with_extension("trt_cache").to_string_lossy().into_owned();
            crate::ort_net::build_trt_profile(&p, true, &cache, &min, &opt, &max).unwrap()
        };
        let mut s1 = build("tier1.onnx", p1(1), p1(128), p1(MAX_ROWS));
        let mut s2 = build(
            "tier2.onnx",
            p2(1, 1),
            p2(1024, 32 * 1024),
            p2(MAX_ROWS, MAX_TOTAL_KIDS),
        );
        let trunk: Trunk = vec![half::f16::from_f32(0.1); c * CELLS].into();
        let batch = |n1: usize, n2: usize, k: usize| Batch {
            ridx: vec![SPARSE_DIM as i32; n1 * MAX_ACTIVE],
            rscal: vec![0.0; n1 * N_SCALARS],
            n1,
            trunk: (0..n2).map(|_| TrunkSrc::Given(trunk.clone())).collect(),
            nidx: vec![SPARSE_DIM as i32; n2 * MAX_ACTIVE],
            nscal: vec![0.0; n2 * N_SCALARS],
            kids: (0..n2).map(|_| (0..k as i32).collect()).collect(),
            slots: SLOTS,
        };
        let shapes: Vec<(usize, usize, usize)> = vec![
            (0, 60, 27),
            (0, 250, 27),
            (0, 1000, 27),
            (6, 0, 0),
            (60, 0, 0),
            (250, 0, 0),
            (40, 300, 27),
        ];
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); shapes.len()];
        for round in 0..32 {
            for (i, &(n1, n2, k)) in shapes.iter().enumerate() {
                let t0 = std::time::Instant::now();
                run_batch(&mut s1, &mut s2, c, false, batch(n1, n2, k)).unwrap();
                if round >= 2 {
                    times[i].push(t0.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        for (&(n1, n2, k), t) in shapes.iter().zip(&mut times) {
            t.sort_by(|a, b| a.total_cmp(b));
            eprintln!(
                "tier1 rows {n1:4} tier2 rows {n2:4} kids/row {k:3}: {:6.2} ms",
                t[t.len() / 2]
            );
        }
    }

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

    /// The same weights exported with an image per row (`LATTICE_A`) and as
    /// distinct images plus indices (`LATTICE_B`) give the same outputs, on
    /// expansions mixing cached and computed trunks across lattices.
    #[test]
    #[ignore]
    fn trunk_gather_parity() {
        let (Ok(a), Ok(b)) = (std::env::var("LATTICE_A"), std::env::var("LATTICE_B")) else {
            eprintln!("set LATTICE_A and LATTICE_B");
            return;
        };
        let ea = OrtLatticeEval::load(Path::new(&a), "trt", true, 1).unwrap();
        let eb = OrtLatticeEval::load(Path::new(&b), "trt", true, 1).unwrap();
        let mut lats = Vec::new();
        for seed in 0..6u64 {
            let s = StepState::begin(&crate::game::MinionsState::with_bid(3), seed);
            let mut l = crate::lattice::Lattice::new(s).unwrap();
            l.descend(&ea, crate::lattice::Cursor::at(0, 0.0), 1.5)
                .unwrap();
            lats.push(l);
        }
        let mut slots = Vec::new();
        let mut nodes = Vec::new();
        for l in &mut lats {
            for n in 0..l.nodes.len().min(6) as u32 {
                if l.nodes[n as usize].kind == crate::lattice::Kind::Inner {
                    slots.push(l.request(n).0);
                    nodes.push((nodes.len(), n));
                }
            }
        }
        let mut reqs = Vec::new();
        let mut k = 0;
        for (li, l) in lats.iter().enumerate() {
            for n in 0..l.nodes.len().min(6) as u32 {
                if l.nodes[n as usize].kind != crate::lattice::Kind::Inner {
                    continue;
                }
                reqs.push(ExpandReq {
                    root: &l.start,
                    node: &l.nodes[n as usize].state,
                    slots: &slots[k],
                    // Odd lattices: computed in the call.
                    trunk: if li % 2 == 0 { l.trunk.clone() } else { None },
                });
                k += 1;
            }
        }
        let (oa, _) = ea.both(&reqs, &[]).unwrap();
        let (ob, _) = eb.both(&reqs, &[]).unwrap();
        let mut worst = 0f32;
        for (x, y) in oa.iter().zip(&ob) {
            worst = worst.max((x.value - y.value).abs());
            for (p, q) in x.logits.iter().zip(&y.logits) {
                worst = worst.max((p - q).abs());
            }
            for (p, q) in x.values.iter().zip(&y.values) {
                worst = worst.max((p - q).abs());
            }
        }
        eprintln!("{} expansions, worst difference {worst}", oa.len());
        assert!(worst < 1e-2, "worst difference {worst}");
    }

    /// Two nets (`LATTICE_A`, `LATTICE_B`; either view) on the same lattice
    /// nodes from random openings: mean per-cell prior total variation,
    /// mean |child value| and |node value| differences, and tier-1 values.
    #[test]
    #[ignore]
    fn view_parity() {
        let (Ok(a), Ok(b)) = (std::env::var("LATTICE_A"), std::env::var("LATTICE_B")) else {
            eprintln!("set LATTICE_A and LATTICE_B");
            return;
        };
        use rand::SeedableRng;
        let ea = OrtLatticeEval::load(Path::new(&a), "trt", true, 1).unwrap();
        let eb = OrtLatticeEval::load(Path::new(&b), "trt", true, 1).unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(9);
        let (mut tv, mut ncell, mut dv, mut nk, mut dn, mut nn, mut d1, mut n1) =
            (0f64, 0usize, 0f64, 0usize, 0f64, 0usize, 0f64, 0usize);
        let mut by_stage = [(0f64, 0usize); 3];
        for g in 0..24u64 {
            let real = crate::selfplay::random_opening(22, 1.0884, &mut rng).0;
            let start = StepState::begin(&real, g);
            if start.is_terminal() || crate::lattice::necro_kill(&start).is_ok() {
                continue;
            }
            let mut l = crate::lattice::Lattice::new(start.clone()).unwrap();
            for _ in 0..40 {
                l.descend(&ea, crate::lattice::Cursor::at(0, 0.0), 1.5)
                    .unwrap();
            }
            let (_, va) = ea.both(&[], &[&start]).unwrap();
            let (_, vb) = eb.both(&[], &[&start]).unwrap();
            d1 += f64::from((va[0].v - vb[0].v).abs());
            n1 += 1;
            let ids: Vec<u32> = (0..l.nodes.len() as u32)
                .filter(|&n| l.nodes[n as usize].kind == crate::lattice::Kind::Inner)
                .take(64)
                .collect();
            let slots: Vec<_> = ids.iter().map(|&n| l.request(n).0).collect();
            let reqs: Vec<ExpandReq> = ids
                .iter()
                .zip(&slots)
                .map(|(&n, sl)| ExpandReq {
                    root: &l.start,
                    node: &l.nodes[n as usize].state,
                    slots: sl,
                    trunk: None,
                })
                .collect();
            let (oa, _) = ea.both(&reqs, &[]).unwrap();
            let (ob, _) = eb.both(&reqs, &[]).unwrap();
            for ((x, y), (sl, r)) in oa.iter().zip(&ob).zip(slots.iter().zip(&reqs)) {
                dn += f64::from((x.value - y.value).abs());
                nn += 1;
                for (p, q) in x.values.iter().zip(&y.values) {
                    dv += f64::from((p - q).abs());
                    nk += 1;
                }
                let mut cells: Vec<u8> = sl.iter().map(|&(c, _)| c).collect();
                cells.sort();
                cells.dedup();
                for c in cells {
                    let ix: Vec<usize> = (0..sl.len()).filter(|&i| sl[i].0 == c).collect();
                    if ix.len() < 2 {
                        continue;
                    }
                    let sm = |lg: &[f32]| {
                        let m = ix.iter().map(|&i| lg[i]).fold(f32::MIN, f32::max);
                        let e: Vec<f64> = ix.iter().map(|&i| f64::from(lg[i] - m).exp()).collect();
                        let z: f64 = e.iter().sum();
                        e.into_iter().map(|v| v / z).collect::<Vec<_>>()
                    };
                    let (pa, pb) = (sm(&x.logits), sm(&y.logits));
                    let t: f64 = pa.iter().zip(&pb).map(|(u, v)| (u - v).abs()).sum::<f64>() / 2.0;
                    tv += t;
                    ncell += 1;
                    let st = match r.node.stage {
                        crate::steps::Stage::Fates => 0,
                        crate::steps::Stage::Moves => 1,
                        _ => 2,
                    };
                    by_stage[st].0 += t;
                    by_stage[st].1 += 1;
                }
            }
        }
        eprintln!(
            "prior TV {:.4} over {ncell} cells (fates {:.4}/{}, moves {:.4}/{}, spawns {:.4}/{}); child value |d| {:.4} over {nk}; node value |d| {:.4} over {nn}; tier-1 value |d| {:.4} over {n1}",
            tv / ncell.max(1) as f64,
            by_stage[0].0 / by_stage[0].1.max(1) as f64, by_stage[0].1,
            by_stage[1].0 / by_stage[1].1.max(1) as f64, by_stage[1].1,
            by_stage[2].0 / by_stage[2].1.max(1) as f64, by_stage[2].1,
            dv / nk.max(1) as f64,
            dn / nn.max(1) as f64,
            d1 / n1.max(1) as f64
        );
    }

    /// Priors vs posteriors at both tiers (`LATTICE_DIR`, `N1`, `POSITIONS`):
    /// tier 1's static value / child values / child priors against its
    /// search; tier 2's head node and kid values against the best complete
    /// turn found below them in the root lattice (static: the turn's tier-1
    /// value; deep: its searched value where searched >= 5 times), and the
    /// per-cell prior against the choice of the best turn found.
    #[test]
    #[ignore]
    fn prior_posterior_diag() {
        let Ok(lat) = std::env::var("LATTICE_DIR") else {
            eprintln!("set LATTICE_DIR");
            return;
        };
        let n1: u32 = std::env::var("N1")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);
        let npos: usize = std::env::var("POSITIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(48);
        use rand::SeedableRng;
        let le = OrtLatticeEval::load(Path::new(&lat), "trt", true, 1).unwrap();
        #[derive(Default)]
        struct Acc {
            x: Vec<f64>,
            y: Vec<f64>,
        }
        impl Acc {
            fn add(&mut self, x: f64, y: f64) {
                self.x.push(x);
                self.y.push(y);
            }
            fn show(&self, name: &str) {
                let n = self.x.len() as f64;
                if n < 2.0 {
                    eprintln!("  {name:44} n {n}");
                    return;
                }
                let mx = self.x.iter().sum::<f64>() / n;
                let my = self.y.iter().sum::<f64>() / n;
                let (mut sxy, mut sxx, mut syy, mut mae) = (0.0, 0.0, 0.0, 0.0);
                for (a, b) in self.x.iter().zip(&self.y) {
                    sxy += (a - mx) * (b - my);
                    sxx += (a - mx) * (a - mx);
                    syy += (b - my) * (b - my);
                    mae += (a - b).abs();
                }
                eprintln!(
                    "  {name:44} n {:6}  MAE {:.3}  bias(prior-post) {:+.3}  corr {:.3}  sd prior {:.3} post {:.3}",
                    n as usize,
                    mae / n,
                    mx - my,
                    sxy / (sxx * syy).sqrt().max(1e-12),
                    (sxx / n).sqrt(),
                    (syy / n).sqrt()
                );
            }
        }
        let mut t1v = Acc::default();
        let mut t1c = Acc::default();
        let (mut t1kl, mut t1top, mut t1n, mut t1pm) = (0f64, 0usize, 0usize, 0f64);
        // Same currency for policy and value: whose top pick is the search's?
        // [prior, static value] x [most visited, best searched (n >= 5)].
        let mut t1pick = [[0usize; 2]; 2];
        let mut t1pick_n = [0usize; 2];
        let (mut rho_p, mut rho_v, mut rho_n) = (0f64, 0f64, 0usize);
        let (mut c2top_v, mut c2top_n) = (0usize, 0usize);
        let mut node_s: [Acc; 3] = Default::default();
        let mut node_d = Acc::default();
        let mut kid_s: [Acc; 3] = Default::default();
        let mut kid_d = Acc::default();
        let (mut cells, mut top, mut mass, mut chance) = (0usize, 0usize, 0f64, 0f64);
        let bucket = |c: usize| {
            if c <= 1 {
                0
            } else if c < 10 {
                1
            } else {
                2
            }
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(17);
        let mut done = 0;
        while done < npos {
            let real = crate::selfplay::random_opening(22, 1.0884, &mut rng).0;
            let start = StepState::begin(&real, done as u64);
            if start.is_terminal() || crate::lattice::necro_kill(&start).is_ok() {
                continue;
            }
            done += 1;
            let (_, v0) = le.both(&[], &[&start]).unwrap();
            let mut t = crate::tier1::Tier1::new(start.clone(), Default::default());
            t.run_inflight(&le, n1, 16).unwrap();
            t1v.add(f64::from(v0[0].v), t.root_value());
            let l = t.lattice(0);
            // Tier 1: children.
            let pri = t.priors(0);
            let ch = &t.nodes[0].children;
            let tot: f64 = ch.iter().map(|&c| f64::from(t.nodes[c as usize].n)).sum();
            let mut deep: HashMap<u32, f64> = HashMap::new();
            let (mut best_p, mut best_n) = (0usize, 0usize);
            for (i, &c) in ch.iter().enumerate() {
                let nd = &t.nodes[c as usize];
                let sv = l.nodes[nd.sink as usize].value.map(f64::from);
                if nd.n >= 5 {
                    deep.insert(nd.sink, 1.0 - nd.q());
                    if let Some(sv) = sv {
                        t1c.add(sv, 1.0 - nd.q());
                    }
                }
                if pri[i] > pri[best_p] {
                    best_p = i;
                }
                if nd.n > t.nodes[ch[best_n] as usize].n {
                    best_n = i;
                }
                let q = f64::from(nd.n) / tot.max(1.0);
                if q > 0.0 {
                    t1kl += q * (q / pri[i].max(1e-12)).ln();
                }
            }
            if !ch.is_empty() {
                t1n += 1;
                t1top += usize::from(best_p == best_n);
                t1pm += pri[best_n];
                let sv: Vec<f64> = ch
                    .iter()
                    .map(|&c| {
                        l.nodes[t.nodes[c as usize].sink as usize]
                            .value
                            .map_or(f64::MIN, f64::from)
                    })
                    .collect();
                let best_v = (0..ch.len())
                    .max_by(|&a, &b| sv[a].total_cmp(&sv[b]))
                    .unwrap();
                t1pick_n[0] += 1;
                t1pick[0][0] += usize::from(best_p == best_n);
                t1pick[1][0] += usize::from(best_v == best_n);
                let srch: Vec<usize> = (0..ch.len())
                    .filter(|&i| t.nodes[ch[i] as usize].n >= 5)
                    .collect();
                if srch.len() >= 2 {
                    let q = |i: usize| 1.0 - t.nodes[ch[i] as usize].q();
                    let best_s = *srch
                        .iter()
                        .max_by(|&&a, &&b| q(a).total_cmp(&q(b)))
                        .unwrap();
                    let bp = *srch
                        .iter()
                        .max_by(|&&a, &&b| pri[a].total_cmp(&pri[b]))
                        .unwrap();
                    let bv = *srch
                        .iter()
                        .max_by(|&&a, &&b| sv[a].total_cmp(&sv[b]))
                        .unwrap();
                    t1pick_n[1] += 1;
                    t1pick[0][1] += usize::from(bp == best_s);
                    t1pick[1][1] += usize::from(bv == best_s);
                    if srch.len() >= 3 {
                        let rank = |f: &dyn Fn(usize) -> f64| {
                            let mut ix = srch.clone();
                            ix.sort_by(|&a, &b| f(a).total_cmp(&f(b)));
                            let mut r = vec![0f64; ch.len()];
                            for (k, &i) in ix.iter().enumerate() {
                                r[i] = k as f64;
                            }
                            r
                        };
                        let rq = rank(&|i| q(i));
                        let rp = rank(&|i| pri[i]);
                        let rv = rank(&|i| sv[i]);
                        let sp = |a: &[f64], b: &[f64]| {
                            let n = srch.len() as f64;
                            let d: f64 = srch.iter().map(|&i| (a[i] - b[i]).powi(2)).sum();
                            1.0 - 6.0 * d / (n * (n * n - 1.0))
                        };
                        rho_p += sp(&rp, &rq);
                        rho_v += sp(&rv, &rq);
                        rho_n += 1;
                    }
                }
            }
            // Tier 2: the root lattice's sinks.
            let sinks: Vec<(crate::lattice::Key, f64, f64)> = l
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.kind == crate::lattice::Kind::Sink)
                .filter_map(|(i, n)| {
                    n.value.map(|v| {
                        let v = f64::from(v);
                        (n.key, v, deep.get(&(i as u32)).copied().unwrap_or(v))
                    })
                })
                .collect();
            let best = |k: &crate::lattice::Key| {
                let (mut c, mut bs, mut bd) = (0usize, f64::MIN, f64::MIN);
                for (sk, sv, dv) in &sinks {
                    if sk.includes(k) {
                        c += 1;
                        bs = bs.max(*sv);
                        bd = bd.max(*dv);
                    }
                }
                (c, bs, bd)
            };
            for x in &l.nodes {
                if x.kind != crate::lattice::Kind::Inner || !x.expanded || x.n == 0 {
                    continue;
                }
                let (c, bs, bd) = best(&x.key);
                if c > 0 {
                    let hv = x.w / f64::from(x.n);
                    node_s[bucket(c)].add(hv, bs);
                    if c >= 10 {
                        node_d.add(hv, bd);
                    }
                }
                let mut i = 0;
                while i < x.kids.len() {
                    let var = x.kids[i].var;
                    let j = i + x.kids[i..].iter().take_while(|k| k.var == var).count();
                    let mut post: Vec<(usize, f64)> = Vec::new();
                    for k in i..j {
                        let kd = &x.kids[k];
                        if kd.dead {
                            continue;
                        }
                        let (c, bs, bd) = best(&x.key.with(kd.var as usize, kd.choice));
                        if c > 0 {
                            kid_s[bucket(c)].add(f64::from(kd.v), bs);
                            if c >= 10 {
                                kid_d.add(f64::from(kd.v), bd);
                            }
                            post.push((k, bs));
                        }
                    }
                    if post.len() >= 2 {
                        let pb = post.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
                        let pa = (i..j)
                            .filter(|&k| !x.kids[k].dead)
                            .max_by(|&a, &b| x.kids[a].logp.total_cmp(&x.kids[b].logp))
                            .unwrap();
                        let pv = post
                            .iter()
                            .map(|&(k, _)| k)
                            .max_by(|&a, &b| x.kids[a].v.total_cmp(&x.kids[b].v))
                            .unwrap();
                        c2top_v += usize::from(pv == pb);
                        c2top_n += 1;
                        cells += 1;
                        top += usize::from(pa == pb);
                        mass += f64::from(x.kids[pb].logp).exp();
                        chance += 1.0 / (i..j).filter(|&k| !x.kids[k].dead).count() as f64;
                    }
                    i = j;
                }
            }
        }
        eprintln!("prior vs posterior, {npos} positions, n1 {n1}, net {lat}");
        eprintln!("tier 1 (values for the mover):");
        t1v.show("root: net value vs search value");
        t1c.show("children n>=5: net value vs searched");
        eprintln!(
            "  children policy: KL(visits || prior) {:.3}, prior top = most visited {:.3}, prior mass on most visited {:.3} ({t1n} roots)",
            t1kl / t1n.max(1) as f64,
            t1top as f64 / t1n.max(1) as f64,
            t1pm / t1n.max(1) as f64
        );
        eprintln!(
            "  top pick = most visited ({} roots): prior {:.3}, static value {:.3}; = best searched child (n>=5, {} roots): prior {:.3}, static value {:.3}",
            t1pick_n[0],
            t1pick[0][0] as f64 / t1pick_n[0].max(1) as f64,
            t1pick[1][0] as f64 / t1pick_n[0].max(1) as f64,
            t1pick_n[1],
            t1pick[0][1] as f64 / t1pick_n[1].max(1) as f64,
            t1pick[1][1] as f64 / t1pick_n[1].max(1) as f64
        );
        eprintln!(
            "  Spearman vs searched value over children n>=5 ({rho_n} roots): prior {:.3}, static value {:.3}",
            rho_p / rho_n.max(1) as f64,
            rho_v / rho_n.max(1) as f64
        );
        eprintln!("tier 2, root lattice (posterior = best complete turn found below):");
        for (b, name) in ["1 turn", "2-9 turns", "10+ turns"].iter().enumerate() {
            node_s[b].show(&format!("node value vs best static, {name}"));
        }
        node_d.show("node value vs best deep, 10+ turns");
        for (b, name) in ["1 turn", "2-9 turns", "10+ turns"].iter().enumerate() {
            kid_s[b].show(&format!("kid value vs best static, {name}"));
        }
        kid_d.show("kid value vs best deep, 10+ turns");
        eprintln!(
            "  per-cell top pick = best turn's choice: prior {:.3}, kid value {:.3} (uniform {:.3}), prior mass on it {:.3} ({cells} cells)",
            top as f64 / cells.max(1) as f64,
            c2top_v as f64 / c2top_n.max(1) as f64,
            chance / cells.max(1) as f64,
            mass / cells.max(1) as f64
        );
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
