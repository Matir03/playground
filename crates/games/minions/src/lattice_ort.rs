//! The turn-lattice nets under ONNX Runtime: `lattice.onnx` from
//! `python/minions_lattice_learner.py`, one graph answering both tiers'
//! requests. A row is a turn start (tier-1 input), a lattice node and the
//! node's children as `cell * 37 + slot`; the graph returns the turn start's
//! tier-1 value, the children's prior logits and values, and the node's
//! value. Tier-1 value requests are rows whose node is the turn start itself
//! with one dummy child. Rows sharing a turn start recompute its trunk.

use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;

use anyhow::Result;
use ort::session::Session;

use crate::latgen::t2_view;
use crate::lattice::{ExpandOut, ExpandReq, LatticeEval};
use crate::steps::{StepState, View, MAX_ACTIVE, N_SCALARS, SPARSE_DIM};

pub const SLOTS: usize = 37;

pub struct OrtLatticeEval {
    sessions: Vec<Mutex<Session>>,
    next: AtomicUsize,
    pub calls: AtomicU64,
    pub rows: AtomicU64,
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
}

impl OrtLatticeEval {
    pub fn load(path: &Path, ep: &str, fp16: bool, cache: &str, sessions: usize) -> Result<Self> {
        let mut v = Vec::new();
        for _ in 0..sessions.max(1) {
            v.push(Mutex::new(crate::ort_net::build(path, ep, fp16, cache)?));
        }
        Ok(OrtLatticeEval {
            sessions: v,
            next: AtomicUsize::new(0),
            calls: AtomicU64::new(0),
            rows: AtomicU64::new(0),
        })
    }

    /// Run the rows: per row (v1, logits, values, v2).
    #[allow(clippy::type_complexity)]
    fn run(&self, rows: Rows) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>)> {
        use ort::value::Tensor;
        let e = |e: ort::Error| anyhow::anyhow!("{e}");
        let b = rows.kids.len();
        let k = rows.kids.iter().map(|x| x.len()).max().unwrap_or(1).max(1);
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
        // Any free session, else wait on one in turn.
        let n = self.sessions.len();
        let start = self.next.fetch_add(1, Relaxed);
        let mut guard = None;
        for j in 0..n {
            if let Ok(g) = self.sessions[(start + j) % n].try_lock() {
                guard = Some(g);
                break;
            }
        }
        let mut sess = match guard {
            Some(g) => g,
            None => self.sessions[start % n]
                .lock()
                .unwrap_or_else(|p| p.into_inner()),
        };
        let outs = sess.run(inputs).map_err(e)?;
        let get = |name: &str| -> Result<Vec<f32>> {
            let (_, v) = outs[name].try_extract_tensor::<f32>().map_err(e)?;
            Ok(v.to_vec())
        };
        let (v1, logit, value, v2) = (get("v1")?, get("logit")?, get("value")?, get("v2")?);
        drop(outs);
        drop(sess);
        self.calls.fetch_add(1, Relaxed);
        self.rows.fetch_add(b as u64, Relaxed);
        let per = |x: &[f32]| -> Vec<Vec<f32>> {
            rows.kids
                .iter()
                .enumerate()
                .map(|(i, ks)| x[i * k..i * k + ks.len()].to_vec())
                .collect()
        };
        Ok((v1, per(&logit), per(&value), v2))
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
        let le = OrtLatticeEval::load(Path::new(&lat), "cuda", false, "/tmp/lat_trt", 1).unwrap();
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
    }
}
