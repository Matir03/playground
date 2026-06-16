//! ONNX Runtime inference backend (CUDA / TensorRT EP, fp16) — an alternative to
//! candle for the batched self-play engine, to use the GPU's tensor cores.
//!
//! Implements [`BatchModel`] via an `ort::Session` over the exported Go9 ONNX
//! graph (input `planes` (B,3,9,9) -> `value` (B,), `policy` logits (B,82)). The
//! policy logits are softmaxed here so the output matches candle's
//! `value_policy_batch` contract (probabilities, not logits).
//!
//! Uses `ort`'s `load-dynamic` feature: the onnxruntime shared lib is found at
//! runtime via `ORT_DYLIB_PATH` (set in /tmp/cuda_env.sh to the pip-installed
//! onnxruntime-gpu, which bundles the CUDA + TensorRT EPs). TensorRT builds a
//! per-shape engine on first use; we cache it on disk to avoid rebuilds.

use std::sync::Mutex;

use ort::ep::cuda::ConvAlgorithmSearch;
use ort::execution_providers::{CUDA, TensorRT};
use ort::session::Session;
use ort::value::Tensor;

use playground_core::batch_search::BatchModel;

use crate::game::N;
use crate::net::ACTIONS;

/// ort's error type isn't Send+Sync, so it can't flow through `?` into anyhow;
/// stringify it.
fn e2a<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("ort: {e}")
}

pub struct OrtModel {
    // Session::run takes &mut self; the inference server is single-threaded but the
    // BatchModel API is &self + Send+Sync, so guard behind a Mutex.
    session: Mutex<Session>,
}

impl OrtModel {
    /// `ep` = "cuda" (fp32) or "trt" (TensorRT, fp16 when `fp16`). `cache_dir` is
    /// where TensorRT caches its built engines.
    pub fn new(onnx_path: &str, ep: &str, fp16: bool, cache_dir: &str) -> anyhow::Result<Self> {
        // Heuristic conv-algorithm selection: the default (Exhaustive) re-autotunes
        // cuDNN for every new batch shape, which is pathological for self-play's
        // varying batch sizes (a 16-game chunk took 75s; GPU 23%).
        let cuda = || CUDA::default().with_device_id(0)
            .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic);
        let providers = match ep {
            "trt" => {
                std::fs::create_dir_all(cache_dir).ok();
                vec![
                    TensorRT::default()
                        .with_device_id(0)
                        .with_fp16(fp16)
                        .with_engine_cache(true)
                        .with_engine_cache_path(cache_dir.to_string())
                        .build(),
                    cuda().build(), // fallback for any subgraph TRT can't take
                ]
            }
            "cuda" => vec![cuda().build()],
            other => anyhow::bail!("unknown --ep {other:?} (use cuda|trt)"),
        };
        let session = Session::builder()
            .map_err(e2a)?
            .with_execution_providers(providers)
            .map_err(e2a)?
            .commit_from_file(onnx_path)
            .map_err(e2a)?;
        Ok(Self { session: Mutex::new(session) })
    }
}

impl BatchModel for OrtModel {
    fn forward(&self, planes: &[f32], b: usize) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let input = Tensor::from_array((vec![b as i64, 3, N as i64, N as i64], planes.to_vec()))
            .map_err(e2a)?;
        let mut sess = self.session.lock().unwrap();
        let outputs = sess.run(ort::inputs!["planes" => input]).map_err(e2a)?;
        let (_vs, vals) = outputs["value"].try_extract_tensor::<f32>().map_err(e2a)?;
        let (_ps, logits) = outputs["policy"].try_extract_tensor::<f32>().map_err(e2a)?;
        let values = vals.to_vec();
        // Softmax each row of policy logits (b × ACTIONS) to match candle's output.
        let mut probs = vec![0f32; b * ACTIONS];
        for i in 0..b {
            let row = &logits[i * ACTIONS..(i + 1) * ACTIONS];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (j, &x) in row.iter().enumerate() {
                let e = (x - m).exp();
                probs[i * ACTIONS + j] = e;
                sum += e;
            }
            let inv = 1.0 / sum.max(1e-12);
            for j in 0..ACTIONS {
                probs[i * ACTIONS + j] *= inv;
            }
        }
        Ok((values, probs))
    }
}
