//! ONNX Runtime inference backend (CUDA / TensorRT EP, fp16) — the fast
//! alternative to candle for self-play, using the GPU's tensor cores. Mirrors
//! `amazons/src/ort_net.rs`, but the graph is plain AZ: two outputs, `value`
//! (B,) and `policy` (B,169) raw logits, so there is no head interleaving.
//!
//! The learner exports `model.onnx` next to `latest.safetensors`; the actor
//! picks it up with `--backend ort`. The onnxruntime shared library is found at
//! runtime via `ORT_DYLIB_PATH`, and TensorRT caches its built engines under
//! `<live>/trt_cache` (first run pays the build, later ones don't).

use std::sync::Mutex;

use ort::ep::cuda::ConvAlgorithmSearch;
use ort::execution_providers::{CUDA, TensorRT};
use ort::session::Session;
use ort::value::Tensor;

use playground_core::batch_search::BatchModel;

use crate::game::{N, PLANES as IN_PLANES};
use crate::mcts::POLICY_DIM;

fn e2a<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("ort: {e}")
}

pub struct OrtModel {
    session: Mutex<Session>,
}

impl OrtModel {
    pub fn new(onnx_path: &str, ep: &str, fp16: bool, cache_dir: &str) -> anyhow::Result<Self> {
        let cuda = || {
            CUDA::default()
                .with_device_id(0)
                .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
        };
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
                    cuda().build(), // fall back per-node where TRT can't build
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
        let input = Tensor::from_array((
            vec![b as i64, IN_PLANES as i64, N as i64, N as i64],
            planes.to_vec(),
        ))
        .map_err(e2a)?;
        let mut sess = self.session.lock().unwrap();
        let outputs = sess.run(ort::inputs!["planes" => input]).map_err(e2a)?;
        let (_vs, vals) = outputs["value"].try_extract_tensor::<f32>().map_err(e2a)?;
        let (shape, policy) = outputs["policy"].try_extract_tensor::<f32>().map_err(e2a)?;
        debug_assert_eq!(policy.len(), b * POLICY_DIM, "policy shape {shape:?}");
        Ok((vals.to_vec(), policy.to_vec()))
    }
}
