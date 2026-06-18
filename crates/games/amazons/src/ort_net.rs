//! ONNX Runtime inference backend (CUDA / TensorRT EP, fp16) for Amazons self-play
//! — the fast alternative to candle, using the GPU tensor cores. Mirrors
//! `go9/ort_net.rs`, but the ONNX graph has THREE outputs: `value` (B,), `queen`
//! logits (B,7200), `arrow` logits (B,7200). They're concatenated into the
//! `POLICY_DIM`-wide raw-logit policy the move-search expects (`value`, `priors`).
//!
//! The onnxruntime shared lib is found at runtime via `ORT_DYLIB_PATH`.

use std::sync::Mutex;

use ort::ep::cuda::ConvAlgorithmSearch;
use ort::execution_providers::{CUDA, TensorRT};
use ort::session::Session;
use ort::value::Tensor;

use playground_core::batch_search::BatchModel;

use crate::game::N;
use crate::moves::{PER_HEAD, POLICY_DIM};

const IN_PLANES: usize = crate::game::PLANES; // 4

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
                    cuda().build(),
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
        let (_qs, queen) = outputs["queen"].try_extract_tensor::<f32>().map_err(e2a)?;
        let (_as, arrow) = outputs["arrow"].try_extract_tensor::<f32>().map_err(e2a)?;
        // Interleave the two heads into one POLICY_DIM-wide logit row per position:
        // [queen(7200) ‖ arrow(7200)], matching `combine_priors`' slicing.
        let mut policy = vec![0f32; b * POLICY_DIM];
        for i in 0..b {
            let dst = &mut policy[i * POLICY_DIM..(i + 1) * POLICY_DIM];
            dst[..PER_HEAD].copy_from_slice(&queen[i * PER_HEAD..(i + 1) * PER_HEAD]);
            dst[PER_HEAD..].copy_from_slice(&arrow[i * PER_HEAD..(i + 1) * PER_HEAD]);
        }
        Ok((vals.to_vec(), policy))
    }
}
