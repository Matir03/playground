//! Latent game state (LGS) — see docs/algorithm.md.
//!
//! An opaque byte buffer produced by the encoder network. Only the model
//! server and the model-client crate interpret the bytes; MCTS treats it as a
//! handle plus shape metadata. Distinct from the concrete `GameState` (the real
//! board), which lives in the proto/game layer.

use bytes::Bytes;

#[derive(Clone, Debug)]
pub struct LatentGameState {
    pub payload: Bytes,
    pub shape: Vec<i32>,
    pub model_version: String,
}

impl LatentGameState {
    pub fn new(payload: impl Into<Bytes>, shape: Vec<i32>, model_version: String) -> Self {
        Self {
            payload: payload.into(),
            shape,
            model_version,
        }
    }
}
