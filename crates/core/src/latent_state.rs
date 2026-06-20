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

    /// Build an LGS from a float vector (little-endian f32 payload, shape `[d]`).
    pub fn from_f32(v: &[f32], model_version: String) -> Self {
        let payload: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Self::new(payload, vec![v.len() as i32], model_version)
    }

    /// Decode the little-endian f32 payload into a float vector.
    pub fn to_f32(&self) -> Vec<f32> {
        self.payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
