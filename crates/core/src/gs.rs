//! Opaque game-state encoding (GS).
//!
//! The layout is owned by whichever model produced the bytes. Only the
//! model server and the model-client crate look inside.

use bytes::Bytes;

#[derive(Clone, Debug)]
pub struct Gs {
    pub payload: Bytes,
    pub shape: Vec<i32>,
    pub model_version: String,
}

impl Gs {
    pub fn new(payload: impl Into<Bytes>, shape: Vec<i32>, model_version: String) -> Self {
        Self {
            payload: payload.into(),
            shape,
            model_version,
        }
    }
}
