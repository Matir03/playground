//! Generated protobuf + tonic stubs for the playground schemas.
//!
//! Modules mirror the .proto package tree. Regenerated at build time by
//! `build.rs`; no hand-written types should live in this crate.

pub mod playground {
    pub mod game {
        tonic::include_proto!("playground.game");
    }
    pub mod record {
        tonic::include_proto!("playground.record");
    }
    pub mod engine {
        tonic::include_proto!("playground.engine");
    }
    pub mod model {
        tonic::include_proto!("playground.model");
    }
    pub mod orchestrator {
        tonic::include_proto!("playground.orchestrator");
    }
}

// Convenience re-exports for the most-used types.
pub use playground::engine as engine_pb;
pub use playground::game as game_pb;
pub use playground::model as model_pb;
pub use playground::orchestrator as orchestrator_pb;
pub use playground::record as record_pb;
