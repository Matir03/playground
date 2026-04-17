//! Generated protobuf + tonic stubs for the playground schemas.
//!
//! Modules mirror the .proto package tree. Regenerated at build time by
//! `build.rs`; no hand-written types should live in this crate.

pub mod playground {
    pub mod game {
        pub mod v1 {
            tonic::include_proto!("playground.game.v1");
        }
    }
    pub mod record {
        pub mod v1 {
            tonic::include_proto!("playground.record.v1");
        }
    }
    pub mod engine {
        pub mod v1 {
            tonic::include_proto!("playground.engine.v1");
        }
    }
    pub mod model {
        pub mod v1 {
            tonic::include_proto!("playground.model.v1");
        }
    }
    pub mod orchestrator {
        pub mod v1 {
            tonic::include_proto!("playground.orchestrator.v1");
        }
    }
}

// Convenience re-exports for the most-used types.
pub use playground::engine::v1 as engine_v1;
pub use playground::game::v1 as game_v1;
pub use playground::model::v1 as model_v1;
pub use playground::orchestrator::v1 as orchestrator_v1;
pub use playground::record::v1 as record_v1;
