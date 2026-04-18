//! Compiles .proto files from ../../schemas/proto into Rust types + tonic
//! service stubs. Requires `protoc` to be installed on the build machine.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root: PathBuf = PathBuf::from("../../schemas/proto");
    let files: Vec<PathBuf> = vec![
        proto_root.join("playground/game/game.proto"),
        proto_root.join("playground/record/record.proto"),
        proto_root.join("playground/engine/engine.proto"),
        proto_root.join("playground/model/model.proto"),
        proto_root.join("playground/orchestrator/orchestrator.proto"),
    ];

    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    println!("cargo:rerun-if-changed=build.rs");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &files.iter().map(|p| p.as_path()).collect::<Vec<_>>(),
            &[proto_root.as_path()],
        )?;

    Ok(())
}
