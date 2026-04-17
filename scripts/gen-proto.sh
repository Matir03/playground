#!/usr/bin/env bash
# Regenerate TS + Python bindings from schemas/proto.
#
# Rust bindings are generated automatically by `crates/protocol/build.rs`
# on the next `cargo build`, so this script only handles the two languages
# that rely on `buf generate`.
#
# Prereqs: `buf` on PATH (https://buf.build), internet access for the first
# run (buf pulls remote plugins).
set -euo pipefail

cd "$(dirname "$0")/../schemas"

echo "==> buf lint"
buf lint

echo "==> buf generate"
buf generate

echo "==> done. Generated:"
echo "    web/src/gen/          (TS)"
echo "    python/src/playground/gen/ (Python)"
echo
echo "Rust regeneration happens via cargo build (see crates/protocol/build.rs)."
