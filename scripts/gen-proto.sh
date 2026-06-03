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

# buf's protocolbuffers/python plugin emits proto-package-rooted cross-imports
# (e.g. `from playground.game import game_pb2`). Our Python output lands under
# `playground/gen/`, so rewrite those to be `playground.gen.`-rooted, matching
# how callers import them (see playground/gen/__init__.py). `from google...`
# and already-rewritten lines are left untouched.
echo "==> fixing python import paths"
gen_root="../python/src/playground/gen/playground"
find "$gen_root" \( -name '*.py' -o -name '*.pyi' \) -print0 \
  | xargs -0 sed -i -E 's/^(from|import) playground\.(game|engine|model|record|orchestrator)([ .])/\1 playground.gen.playground.\2\3/'

echo "==> done. Generated:"
echo "    web/src/gen/          (TS)"
echo "    python/src/playground/gen/ (Python)"
echo
echo "Rust regeneration happens via cargo build (see crates/protocol/build.rs)."
