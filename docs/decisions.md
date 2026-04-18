# Decisions

ADR-style log. Each entry: context, decision, consequences. Append only —
reverse a decision by adding a new entry that supersedes it.

---

## D1 — Python owns the model

**Context.** Self-play needs fast NN eval. Options were (a) run models
in-Rust via candle/tch/ONNX, (b) run models in Python behind a server,
(c) mixed.

**Decision.** All NN code lives in Python. Rust engines talk to a Python
**model server** over gRPC (`ModelService`). The `model-client` Rust crate
does micro-batching to amortize the boundary crossing.

**Consequences.**
- Model architecture iteration stays ergonomic (PyTorch, freely).
- We pay a real cost on self-play throughput from the cross-process hop
  and batching coordination; we accept this up front and plan to measure.
- If this becomes the bottleneck, we can swap the model server for an
  in-Rust ONNX runtime without changing the protocol — `model-client`
  just gets a new backend. This is a reason the boundary exists.

---

## D2 — One Rust binary per game, shared core crate

**Context.** The games differ in state, move encoding, and performance
profiles. We want clean isolation between games so changes to one don't
destabilize another.

**Decision.** A binary crate under `crates/games/<name>` per game. All
game-agnostic code (MCTS, KR-UCT, progressive widening, recording,
ModelClient trait) lives in `crates/core`, with tonic-based server
glue in `crates/engine-server`.

**Consequences.**
- Each binary is small and game-focused.
- Core changes ripple cleanly to all games with a single `cargo build`.
- Engine-server is generic over the `Game` trait; adding a new game is
  implementing the trait and wiring one `main.rs`.

---

## D3 — Protobuf everywhere, via buf

**Context.** Three languages (Rust, Python, TypeScript) need to agree on
wire + record formats. On-disk records also cross a language boundary
(Rust writes, Python reads).

**Decision.** Protobuf for all wire and record formats.
`schemas/proto` is the source of truth. Codegen via `buf generate`
(`buf.gen.yaml`) for TS and Python, via `tonic-build` in `build.rs`
for Rust.

**Consequences.**
- One schema, three language clients, generated.
- `buf lint` + `buf breaking` gate schema changes in CI.
- Generated code is gitignored; regenerate after every proto change.

---

## D4 — gRPC-Web from the browser

**Context.** Browser can't speak native gRPC. We still want
Connect-ES-style typed clients in TS.

**Decision.** Rust engine servers wrap tonic with `tonic-web`; the TS
client uses `@connectrpc/connect-web` with the gRPC-Web transport.
Python-to-Rust and Rust-to-Python use native gRPC (no tonic-web).

**Consequences.**
- No Envoy or sidecar proxy. One server, two transports.
- We don't use the Connect wire format on the hot path; we use gRPC /
  gRPC-Web. The Connect *developer experience* is what we wanted; the
  wire choice is pragmatic.
- If Rust ConnectRPC server libraries mature, switching is low-risk (same
  protos, different transport annotation).

---

## D5 — Long-lived engine server, not UCI subprocess per move

**Context.** Batching NN evals across multiple in-flight games is the only
way to get reasonable self-play throughput. A fresh UCI subprocess per move
can't batch.

**Decision.** Engine binaries are long-lived servers with three modes:
interactive, batched self-play, and match. A `stdio` mode exists for
debugging and future tools, but the hot path is always the long-lived
server.

**Consequences.**
- The `model-client` crate can batch NN calls across the games currently
  running in one engine process.
- Orchestrator supervises a pool of engine processes instead of spawning
  per game.

---

## D6 — MCTS lives in `core`, no I/O

**Context.** The algorithm is the interesting part. We want to unit-test
it ruthlessly without spinning up a tokio runtime or a model server.

**Decision.** `crates/core` is pure logic — `Game` trait, MCTS, KR-UCT,
progressive widening, recording. It depends on no tonic, no tokio, and
takes any `ModelClient: async_trait` as input. Tests use a deterministic
mock `ModelClient`.

**Consequences.**
- Game correctness, MCTS correctness, and recording correctness are
  provable in-process.
- `engine-server` is the only crate coupling algorithm to transport.

---

## D7 — No package versioning in schemas

**Context.** The usual protobuf pattern tacks `.v1` on every package so
breaking changes can introduce `.v2` without disturbing old consumers. That
assumes consumers you don't control. We don't have any — it's a monorepo,
all three language bindings are generated from the same source, and a
breaking schema change reves the Rust server, the Python services, and the
web bundle in the same PR.

**Decision.** Packages are unversioned (`playground.engine`, not
`playground.engine.v1`). Evolution rules (never reuse field numbers,
additive-only over the wire for on-disk records) still apply, but the
"keep v1 frozen, introduce v2" dance doesn't.

**Consequences.**
- One less path segment in every import.
- No `buf breaking` gate — intentional. If we ever publish a client anyone
  else consumes, reintroduce versioning at that boundary only.
- Schema changes are just code changes; review them on the merits.
