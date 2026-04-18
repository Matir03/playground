# Architecture

## Component map

```
              ┌────────────────────────────────────────────────────┐
              │                     web (React)                    │
              │   Connect-ES client ── gRPC-Web over HTTP/1.1      │
              └──────────────┬───────────────────────┬─────────────┘
                             │                       │
                             ▼                       ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │  Rust engine binary      │    │  Python orchestrator     │
        │  (one per game)          │    │  (admin RPC)             │
        │  tonic + tonic-web       │    │                          │
        │  EngineService           │    │  OrchestratorService     │
        └──────────┬───────────────┘    └───────────┬──────────────┘
                   │                                │
             gRPC (ModelService)                    │ manages:
                   │                                │  - self-play workers
                   ▼                                │  - replay DB
        ┌──────────────────────────┐                │  - model registry
        │  Python model server     │◄───────────────┘  - training trigger
        │  grpcio                  │                │  - eval tournaments
        │  ModelService            │                │
        │  (encoder/eval/          │                │
        │   policy/reifier heads)  │                │
        └──────────────────────────┘                │
                                                    ▼
                                     ┌──────────────────────────┐
                                     │  Training                │
                                     │  (reads replay DB,       │
                                     │   writes model registry) │
                                     └──────────────────────────┘
```

## Component responsibilities

### `crates/core` (Rust, no I/O)
- `Game` trait: move type, state type, legal moves, apply, terminal, features.
- MCTS over GS: tree node structure, selection/expansion/backprop, KR-UCT,
  progressive widening. Generic over `Game` and `ModelClient`.
- Game-state encoding abstraction (`Gs`) — opaque byte buffer + shape metadata.
- Recording schema (converts in-memory tree → `record.proto` messages).
- `ModelClient` trait: async methods mirroring `ModelService` RPCs.
- **No I/O, no tokio runtime, no tonic.** Keeps the algorithm unit-testable.

### `crates/protocol` (Rust)
- tonic-build-generated types from `schemas/proto/**/*.proto`.
- Thin re-exports so downstream crates say `use playground_protocol::engine_pb::*`.

### `crates/engine-server` (Rust)
- Implements `EngineService` generically over `G: Game + ModelClient`.
- Modes (see [protocols.md](protocols.md)):
  1. **Interactive** — single game, human/UI-driven, one move at a time.
  2. **Batched self-play** — N parallel games, NN eval calls fan into a
     shared batcher before hitting the model server.
  3. **Match** — two engine endpoints play each other; records results.

### `crates/games/<name>` (Rust binaries)
- One binary per game.
- Wires the game's concrete `Game` impl into `engine-server::run`.
- Exposes a CLI with subcommands for the three server modes above.

### `crates/model-client` (Rust)
- Implements `core::ModelClient` by calling the Python `ModelService` over
  gRPC with request batching.
- This is the *only* place NN calls cross a process boundary.

### `python/src/playground/model_server`
- Serves `ModelService` via grpcio.
- Holds the current model weights in-process. Reloads weights on
  `ReloadWeights` RPC (issued by orchestrator after a new checkpoint lands).

### `python/src/playground/orchestrator`
- Owns the self-play loop (see [training.md](training.md)).
- Spawns/supervises engine binaries in batched self-play mode.
- Writes game records to the replay DB.
- Triggers training jobs, promotes checkpoints.
- Schedules eval tournaments.
- Exposes `OrchestratorService` for the web admin UI.

### `python/src/playground/training`
- Pure training loops. Reads records from the replay DB, produces new
  checkpoints, writes metadata to the model registry.

### `python/src/playground/eval`
- Round-robin and pairwise match runners.
- ELO fitting (BayesElo / Elo-MMR).

### `web`
- React SPA. Connect-ES clients for both `EngineService` (playing games) and
  `OrchestratorService` (admin / monitoring).
- Any player slot can be `human`, `model://<name>@<version>`, or `engine://<uri>`.

## Data flow: one self-play game

1. Orchestrator selects a game + model version, spawns or picks an engine
   worker in batched self-play mode.
2. Engine worker starts N parallel games internally.
3. For each GS node expansion, engine asks `ModelClient` for encode/eval/policy.
   `model-client` batches across games and hits the Python model server.
4. Every K visits, engine reifies — calls `reify` on the model server, applies
   the concrete move, encodes the new state, attaches to the tree.
5. When a game ends, engine serializes the full tree + reification history
   as a `GameRecord` and writes it to the shared replay store.
6. Orchestrator notices the record, updates stats, possibly triggers training.

## Data flow: training → new model version

1. Training reads a sliding window of recent `GameRecord`s.
2. Trains encoder+reifier first, then (once stable) adds eval and policy.
3. Writes a new checkpoint under `data/models/<name>/<version>/`.
4. Orchestrator issues `ReloadWeights` to running model servers (or spawns
   a new one and drains old one).
5. Engines continue self-play with the new model; new `GameRecord`s are
   tagged with the new model version.
