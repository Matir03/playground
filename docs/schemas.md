# Schemas

All wire and on-disk formats are protobuf. Source of truth: [`schemas/proto`](../schemas/proto).

## Layout

```
schemas/proto/playground/
  game/v1/game.proto             # shared: GameState, Move, Player, Outcome, GameId
  record/v1/record.proto         # on-disk training records
  engine/v1/engine.proto         # EngineService + its message types
  model/v1/model.proto           # ModelService + its message types
  orchestrator/v1/orchestrator.proto
```

## Conventions

- **Versioned packages.** Every package ends in `.vN`. Breaking changes bump N.
- **Split service from shared types.** `game/v1` holds types that *multiple*
  services use (GameState, Move). Service-specific RPC messages live under
  the service's own package. Rationale: record format and RPC surface evolve
  at different rates; keeping them in separate files means changing one
  doesn't churn the other.
- **No global `enum`s in shared files** except for the small player/outcome
  stuff. Game-specific things stay opaque — see below.
- **Bytes for game-specific state.** `GameState.payload` is `bytes` with a
  sibling `game_id` field indicating how to interpret it. Each game crate
  documents its own payload encoding (and ideally provides a sub-proto for
  inspection / debugging — but that's optional).
- **Bytes for GS.** The GS encoding is `bytes` + shape hint; only the model
  server and `model-client` know the tensor layout.

## Evolution rules (buf-breaking compatible)

- Never reuse a field number.
- Never change the wire type of an existing field.
- New fields are always `optional` (proto3 presence) unless clearly required
  for all future records.
- When we truly need a break, create `v2` and keep `v1` until consumers
  migrate. Do not edit `v1` after first production use.
- `buf breaking` is our gatekeeper; run it in CI before we merge PRs that
  touch `schemas/proto`.

## Training record format

The `GameRecord` message is how self-play feeds training. Key design points:

- It carries the **full GS-level MCTS tree** that ran during play, including
  per-node visits/values/priors.
- For each GS node, any **reified concrete children** are recorded with:
  their concrete state, the move that was applied, and the action-head
  vector that was sampled (plus noise scale) to reify it.
- The final game **Outcome** is the ground-truth signal for the eval head.

The exact proto is in [`record/v1/record.proto`](../schemas/proto/playground/record/v1/record.proto). Revisit the schema
after M2 (Connect 4 training) when we have a concrete sense of what the
training loop actually uses.

## Storage

Records are written as length-delimited protobuf files, one file per game,
sharded into directories by `run_id/date/`. A tiny sqlite index tracks
`run_id`, `model_version`, `game_id`, `outcome`, `file_path`. This keeps
append throughput high (no DB on the hot path) and queries cheap.
