# Schemas

All wire and on-disk formats are protobuf. Source of truth: [`schemas/proto`](../schemas/proto).

## Layout

```
schemas/proto/playground/
  game/game.proto                # shared: GameState, Move, Player, Outcome, GameId
  record/record.proto            # on-disk training records
  engine/engine.proto            # EngineService + its message types
  model/model.proto              # ModelService + its message types
  orchestrator/orchestrator.proto
```

## Conventions

- **No package versioning.** This is a monorepo; all clients rev with the
  schemas. See [decisions.md#d7](decisions.md) for the rationale.
- **Split service from shared types.** `game` holds types that *multiple*
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
- **Bytes for LGS.** The LGS encoding is `bytes` + shape hint; only the model
  server and `model-client` know the tensor layout.

## Evolution rules

- Never reuse a field number.
- Never change the wire type of an existing field.
- New fields are always `optional` (proto3 presence) unless clearly required
  for all future records.
- Because everything rev's together, breaking changes are fine — just update
  all callers in the same PR. `buf breaking` is *not* gating us here; its job
  would be to protect external consumers, and we have none.
- If/when we grow an external consumer, introduce versioned packages at
  that boundary only.

## Training record format

The `GameRecord` message is how self-play feeds training. Key design points:

- It carries the **full latent-space MCTS tree** that ran during play,
  including per-node visits/values and the sampled candidate targets.
- For each node, its **reified concrete children** are recorded with: the
  concrete state, the move that was applied, the re-encoded child latent, and
  the **merged visit count** (counting aliased candidates). Enough to rebuild
  the visit distribution `π_i` and the reifier pairs `(S, A_i)`.
- The final game **Outcome** is the ground-truth signal for the eval head.

  See [algorithm.md](algorithm.md#search--training-targets) for how these fields
  become training targets.

The exact proto is in [`record/record.proto`](../schemas/proto/playground/record/record.proto). Revisit the schema
after M2 (Connect 4 training) when we have a concrete sense of what the
training loop actually uses.

## Storage

Records are written as length-delimited protobuf files, one file per game,
sharded into directories by `run_id/date/`. A tiny sqlite index tracks
`run_id`, `model_version`, `game_id`, `outcome`, `file_path`. This keeps
append throughput high (no DB on the hot path) and queries cheap.
