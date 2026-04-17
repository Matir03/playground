# Training

## The self-play loop

Driven by `python/src/playground/orchestrator`. One "run" looks like:

```
for epoch in run:
    kick off N engine workers in batched self-play mode
    each worker plays M games against the current model
    records land in data/runs/<run_id>/games/
    when >= G new records since last training step:
        training reads a sliding window of records
        produces new checkpoint under data/models/<name>/<version>/
        orchestrator issues ReloadWeights to model server(s)
        engines continue with new model
    periodically: kick off eval tournament against fixed baselines
```

Parameters (config-driven):
- `N` worker processes, `M` games per worker per epoch.
- `G` games-between-training-steps; sliding window size `W`.
- Eval cadence (games or wall time).

## Replay storage

See [schemas.md](schemas.md#storage). Append-only files + a thin sqlite
index. Training reads the index, picks a window by recency or by a
configurable mix, and streams the selected files.

## Model registry

```
data/models/<name>/<version>/
  checkpoint.safetensors
  config.json
  meta.json               # parent version, training step, run id, timestamps
```

Versions are opaque IDs (ULID or hash). "Current" is a symlink. Promotion
is a single atomic symlink swap + a `ReloadWeights` RPC.

## Training stages

Per [algorithm.md](algorithm.md):

1. **Encoder + reifier.** Pair data mined from recorded games: for every
   observed move M : A → B, target `reify(encode(A), encode(B)) = M`.
   No MCTS targets needed here.
2. **Encoder + eval.** Once encoder is somewhat healthy, fold in value
   regression. Targets: concrete-tree value estimates, with game-end nodes
   clamped to true outcome; optionally regularized against true final result.
3. **Policy.** Only once encoder is stable. Target is the concrete-MCTS
   visit policy re-encoded into GS space (see algorithm.md).

Each stage has its own training script / config; orchestrator decides which
stage to run based on run config + current training step.

## Hot-swapping models

Two paths, both fine, switch per situation:

- **In-place**: model server reloads weights on `ReloadWeights`. Cheap, but
  in-flight requests may complete with the old weights. Fine for self-play
  since we tag records with the model version actually used.
- **Blue/green**: orchestrator spawns a new model server, health-checks it,
  points engines at it, then drains the old one. Use for eval where we
  don't want any mixing.

The engine tags every `GameRecord` with `model_version` as reported by the
model server's `Health` endpoint at game start. That's enough to
reconstruct which games came from which weights.
