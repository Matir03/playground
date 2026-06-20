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

## Training objective

One combined loss with weighted terms, see [algorithm.md](algorithm.md#training)
for the exact math. Five terms:

1. **Value** — `L_v = (v(E(S)) − z)²`, MSE through encoder + eval; `z` is the
   bootstrapped search value, terminal nodes clamped to true outcome.
2. **Reifier** — `L_ρ`: for every visited child, recover its concrete move from
   the encoder's own endpoint embeddings, `ρ(E(S), E(S_i)) = A_i`.
   Autoregressive, legality-masked; trained jointly with the encoder.
3. **Policy** — `L_π`: match the policy MDN to the search's visit distribution
   over next-LGS (the "true kernel distribution"). Default visit-weighted NLL;
   MMD-with-`κ` as the faithful alternative.
4. **Consistency** — `L_c`: over random *cross-dataset* state pairs, make latent
   distance predict functional distance,
   `(α‖E1−E2‖ − [|v(E1)−v(E2)| + λ_p D(π(E1),π(E2))])²`. Gradients through all
   three of encode/eval/policy. This is what makes the latent metric smooth
   enough for KR-UCT — keep it on throughout.
5. **Regularization / anti-collapse** — weight decay, latent bounding, mixture
   entropy, VICReg-style variance terms.

### Curriculum

Stage in roughly this order; each stage is a config that adjusts term weights,
not a separate script:

1. **Encoder + reifier + value first.** Get an invertible, value-aware latent.
   No policy target needed yet.
2. **Add the policy MDN** once the latent is stable (otherwise the visit-policy
   target is noise).
3. **Consistency on throughout**, weight `λ_c` ramped — watch reifier recall as
   it rises (the encoder must stay invertible; see algorithm.md "central
   tension").

The orchestrator decides the active stage from run config + current training
step.

### Efficiency optimizations

Add small-weight **auxiliary targets** (opponent-reply policy, short-term value,
game-specific dense targets like ownership) to the objective — cheap
representation-learning wins. On the self-play side, **playout cap randomization**
governs which moves yield policy targets, and **forced playouts + policy target
pruning** shape the visit-count target before it's recorded. The encoder trunk
uses **global pooling**. All four are specified in
[algorithm.md](algorithm.md#self-play-and-training-optimizations).

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
