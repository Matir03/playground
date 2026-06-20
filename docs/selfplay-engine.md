# LGS self-play engine (implemented, Connect 4)

How the reification-based latent-space algorithm ([algorithm.md](algorithm.md))
is actually run for self-play. This documents the **implemented** Connect 4 stack
(`crates/games/connect4` + `python/...`), which coexists with the discrete-AZ
baseline rather than replacing it.

## Two network implementations, kept in parity

The net has heads over a tanh LGS latent (`d = 64`): encoder (ResNet trunk +
**global pooling**) → LGS; **eval**; **MDN policy** (`K=8` diagonal Gaussians
over next-LGS); **reifier** (`[LGS‖next-LGS] → 7 column logits`); aux heads.

| Where | File | Role |
|-------|------|------|
| PyTorch | [c4_lgs_net.py](../python/src/playground/model_server/c4_lgs_net.py) | source of truth; training; MDN NLL + closed-form MMD helpers |
| candle  | [lgs_net.rs](../crates/games/connect4/src/lgs_net.rs) | in-process inference for self-play (`LgsModel`, `LgsServer`) |

A gated test (`lgs_net::tests::parity_with_python_lgs_net`) checks the candle net
matches PyTorch on value/LGS/MDN within 1e-4 for a fixed checkpoint.

## Core interface (game-agnostic)

[crates/core/src/model.rs](../crates/core/src/model.rs):
- **`MixturePolicy`** — diagonal Gaussian mixture over next-LGS, with
  allocation-free `log_density` and `sample`.
- **`LatentModel`** trait — `encode` / `eval` / `policy` (MDN) / `reify`.
  Separate from the discrete `ModelClient`, so go9/amazons are unaffected.

The search lives in [latent_mcts.rs](../crates/core/src/latent_mcts.rs)
(KR-PUCT, reify-on-2nd-visit, aliasing, progressive widening, forced playouts)
and is **async over `LatentModel`** — the same code serves both the simple
single-game driver and the throughput engine below.

## Throughput: central-batched lockstep self-play

The throughput path (`connect4-engine lgs-gen --concurrency … --threads …
--max-batch …`, in [lgs_selfplay.rs](../crates/games/connect4/src/lgs_selfplay.rs))
mirrors the discrete `core::batch_search` design, adapted to the four LGS op
types. Goal: keep a fixed-size GPU batch full while doing tree work on all cores.

```
                          shared, sharded
        worker 0 ─┐       transposition cache
        worker 1 ─┤  (encode/policy by state)
          …       │            ▲
        worker T ─┘   miss     │ inline hits
                       │       │
                  wave │       │            ONE inference server thread
                       ▼       │            (the only GPU consumer)
        ┌───────────────────────────────┐   coalesce waves ≤ max_batch
        │ central inference server       │── encode/eval/policy/reify
        │ (1 thread, batched forwards)   │   one batched forward per type
        └───────────────────────────────┘   → scatter results back
```

Key properties (and why):
- **Pool ≫ batch, continuous replenishment.** Each worker drives `in_flight`
  games in **lockstep** (one pending net op per active game per round). The total
  pool = `threads × in_flight` is kept far larger than `max_batch`; finished
  games are immediately replaced from a shared `AtomicUsize` budget, so batches
  stay full and shared positions stay co-resident (better cache hits).
- **One central inference server thread.** Workers submit whole *waves* of op
  *misses*; the server coalesces waves across workers up to `max_batch` and runs
  **one batched forward per op type**. A single GPU consumer avoids the
  catastrophic serialization of concurrent CUDA launches from many threads
  (measured: multi-thread GPU forwards collapse to ~1 game/s).
- **Async search, manual lockstep executor.** Workers reuse the async
  `latent_mcts::search` unchanged; a no-op-waker executor polls each game's
  future to its next op (registered via a per-slot `Lane`), so there is no tokio
  runtime / per-op channel churn on the hot path.
- **Sharded transposition cache.** `encode` and `policy` are only ever called on
  *real* board states (which Connect 4 transposes heavily), so they are cached by
  a hash of the input floats in a 64-shard map with per-key short locks. `eval`
  and `reify` are **not** cached — they hit unique sampled latents and would
  pollute the cache.

### Record format (v2)

Self-play writes a fixed-stride binary dump (see `lgs_selfplay.rs`):
`parent planes | z | MAX_CHILDREN × [col, visits, child planes]` per move.
Concrete child states are stored (not stale latents) so the trainer re-encodes
parents and children with the *live* encoder — the MDN/reifier/consistency
targets must live in the current latent space.

## Training

[c4_lgs_train.py](../python/c4_lgs_train.py): `--init` seeds a random checkpoint;
training re-encodes parents+children and fits **value MSE + MDN-policy NLL +
reifier CE + consistency** (+ mixture entropy), exporting a candle-loadable
`checkpoint.safetensors` + `config.json`.

Two empirical fixes (see [training.md](training.md)):
- **Per-dimension MDN NLL.** A `d`-dim Gaussian log-density is `O(d)` and swamped
  the value/reifier gradients on the shared encoder; normalizing by `d` puts all
  terms on an `O(1)` scale.
- **Consistency defaults OFF.** The exact-match form collapses the latent while
  the value head is ≈constant (`d_func≈0 ⇒ d_lat→0`), killing value/reifier
  learning — the docs' "central tension" made real. Re-enable later as a ramped
  anti-collapse (Lipschitz) term.

## Performance characteristics (RTX 4060 laptop, 32ch/3blk, 48 sims)

| Self-play design | games/s |
|---|---|
| naive single-thread | 0.2 |
| async per-op server (CPU / GPU) | 5.4 / 10.6 |
| per-thread lockstep + cache (CPU) | 14.0 |
| central server, pool≫batch (GPU) | 17.6 |
| **+ sharded cache (GPU)** | **21.6** (~24k sims/s) |

**The workload is CPU-tree-bound, not GPU-bound, for a net this small.** During a
GPU run the GPU sits at ~35% / ~19 W; the cost is the KR-PUCT search + the single
server thread's marshaling, not the matmuls. The transposition cache hits only
~25% (self-play games diverge). Consequences:
- A faster GPU backend (fp16 / TensorRT) does **not** help here — the GPU is
  starved. It would matter only with a materially heavier encoder.
- The current ceiling (~24k sims/s) is set by per-sim CPU search cost and the
  serial inference-server marshaling; further gains need a cheaper encoder (to
  enter the GPU-bound regime) or a leaner server protocol (flat result buffers).

## Knobs

| Flag | Meaning |
|------|---------|
| `--concurrency` | games in flight **per worker** (≈ each worker's wave size) |
| `--threads` | worker threads (parallel tree work) feeding the server |
| `--max-batch` | fixed batch the server coalesces up to (256 default) |
| `--sims` | simulations per move; `--temp-moves` | opening plies sampled ∝ visits |
| `--c-puct`, `--kernel-bandwidth` (`h`) | KR-PUCT exploration + kernel scale |
| `--cuda` | run the server's forwards on the GPU (single consumer) |

Keep `threads × concurrency ≫ max_batch` so the pipeline stays full.
