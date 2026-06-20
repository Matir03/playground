# Protocols

All cross-process communication uses protobuf-defined gRPC services. The web
frontend uses gRPC-Web (via `tonic-web` on the Rust side and `@connectrpc/connect-web`
in the browser). Python-to-Rust and Rust-to-Python use native gRPC.

Source of truth: [`schemas/proto`](../schemas/proto).

## Services

| Service              | Host                    | Callers                           |
|----------------------|-------------------------|-----------------------------------|
| `EngineService`      | Rust game binary        | Web UI, orchestrator, other engines (matches) |
| `ModelService`       | Python model server     | Rust engine binaries (via `model-client`) |
| `OrchestratorService`| Python orchestrator     | Web UI (admin)                    |

### `EngineService`

- `NewGame(NewGameRequest) → GameHandle` — start a new interactive game.
- `ApplyMove(ApplyMoveRequest) → GameState` — play one move.
- `GetState(GameHandle) → GameState` — poll current state.
- `RequestMove(RequestMoveRequest) → stream MoveProgress` — engine thinks;
  streams intermediate MCTS stats and finalizes with chosen move.
- `PlayMatch(MatchSpec) → stream MatchProgress` — engine-vs-engine or
  engine-vs-model match; streams per-move updates; terminates with outcome.
- `StartSelfPlay(SelfPlaySpec) → stream SelfPlayProgress` — batched self-play
  mode; streams completed-game notifications; writes `GameRecord`s to a path
  specified in the request.

### `ModelService`

- `Encode(EncodeRequest) → EncodeResponse` — concrete state → LGS.
- `Eval(EvalRequest) → EvalResponse` — LGS → scalar value.
- `Policy(PolicyRequest) → PolicyResponse` — LGS → B (action, next-LGS) pairs.
- `Reify(ReifyRequest) → ReifyResponse` — (LGS_a, LGS_b) → concrete move.
- `EncodeBatch / EvalBatch / PolicyBatch / ReifyBatch` — batched variants
  for throughput. The `model-client` Rust crate does the micro-batching.
- `ReloadWeights(ReloadWeightsRequest) → ReloadWeightsResponse` — swap to a
  new checkpoint in-place. Returns the loaded version id.
- `Health(Empty) → HealthResponse` — liveness + current model version.

### `OrchestratorService`

Admin surface — not called on the hot path. Used by the web UI.
- `ListRuns`, `GetRun`, `StartRun`, `StopRun`
- `ListModels`, `GetModel`, `PromoteModel`
- `ListEvals`, `StartEval`, `GetEloHistory`

Exact shapes evolve; see [schemas.md](schemas.md) for evolution rules.

## Engine binary CLI

Each `crates/games/<name>` binary exposes the same subcommands:

```
<game>-engine serve      --port 7100                 # interactive + match
<game>-engine self-play  --port 7100 --workers 16    # adds batched self-play
<game>-engine match      --white <uri> --black <uri> # client-side match runner
<game>-engine stdio                                   # UCI-like text protocol for tools
```

The `stdio` mode is a convenience for debugging and for tools that can't
speak gRPC; it carries the same semantics as `EngineService` but framed as
newline-delimited JSON (or text commands — TBD; decide when we wire up a
second tool that needs it).

## Why gRPC-Web from the browser?

We wanted the Connect-ES developer experience (typed clients, streaming,
nice errors) from TS. Connect-ES speaks three transports: Connect, gRPC, and
gRPC-Web. gRPC-Web is the one browsers can speak directly to a
`tonic-web`-wrapped Rust server without an extra proxy. See
[decisions.md](decisions.md#d4-grpc-web-from-the-browser).
