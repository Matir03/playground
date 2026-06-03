# playground

Research playground for a custom MuZero variant. We validate the algorithm on a
sequence of progressively richer 2-player abstract strategy games
(Connect4 → 9×9 Go → Amazons → Arimaa) before applying it to the target game.

- Rust engines per game with shared core libs (MCTS, recording, server).
- Python owns everything model-related: training, model server, orchestrator.
- Protobuf schemas via `buf`, typed clients on all three sides.
- ConnectRPC-compatible wire (gRPC-Web from Rust `tonic-web`).
- React web frontend for viewing and playing games with a human or model on either side.

See [docs/](docs/) for the full design. Start with [docs/overview.md](docs/overview.md).

## Running components

Most commands below are milestone-gated and will exit with a "implement at Mx"
message until the corresponding milestone lands (see
[docs/overview.md](docs/overview.md)). What *is* working today: the Rust
workspace builds, generated bindings for all three languages, the engines'
`Health` RPC, the full interactive-play surface (`NewGame` / `GetState` /
`ApplyMove` / `RequestMove`, with server-side legality + win detection), a
playable Connect 4 board in the web app, and a **perfect Connect 4 engine**
you can play against. No model server is required — the engine connects to it
lazily, and `RequestMove` is served by the built-in solver.

### Play Connect 4 end-to-end (no Python needed)

```bash
# One-time: fetch the opening book (~32 MB; makes opening play instant).
scripts/fetch-connect4-book.sh

# Terminal A — engine (human/perfect-engine play needs no model server)
cargo run --release -p connect4-engine -- serve --addr 127.0.0.1:7100

# Terminal B — web UI, then open http://127.0.0.1:5173
cd web && npm install && npm run dev
```

In the UI each seat (Yellow / Red) is a **human** or the **perfect engine**.
The Rust engine owns the rules; engine seats play optimally.

### The perfect Connect 4 engine

`RequestMove` is answered by a bitboard negamax solver (alpha-beta +
transposition table + move ordering) — the algorithm from Pascal Pons'
[Connect 4 tutorial](http://blog.gamesolver.org). It is **result-perfect**:
it always takes a win, blocks every immediate threat, and never loses a
position it can hold. Proving the value of a near-empty board is intractable
in real time, so the engine consults a precomputed [opening book][book]
(`data/7x6.book`, depth 14) for shallow positions and solves the rest live
(fast once the board is past the opening). Run `--release`; the book is
gitignored — fetch it with `scripts/fetch-connect4-book.sh` (or point at your
own copy with `--book <path>`). Without a book the engine still plays
perfectly but is very slow in the first ~14 plies.

[book]: https://github.com/PascalPons/connect4

### The neural-net engine (random weights)

Each seat in the UI can also be a **neural net** instead of the solver. NN
moves are served by `RequestMove` doing a 1-ply lookahead over the model's
`Encode` + `Eval` heads, so they need the Python **model server** running:

```bash
cd python
python -m venv .venv && . .venv/bin/activate
pip install -e '.[model]'
playground-model-server --game connect4 --port 7200   # random weights

# engine must point at it (this is the default --model-uri):
cargo run --release -p connect4-engine -- serve \
    --addr 127.0.0.1:7100 --model-uri http://127.0.0.1:7200
```

The net is **untrained** (random init), so it plays weakly — the point is to
exercise the full engine ↔ model-server path. The reification-based MuZero
variant (`Policy`/`Reify` heads + KR-UCT MCTS over GS) is wired in the proto
and model server but not yet used for move selection; that lands with M4 (see
[docs/algorithm.md](docs/algorithm.md)).

### Prerequisites

| Tool     | Used for                                 | Install                    |
|----------|------------------------------------------|----------------------------|
| rustup / cargo | Rust engine binaries               | <https://rustup.rs>        |
| protoc   | tonic-build invokes it for Rust codegen  | `apt install protobuf-compiler` |
| buf      | TS + Python proto codegen                | `brew install buf`         |
| python   | ≥ 3.11                                   |                            |
| node / npm | web frontend                           |                            |

### Regenerate proto bindings

Run after any change under [schemas/proto/](schemas/proto/).

```bash
scripts/gen-proto.sh
```

Regenerates TS into `web/src/gen/` and Python into
`python/src/playground/gen/playground/`. Both are gitignored. Rust regenerates
automatically on `cargo build` via [crates/protocol/build.rs](crates/protocol/build.rs).

### Rust engines

Each game has its own binary. All four speak the same `EngineService` on
gRPC + gRPC-Web. Default ports:

| Game      | Binary            | Engine port | Expects model server at |
|-----------|-------------------|-------------|-------------------------|
| Connect 4 | `connect4-engine` | `7100`      | `http://127.0.0.1:7200` |
| Go 9×9    | `go9-engine`      | `7101`      | `http://127.0.0.1:7201` |
| Amazons   | `amazons-engine`  | `7102`      | `http://127.0.0.1:7202` |
| Arimaa    | `arimaa-engine`   | `7103`      | `http://127.0.0.1:7203` |

```bash
# Build all engines.
cargo build --release

# Interactive server (one game in flight).
cargo run --release -p connect4-engine -- serve \
    --addr 127.0.0.1:7100 \
    --model-uri http://127.0.0.1:7200

# Same, with batched self-play mode enabled.
cargo run --release -p connect4-engine -- self-play \
    --addr 127.0.0.1:7100 \
    --model-uri http://127.0.0.1:7200 \
    --workers 16

# Client-side match runner (stubbed until M2).
cargo run --release -p connect4-engine -- match \
    --first-player-uri model://v1 \
    --second-player-uri engine://127.0.0.1:7100 \
    --num-games 20

# Check a running engine is alive (requires grpcurl).
grpcurl -plaintext 127.0.0.1:7100 playground.engine.EngineService/Health
```

Full protocol surface is in [docs/protocols.md](docs/protocols.md).

### Python — model server

One process per game (the server is configured for a specific game and
architecture). Connect 4 is implemented today with **random initial weights**
(torch); all `ModelService` RPCs respond. Other games are not wired yet.

```bash
cd python
python -m venv .venv && . .venv/bin/activate
pip install -e '.[model,orchestrator,training,eval,dev]'

playground-model-server --game connect4 --port 7200
```

### Python — orchestrator

Owns the self-play loop + admin RPC for the web dashboard. Stub until M2.

```bash
# Run the OrchestratorService (admin surface for the web UI).
playground-orchestrator serve

# Kick off a self-play run with a config file.
playground-orchestrator start-run --game connect4 --config configs/connect4-m2.yaml
```

### Python — training

Trains one head stage at a time (see [docs/algorithm.md](docs/algorithm.md)
for what each stage targets). Stub until M2.

```bash
playground-train --game connect4 --stage encoder-reifier \
    --records-dir data/runs/<run_id>/games \
    --out data/models/connect4/<new_version>
```

### Python — eval

Runs a tournament and fits ELO (see [docs/evaluation.md](docs/evaluation.md)).

```bash
playground-eval --game connect4 \
    --player model://connect4@v17 \
    --player model://connect4@v16 \
    --player external://perfect-solver \
    --games-per-pair 40
```

### Web frontend

```bash
cd web
npm install
npm run dev       # Vite dev server on http://127.0.0.1:5173
npm run build     # production bundle into web/dist
npm run typecheck # tsc --noEmit; run this after regenerating protos
```

The app pings `EngineService.Health` on each engine port at load time, so
start whichever engines you want to exercise before opening the page.

### Full local stack (once M2 lands)

```bash
# Terminal A — model server for connect4
playground-model-server --game connect4 --port 7200

# Terminal B — engine (batched self-play mode)
cargo run --release -p connect4-engine -- self-play \
    --addr 127.0.0.1:7100 --model-uri http://127.0.0.1:7200

# Terminal C — orchestrator admin surface
playground-orchestrator serve

# Terminal D — web UI
cd web && npm run dev
```
