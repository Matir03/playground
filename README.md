# playground

Research playground for a custom MuZero variant. We validate the algorithm on a
sequence of progressively richer 2-player abstract strategy games
(Connect4 → 9×9 Go → Amazons → Arimaa → Hex 13×13) before applying it to the
target game.

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
moves run the reification-based MCTS from [docs/algorithm.md](docs/algorithm.md):
search proceeds in GS (game-state latent) space — `policy` expands children,
`eval` scores leaves, and promising edges are **reified** (`reify`) into concrete
moves that advance the real game and are re-`encode`d. So NN play needs the
Python **model server** running:

```bash
cd python
python -m venv .venv && . .venv/bin/activate
pip install -e '.[model]'

# One model server per net (different ports / checkpoints).
playground-model-server --game connect4 --port 7200 \
    --checkpoint data/models/connect4/random/checkpoint.safetensors &
playground-model-server --game connect4 --port 7201 \
    --checkpoint data/models/connect4/bc-perfect/checkpoint.safetensors &
playground-model-server --game connect4 --port 7202 \
    --checkpoint data/models/connect4/selfplay/checkpoint.safetensors &

# One engine, wired to all three under names the UI selects.
cargo run --release -p connect4-engine -- serve --addr 127.0.0.1:7100 \
    --model random=http://127.0.0.1:7200 \
    --model trained=http://127.0.0.1:7201 \
    --model selfplay=http://127.0.0.1:7202
```

In the UI each seat can be `NN — random`, `NN — trained (cloned solver)`, or
`NN — self-play`; the engine routes the move to the matching model server. The
model server reads the net's architecture from a sibling `config.json` (the
larger self-play ResNet vs the small cloned/random nets), so one server binary
serves any checkpoint.

#### Embedded (in-process) NN — no Python, no gRPC round-trips

The MCTS above evaluates the net by a gRPC call **per node** to the Python model
server, so the round-trips cap the search (~310–360 visits/s at a 1 s budget).
You can instead run the net **inside the engine** with `--model-local
NAME=CHECKPOINT`: the same `.safetensors` checkpoints are loaded with
[`candle`](https://github.com/huggingface/candle) and inference is a local
function call (architecture still read from the sibling `config.json`). No Python
process is involved.

```bash
cargo run --release -p connect4-engine -- serve --addr 127.0.0.1:7100 \
    --model-local random=data/models/connect4/random/checkpoint.safetensors \
    --model-local trained=data/models/connect4/bc-perfect/checkpoint.safetensors \
    --model-local selfplay=data/models/connect4/selfplay/checkpoint.safetensors
```

The UI is unchanged — it just selects a name. Removing the round-trips lets the
same 1 s budget reach far more visits: ~12,800/s for the small nets (≈41×) and
~630/s for the larger self-play ResNet (≈1.7×). `--model-local` and `--model`
(remote) can be mixed in one engine. The net runs single-threaded by default
(batch-1 inference on the 6×7 board is faster without rayon's threading
overhead); override with `RAYON_NUM_THREADS`.

The UI shows per-move visit and reification counts, and a per-seat **search
limit** for the NN — either a time budget (default **1 second/move**) or a fixed
visit count (`RequestMove`'s `max_millis` / `max_visits`; a time budget runs
uncapped rollouts until it elapses). The MCTS itself lives in
`crates/core/src/mcts.rs` and uses **KR-UCT** selection (kernel regression over
action vectors; σ → 0 recovers PUCT). Noisy reified siblings / progressive
widening (see docs/algorithm.md) are the remaining refinements, relevant for the
larger games.

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

Each game has its own binary. All of them speak the same `EngineService` on
gRPC + gRPC-Web. Default ports:

| Game      | Binary            | Engine port | Expects model server at |
|-----------|-------------------|-------------|-------------------------|
| Connect 4 | `connect4-engine` | `7100`      | `http://127.0.0.1:7200` |
| Go 9×9    | `go9-engine`      | `7101`      | `http://127.0.0.1:7201` |
| Amazons   | `amazons-engine`  | `7102`      | `http://127.0.0.1:7202` |
| Arimaa    | `arimaa-engine`   | `7103`      | `http://127.0.0.1:7203` |
| Hex 13×13 | `hex-engine`      | `7104`      | `http://127.0.0.1:7204` |

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

### Hex 13×13 and MoHex

The Hex rules live in [crates/games/hex](crates/games/hex) (`hex-engine`).
Black (First) connects top↔bottom, White (Second) left↔right — benzene's
convention, so our board and MoHex's agree cell for cell (`a1`..`m13`). The
swap rule is off by default; see `ALLOW_SWAP` in `src/game.rs` for what that
costs and how to turn it on. No AZ net yet: `RequestMove` is answered by a
placeholder, and the strength work comes next.

The web UI has a playable board (`cd web && npm run dev`) with three seat
options per colour: human, the placeholder, and **MoHex**.

**MoHex 2.0** (benzene) is the reference engine — the Hex equivalent of go9's
GNU Go. Build it once; the script clones, patches the 2015 source for a modern
toolchain (see [scripts/patches](scripts/patches)), pulls Boost + Berkeley DB
into a micromamba env if the system lacks them, and symlinks the binary into
`~/.local/bin`:

```bash
scripts/fetch-mohex.sh

# Serve with MoHex as a selectable opponent (UI seat "MoHex 2.0"); the seat's
# seconds/move becomes its per-move time budget.
cargo run --release -p hex-engine -- serve --addr 127.0.0.1:7104 \
    --mohex-bin "$(which mohex)" --mohex-playouts 10000

# Cross-check our rules against MoHex: random games, ply-by-ply agreement on
# who has connected. This is the real test that the board geometry is right.
cargo run --release -p hex-engine -- verify --games 40

# Play MoHex (here against a random stand-in until the net lands).
cargo run --release -p hex-engine -- match --games 4 --mohex-playouts 1000

# Rules throughput — the floor for AZ self-play (~4.5M moves/s single-threaded).
cargo run --release -p hex-engine -- bench --games 3000
```

`HEX_GTP_DEBUG=1` traces the GTP conversation and keeps MoHex's own search log.

### Hex AlphaZero

Plain AlphaZero — a direct policy/value net over real board states, PUCT search,
self-play, train, repeat. No latent space, no reifier, no encoding: the
MuZero-variant machinery the other games exercise is deliberately absent here,
so Hex is the control experiment.

* **Net** (`crates/games/hex/src/net.rs`, `python/src/playground/model_server/hex_net.py`)
  — ResNet trunk, one policy logit per cell (169) and a scalar value. The two
  files must stay in sync: they load the same `.safetensors`.
* **Input** — 2 planes (mine, theirs) in the *canonical frame*: when White is to
  move the board is transposed, so the net always sees itself as the player
  connecting top↔bottom. Hex's transpose+colour-swap symmetry makes the colours
  literally the same problem, which is why there is no side-to-move plane.
  `game::canon_index` maps policy indices back to real cells.
* **Search** — `MoveGame` on `core::batch_search`, the same GPU-saturating engine
  amazons uses: hundreds of games in flight, their leaf evaluations coalesced
  into one forward. Priors are a softmax over the legal cells only.
* **Training example** — one per ply: canonical planes, the game outcome z from
  that position's mover's point of view (±1 — Hex has no draws), and the MCTS
  visit distribution. Dense, 169 floats.

```bash
# Terminal A — the actor. Writes chunks into <live>/buffer, hot-reloads weights.
# (Needs a gen-0 net; the learner writes one on its first run.)
source /tmp/cuda_env.sh   # this box: see the CUDA build notes
cargo build --release -p hex-engine --features cuda
./target/release/hex-engine selfplay-loop --live data/models/hex_az/live --cuda \
    --concurrency 384 --max-batch 128 --node-budget 400

# Terminal B — the learner. Trains a generation per burst of fresh self-play.
# It writes both latest.safetensors (candle) and model.onnx (ort) each snapshot.
PYTHONPATH=python/src python/.venv/bin/python python/hex_az_learner.py \
    --live data/models/hex_az/live

# Or let the orchestrator drive the cadence (learner + eval per generation):
PYTHONPATH=python/src python/.venv/bin/python python/hex_train_cycle.py \
    --live data/models/hex_az/live --cuda --period 900
```

Rating a checkpoint, two ways — relative (has it improved?) and absolute (is the
whole run going anywhere?):

```bash
hex-engine eval-match --a <new.safetensors> --b <old.safetensors> --cuda --games 32
hex-engine vs-mohex  --model <ckpt> --cuda --games 10 --sims 400 --mohex-playouts 1000
```

#### Inference backends

Self-play runs the net through one of two backends, chosen with `--backend`:

| Backend | Weights file | Forward (batch 256) | Self-play |
|---------|--------------|--------------------:|----------:|
| `candle` (default) | `latest.safetensors` | 4.6k pos/s | 3.0k nodes/s |
| `ort --ep cuda` | `model.onnx` | 38k pos/s | 26k nodes/s |
| `ort --ep trt` (fp16) | `model.onnx` | **158k pos/s** | **38–63k nodes/s** |

Measured on a laptop RTX 4060 with a 64ch/6b net, `--node-budget 200`,
`--concurrency 768 --max-batch 256` (the top self-play figure at 3072/768).
TensorRT is ~15× candle end to end, which is the difference between a
generation per day and one per hour. The self-play range is run-to-run
variance on a thermally-capped laptop GPU, so compare backends within a
sitting, not across.

**Why self-play nodes/s is far below the raw forward rate**, and why that's
not a contradiction: in Hex every search node is one network evaluation
(`cache hits 0%` — no transpositions worth the lookup), so the two figures
are the same unit. `selfplay-bench`'s breakdown shows where the difference
goes — of the single inference thread's time, only ~46% is inside `forward`;
~49% is *dispatch* (gathering each leaf batch and scattering results back to
their games) and ~5% starved. The workers themselves are nearly idle (8.5
core-seconds out of 220 available), so the limit is that one thread's serial
bookkeeping, not tree search. `net-bench` is an upper bound by construction:
one preallocated buffer, no encoding, no scatter, nothing else running.

ORT needs its shared libraries located at runtime:

```bash
source scripts/ort-env.sh    # ORT_DYLIB_PATH + the CUDA/TensorRT libs from the venv
./target/release/hex-engine selfplay-loop --live data/models/hex_az/live \
    --backend ort --ep trt --concurrency 768 --max-batch 256 --node-budget 400
```

Without it ORT quietly falls back to the CPU provider — slower than candle on
the GPU — so check the log for `CUDAExecutionProvider` registration errors if
the numbers look wrong. TensorRT builds an engine per batch shape on first use
(a few seconds) and caches it under `<live>/trt_cache`. fp16 costs about 4
decimal places on the value head and nothing on the move ranking:
`net-eval --backend ort --ep trt` against `--backend candle` is the check.

One upstream wart: dropping a TensorRT session aborts the process with heap
corruption about one time in three (never under a debugger — it's a race in
ORT's shutdown; the CUDA EP and candle are clean). `OrtModel` therefore leaks
its session rather than dropping it, which costs one session per hot-reload
and keeps the actor alive across weight updates. See `ort_net.rs`.

Two throughput probes, because "self-play is slow" has two very different
causes: `net-bench` times the raw forward (the ceiling), `selfplay-bench` the
whole search (what you actually get).

`hex-engine net-eval --model <ckpt> --moves g7,d4` prints the net's value and top
priors for a position — the quickest way to see what a checkpoint believes.

Play a trained net in the UI with
`hex-engine serve --model-local nn=<ckpt> --cuda`.

Interactive MoHex play goes through `RequestMove(player = EXTERNAL)`, a
protocol-level seat for "the reference engine this game crate drives itself"
(`Solver::choose_move_external`; the natural home for GNU Go in go9 too). The
engine holds one MoHex process and syncs it to each position — incrementally
when the position extends what it last saw, so MoHex keeps its search tree.
Without `--mohex-bin` those requests fail with a failed-precondition rather
than quietly falling back to a weaker mover.

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

The full self-play → encoder/reifier/eval/policy stage schedule (see
[docs/algorithm.md](docs/algorithm.md)) is still M4. What works today is a
**behavioral-cloning** pipeline for Connect 4: the random net plays itself, and
every position is annotated by the perfect solver — its best move (policy
target) and its **evaluation** of the position (value target). We train on the
solver's annotation, not the realized game scores, so the value target is a
graded assessment rather than "who won this random game".

```bash
# 0. Save the random-init net (kept as a baseline / opponent).
playground-train save-random          # -> data/models/connect4/random/checkpoint.safetensors

# 1. Generate data. Needs a running engine + the random model server.
#    Random-net self-play, annotated by the perfect bot's policy + value.
playground-train gen-data --games 300 # -> data/datasets/connect4_annotated.npz

# 2. Train (encoder + eval + policy heads) into the model registry.
playground-train train --data data/datasets/connect4_annotated.npz \
    --epochs 40                       # -> data/models/connect4/bc-perfect/checkpoint.safetensors
```

`train` prints the held-out move-match rate against the perfect bot. The model
server's `--checkpoint` (and the `ReloadWeights` RPC) load any `.safetensors`
checkpoint; reify is a structural column decode for Connect 4, so the trained
policy+value heads are what drive the NN's MCTS. Serve the random and trained
nets side by side (see "the neural-net engine" above) to pick between them in
the UI.

There is also a **pure self-play** trainer (AlphaZero-style, no solver): a
larger residual net plays itself with an in-process PUCT MCTS and trains on its
own games — policy target = MCTS visit counts, value target = game outcome —
iterated over generations from random init. It saves a `config.json` recording
the architecture so the model server can serve the bigger net.

```bash
playground-train selfplay --generations 10 --games-per-gen 120 --sims 60 \
    --channels 64 --blocks 5     # -> data/models/connect4/selfplay/checkpoint.safetensors
```

Each generation prints its win rate against the random baseline. This is a
separate, larger network; the random and cloned-solver nets are untouched.

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
start whichever engines you want to exercise before opening the page. Engines
are reached through same-origin proxy paths (`/c4`, `/go9`, `/a`, `/hex` — see
[web/vite.config.ts](web/vite.config.ts)), so restart the dev server after
adding one. Playable boards today: Connect 4, Go 9×9, Amazons, Hex 13×13.

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
