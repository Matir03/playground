# Minions: best-first step search ("steps" mode)

A second training setup for Minions beside the sampled-AlphaZero stack
(diffusion proposer + energy head + sampled MCTS). Both share the rules, the
realizer core and the ORT/TensorRT plumbing; nothing here changes the old
stack. Code: `crates/games/minions/src/{steps,bfs,stepsp}.rs`,
`python/minions_steps_learner.py`; subcommands `steps-selfplay`, `steps-check`.

## Idea

The search, not the game, produces the training data. Every node the search
descends into often enough (`--min-visits`, default 20) becomes a value
record labelled with its backed-up value. Games only decide which positions
get searched. There is no policy: every option at every step is enumerated
and statically evaluated.

## The step tree (`steps.rs`)

A turn is built one variable at a time:

1. **Fates.** Each enemy piece, in a per-turn shuffled order, is declared
   stay / kill / unsummon.
2. **Moves.** Each friendly unit, in a per-turn shuffled order, picks one of
   its 37 destination offsets.
3. **Spawns.** Each open spawn cell, in a shuffled order, gets nothing or one
   affordable unit.

The orders are seeded by the position and a per-game salt, so a state always
rebuilds identically. Between moves and spawns the attack part is realized;
spawns then run on the real spawn-phase position.

**Witnesses: the tree has no dead ends.** Every option offered carries a
*witness* -- a complete fates+moves declaration that extends the decisions so
far with that option and realizes in full. Options are checked in this order:
the parent's witness with this option swapped in (a single realize), else a
freshly built witness (attackers placed on distinct hexes in range of their
targets by a small positional search, units forced off claimed origins
relocated, everyone else staying). Cheap filters run first: reachability,
claimed/occupied hexes, move cycles, a vertex-disjoint-paths max flow for
units whose origins get claimed, a deficiency bound on the kills. The option
the current witness itself uses is always offered, so some option always
exists. Measured: 0 dead ends in self-play (earlier relaxation-only versions
hit ~0.8 per turn in random walks and made the post-search play-through walk
thrash).

*Tradeoff:* the witness is conservative. Legal options with no witness found
(within `WITNESS_ATTEMPTS` / `POSITION_BUDGET`) are dropped. Options per step
fell from ~4.2 to ~2.3 in random walks; part of that was genuinely
infeasible kills, part is lost legal turns (crowded formations that need
shuffles). Not yet quantified.

**Realizer fallback.** `turn::realize` plans all kills at once and misses
turns where one kill must open a hex that a unit moves into before attacking
(warg kills A, serpent steps onto A's hex, serpent kills B). `attack_part`
retries with a sequential realizer (moves, one kill, repeat; each target
tried first in turn).

**Forced steps** (one option) collapse at expansion time, never across a
turn boundary, so a node's mover never changes under it.

**Features** (`View`): the turn so far *projected* onto the board (fated
enemies removed with their rebate/reinforcement, decided units at their
destinations, spawns placed) plus marker planes: decided friends, decided
enemies, decided spawn cells, and the next variable to decide. Sparse layout
`plane * 49 + cell` over 17 planes (12 unit planes, graveyards, 4 markers) =
833 bits, plus 16 scalars (money, reinforcements, turns left, stage one-hot).
Dense net input is those 849 values.

## The search (`bfs.rs`)

Best-first over the step tree with **K heads** per iteration:

- Walk the PV from the root (best child for each node's mover).
- Candidates are all live children of all PV nodes, scored by how far their
  value falls below the PV child's at the same node; the PV's own end has gap
  0. The K smallest gaps are taken; each descends along its own PV to an
  unexpanded leaf; duplicates dropped.
- Every leaf is expanded (all options enumerated, every child evaluated) in
  one batched evaluator call.
- **Backup:** a node's value is the visit-weighted power mean (p = 2) of its
  children's values from its mover's point of view, weights `1 + descents`.
  (Pure descent weights would leave a freshly expanded node undefined, since
  none of its children has been descended into yet; the +1 counts each
  child's own evaluation.)
- Values are win probabilities for the node's mover; `flip` marks a child
  whose turn changed hands.

Node states are stored at creation (no replays); a move keeps the chosen
subtree (`reroot`).

## Self-play (`stepsp.rs`, `steps-selfplay`)

- Random starting positions: `selfplay::random_opening` (random bid, a
  geometric number of random turns).
- Per move: search until `--expansions` leaf expansions (or `--move-secs`,
  or the candidate pool starves -- the PV runs into decided lines).
- Play one whole turn step by step: descents with temperature + Dirichlet
  noise; by value where nothing below was descended; dead picks backed out.
- Records: every expanded node with `>= --min-visits` descents, written when
  it leaves the tree (not in the kept subtree) or at game end. 272-byte
  records, layout in `stepsp.rs`.

## Evaluators

| evaluator | where | notes |
|---|---|---|
| `OldNetEval` | GPU (TRT) | the sampled-AZ champion's value head on the projected board; the bootstrap, no step training needed |
| `OnnxEval` | GPU (TRT) | any step-trained net exported by the learner; one call per worker expansion, sessions pooled |
| `BatchedOnnxEval` | GPU (TRT) | same nets behind a shared queue (`--batcher N` runner threads): workers featurize and enqueue, a runner drains up to 1024 rows into one padded call (shapes 256 / 1024) |
| `CpuMlpEval` | CPU | the learner's `mlp:` nets from `mlp.bin`; sparse first layer; matches ONNX to 2e-7 (`steps-check`) |

## Bootstrapping

Generation 0 uses `OldNetEval`. This replaces the planned diffusion-tree
seeding: the projected board is a legal-looking position, so the old value
head gives a usable static value for every partial-turn node with no extra
machinery, and the search's backups then correct it. The first step-trained
nets train on that data.

## Measurements (2026-09-27, laptop 4060, 8 workers x 32 heads)

- With `OldNetEval`: ~39-44k expansions/s, ~160-180k evals/s, ~2.2k
  records/s; eval 88-89% of worker time (GPU-bound).
- Records are ~113 per move at 2000 expansions/move.

(See the net study section below for fit vs speed.)

## Net study: fit vs speed vs strength (gen0 data, 2026-09-27)

Generation 0: 3.0M records from 548 games, the old champion as evaluator,
6000 expansions/move. Each architecture trained 4 epochs
(`data/models/minions/steps/study.sh`); holdout = whole chunk files (other
games); speed = end-to-end `steps-selfplay` with that net as the evaluator
(GPU nets via TensorRT fp16 at a fixed 256-row shape, CPU MLPs with 10
workers); strength = 50 seat-swapped greedy pairs vs the old champion from
random openings (SE ~0.05).

| arch | where | params | holdout MSE | greedy vs old | expansions/s | evals/s |
|---|---|---|---|---|---|---|
| mlp 256,32,32 | CPU | 227k | 0.0141 | 0.31 | **227k** | **987k** |
| mlp 512,64,32 | CPU | 470k | 0.0158 | 0.34 | 166k | 728k |
| mlp 1024,256,64 | CPU | 1.15M | 0.0170 | 0.32 | 89k | 381k |
| **res 32x4** | GPU | 135k | 0.0096 | **0.43** | 63k | 288k |
| res 64x6 | GPU | 514k | **0.0094** | 0.41 | 49k | 231k |
| res 96x8 | GPU | 1.41M | 0.0106 | 0.32 | 21k | 105k |
| fcres 64x6 | GPU | 3.16M | 0.0153 | 0.36 | 48k | 215k |
| old champion (bootstrap) | GPU | 2.25M | 0.0101* | 0.50 | 41k | 170k |

\* static values vs targets its own search produced -- not a fair holdout.

Readings:

- **Data-limited.** Every family gets *worse* with capacity (MLP 0.0141 ->
  0.0170; ResNet 64x6 -> 96x8; the 3.2M-param fcres worst of the convs).
  548 games is little data, however many records it yields.
- **Convolution's prior wins at this size**: the 135k-param res 32x4 fits
  and plays best.
- **The CPU MLP is the throughput outlier**: 5.5x the old-net pipeline and
  3.6x res 32x4, CPU-bound (expand 70%). If more data closes its fit gap it
  is the obvious engine; re-test it once the buffer holds more games.
- The FC-first-layer ResNet (the "NNUE on GPU" idea) buys nothing here: the
  dense 849 -> 49*ch layer is most of its parameters and overfits.
- Fixed ONNX shapes matter: variable batch sizes made TensorRT build an
  engine per shape (res 32x4 measured 2.3k expansions/s before padding every
  call to 256 rows, 63k after).

The generation loop (`data/models/minions/steps/cycle.sh`) starts from res
32x4: self-play 1.5M records with the latest net, train on a 4-generation
window warm-started from it, log greedy matches vs the previous generation
and vs the old champion (not gating).

## Batching across workers (2026-09-27)

With one ONNX call per worker expansion (~150 rows padded to 256) the GPU sat
at ~46% while eval took 70-85% of worker time: latency-bound, not
throughput-bound. Session count barely mattered (8 workers: 2 sessions 67k,
8 sessions 77k expansions/s; 16 x 16 73k). A shared queue drained by one or two
runner threads into 1024-row calls (`BatchedOnnxEval`) fixed it. ABBA,
res 32x4, 130 s per arm:

| config | expansions/s |
|---|---|
| 8 workers, 8 sessions (per-worker calls) | 79k / 83k |
| 16 workers, 2 runners | 114k / 114k |
| 16 workers, 1 runner | 129k / 125k |
| 24 workers, 2 runners | 138k / 144k |
| 32 workers, 2 runners | 150k / 155k |
| 48 workers, 2 runners | 157k / 150k |
| 32 workers, 1 runner | 133k / 134k |

GPU still at ~44% and 42 W with CPU ~22% idle at 32 x 2: per-call cost, not
compute. **Sparse input** (`value_sparse.onnx`: `idx` [B, 96] int32 active
features + `scal` [B, 16], scattered to the dense row inside the graph; ~9x
fewer bytes per row) on top, ABBA, 32 workers:

| input, runners | expansions/s |
|---|---|
| dense, 2 | 154k / 152k |
| **sparse, 2** | **185k / 191k** |
| sparse, 3 | 169k / 167k |
| dense, 3 | 130k / 133k |

The learner exports both graphs; `BatchedOnnxEval` picks the mode from the
graph's inputs; the cycle uses the sparse graph when present. Dense vs sparse
under TensorRT fp16: max |diff| 4.9e-4 over 3000 positions (`steps-check`).

## Matches

`steps-match` plays seat-swapped pairs from shared random openings, either
greedy on static values (`--expansions 0`) or by search (`--expansions N`:
a fresh tree per turn, then the PV to the end of the turn), pairs in
parallel (`--threads`). Greedy matches are mostly decided by the opening
(76 of 100 pairs split 1-1 at gen 1), so the cycle also logs search matches
vs the previous generation and vs gen 0.

## Playing it in the UI

`serve --steps-net <net> [--steps-secs 5] [--steps-heads 32]` makes the UI's
external-engine seat (`ENGINE_PLAYER_EXTERNAL`) the step search: per turn a
fresh tree searched for the UI's `max_millis` (else `--steps-secs`), then the
principal variation's turn. `StepState::done` carries the finished turn's
declaration across the turn boundary and `steps::turn_actions` realizes it
into atomic moves (the sequential fallback included), handed out one per
request like the sampled-AZ seat's plans.

## Generation log

Scores are A's over 200 games (100 seat-swapped pairs, SE ~0.035). Search
matches: 2000 expansions per turn for both sides. From gen 2: 2M records per
generation, 6-generation window, 1 epoch at lr 1e-3 warm-started, best-holdout
checkpoint, 32 workers x 2 batcher runners.

| gen | self-play | holdout MSE (init -> kept) | greedy vs prev | search vs prev | search vs gen0 | greedy vs old champion |
|---|---|---|---|---|---|---|
| 0 | old champion, 3.0M records / 548 games | 0.0096 | -- | -- | -- | 0.43 (50 pairs) |
| 1 | 1.5M / 344 games, 68k exp/s | 0.0076 | 0.47 | -- | -- | 0.355 |
| 2 | 2.0M / 489 games, 140-149k exp/s | 0.0069 -> 0.0063 | 0.505 | 0.55 | **0.595** | 0.395 |
| 3 | 2.0M / 446 games | 0.0060 -> 0.0060 (step 0 kept: gen3 = gen2) | 0.515 | 0.54 | 0.56 | 0.38 |
| 4 | 2.0M / 431 games | 0.0058 -> 0.0057 | 0.46 | 0.46 | 0.54 | 0.40 |
| 5 | 2.0M / 469 games, sparse graph 170-193k exp/s; EMA from here | 0.0051 | 0.445 | (failed) | 0.535 | 0.345 |
| 6 | 2.0M / 472 games; gen 0 data leaves the window | 0.0013 | 0.47 | (failed) | 0.50 | 0.405 |
| 7 | 2.0M / 506 games | 0.0015 | 0.485 | (failed) | -- | -- |
| 8 | 2.0M; records carry game results from here | 0.0014 | 0.47 | (failed) | 0.55 | 0.34 |
| 9 | outcome-mixed targets from here (w 0.5, decay 0.9) | 0.0016 | 0.525 | (failed) | 0.55 | 0.305 |
| 10 | | 0.0016 | 0.525 | 0.50 | 0.585 | 0.375 |

"(failed)": the search match between two nets died on TensorRT CUDA error
906 until c37aed0 (one session per batch shape). Gen 9 and 10 kept an early
checkpoint judged on pure search targets; from gen 11 the holdout is judged
on the same mixed target the net trains on. Search vs greedy with the same
net: 0.485 (gen 9), 0.512 (gen 10).

**Training fixes after gen 4.** Learner experiments (gen 3 init, gen 0-4
data, same holdout): plain 1 epoch best-checkpoint 0.00561; with a weight EMA
(`--ema 0.999`, evaluated and exported) 0.00537; EMA at lr 3e-4 0.00544; EMA
over 2 epochs 0.00540. The cycle uses EMA 0.999 from gen 5.

Gen 3 is gen 2's weights (no checkpoint beat the init on the holdout), so
its 0.54 vs gen 2 is a same-net match: a calibration of the noise. Strength
vs gen 0 has been flat since gen 2; warm-started training barely moves the
holdout (the peak of the one-cycle schedule knocks the net off and one
epoch does not recover).

## Diagnosis: the loop reached a fixed point (gen 5-8)

Search matches vs gen 0 stayed at 0.50-0.55 from gen 2 to gen 8 while the
holdout kept improving (MSE 0.0051 at gen 5; 0.0013-0.0015 from gen 6, when
the old champion's gen 0 data left the window and the holdout became
self-generated data only, r = 0.996).

The direct test is search vs greedy step picking **with the same net**
(gen 5, 100 pairs each, SE ~0.035):

| A | B | A score |
|---|---|---|
| search 2000, p = 2 | greedy | 0.47 |
| search 6000, p = 2 | search 2000, p = 2 | 0.46 |
| search 2000, p = 8 / 32 / max | greedy | 0.51 / 0.59 / 0.53 (60 pairs) |
| search 2000, p = 32 | search 2000, p = 2 | 0.515 |
| search 6000, p = 32 | greedy | 0.505 |
| search 2000, p = 32 | search 2000, max | 0.53 |
| **material evaluator**: search 2000, p = 2 | greedy | **0.73** (50 pairs) |
| material evaluator: search 2000, max | greedy | 0.46 (50 pairs) |

So the search works (it beats greedy by a wide margin on a crude evaluator)
but adds nothing on top of the trained net: the net has learned to predict
its own search's backed-up values, so greedy picking is already the
distilled search, and each generation's targets equal the previous
generation's predictions except where a tree reaches terminal positions.
New information enters only through those, roughly a turn of look-ahead per
generation. (Selective expansion also makes max backup over-optimistic:
it loses to p = 2 with the material evaluator.)

**Grounding experiment (from gen 9).** Records now carry the game's result
for the record's mover and the turns to the end (`stepsp::outcome_code`, in
the former pad; 0 = unknown). The cycle trains on
`(1 - w) * search value + w * result` with `w = 0.5 * 0.9^turns_to_end`
(`--outcome-weight 0.5 --outcome-decay 0.9`) where the result is known
(gen 8 on). This departs from the design's "games are exploration only":
self-play still picks moves with temperature 1 + Dirichlet at every step, so
results measure that exploratory policy. The cycle also logs search vs
greedy with the same net each generation (the improvement the search still
offers).
