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
| `OnnxEval` | GPU (TRT) | any step-trained net exported by the learner |
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
