# Minions: turn-lattice two-tier search (design, 2026-10-03)

Replaces the steps-mode resample/two-tier experiments
([minions-steps.md](minions-steps.md), last section, shelved). This is the
user's design, written down with the choices that were left open. Items
marked **(default)** are my reading where the spec left room. They are easy
to change.

## The turn lattice

A turn is a set of *commitments*: each enemy piece gets a fate, each
friendly unit a destination, and each spawn cell a spawn or nothing. A
**lattice node** is a partial commitment that is still completable to a
legal turn. The nodes are ordered by inclusion, which only compares nodes
whose shared plies are committed the same way. As a digraph, the source is
the start-of-turn position and the sinks are full turns. Each edge goes to a
node with exactly one more commitment.

Unlike the steps tree, there is no fixed variable order. A node's children
are every feasible (hex, choice) for every undecided hex of the current
phase.

- **Phases.** Fates first, until every enemy is fated. Then moves, until
  every friendly unit has a destination. The attack part is then realized,
  and spawns run on the real spawn-phase board until none remain available.
- **Necromancer.** Its fate is never a choice. If the enemy necromancer can
  be killed this turn, the tier-1 node is terminal with its score and no
  tier-2 search runs. Otherwise it is spared implicitly. The same rule makes
  a full turn terminal: if the opponent can kill our necromancer in the
  resulting position, the turn is scored as a loss on the spot rather than
  by the net. This biases tier-2 values, which is fine because they are only
  compared with other tier-2 values.
- **Spawns** (confirmed 2026-10-03). Each spawn cell's choices are "nothing" or an
  affordable unit. A cell with only "nothing" left is committed
  automatically. The node is a full turn once no undecided cell has a real
  option.
- **Feasibility** reuses the steps machinery unchanged: witnesses, the
  exact search and the lazy witness. Its checks already take the decided
  sets as bitboards, so they are order-free. A child (h, c) is computed by a
  `StepState` with h swapped into the current slot.

**Key / inclusion test.** Variables are indexed per tier-1 node: enemies,
then friends, then spawn cells in sorted order for that fates+moves outcome.
Spawn indexing is only ever compared between nodes whose fates and moves
agree, since other pairs fail inclusion earlier. A node is a `mask` (one bit
per variable) plus `packed` (6 bits per variable for the choice). X ⊇ P
holds iff:

- `P.mask & !X.mask == 0`, and
- `(X.packed ^ P.packed) & field(P.mask) == 0`.

`field(P.mask)` is computed once per selection, so the test is a handful of
word operations.

## Tier 2: prior-based MCTS over the lattice

- **Selection.** Starting at the root, take the child with the highest PUCT
  score for the mover (the spec's "lowest" was a slip, confirmed), and
  expand along the way. The descent stops at a sink, i.e. a full turn.
  Every descent is a whole root-to-turn path: a descent, not a single
  expansion.
- **Expanding a node** is one tier-2 net call. Per undecided hex it returns
  a prior over that hex's choices (softmax per hex, so each hex sums to 1)
  and a value for each child. It also returns a value for the node itself.
  The expansion counts as one direct visit to the node, at its own value,
  and one direct visit to each child, at the child's predicted value.
- **A sink** is valued by the tier-1 value net on the resulting position
  (1 − v, flipped to the mover), or by its terminal score. That is one
  direct visit.
- **Lazy sublattice propagation.** A visit to a node is a visit to every
  node below it, i.e. every subset. There is no backup along the path.
  Statistics are N(P) = Σ_{X ⊇ P} n_X and W(P) = Σ_{X ⊇ P} w_X over the
  direct-visit list, which has O(N) entries.
  - To select at P, one scan of the list computes N(P) and every child's
    (N, W) at once. Each X ⊇ P credits child P + (h, X[h]) for each hex h
    in X \ P that is allowed at P.
  - An expanded node's child visits are stored as its value array, not as
    separate entries. Each child of E credits P + (h', c') directly, and
    together they credit E's own extra commitments with weight |children|.
  - Also, X ⊉ P with |P \ X| = 1 contributes through the single child of X
    that completes P.
  - Cost is O(N) per selection step, times ≤ H credits per entry (more for
    expanded entries' child arrays), over a depth of O(H).
- **PUCT prior.** p(h, c) as is: each hex's children are a disjoint set
  whose priors sum to 1 within it. The single PUCT comparison over all
  children just picks a hex and its choice at once, and the hex-picking
  procedure may change later (the user's call, 2026-10-03).

## Tier 1: AlphaZero-style MCTS over turns, with progressive widening

- A tier-1 node is a start-of-turn position. Its children are the full
  turns found so far.
- **Widening.** When the child count is below k(N) = ⌈c_pw · N^α⌉
  (defaults c_pw 1, α 0.5), the visit widens: one tier-2 descent in that
  node's lattice ends at a full turn. A repeat is retried up to T = 20
  times. After that the node stops widening and keeps its children.
  Repeats still add tier-2 visits.
- **Initial expansion (default).** The first visit to a tier-1 node
  performs its first widening. Both an initial expansion and a widening
  count as one tier-1 visit, backing up the new child's value.
- **Child prior.** Accumulate log-odds along the lattice path, i.e. the sum
  of log p(choice | hex) over the plies. If several descents reached the
  same turn by different paths, average in log-odds space (a geometric mean
  in probability space). Then normalize over the node's children.
- **Backup.** Mean (AlphaZero), with the side flipped at every tier-1 edge.

## Generation

1. **Opening.** Play k turns from the start by expanding lattice nodes ply
   by ply and picking via the ordinal distribution over children ranked by
   value. k = 30 - G with G geometric (mean 10 at first, so the mean start
   is turn 20 of `MAX_TURNS` = 30): end-heavy, loosened toward the start
   over training by raising G's mean (`--from-end-mean`). The user meant G
   as the start turn itself but preferred this reading (2026-10-03).
2. **Each turn: tier-1 search** with N1 ≈ 400 tier-1 visits. The root value
   afterwards is a **tier-1 value target**. The next turn is chosen from the
   children's visits with temperature + Dirichlet noise.
3. **Each turn: full tier-2 search**, reusing the tier-1 search's lattice
   at the root.
   1. Run N2 = 20 descents to a turn (max PUCT; sinks valued by tier-1
      value or terminal score).
   2. Lock one ply: the hex uniformly from the allowed ones, its choice by
      temperature-moderated child visits + Dirichlet.
   3. Re-root there and repeat until a full turn.

   Each locked-from node gives a **tier-2 record**: per-hex child visit
   distributions (including indirect visits) as prior targets, children's
   Q as post-move value targets, and the node's Q as its partial-move value
   target.
4. Continue until the game ends. The game result is not trained on.

**Throughput comes first,** before any learning use:
- many concurrent games;
- batches built across games;
- the next batch prepared while the current one runs on the GPU;
- each turn's tier-2 search running alongside the next turn's tier-1
  search.

## Nets

- **Tier-1 trunk:** relatively big, run once per tier-1 node. Input is the
  start-of-turn position. Output is an image tensor (C × 7 × 7, same
  geometry).
- **Tier-1 value head:** can be big. A tier-1 energy head is optional,
  maybe later.
- **Tier-2 net:** small, called once per lattice expansion. Inputs are the
  trunk output plus partial-commitment channels. Default: the steps-mode
  `View` planes, i.e. projected board and decided markers without the
  `next` plane, since spawns run on a board the trunk never saw. Outputs
  per hex:
  - prior logits over that hex's choices (pre-softmax, combined in
    log-odds);
  - one value per choice (the post-commit value);
  - plus one scalar partial-move value for the node.

  Slots: fates use 3, moves use the 37 `MOVE_DEST` offsets, and spawns use
  the unit index.
- **Trunk cache:** the trunk output stays cached per tier-1 position (the
  sampled-AZ stack's LRU trunk cache in `ort_net.rs` is the model). Tier-2
  batches gather from it instead of shipping features over PCIe.
- **Bootstrap (default):**
  - Tier-1 trunk + value from the steps net (gen 537), whose stage-0 input
    *is* a start-of-turn position.
  - Tier-2 values distilled from the steps net on lattice nodes (its `View`
    of a partial commitment is order-free once `next` is dropped).
  - Tier-2 priors start uniform (zero logits).

## Implementation plan

1. **Lattice core** (`lattice.rs`, CPU only): node/key, order-free child
   enumeration with witnesses, necromancer rule, sink detection. Tests:
   - different orders give equal keys;
   - random walks reach realizable turns;
   - on small positions, the reachable full turns match the steps tree's.
2. **Tier-2 MCTS** with lazy propagation, on a mock evaluator. Test: lazy
   stats equal brute-force upward-closure sums.
3. **Tier-1 MCTS** with progressive widening and log-odds priors, mock
   evaluator.
4. **Nets:**
   - Python model: trunk, tier-1 value, tier-2 head.
   - Two ONNX graphs: trunk+value, and tier-2 taking cached trunk features.
   - ORT wrappers.
   - The bootstrap.
5. **Generation pipeline:** concurrent games, cross-game batching, overlap
   of tier-1/tier-2, record format. Measure and optimize throughput before
   using the data.
6. **Learner and cycle.**

## Implementation status (2026-10-03)

Built (all CPU-side logic unit-tested against mock evaluators):

| piece | where |
|---|---|
| lattice, tier-2 PUCT, lazy propagation (scan checked against brute force), walk | `lattice.rs` |
| tier-1 PW MCTS, log-odds priors | `tier1.rs` |
| generation driver (`lattice-gen`), t1 / t2 record writers | `latgen.rs` |
| steps-net bridge evaluator (bootstrap: child values from child views, uniform priors) | `lattice.rs` `StepsBridge` |
| `lattice.onnx` evaluator: one graph for both tiers, merging batch queue | `lattice_ort.rs` |
| tier-2 learner + export (frozen tier-1 trunk from a steps `net.pt`) | `python/minions_lattice_learner.py` |

Choices made while building:
- Concurrency: each search keeps `--inflight` tier-1 visits in flight, and
  each walk keeps the same number of descents. Tier 1 uses virtual loss on
  the path and counts widenings in flight. In the lattice, the virtual loss
  is a pending zero-value direct visit at the descent's current node, which
  the lazy propagation charges to every node below it. One request per node
  is in flight at a time (busy flags).
- The search reuses the chosen child's lattice. The root's lattice goes to
  that turn's walk.
- `lattice.onnx` must run under TensorRT with one explicit shape profile.
  The CUDA EP returned wrong values (NaN, or values that depended on the
  rest of the batch) for some (rows, kids) shapes whose inputs were fine on
  the CPU EP.
- Witnesses are stored only where they differ from the node's own witness
  with the child's choice swapped in (~4% of children).

Bootstrap data: 15 min of `lattice-gen` with the gen-537 bridge gave 58.7k
tier-2 records (20 kids/record). Tier-2 32x2 trained 8 epochs on it:
- prior excess CE (vs the visit shares, minus their entropy): 0.271 → 0.097;
- child-value BCE: 0.709 → 0.609;
- node-value BCE: 0.691 → 0.601.

Throughput (search only, `--from-end-mean 1000`, 16 workers x 8 games):

| evaluator | in flight | expansions/s | turns/s | RSS |
|---|---|---|---|---|
| bridge, 8 x 16 games, 1 at a time | 1 | 36-47k | 4.2 | 10 GB |
| bridge, 8 x 8 games | 8 | 47k | 6.3 | 7.6 GB |
| lattice.onnx, direct sessions | 8 | 45-54k | 2.2-4 | 12 GB |
| lattice.onnx, merging queue | 8 | 108k | 8.7 | 8.9 GB |
| lattice.onnx, merging queue, dynamic batching (1024 rows / 1.5 ms) | 32 | 153k | 12.3 | 8.8 GB |
| tier1.onnx + tier2.onnx, cached trunk, flat kid index | 32 | 110-150k | 12.6 | 9.1 GB |

Where the time goes now:
- The single runner is ~85-90% busy inside `Session::run`. Per-call host
  overhead dominates: 0.8 ms for 64 rows, 2.25 ms for 1024, roughly doubled
  when the CPU is loaded, which is the laptop's shared power and core
  budget.
- More runners only slow each call. Bigger calls (more rows in flight) are
  what helps.
- Worker thread time: lattice prepare (children and witnesses) ~20%, stats
  scans ~15%.

Memory: about 4.3 GB is ORT/TensorRT. A live lattice node costs ~1.4 KB:
the StepState is 880 B, and each child takes 16 B. A
`Vec<(Kid, TurnSpec)>::into_iter().collect()` once kept ~10x the children's
bytes (in-place collect reuses the allocation).

Two graphs (tier1.onnx: board → value + f16 trunk image; tier2.onnx: trunk
+ partial-commit channels + a flat kid index `row*1813 + cell*37 + slot`)
with the trunk cached per lattice root were roughly neutral: tier 1 is 21%
of runner time, and the 3.1 KB/row fp16 trunk upload costs about what the
recomputation saved. The runner is still ~85% inside `Session::run` at
~110 calls/s, ~1460 rows/call, 7.4 ms/call. The bottleneck is per-row host
and transfer overhead, not GPU compute (GPU 16-40% busy).

Untried: int16 indices, a narrower trunk slice for tier 2, cheaper
per-node CPU (incremental children, a cheaper stats scan, a smaller
per-node StepState).

Throughput is good enough to train on (~12 turns/s, ~220 tier-2 records/s);
the next step is the cycle from random init.

## Training cycle (from random init, started 2026-10-03 22:20)

`data/models/minions/lattice/cycle.sh` (untracked, like the steps cycle),
run detached with `GEN_FLAGS="--games 4"`:
- gen 0: random tier 1 `res:64x6` and tier 2 `32x2`
  (`minions_lattice_learner.py --export-only`; tier-2 priors start uniform).
- Each generation: `lattice-gen` for 600 s with the newest finished nets.
  The opening's from-end mean is 10 + 0.2 G, capped at 30.
- LAG = 1: net G-1 trains beside generation G, which uses net G-2.
- Joint training on the last 4 generations: 2 passes over the tier-2
  records, plus 256 tier-1 records per step. The tier-2 losses train the
  tier-1 trunk too, so it learns from far more than the ~12k tier-1
  records a generation gives. Warm start from the previous generation.
- Tier-1 targets: the root plus every tier-1 turn with at least 16 visits
  (`--t1-min-visits`), about 11 per searched turn.
- Eval every 5 generations: `steps-match` of the new nets against
  generations g-5, g-10, g-20 and g-40, 4 pairs each (random openings,
  seats swapped), 100 tier-1 visits a turn. A lattice net directory passed
  to `steps-match` / `steps-vs-mcts` plays by tier-1 search; its tier-1 net
  is also exported as a steps value net for bids.

Random-init nets behave differently from the distilled ones. Uniform priors
spread every descent, so each lattice holds about twice the nodes.
`--games 8` reached 14.5 GB RSS in 3 minutes and was still growing.
`--games 4` plateaus at 6.3 GB and is about 3x faster per generation.
Gen 1 (random nets): 1010 searched turns, 26k tier-2 and 11.6k tier-1
records in 600 s; GPU ~33% busy. Generation is CPU-bound
(prepare 38%, stats scans 13%), and training takes seconds.

The search is sound but proposal-bound. With the distilled tier 2 (`lt1`),
lattice search lost 2-14 to gen-537 step-greedy at both 50 and 800 tier-1
visits. `lattice_search_diag` shows why: the best root child's static
value is 0.46/0.64/0.73/0.47 on four positions, vs 0.52/0.63/0.82/0.59 for
the greedy turn. Tier-1 search ranks the turns it is given well, but PW
gives it only sqrt(n) of them, all proposed by tier 2.
