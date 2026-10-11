# Minions lattice: the recipe that produced gen 660, and the current one

Gen 660 (`weights/minions/lattice-gen660/`) has been the champion since
2026-10-09 20:39. No later generation has beaten it. This note records how 660
was made and how the current cycle (`data/models/minions/lattice537/cycle.sh`,
restarted 2026-10-11 14:00 at gen 845) differs, so each difference can be
judged as a suspect.

Sources: `cycle.sh`'s dated header, `cycle.log`,
`nets/gen660/meta.json`, and the archived generation logs in
`data_biased/` (gens 742-753, the same self-play settings as 653-660).

## Gen 660

**Lineage**
- Pre-trained from the Athenan/steps gen 537 net.
- Grown to the large nets at gen 531, by function matching gen 526:
  - trunk `pos:96x8`, full-turn positions only;
  - tier 2 `96x6`, the lattice view.
- New rules from gen 547: cyclic moves, ghost rebate 0.
- Tier-1 prior uniform since gen 424 (`--value-prior-tau 100`).

**Training of 660**
- Started from gen 659: a chain, where each generation trains from the
  previous one.
- Data: the 8-generation window, gens 653-660.
  - 264,904 tier-1 records;
  - 70,773 tier-2 records;
  - 131 steps, 1 epoch.
- Optimizer: AdamW with a one-cycle LR schedule (peak 1e-3), new every
  generation; no optimizer state is carried over.
- `--lview --t1-node --energy-weight 0.2`, tier-1 batch 256.
- No weight averaging.

**Self-play that made the window**
- Played by the newest chain net, lagged: gen G's data came from net G-2, so
  660's window came from nets 651-658. Not by the champion (556 at the time).
- 400 s per generation: 16 workers × 4 games, 64 tier-1 visits in flight.
  - That in-flight count raced progressive widening: visits piled onto the
    first-found turns.
  - Fixed later in 9cbde00. 660's training data was made with the racy search.
- Search budget and settings:
  - n1 800 tier-1 visits a turn, n2 20 tier-2 expansions per tier-1 visit;
  - widening k = ⌈1 · N^0.5⌉;
  - one runner session.
- Turn choice: one search a turn. The turn is sampled from the root's visit
  counts at temperature 1, with Dirichlet(0.3) noise at weight 0.25.
- Openings: from a saved pool. Start turn geometric with mean 10, opening
  plies played from the net's values. No evenness filter.
- Handicap: one per generation, the net's own bid.
- Turn cap 30.

**Tier-2 data**
- A separate rooted phase after self-play: up to 190 s.
- Nodes come from the path to each played turn, kept with probability 0.15.
- A from-scratch tier-2 search is rooted at each node.
- Targets: the best complete turn per kid ("max"), per-cell softmax at
  τ 0.02.

**Gate**
- Every 4 generations, 128 games against the champion at n1 100.
- Search: widening (1, 0.5), 64 in flight (the racy search).
- 660 promoted with 0.562 against 556.

## The current recipe (from gen 845)

**Start**
- Chain restarted from 660's weights.
- Gens 844-848 are copies of 660 until the window holds 4 generations of new
  data (`MIN_WINDOW 4`).

**Training**
- Same learner settings: window 8, 1 epoch, AdamW one-cycle peak 1e-3, new
  each generation, tier-1 batch 256.
- Tier 2 trains on the mean targets (`--t2-sub t2m`).
- No weight averaging.

**Self-play**
- Played by the champion only (`SP_CHAMP=1`), starting with 660.
- 400 s per generation: 16 workers × 16 games, 8 tier-1 visits in flight
  (race fixed), 60 s ramp, two runner sessions.
- Search budget and settings:
  - n1 800, n2 20;
  - widening k = ⌈4 · N^0.5⌉;
  - uniform tier-1 prior.
- Turn choice: anchored (`--anchor`).
  1. A turn is picked from the root's visits with the same temperature and
     noise.
  2. A rooted tier-2 search runs at a node on the path to that turn.
  3. The played turn is drawn from the complete turns that search reached.
- Openings: from a pool refilled with even openings.
  - Start turn uniform on 0-28.
  - Each opening commitment is weighted by exp(−|v − 0.5| / 0.05).
  - Openings ending more than 0.1 from 0.5 are dropped.
- Handicap: random per game, N(3.5, 1) rounded and clipped to the bid range.
- Turn cap 30.

**Tier-2 data**
- The same rooted phase (190 s), with nodes kept with probability 0.5.
- Mean targets: each kid's mean over the complete turns reached.

**Gate**
- The champion is the gate. Every 4 generations, 128 games at n1 100.
- Search: widening (4, 0.5) on both seats, 16 in flight.
- Openings: 2 net-sampled turns at handicap 3.
- Promotion at ≥ 0.55.
- Self-checks against gen g − 8. Money probe every 8 generations.

## Differences, as suspects

| | Gen 660 | Now |
|---|---|---|
| Who self-plays | newest chain net (lagged by 2) | champion only |
| In flight (self-play) | 64, racy widening | 8, fixed |
| Widening | ⌈1 · N^0.5⌉ | ⌈4 · N^0.5⌉ |
| Turn choice | sampled from root visits + noise | anchored rooted search |
| Tier-2 nodes kept | 0.15 | 0.5 |
| Tier-2 targets | max (best completion) | mean |
| Openings | geometric mean 10, unfiltered | uniform 0-28, even-filtered |
| Handicap | one per generation, the net's bid | random per game, N(3.5, 1) |
| Games per worker | 4 | 16 |
| Gate search | (1, 0.5), 64 in flight | (4, 0.5), 16 in flight |
| Gate openings | old format | 2 net-sampled turns, handicap 3 |

What has been measured between the two:
- Lattice vs a random commitment order: the lattice wins (0.646 / 0.633).
- Mean vs max tier-2 targets: mean about equal or better (0.516, 0.445,
  0.516 head to head).
- Weight averaging: neutral (0.492, 0.484).
- Widening c=4: +0.15 to +0.22 at the search level.
- The race fix: about serial strength at 16 in flight.

Never isolated:
- self-play by the champion vs by the chain;
- anchored vs plain turn choice;
- even openings;
- random handicaps;
- the tier-2 node fraction.

Every successor trained under the newer settings has scored 0.29-0.45
against 660 in gates. Older nets also lose to 660 in the current gate format
(gen 556: 0.383, gen 648: 0.430), so part of its margin is that format, and
660 itself looks like a strong outlier of its lineage.
