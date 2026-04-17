# Evaluation

## Goals

1. Produce an ELO curve over training to see that the model is actually
   improving.
2. Detect regressions quickly (current checkpoint lost to checkpoint from
   two epochs ago by > X points → pause training, investigate).
3. Compare against external benchmark engines where one exists
   (e.g. Connect 4 is solved → any perfect-play Connect 4 solver; Go has
   KataGo / GNU Go; Arimaa has Bot_Fairy and similar).

## Match runner

`python/src/playground/eval` runs matches by calling `EngineService.PlayMatch`
on either end. Each "player" is a URI:

- `human://` (UI only)
- `model://<name>@<version>` — engine configured to use that model version
- `engine://<uri>` — arbitrary engine speaking `EngineService`
- `external://<adapter>` — external engine adapter (separate binary that
  speaks our protocol on one side and the target engine's native protocol
  on the other; we write these as needed).

The runner writes match results to the same sqlite index used by the replay
store (different table) so the web UI can query everything from one place.

## ELO fitting

Two options, both implemented:
- **BayesElo-style MAP** (simple, fast, good for quick charts).
- **Elo-MMR** or similar (proper uncertainty, used when we trust the output
  enough to gate training decisions).

External anchors (where available) are pinned at a known rating so our
relative ELOs have absolute meaning. Without anchors, ratings are only
valid within a single tournament.

## Tournament shapes

- **Training ELO curve.** Every K games, the latest checkpoint plays a
  fixed pool (every Nth past checkpoint + any external anchors). Small
  round-robin; append results.
- **Release eval.** Before calling any variant "working", run a longer
  round-robin to bound the uncertainty.
- **A/B comparisons.** When changing a hyperparameter (K, B, noise schedule),
  spin up a run with the change and compare resulting ELO curve to control.

## What "good" looks like, per milestone

- **M2 (Connect 4):** reaches near-perfect play vs. a solved-game baseline
  within a few hours on a single machine. If not, plumbing is wrong.
- **M3 (Go 9×9):** beats a random-play baseline, then a simple Monte Carlo
  baseline, then shows continuous improvement over checkpoints.
- **M4 (reifier+KR-UCT on Connect 4/Go):** matches direct-action MCTS
  on strength (within noise) at similar compute. If it's *worse*, the
  algorithm is broken; if it's *better*, something is probably wrong too.
- **M5 (Amazons) / M6 (Arimaa):** non-trivial play; KR-UCT progressive
  widening behaves sensibly under load (tree doesn't explode, reification
  budget is actually spent).
