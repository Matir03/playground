# Overview

## Goal

Develop and validate a custom MuZero variant (see [algorithm.md](algorithm.md))
on a sequence of 2-player abstract strategy games of increasing complexity,
before applying the algorithm to the primary target board game.

## Validation sequence

| # | Game     | Branching factor (rough) | State space | Why it's here |
|---|----------|--------------------------|-------------|----------------|
| 1 | Connect 4 | ~7 | Solved, small | Sanity check: can we learn a known game end-to-end? |
| 2 | Go 9×9   | ~80 | Large | Standard RL benchmark; validates MCTS + policy/value at scale. |
| 3 | Amazons  | ~1000+ | Large, high branching | First compound-move reifier (move queen + shoot arrow); rollout throughput. |
| 4 | Arimaa   | ~17000 (per step × 4 steps) | Very large, compound moves | The real stress test for reification-based action decoding. |
| 5 | Hex 13×13 | ~169 | Large, no draws | Pure connection game: trivial rules, deep strategy, and a strong reference engine (MoHex) to measure absolute strength against. |

Connect 4 and Go validate the *training* and *plumbing*. Amazons and Arimaa
validate the *reifier* and *KR-UCT progressive widening* approach — they have
action spaces that are the core reason the variant exists.

## Repository layout

```
playground/
  crates/               # Rust engines + shared libs
    core/               # Game trait, MCTS, LGS abstraction, ModelClient trait
    protocol/           # tonic-generated types from schemas/proto
    engine-server/      # tonic server implementing EngineService, generic over Game
    model-client/       # tonic client to Python ModelService, implements core::ModelClient
    games/
      connect4/  go9/  amazons/  arimaa/  hex/
  python/
    src/playground/
      model_server/     # hosts encoder/eval/reifier/policy NNs
      orchestrator/     # self-play loop, match scheduler, replay writer
      training/         # training loops for each NN head
      eval/             # tournament runner + ELO
      gen/              # generated from schemas/proto
  web/                  # React + Connect-ES frontend
    src/gen/            # generated from schemas/proto
  schemas/
    proto/              # .proto source of truth
    buf.yaml  buf.gen.yaml
  docs/
  scripts/
  data/                 # games DB, model registry (gitignored)
```

## Milestones

1. **M0 — Scaffold (this PR).** Workspace compiles end-to-end with stubs. A
   web client can open a game, send a move, get a reply. No game logic yet.
2. **M1 — Connect 4 engine.** Full rules, legal move gen, terminal detection,
   CLI self-play with a random model. Web UI plays it human-vs-human.
3. **M2 — Connect 4 training.** Stub encoder/eval/policy (no reifier needed —
   action space is small). First full self-play → training → eval loop.
4. **M3 — Go 9×9 engine + training.** Exercises scale.
5. **M4 — Reifier + KR-UCT.** First version of the custom MCTS; validated on
   Connect 4 and Go as regressions against direct-action MCTS.
6. **M5 — Amazons.** First real test of the reifier (compound move = place
   queen + shoot arrow).
7. **M6 — Arimaa.** Four-step compound turns; progressive widening earns keep.
8. **M7 — Target game.** Out of scope here.

Each milestone has a brief "definition of done" in the relevant section of
[training.md](training.md) or the game-specific doc (added as we go).
