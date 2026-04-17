# Algorithm: Reification-based MuZero variant

This document describes the custom MuZero variant we're validating. It is the
source of truth for the algorithm; code comments should reference this file
rather than restating intent.

## Motivation

Classical MuZero models transitions in a learned latent space and never needs
to decode a latent back to an action. That works when you can afford a policy
head over the full concrete action space. Games like Amazons and especially
Arimaa have action spaces so large (or so combinatorially structured) that a
full concrete policy head is impractical.

Our variant sidesteps that by keeping MCTS in a learned *game state* (GS)
latent space, and decoding the edge between two GS nodes back into a concrete
move on demand (*reification*). The concrete game then advances, is encoded
again, and the search continues.

## Neural network components

All four networks operate over a latent **game state encoding** (GS).

| NN        | Input          | Output                                              |
|-----------|----------------|-----------------------------------------------------|
| encoder   | concrete state | GS                                                  |
| eval      | GS             | scalar value (expected outcome for player-to-move)  |
| policy    | GS             | B "best" (action, next-GS) pairs — two heads        |
| reifier   | (GS_a, GS_b)   | concrete move M such that apply(M, a) ≈ b           |

Two-head policy: the *state* head is used during MCTS rollouts (fast path — no
reification needed to expand a child). The *action* head is used during
reification to provide an initial concrete move hypothesis.

## MCTS (high level)

1. MCTS runs over GS nodes — each node stores a latent GS and MCTS statistics
   (visits, value, prior).
2. Expansion uses the policy state-head: a new GS node gets up to B children
   as (prior, next_GS) without ever touching the concrete game.
3. **Reification trigger.** Every K visits to a node, reify one of its child
   edges: use the policy action-head to propose a concrete move, apply it
   against the concrete game, encode the resulting state, and attach that
   (concrete-state, GS) pair to the child.
4. **Reified siblings.** Each time a child is reified, we also add a reified
   *sibling* by perturbing the action with noise. Noise scale grows with the
   number of already-reified siblings of that edge (more siblings = more
   exploration around that action).
5. Use **KR-UCT** — kernel regression UCT — for child selection among reified
   siblings, giving us continuous/structured action smoothing.
   Reference: <https://www.ijcai.org/Proceedings/16/Papers/104.pdf>.
6. **Progressive widening** on the reification threshold: the K needed to
   trigger another reification grows with node visit count, keeping the
   effective branching factor under control.

## Training

Targets are derived by converting the GS-level MCTS tree into a *concrete*
MCTS tree, using the reifications accumulated during search. At game-end
nodes, synthetic targets are replaced with true game outcomes.

### encoder + eval
- Target: post-MCTS value of the concrete tree, with terminal nodes replaced
  by true outcomes.
- Initial bootstrapping: intermediate non-terminal nodes can use heuristic
  values until the eval head stabilizes.
- Can additionally train on true final game result as a regularizer.

### encoder + reifier
- Target: for every concrete move M : A → B observed in play,
  `reify(encode(A), encode(B)) = M`.
- Trained jointly with encoder — encoder must learn representations that are
  decodable back to moves.

### policy
- Target: the concrete MCTS visit-count policy re-encoded into GS space
  (i.e. for each child in the concrete tree, encode its resulting state and
  use the visit count as a target over (action, next-GS) pairs).
- **Train only after encoder is healthy** — otherwise the policy target is
  noise.

## Open questions / knobs

- `K` (reification trigger interval) and its progressive-widening schedule.
- `B` (policy head width). Larger B = richer MCTS expansion but heavier NN.
- Noise distribution and scale growth for reified siblings.
- KR-UCT kernel choice (currently: follow the 2016 paper; revisit per game).
- When to freeze encoder vs keep updating during policy training.
- Whether to share trunks across encoder/eval/policy/reifier or keep them
  independent.

We expect to learn most of these empirically on Connect 4 and Go before
adding Amazons / Arimaa to the mix.
