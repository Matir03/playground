# Algorithm: Reification-based latent-space search

This document describes the custom MuZero/AlphaZero variant we're validating.
It is the source of truth for the algorithm; code comments should reference
this file rather than restating intent.

> **Naming.** The learned per-position embedding is the **latent game state**
> (**LGS**) — `LatentGameState` in the code. The algorithm treats it as a point
> in a continuous space, which is why the search and losses are written as
> geometry over LGS.

## Motivation

Classical AlphaZero/MuZero put a softmax policy head over the *concrete* action
space. That works for Connect 4 (7 actions) and is tolerable for Go 9×9 (~82).
It breaks for games whose per-turn action space is enormous or combinatorial:

| Game    | Per-turn actions (rough)                  |
|---------|-------------------------------------------|
| Amazons | ~1,000–2,000 (move queen × shoot arrow)   |
| Arimaa  | ~17,000 average, >60,000 peak (4 steps)   |

A discrete head over 17k+ structured actions is impractical to learn and to
search. Our variant sidesteps the concrete action space entirely during search:

1. Encode the whole position into a continuous **latent game state** (LGS).
2. Run MCTS in LGS space, treating *"which next-LGS to move to"* as a
   **continuous action**, searched with **KR-UCT** (kernel-regression UCT).
3. Decode a chosen LGS→LGS edge back into a concrete move only when the search
   commits to it — **reification** — using a learned inverse model. Apply the
   concrete move, re-encode, continue.

The policy is therefore not a distribution over actions but a **distribution
over next-LGS** (a density on R^d), which we represent as a mixture of Gaussians.
The concrete action space never appears in the network's output dimension; its
size only shows up in the (autoregressive, legality-masked) reifier.

## Does this make sense for a game like Arimaa?

Short answer: **yes, it's a coherent bet, and Arimaa is exactly the regime that
justifies it — but it rests on two load-bearing assumptions that we must verify
on the small games before trusting it on Arimaa.** This section is the honest
validation the design hinges on.

### Why the shape fits
- Arimaa's cost is a huge, *compositional* action space (4 ordered steps with
  push/pull), not a deep tree. Moving the branching factor out of the network
  output and into a learned inverse model is precisely the trade this variant
  makes. The output dimension becomes `d` (latent size), independent of 17k+.
- Continuous-action MCTS (KR-UCT) was built for exactly "too many / structured
  actions to enumerate, but nearby actions behave similarly." If the latent is
  smooth (see below), Arimaa moves that lead to similar positions sit close in
  LGS and share statistics through the kernel — we get generalization across
  moves *inside a single search*, which a discrete head cannot.

### The two load-bearing assumptions
1. **The latent metric is smooth w.r.t. value/policy.** KR-UCT only helps if
   Euclidean distance in LGS predicts "these two next-states play alike."
   Nothing makes that true for free; the **consistency loss** (below) is what
   forces it. If it fails, kernel sharing averages unrelated states and search
   degrades to noisy random sampling.
2. **The reifier covers the moves search needs.** Search can only ever play a
   move the reifier can produce from some sampled next-LGS. If a tactically
   essential Arimaa move (a goal-line trap, a specific freeze-and-pull) lives in
   a latent region the policy never proposes *and* the reifier can't decode, it
   is invisible to search. Coverage, not just accuracy, is the metric that
   matters for the reifier.

### Honest risks and the baseline to beat
- **Tactics through a fuzzy channel.** Continuous relaxation + sampling is great
  for smooth/positional decisions and weak exactly where a single precise move
  matters. Arimaa is more positional than, say, chess, which helps — but goal
  tactics are real. Mitigation: the engine is authoritative (reifier proposes,
  engine validates), and we always reify before committing, so we never *play*
  an illegal/hallucinated move; the risk is failing to *find* the sharp one.
- **More moving parts.** Two extra learned components (a smoothness-regularized
  encoder and an inverse model) and a metric assumption, versus a plain policy.
- **The competitor is a factored autoregressive policy** — predict the 4 steps
  one at a time with a small per-step head (à la AlphaStar action heads). It's
  simpler and a strong baseline. The reification approach's claim is
  *generality*: one game-agnostic mechanism for any large/structured action
  space, no hand-designed action factorization. We should treat the factored
  autoregressive policy as the baseline reification must justify itself against
  on Amazons/Arimaa, not as a strawman.

### What we verify before Arimaa (on Connect 4 / Go, where ground truth exists)
- **Metric smoothness:** correlation between `‖E(S1)-E(S2)‖` and the true
  value/policy difference across random pairs. Must be high once trained.
- **Reifier recall:** fraction of MCTS-visited concrete moves the reifier
  recovers from the corresponding sampled next-LGS (target ≈ 1.0).
- **Regression vs. discrete MCTS:** on Connect 4/Go the latent-search engine
  must match a plain discrete-action AZ of equal compute. If it can't match
  where the discrete head is feasible, it won't surprise us where it isn't.

If those three checks pass on the small games, the Arimaa bet is justified. If
the metric correlation is poor or reifier recall is low, the idea is falsified
cheaply, before we pay Arimaa's cost.

## Notation

| Symbol         | Meaning                                                       |
|----------------|--------------------------------------------------------------|
| `S`            | concrete game state                                          |
| `e = E(S)`     | latent game state, `e ∈ R^d` (encoder)                            |
| `v(e)`         | scalar value in [-1,1] for the player to move (eval)         |
| `p_θ(·∣e)`     | policy: a density over next-LGS, `R^d → ℝ_{≥0}`               |
| `ρ(e, e′)`     | reifier: concrete move realizing the edge `e → e′`           |
| `κ(x,y)`       | kernel `exp(−‖x−y‖² / 2h²)`, shared everywhere (bandwidth h) |

The single kernel `κ` (one bandwidth `h`) ties the whole system together: it is
the KR-UCT kernel **and** the metric the policy/consistency losses are written
in. Keep them the same — that is what makes the search and the training agree on
what "close" means.

## Neural network components

All four heads operate over the latent game state (LGS). They may share a trunk; see
open questions.

| NN       | Input            | Output                                                  |
|----------|------------------|--------------------------------------------------------|
| encoder  | concrete state S | latent `e ∈ R^d`                                       |
| eval     | `e`              | scalar value `v(e)` for player to move                  |
| policy   | `e`              | parameters of a mixture of `K` Gaussians over `R^d`     |
| reifier  | `(e, e′)`        | concrete move `M` with `apply(M, decode-of e) ≈ e′`     |

### Policy head — the next-LGS distribution (Mixture Density Network)

The policy is a **Mixture Density Network**. From `e` it emits, for each of `K`
components `k`:

- mixture weight `w_k` — softmax over `K` logits, `Σ_k w_k = 1`;
- mean `μ_k ∈ R^d`;
- (diagonal) std `σ_k ∈ ℝ_{>0}^d` — via `softplus`, floored at `σ_min`.

so that

```
p_θ(x ∣ e) = Σ_k w_k · N(x ; μ_k, diag(σ_k²)),   x ∈ R^d.
```

Diagonal covariance keeps the head at `K·(1 + 2d)` outputs. If component-wise
correlations matter we can upgrade to low-rank-plus-diagonal `Σ_k = D_k + V_k
V_kᵀ` with `V_k ∈ R^{d×r}`, small `r`; start diagonal. `K` in the low tens.

The policy is used three ways: (a) to **seed** new continuous candidates during
widening (sample `x ~ p_θ`), (b) as the **prior term** in KR-PUCT, and (c) it is
trained to match the search's own visit distribution (below).

### Reifier — inverse model, autoregressive and legality-masked

`ρ(e, e′)` predicts the concrete move that takes the position behind `e` to the
position behind `e′`. For compound moves (Amazons: move+shoot; Arimaa: up to 4
steps) it is **autoregressive over sub-actions**, conditioned on `(e, e′)` and
the partial move so far, decoding one sub-action at a time. **The engine is
authoritative**: at each sub-action the legal set masks the reifier's logits, so
the output is always a legal move (or a legal "stop"). Optional beam search over
the reifier yields several concrete candidates from one target `e′`.

Crucially, `e′` is only a *target*. After `ρ` produces a concrete move `M`, we
`apply` it and **re-encode** the real resulting state: `e″ = E(apply(M,S))`.
`e″` (not the sampled `e′`) becomes the child's latent — reification *snaps the
continuous proposal onto the reachable discrete manifold*. This projection is
what makes the continuous relaxation legitimate.

## MCTS in latent space

Each tree node holds a latent `e`, MCTS stats (visit count, value sum), and a
set of **candidate children**. A candidate is a point `c ∈ R^d` in next-LGS
space; candidates are born *unreified* (just a sampled target) and may later
become *reified* (a real, applied, re-encoded child).

### Selection — KR-PUCT

KR-PUCT = KR-UCT value/visit smoothing + a PUCT prior term, with the prior taken
from the policy density. For the current candidate set `C` at a node, define the
kernel-regressed visit mass and value of a candidate `x`:

```
N(x) = Σ_{c∈C} κ(x,c) · n(c)                      (kernel visit mass)
Q(x) = ( Σ_{c∈C} κ(x,c) · W(c) ) / N(x)           (kernel-regressed value)
```

where `n(c)`, `W(c)` are the visit count and backed-up value sum at `c`. Select

```
x* = argmax_{x∈C}  Q(x) + c_puct · P(x) · √(Σ_{x'∈C} N(x')) / (1 + N(x))
P(x) = p_θ(x ∣ e) / Σ_{x'∈C} p_θ(x' ∣ e)          (policy prior over candidates)
```

`κ` shares statistics across nearby candidates (the KR-UCT idea); `P(x)`
injects the learned policy (the PUCT idea) in LGS space. `h → 0` recovers plain
PUCT over the discrete candidate set. Reference (KR-UCT):
<https://www.ijcai.org/Proceedings/16/Papers/104.pdf>.

### Progressive widening — sampling new candidates from the policy

Add a fresh candidate when the node is under-widened, `|C| < β · n_node^α`
(`α∈(0,1)`, classic PW). The new candidate is sampled `x_new ~ p_θ(·∣e)` — "MCTS
seeded with the original policy." It starts unreified with zero visits.

### Reification trigger and aliasing

We do not reify on first touch (a single visit may be exploration noise). **On a
candidate's 2nd visit, reify it:**

1. Compute the concrete move `M = ρ(e, x)`, apply it, re-encode `e″ = E(S″)`.
2. **Dedup against existing reified children.** If `e″` matches an existing
   reified child `c_j` (same concrete state, or `‖e″ − c_j‖ < ε`), do **not**
   add a node: mark this candidate `x` an **alias** of `c_j`. Otherwise create a
   new reified child at `e″` and move the candidate's accumulated stats onto it.
3. Backups thereafter accrue to the reified node. **Aliases share the reified
   node's `(n, W)`** when they appear in the kernel sums `N(·)`, `Q(·)` — i.e.
   the merged visit count is used as the count of every alias for kernel
   regression. (The user-facing rule: *"when widening, use visits on reified
   nodes as visit counts on their aliases as well."*)

Aliasing matters because the sampled-latent → concrete-state map is many-to-one:
many nearby targets `x` reify to the same move. Without merging we'd fragment
visits across duplicates and let the kernel double-count one real child as a
dense cluster, corrupting both `N` and `Q`.

### One playout, end to end

1. Descend by KR-PUCT to a leaf candidate.
2. If the candidate is unvisited: evaluate the (still latent) target — back up
   `v(x)` — and return (cheap, no concrete game touched).
3. If it's the candidate's 2nd visit: reify (with aliasing), evaluate `v(e″)`,
   back up.
4. If the node is under-widened: sample a new candidate from `p_θ`.
5. Backpropagate value to the root (negamax sign per ply).

Terminal handling: a reified child that is concretely terminal backs up the true
game result, not `v`.

## Search → training targets

At the root (and every reified node we record), the search yields a set of
reified children `{c_i}` with **merged** visit counts `{n_i}` and the concrete
move `{A_i}` that reified each. From these:

- **Visit policy** `π_i = n_i / Σ_j n_j` — an empirical distribution over the
  visited next-LGS points `{c_i}`. Its kernel-smoothed form
  `q(x) ∝ Σ_i n_i κ(x, c_i)` is the **"true kernel distribution"** the policy
  head is trained to match.
- **Value target** `z` — the bootstrapped search value, with terminal nodes
  clamped to the true outcome. Optionally regularized toward the final game
  result.
- **Reification pairs** `(S, A_i)` for every visited child — supervision for the
  inverse model.

## Training

One combined objective; weight the terms. Run encoder+reifier+value first, fold
in the policy once the latent is stable, keep the consistency term on throughout
(ramp its weight). All terms backprop into the encoder unless noted.

Let `S` be a recorded state, `e = E(S)`, `{(c_i, n_i, A_i)}` its reified
children, `π_i = n_i/Σn`, value target `z`.

**1. Value (MSE on encode+eval).**
```
L_v = ( v(e) − z )²
```

**2. Policy — distance to the true kernel distribution.** The clean, tractable
objective is the **visit-weighted negative log-likelihood** of the visited
next-LGS points under the mixture (weighted MDN MLE):
```
L_π = − Σ_i π_i · log p_θ(c_i ∣ e)
```
This is `KL(π̂ ‖ p_θ)` up to a constant, with `π̂` the empirical visit measure;
`log p_θ` is a stable `logsumexp` over components. To match the *smoothed* kernel
target `q` (and avoid the MDN spiking on individual reified points, which are
themselves noisy samples of a smooth intent), jitter targets by the kernel:
draw `c_i` w.p. `π_i`, add `N(0, h²I)`, take NLL — i.e. fit to samples from `q`.

*Faithful alternative.* Since both `q` and `p_θ` are Gaussian mixtures and `κ`
is Gaussian, the **MMD²** between them with kernel `κ` is closed form (every term
is a Gaussian–Gaussian convolution integral):
```
L_π^MMD = E_{x,x'~p_θ} κ(x,x') − 2 E_{x~p_θ, y~q} κ(x,y) + E_{y,y'~q} κ(y,y')
```
This *literally* is "distance to the true kernel distribution" in the same
kernel the search uses, and resists mode collapse better than NLL, at higher
cost (`O(K² + K·#children)` Gaussian evals). Default to weighted NLL; switch to
MMD if the MDN collapses or under-disperses.

**3. Reifier (inverse model).** For every visited child, the reifier must
recover the concrete move from the *encoder's own* embeddings of the endpoints:
```
L_ρ = Σ_i π_i · ℓ_act( ρ(e, E(S_i)),  A_i )
```
where `S_i = apply(A_i, S)` and `ℓ_act` is the autoregressive decoding loss
(per-sub-action cross-entropy under the legal mask). Trained jointly with the
encoder — the encoder must keep enough transition information to be invertible.

**4. Consistency — shape the latent metric (the key regularizer).** For random
pairs `(S1, S2)` drawn from the *whole* training set, with `E1=E(S1)`,
`E2=E(S2)`, make latent distance predict functional distance:
```
d_lat  = ‖E1 − E2‖₂
d_func = |v(E1) − v(E2)| + λ_p · D( p_θ(·∣E1), p_θ(·∣E2) )
L_c    = ( α · d_lat − d_func )²
```
`D` is a divergence between the two predicted mixtures; use **MMD with the same
`κ`** (closed form for Gaussian mixtures) so the whole system speaks one kernel.
`α` is a **learnable scalar** absorbing the arbitrary latent-norm scale (without
it, demanding `d_lat == d_func` over-constrains the geometry). Gradients flow
through **all three** of `E`, `v`, `p_θ`. Pairs come from across the dataset, not
just within a game, so the metric is globally calibrated. Robustify with a Huber
on the residual; a scale-free variant maximizes batch correlation
(Pearson/Spearman) between `d_lat` and `d_func` instead of squared matching.

This term is *why KR-UCT is allowed to work*: it forces `‖·‖` in LGS to be a
faithful stand-in for "these positions play alike," which is exactly the
precondition the kernel needs.

**5. Regularization & anti-collapse.** Weight decay; bound the latent
(LayerNorm or project to a ball) so `h` and `α` are stable; entropy bonus on the
mixture weights to keep components from collapsing; VICReg-style
variance/covariance terms on a batch of latents to prevent representational
collapse.

**Total:** `L = L_v + λ_π L_π + λ_ρ L_ρ + λ_c L_c + reg`.

### The central tension to watch

`L_c` *pulls together* states with equal value and policy — even concretely
different ones. `L_ρ` *pushes apart* any two states linked by a move, because
the encoder must retain enough information to decode that move. These are
compatible only if the latent has the capacity to be **metrically smooth in
value/policy yet still invertible w.r.t. transitions**. If reifier recall drops
as `λ_c` rises, the encoder is collapsing useful structure — back off `λ_c` or
grow `d`. This trade-off is the thing to instrument first.

## Self-play and training optimizations

These are the levers that make self-play affordable and learning sample-
efficient. Most are adapted from KataGo; each interacts with the reifier /
KR-PUCT / MDN structure in a way worth spelling out.

### Playout cap randomization (PCR)

Reification is the expensive part of a playout (a `reify` call + `apply` +
re-`encode`). We can't afford full search on every move. Per move, search to a
**small cap `n_fast`** with high probability `1 − p`, and to the **full cap
`n_full`** with probability `p` (KataGo uses `p ≈ 0.25`). **Only full-search
moves contribute a policy target**; every move (fast or full) contributes a
value target. This decouples "lots of cheap value data" from "few high-quality
policy targets."

It pays off more here than in discrete AZ for two reasons: (a) reifications
concentrate on exactly the moves we train the policy on; (b) fast moves can run
almost entirely in cheap latent expansion — keep widening/reification minimal —
so the fast/full cost gap is larger than the playout ratio alone suggests. Skip
root Dirichlet/exploration noise on fast moves (they aren't policy training
data).

### Forced playouts + policy target pruning

**Forced playouts.** Each reified candidate that has received ≥1 visit is forced
to get at least `n_forced(x) = ⌈ k · P(x) · Σ N ⌉` visits before KR-PUCT is
allowed to starve it (`P(x)` the policy prior, `Σ N` the node's kernel visit
mass). This guarantees low-prior candidates — and freshly widened/reified ones —
are actually explored, so the MDN prior can't quietly dominate the search early.

**Policy target pruning.** Forced exploration would bias the visit-count policy
target toward moves that didn't earn their visits. So before forming `π_i`,
**subtract** forced playouts from each reified child down to the visit count at
which plain KR-PUCT *would* have selected it on its own merits — never pruning
the single most-visited child — then renormalize. The result is a sharp,
unbiased kernel-distribution target that still came from a well-explored tree.

Two adaptations for this algorithm: the "count KR-PUCT would have chosen" must be
computed from the **kernel-regressed** `Q(·)`, `N(·)` (selection runs on the
smoothed quantities, not raw per-child stats); and with aliasing, prune on the
**merged reified node**, since its shared count is what feeds the kernel sums.

### Global pooling in the encoder trunk

Board games carry global quantities — material balance, who's ahead, ko /
repetition state, Arimaa goal-distance — that stacked local convolutions
aggregate poorly. Add **global pooling** layers to the encoder trunk: pool
channel features across the board (mean and max, optionally scaled by board
size) and broadcast them back as per-channel biases (KataGo-style).

This matters doubly in our design: the LGS is a *fixed-dimension global summary*
that `eval` and the **consistency loss** both lean on, and the latent metric can
only reflect global value differences if the encoder can actually compute global
features. Cheap; include it from the start for Go / Amazons / Arimaa.

### Auxiliary targets

Extra heads trained from data we already have. They regularize the encoder and
speed learning for near-zero self-play cost; most are dropped at inference.

- **Opponent-reply (auxiliary policy) target.** Predict the policy at the *next*
  position — an auxiliary MDN over the next-next-LGS — trained against the search
  policy at the child actually played. Forces the encoder to model the
  opponent's options, not just the side to move. (KataGo's largest aux win.)
- **Short-term value.** Predict `v` one ply ahead (at the chosen child); a cheap
  self-consistency signal that sharpens the value head and complements `L_c`.
- **Game-specific dense targets.** Where a game offers per-square supervision —
  Go ownership, Amazons region/territory control, Arimaa material & goal-distance
  — predict it. Dense targets teach the encoder far more per position than a
  single scalar value and noticeably improve representation quality; fold in per
  game.

All are added to the total objective as small-weight auxiliary losses sharing the
trunk — the cheapest sample-efficiency lever available.

## Open questions / knobs

- **Kernel bandwidth `h`** — couples search, policy target, and consistency
  metric. Per-game; possibly annealed or adapted to local latent density.
- **`K`** (mixture components), **`d`** (latent dim), and the diagonal-vs-
  low-rank covariance choice.
- **PW schedule** `(α, β)` and `c_puct`; **reify-on-2nd-visit** threshold and the
  alias dedup radius `ε`.
- **Policy loss**: weighted NLL vs MMD; how much kernel jitter.
- **Consistency loss**: squared-match-with-learnable-`α` vs batch-correlation;
  weight `λ_c` schedule; `λ_p` (value/policy balance in `d_func`).
- **Curriculum**: when to enable the policy head; whether to freeze the encoder
  during policy fitting.
- **Trunk sharing** across the four heads vs independent towers.
- **Reifier coverage** vs accuracy; whether to beam-search and add multiple
  reified children per target.
- **PCR**: fast/full caps `n_fast`, `n_full` and full-search probability `p`.
- **Forced playouts**: constant `k`; how aggressively to prune the policy target.
- **Auxiliary losses**: which heads to keep, and their weights.

We expect to pin most of these empirically on Connect 4 and Go (where a discrete
head is feasible and gives ground truth) before trusting Amazons / Arimaa.
