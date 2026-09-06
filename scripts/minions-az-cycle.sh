#!/usr/bin/env bash
# Serial generate -> train -> eval for the Minions sampled-AZ run.
#
# Strictly serial on purpose: on a thermally capped laptop, contention costs
# more than the idle time it saves, and it makes every measurement ambiguous
# about what else was running. Same rationale as scripts/hex-az-cycle.sh.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"

LIVE="${LIVE:-$repo/data/models/minions/live}"
ENGINE="${ENGINE:-$repo/target/release/minions-engine}"
PYTHON="${PYTHON:-$repo/python/.venv/bin/python}"
export PYTHONPATH="$repo/python/src"

CHANNELS="${CHANNELS:-96}"
BLOCKS="${BLOCKS:-8}"
# Recorded examples per generation. Raised 20000 -> 80000 when playout-cap
# randomization started recording fast positions too: ~4x more examples now come
# out of the same amount of search, so holding the old target would have cut the
# self-play per generation 4x instead of quadrupling the data from it.
#
# 80000 -> 250000 to amortise the restored promotion gate. Measured at 80000:
# self-play ~12-13 min, training ~80s, the 40-game champion match ~3m50s, so
# eval was ~21% of wall clock (~30% on greedy generations) -- close to the 29%
# that motivated dropping the gate in the first place. Tripling the generation
# stretches self-play to ~40 min against a fixed eval cost, putting eval near 8%.
TARGET_EXAMPLES="${TARGET_EXAMPLES:-250000}"
# Batch sizing, measured on the 7x7 net (ABBA pairs, quiet machine).
#
# The heads run at `MAX_BATCH * CANDIDATES` rows, but the *trunk* runs at
# MAX_BATCH itself — at 32 it sat at 48k rows/s against a 102k ceiling, less than
# half. Raising it to 128 helps the trunk and both heads at once: +17% median
# (3/3 pairs). 256 was then measured and is *worse* (-6.5%, -4.4%, 0.0%), so this
# is the operating point, not a floor to keep raising.
#
# CONCURRENCY tracks MAX_BATCH: `pick_mode` only guarantees zero padding while
# the pool holds >= 2 * MAX_BATCH slots, and padding is discarded GPU work.
# `pick_mode` guarantees zero padding only while each worker owns
# `2 * MAX_BATCH` slots, so concurrency has to scale with the batch:
# 2 * 768 * 4 workers = 6144.
#
# Measured ABBA against 128, both reps agreeing in sign: 192 +16/+20%,
# 256 +37/+45%, 384 +15/+19% and 512 +29/+30% over 256, 768 +44/+40% over 256.
# At 1024 and 1536 the reps disagree in sign, so 768 is the plateau. Note the
# old note in memory had 256 at -5%; that was true of a different inner loop,
# and re-measuring after changing the loop is the whole point.
CONCURRENCY="${CONCURRENCY:-6144}"
MAX_BATCH="${MAX_BATCH:-768}"
# Pool workers, each owning a disjoint slice of the pool so one worker's CPU
# phases (commit, realize, playout) overlap another's GPU work. Measured on the
# 7x7 net, ABBA pairs on a quiet machine: 1 -> 2 is +17.5% (3/3), 2 -> 4 a
# further +5% (3/3). Returns diminish because the backend serialises the device
# behind one session mutex, so extra workers only fill gaps rather than adding
# GPU throughput. At CONCURRENCY 1024 each of 4 workers holds 256 slots, exactly
# the 2 * MAX_BATCH `pick_mode` needs to keep padding at zero — verified 0%.
WORKERS="${WORKERS:-4}"
SIMS="${SIMS:-400}"
FAST_SIMS="${FAST_SIMS:-100}"
CANDIDATES="${CANDIDATES:-16}"
T_STEPS="${T_STEPS:-4}"
# Fast (playout-cap) searches produce no training data, so they get a cheaper
# sampler. Measured +13% end to end in a paired A/B.
T_STEPS_FAST="${T_STEPS_FAST:-2}"
# Same argument applied to the candidate count, which multiplies T in the
# diffusion cost. At FAST_SIMS visits a 32-wide candidate set is spread far too
# thin to form a tree anyway, so this is cheaper *and* better shaped.
CANDIDATES_FAST="${CANDIDATES_FAST:-8}"
# Eval is a *yardstick*, so its search is pinned independently of self-play's.
# Deliberately stronger than self-play: the question a match answers is "which
# net is better", and a deeper search reads that off the weights with less noise
# than the shallow search self-play can afford. Eval has no fast/full split, so
# EVAL_SIMS applies to every move.
EVAL_CANDIDATES="${EVAL_CANDIDATES:-24}"
EVAL_SIMS="${EVAL_SIMS:-800}"
RANDOM_FRACTION="${RANDOM_FRACTION:-0.25}"
# Random openings: N random legal turns before the search takes over, none of
# them recorded. Keeps the training distribution moving when self-play comes
# from one frozen net — a Hex run stalled at parity without this. Truncated
# geometric on 0..=MAX with P(n) proportional to DECAY^-n. 15/1.1335 has mean 5,
# a quarter of the 20-turn cap — an opening cannot outlast the game.
RANDOM_OPENING_MAX="${RANDOM_OPENING_MAX:-15}"
RANDOM_OPENING_DECAY="${RANDOM_OPENING_DECAY:-1.1335}"
# Visits guaranteed to each forced-exploration candidate, regardless of prior.
# KataGo's sqrt(k*P*N) floor only keeps an already-tried candidate alive, so it
# cannot protect the ones the energy head scores near zero — which are exactly
# the ones exploration is for. Pruned back out of the training target.
MIN_EXPLORE_VISITS="${MIN_EXPLORE_VISITS:-2.0}"
# Exactly one generation. The retention limit that matters is how many
# promotions back the oldest data reaches, and at 250000 that is zero: every
# training step sees only what the current champion just played. The 1M buffer
# held four generations, so up to three promotions of drift.
MAX_EXAMPLES="${MAX_EXAMPLES:-250000}"
# One full pass over the buffer per generation: 488 * 512 ~= MAX_EXAMPLES.
#
# The learner samples uniformly *from the whole buffer*, not from the newest
# generation, so what matters is `steps * batch / buffer_size` — the expected
# number of times each example is drawn. At 1875 steps and batch 128 that was
# 0.24 per generation, and with examples living four generations, roughly *one*
# draw over an entire lifetime. Every one of those positions cost 400
# simulations to produce and was looked at once. A full pass takes the lifetime
# figure to ~4, which is the usual range for AlphaZero-style setups.
STEPS="${STEPS:-488}"
# The learner idles until the buffer holds this many examples. Tie it to the
# actor's target so a cycle can never deadlock waiting for data that a completed
# generate phase was never going to produce.
# Explicitly *below* TARGET_EXAMPLES now that MAX_EXAMPLES equals it: the
# pruner trims to at most MAX_EXAMPLES, so the buffer can settle just under
# 250000 and a MIN_SAMPLES of exactly 250000 would wait for data no completed
# generate phase will ever produce.
MIN_SAMPLES="${MIN_SAMPLES:-200000}"
# 512 measured 1.30x the samples/s of 128 (paired ABBA on a cooled, idle GPU:
# 1273 -> 1657). 1024 was 0.98x of 512, so this is the knee, not a ceiling.
BATCH="${BATCH:-512}"
# Back to 1e-3, sqrt(1/4) for the 4x smaller buffer. It went to 2e-3 when the
# batch went 128 -> 512, which cut gradient noise; a single-generation buffer
# puts correlation back in — every step now draws from one champion's games
# rather than four generations' worth — so the same damping applies in reverse.
LR="${LR:-1e-3}"
# Compare the newest net against the one this many generations back. One
# generation apart is inside the noise of a 40-game match; five gives the
# comparison enough separation while keeping the diagnostic frequent.
EVAL_LAG="${EVAL_LAG:-5}"
# GFlowNet trajectory-balance weight for the sampler. TB settles around 7-9
# while the other losses sit at 0.5-3, so 1.0 would swamp them.
TB_WEIGHT="${TB_WEIGHT:-0.3}"
EVAL_GAMES="${EVAL_GAMES:-40}"
# Games against the greedy yardstick. 12 was too few to read: across generations
# 13-22 the margin swung between -1.0 and -11.0 with no real trend, which is
# enough noise to fake both a rally and a collapse — it did, and both were
# reported before the full history showed a flat plateau. This is the only
# opponent outside the self-play lineage, so it needs to be trustworthy.
GREEDY_GAMES="${GREEDY_GAMES:-30}"
PROMOTE="${PROMOTE:-0.52}"
# Inference backend. `ort` with the TensorRT EP and fp16 is the fast path:
# +73% end to end against the CUDA EP (ABBA pairs, 516->894 and 524->908
# expansions/s) and 1.8x on isolated forwards. Needs `source scripts/ort-env.sh`
# so the ort crate can dlopen libonnxruntime and its CUDA/TRT providers.
#
# The win is kernel *fusion* — fewer launches — and NOT the tensor cores as
# such. TF32 on the CUDA EP uses tensor cores without fusing and measured -7%
# over four ABBA pairs; at 10x10 these convs are launch- and occupancy-limited,
# not FLOP-bound. Engines are content-keyed per generation, rebuilt in seconds.
BACKEND="${BACKEND:-ort}"
EP="${EP:-trt}"
FP16_FLAG="${FP16_FLAG:---fp16}"
CUDA_FLAG="${CUDA_FLAG:-}"   # candle only; needs a --features cuda build

# The ort crate is built with `load-dynamic`: without ORT_DYLIB_PATH and the
# provider libraries on LD_LIBRARY_PATH, session creation blocks forever with
# zero CPU rather than failing, which is indistinguishable from a hung actor.
# Source it here so the default BACKEND=ort path just works.
#
# NB: must not be run through a pipe — `source x | tail` puts the exports in a
# subshell and they never land.
if [[ "$BACKEND" == "ort" ]]; then
  source "$repo/scripts/ort-env.sh" >/dev/null 2>&1 || {
    echo "FATAL: scripts/ort-env.sh failed; ORT cannot dlopen libonnxruntime" >&2
    exit 1
  }
fi

mkdir -p "$LIVE"

case "${1:-run}" in
  stop)   touch "$LIVE/PAUSE"; echo "PAUSE set"; exit 0 ;;
  status) cat "$LIVE/scores.json" 2>/dev/null || echo "no scores yet"; exit 0 ;;
esac
rm -f "$LIVE/PAUSE"

# Bootstrap generation 0 so the actor has weights to load.
if [[ ! -f "$LIVE/latest.safetensors" ]]; then
  echo "== seeding generation 0 =="
  "$PYTHON" -c "
from playground.model_server.minions_net import MinionsNet, save_random, export_onnx_atomic
import os, shutil, torch
from safetensors.torch import load_file
d = '$LIVE'
p = save_random(d, channels=$CHANNELS, blocks=$BLOCKS, seed=0)
shutil.move(p, os.path.join(d, 'latest.safetensors'))
# The actor runs *before* the first training step, so with --backend ort it
# needs the three graphs to exist already; the learner only writes them after
# a generation. Without this, generation 1 dies on a missing trunk.onnx.
net = MinionsNet(channels=$CHANNELS, blocks=$BLOCKS)
net.load_state_dict(load_file(os.path.join(d, 'latest.safetensors')))
export_onnx_atomic(net, d)
print('seeded', d)"
fi

gen=0
while true; do
  gen=$((gen + 1))
  echo "== cycle $gen: generate =="
  "$ENGINE" selfplay-loop --live "$LIVE" \
    --concurrency "$CONCURRENCY" --max-batch "$MAX_BATCH" --workers "$WORKERS" \
    --sims "$SIMS" --fast-sims "$FAST_SIMS" --candidates "$CANDIDATES" \
    --candidates-fast "$CANDIDATES_FAST" \
    --t-steps "$T_STEPS" --t-steps-fast "$T_STEPS_FAST" \
    --random-fraction "$RANDOM_FRACTION" \
    --random-opening-max "$RANDOM_OPENING_MAX" \
    --random-opening-decay "$RANDOM_OPENING_DECAY" \
    --min-explore-visits "$MIN_EXPLORE_VISITS" \
    --max-examples "$MAX_EXAMPLES" --target-examples "$TARGET_EXAMPLES" \
    --seed "$gen" --backend "$BACKEND" --ep "$EP" $FP16_FLAG $CUDA_FLAG

  echo "== cycle $gen: train =="
  "$PYTHON" "$repo/python/minions_az_learner.py" --live "$LIVE" \
    --channels "$CHANNELS" --blocks "$BLOCKS" \
    --steps "$STEPS" --batch "$BATCH" --lr "$LR" --min-samples "$MIN_SAMPLES" \
    --tb-weight "$TB_WEIGHT" --t-steps "$T_STEPS" \
    --exit-after-gens 1

  # Degeneracy check. Gold adjudication at the 40-turn cap can be farmed by
  # hoarding, and that failure is invisible in the loss curves — every head keeps
  # improving while the games become worthless. Never fatal: it appends to
  # <live>/watchdog.jsonl and warns, so the trend is on the record either way.
  "$PYTHON" "$repo/python/minions_watchdog.py" --live "$LIVE" || true

  # Promotion is gated on a match against the reigning champion, every
  # generation. Auto-promotion was tried and abandoned: with nothing gating the
  # ladder, forty generations of falling losses produced a net that lost 180
  # straight games to greedy, because every generation entered self-play
  # regardless of whether it was actually stronger than the one before it.
  #
  # The gate's known weakness stands — a 40-game match is noisy, and it will
  # sometimes reject a generation that was fine. That is the trade being made
  # deliberately: a rejected good generation costs one cycle, an accepted bad
  # one poisons the buffer that trains all its successors.
  echo "== cycle $gen: eval vs champion (gates promotion) =="
  # Greedy is a slow external yardstick, not a rung on the ladder. The eval
  # loop runs it only when the candidate is promoted, so the games are offered
  # unconditionally here and it decides — a schedule in this script would be a
  # second, disagreeing mechanism.
  "$PYTHON" "$repo/python/minions_eval_loop.py" --live "$LIVE" \
    --engine "$ENGINE" --games "$EVAL_GAMES" --candidates "$EVAL_CANDIDATES" \
    --sims "$EVAL_SIMS" --greedy-games "$GREEDY_GAMES" \
    --t-steps "$T_STEPS" --promote-threshold "$PROMOTE" --once \
    --backend "$BACKEND" --ep "$EP" $FP16_FLAG || {
      # Never let evaluation kill the run. The script is `set -e`, so a bad
      # eval used to take the whole cycle down with it — an unattended run
      # died at cycle 3 that way, losing the night. A skipped eval means the
      # champion simply stands for another generation.
      echo "WARNING: cycle $gen eval failed; champion unchanged" >&2
    }
done
