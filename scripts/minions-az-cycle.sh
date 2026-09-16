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
#
# 250000 -> 200000 (user, 2026-09-13), with the buffer at five generations.
# 100k since 2026-09-16: at 800 sims with no fast searches a position costs
# ~4.6x what it did (800 against a ~175 average), so 200k would take ~2 h per
# generation. At 100k over a 500k buffer an example still lives five gens.
TARGET_EXAMPLES="${TARGET_EXAMPLES:-100000}"
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
# 800 since 2026-09-16 (was 400): deeper search per position, at roughly double
# the self-play wall clock. EVAL_SIMS stays 400 so tournament ratings remain
# comparable with the whole ladder from gen362 on.
SIMS="${SIMS:-800}"
FAST_SIMS="${FAST_SIMS:-100}"
# 1.0 since 2026-09-16 (user): no playout-cap randomization -- every position
# gets the full search and a policy target. The actor's own default is 0.25,
# which is what produced the ~24% full rows before this. FAST_SIMS,
# CANDIDATES_FAST and T_STEPS_FAST are dead while this is 1.0.
FULL_SEARCH_PROB="${FULL_SEARCH_PROB:-1.0}"
# 2026-09-15 (user): 16 diffusion candidates plus 8 random-playout explore
# candidates per full-search node -- 24, with RANDOM_FRACTION 1/3. Each
# candidate is random by a weighted coin, so a node gets 8 in expectation.
CANDIDATES="${CANDIDATES:-24}"
T_STEPS="${T_STEPS:-4}"
# Fast (playout-cap) searches produce no training data, so they get a cheaper
# sampler. Measured +13% end to end in a paired A/B (at T 2).
#
# 2026-09-15 (user): fast searches run the full search's settings --
# refinement included -- with only the budget cut: 100 simulations, T 3
# (refinements redraw at RESAMPLE_T) and no forced visits (search_params).
# They draw 16 diffusion candidates and no random ones: injection is
# full-search only (the user kept it that way).
T_STEPS_FAST="${T_STEPS_FAST:-3}"
CANDIDATES_FAST="${CANDIDATES_FAST:-16}"
# Refinement: every RESAMPLE_EVERY visits to a node, the visit re-masks a
# uniform share in [0, RESAMPLE_FRAC] of PUCT's pick and redraws it with
# RESAMPLE_T diffusion rounds instead of descending. Converged by head-to-head
# on gen343 (400 sims, 16 candidates): every 2 / frac 1.0 / T 2 at every depth
# beat plain search 75-5, and plain search at twice the simulations 72-8.
RESAMPLE_EVERY="${RESAMPLE_EVERY:-2}"
RESAMPLE_FRAC="${RESAMPLE_FRAC:-1.0}"
RESAMPLE_T="${RESAMPLE_T:-2}"
# KR-PUCT bandwidth. 0 = plain PUCT. Every sigma tried lost to 0 on top of
# refinement, less the smaller it got: 0.25 0.275, 0.1 0.425, 0.05 0.425,
# 0.03 0.475 (80 games each).
KERNEL_SIGMA="${KERNEL_SIGMA:-0}"
# Eval is a *yardstick*, so its search is pinned independently of self-play's.
# Deliberately stronger than self-play: the question a match answers is "which
# net is better", and a deeper search reads that off the weights with less noise
# than the shallow search self-play can afford. Eval has no fast/full split, so
# EVAL_SIMS applies to every move.
#
# 2026-09-12: set to 400 sims / 16 candidates by the user -- self-play's full
# search budget -- with the same diffusion length as self-play.
EVAL_CANDIDATES="${EVAL_CANDIDATES:-16}"
EVAL_SIMS="${EVAL_SIMS:-400}"
EVAL_T_STEPS="${EVAL_T_STEPS:-$T_STEPS}"
RANDOM_FRACTION="${RANDOM_FRACTION:-0.3333333}"
# Random openings: N random legal turns before the search takes over, none of
# them recorded. Keeps the training distribution moving when self-play comes
# from one frozen net — a Hex run stalled at parity without this. Truncated
# geometric on 0..=MAX with P(n) proportional to DECAY^-n. 15/1.1335 has mean 5,
# a quarter of the 20-turn cap — an opening cannot outlast the game.
RANDOM_OPENING_MAX="${RANDOM_OPENING_MAX:-22}"
RANDOM_OPENING_DECAY="${RANDOM_OPENING_DECAY:-1.0884}"
# Visits guaranteed to each forced-exploration candidate, regardless of prior.
# KataGo's sqrt(k*P*N) floor only keeps an already-tried candidate alive, so it
# cannot protect the ones the energy head scores near zero — which are exactly
# the ones exploration is for. Pruned back out of the training target.
MIN_EXPLORE_VISITS="${MIN_EXPLORE_VISITS:-2.0}"
# Flat forced-exploration quota, replacing the prior-proportional KataGo floor.
# These visits are invisible to PUCT and subtracted from the policy target, so
# they buy a value estimate and nothing else. Capped in the search at half the
# simulation budget.
#
# 2026-09-15 (user): 10 per random explore candidate (was 30), and only those:
# ~8 of them is ~80 of 400 simulations.
FORCED_VISITS="${FORCED_VISITS:-10}"
# Non-explore candidates drawn at random to also get a quota. 0 since
# 2026-09-15 (user): only the random explore candidates are forced. The root
# sweep's one seeded visit per candidate stays (the user kept it).
FORCED_EXTRA="${FORCED_EXTRA:-0}"
# Five generations: one tournament round of the champion's own games, since
# only the champion plays self-play between tournaments (SELFPLAY_NET).
# 500k since 2026-09-16, halved with FULL_SEARCH_PROB at 1.0: every row now
# carries its root's children (~103 at 17.0 KB/row, against 5.2 KB when three
# quarters of the rows were fast and stored none). 1M all-full rows would be
# 15.8 GiB of buffer, and the learner holds it all in RAM on a 24 GB box.
MAX_EXAMPLES="${MAX_EXAMPLES:-500000}"
# 488 * 512 = 250k samples per generation -- a quarter of the buffer -- so each
# example is drawn about 1.25 times over its five-generation life.
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
# Must stay below TARGET_EXAMPLES.
MIN_SAMPLES="${MIN_SAMPLES:-75000}"
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
#
# 0 since 2026-09-15: the TB objective damaged the sampler's conditionals, and
# stays off until it is replaced.
TB_WEIGHT="${TB_WEIGHT:-0}"
EVAL_GAMES="${EVAL_GAMES:-40}"
# Promotion mode, see the eval step. 0: no gate, a tournament every
# MEASURE_EVERY generations. 1: the per-generation gate.
GATE="${GATE:-0}"
MEASURE_EVERY="${MEASURE_EVERY:-5}"
# Who plays self-play between tournaments: the champion, or each new generation.
# newest since 2026-09-15: strength was flat for three tournaments (gen371
# onward) while the losses kept improving. Five generations of self-play on
# one frozen champion give the buffer a single net's distribution, the same
# shape of stall seen in the Hex run. Tournaments still rate and crown.
SELFPLAY_NET="${SELFPLAY_NET:-newest}"
# Tournament matches at once, at most. Each eval-match process holds ~2 GB, so
# on this 23 GB box the eval loop's RAM floor usually binds first.
EVAL_PARALLEL="${EVAL_PARALLEL:-10}"
# Games against the greedy yardstick. 12 was too few to read: across generations
# 13-22 the margin swung between -1.0 and -11.0 with no real trend, which is
# enough noise to fake both a rally and a collapse — it did, and both were
# reported before the full history showed a flat plateau. This is the only
# opponent outside the self-play lineage, so it needs to be trustworthy.
#
# Played only every 5th champion now (`--greedy-every` in the eval loop), as a
# check that the ladder is not regressing against a fixed opponent, and never
# rated: the champions beat greedy 0.90-1.00, where 30 games carry about one
# unit of Fisher information, so rating them added noise and -- landing after
# the promotion decision -- made the fit a candidate was judged on differ from
# the one recorded for it.
# 0 since 2026-09-15: the greedy yardstick stopped carrying signal -- the
# champions beat it 0.90-1.00 and every game costs eval time -- so it is no
# longer played. Set GREEDY_GAMES to a positive number to bring it back.
GREEDY_GAMES="${GREEDY_GAMES:-0}"
# A pre-filter, not the decision, since the rating gate landed. It exists only
# to avoid paying four ladder matches for a candidate that is plainly worse, so
# it sits below parity: a candidate that lost the head-to-head 0.45 has said
# little over 40 games, and the ladder plus the Bradley-Terry fit over the whole
# record is a far better judge of it than this number is. Anything that clears
# this still has to top the ladder to be promoted, and when an older net tops
# it instead, that net is crowned champion (`crown_ladder_top`).
PROMOTE="${PROMOTE:-0.40}"
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
  # Seed from the run's generation, not this script's counter: `gen` restarts
  # at 1 on every launch, and self-play on an unchanged champion is
  # deterministic in the seed, so a relaunch replayed the first cycle's games
  # into the buffer as exact duplicates.
  seed=$(grep -o '"generation": *[0-9]*' "$LIVE/config.json" | grep -o '[0-9]*$')
  "$ENGINE" selfplay-loop --live "$LIVE" \
    --concurrency "$CONCURRENCY" --max-batch "$MAX_BATCH" --workers "$WORKERS" \
    --sims "$SIMS" --fast-sims "$FAST_SIMS" --candidates "$CANDIDATES" \
    --full-search-prob "$FULL_SEARCH_PROB" \
    --candidates-fast "$CANDIDATES_FAST" \
    --t-steps "$T_STEPS" --t-steps-fast "$T_STEPS_FAST" \
    --resample-every "$RESAMPLE_EVERY" --resample-frac "$RESAMPLE_FRAC" \
    --resample-t "$RESAMPLE_T" --kernel-sigma "$KERNEL_SIGMA" \
    --random-fraction "$RANDOM_FRACTION" \
    --random-opening-max "$RANDOM_OPENING_MAX" \
    --random-opening-decay "$RANDOM_OPENING_DECAY" \
    --min-explore-visits "$MIN_EXPLORE_VISITS" \
    --forced-visits "$FORCED_VISITS" --forced-extra "$FORCED_EXTRA" \
    --max-examples "$MAX_EXAMPLES" --target-examples "$TARGET_EXAMPLES" \
    --seed "${seed:-$gen}" --backend "$BACKEND" --ep "$EP" $FP16_FLAG $CUDA_FLAG

  echo "== cycle $gen: train =="
  "$PYTHON" "$repo/python/minions_az_learner.py" --live "$LIVE" \
    --channels "$CHANNELS" --blocks "$BLOCKS" \
    --steps "$STEPS" --batch "$BATCH" --lr "$LR" --min-samples "$MIN_SAMPLES" \
    --tb-weight "$TB_WEIGHT" --t-steps "$T_STEPS" \
    --exit-after-gens 1

  # Losses for this generation beside the recent ones and the last few
  # champions. A single generation's numbers are noise; the table is what makes
  # the direction of travel, and the champion rows, readable.
  "$PYTHON" "$repo/python/minions_loss_table.py" --live "$LIVE" || true

  # Degeneracy check. Gold adjudication at the 40-turn cap can be farmed by
  # hoarding, and that failure is invisible in the loss curves — every head keeps
  # improving while the games become worthless. Never fatal: it appends to
  # <live>/watchdog.jsonl and warns, so the trend is on the record either way.
  "$PYTHON" "$repo/python/minions_watchdog.py" --live "$LIVE" || true

  # Default (GATE=0): no gate. Self-play runs on the champion alone, so the
  # buffer (MAX_EXAMPLES, five generations) is the champion's own games. The
  # learner trains one continuous run from the champion, and every generation
  # is snapshotted as a candidate. Every MEASURE_EVERY-th generation ends in a
  # tournament -- the candidates against each other and the leaderboard's top
  # five, 10 games a pairing, in parallel -- and the best-rated fp16 net
  # becomes the champion: self-play and the learner both restart from it.
  # SELFPLAY_NET=newest puts each new generation into self-play instead
  # (AlphaZero-style), as the run did from gen342 to gen361.
  #
  # History: the per-generation gate cost ~24 of every ~51 minutes, and under
  # ORT it never worked -- its champion seat loaded the candidate's own graphs
  # (see minions_elo.self_match). GATE=1 restores it, now playing best/.
  if [[ "$GATE" == 0 ]]; then
    echo "== cycle $gen: eval (tournament every $MEASURE_EVERY; self-play on $SELFPLAY_NET) =="
    EVAL_MODE=(--ungated --measure-every "$MEASURE_EVERY" --eval-parallel "$EVAL_PARALLEL"
               --self-play-net "$SELFPLAY_NET")
  else
    echo "== cycle $gen: eval vs champion (gates promotion) =="
    EVAL_MODE=()
  fi
  # Greedy is a slow external yardstick, not a rung on the ladder. The eval
  # loop runs it only when the candidate is promoted, so the games are offered
  # unconditionally here and it decides — a schedule in this script would be a
  # second, disagreeing mechanism.
  "$PYTHON" "$repo/python/minions_eval_loop.py" --live "$LIVE" \
    --engine "$ENGINE" --games "$EVAL_GAMES" --candidates "$EVAL_CANDIDATES" \
    --sims "$EVAL_SIMS" --greedy-games "$GREEDY_GAMES" \
    --t-steps "$EVAL_T_STEPS" --promote-threshold "$PROMOTE" --once \
    --backend "$BACKEND" --ep "$EP" $FP16_FLAG ${EVAL_MODE[@]+"${EVAL_MODE[@]}"} || {
      # Never let evaluation kill the run. The script is `set -e`, so a bad
      # eval used to take the whole cycle down with it — an unattended run
      # died at cycle 3 that way, losing the night. A skipped eval means the
      # champion simply stands for another generation.
      echo "WARNING: cycle $gen eval failed; champion unchanged" >&2
    }
done
