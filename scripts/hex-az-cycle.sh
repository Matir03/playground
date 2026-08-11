#!/usr/bin/env bash
# Hex AlphaZero as a strictly serial loop:  generate -> train -> eval -> repeat.
#
# Nothing overlaps. Each phase is a process that runs to completion and exits, so
# the GPU is doing exactly one thing at a time:
#
#   generate  actor plays until it has written --target-examples, then exits
#   train     learner trains one generation from the buffer, then exits
#   eval      every --eval-every cycles: candidate vs champion (the gate), and
#             MoHex if it promotes; then exits
#
# (The evaluator's actor-pause handshake is vestigial here — nothing else is
# running — so it is called with --pause-timeout 0 rather than waiting 90s for a
# reply that can't come.)
#
# The alternative — actor and learner running continuously side by side — keeps
# the GPU busier on paper but has them fighting for it, and on a thermally
# capped laptop contention costs more than the idle time it saves. It also makes
# every measurement ambiguous: you can't tell whether a slow generation was the
# search or the trainer.
#
#   scripts/hex-az-cycle.sh          # run until stopped (Ctrl-C or `stop`)
#   scripts/hex-az-cycle.sh stop
#   scripts/hex-az-cycle.sh status
#
# Needs /tmp/cuda_env.sh and scripts/ort-env.sh sourced (the script sources them).
set -o pipefail  # not -u: the env scripts append to possibly-unset paths

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVE="${LIVE:-$repo/data/models/hex_az/live}"
LOGS="$LIVE/logs"
ENGINE="$repo/target/release/hex-engine"
PYTHON="${PYTHON:-$repo/python/.venv/bin/python}"

# Generate: how much fresh data per cycle, and how it's played.
TARGET_EXAMPLES="${TARGET_EXAMPLES:-100000}"
CONCURRENCY="${CONCURRENCY:-768}"
MAX_BATCH="${MAX_BATCH:-256}"
NODE_BUDGET="${NODE_BUDGET:-400}"
FAST_NODE_BUDGET="${FAST_NODE_BUDGET:-100}"
FULL_SEARCH_PROB="${FULL_SEARCH_PROB:-0.25}"
C_PUCT="${C_PUCT:-1.5}"
CHUNK_SECS="${CHUNK_SECS:-60}"
# The buffer window that matters is examples, not chunks: a chunk is 60s of play,
# so its size tracks GPU speed. When the GPU clamped to 210MHz, 40 chunks fell
# from 270k examples to 63k, reuse went 0.6x -> 2.4x, and the value loss jumped
# 0.09 -> 0.68. Chunk count is now just a backstop.
MAX_CHUNKS="${MAX_CHUNKS:-200}"
MAX_EXAMPLES="${MAX_EXAMPLES:-300000}"
BACKEND="${BACKEND:-ort}"
EP="${EP:-trt}"

# Train.
CHANNELS="${CHANNELS:-64}"
BLOCKS="${BLOCKS:-6}"
STEPS="${STEPS:-600}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# Eval / gate.
EVAL_EVERY="${EVAL_EVERY:-3}"          # cycles between gate matches
# 40 games at a 52% bar. The gate has two failure modes and they pull opposite
# ways: promote a worse net and the run drifts down (the 2026-08-09 failure);
# refuse a better one and the champion freezes, which caps the *data* — the
# actor plays from best.*, so a stale champion means stale examples for every
# generation after it. 24 games at 55% turned out to sit on the wrong side of
# that: two candidates scored 0.458 and 0.542, both a coin flip from even, and
# the champion sat for seven generations while the losses kept falling.
# AGZ afforded 400-game matches; at this scale the honest choice is a lower bar
# with more games. (SPRT would be the principled fix — play until decisive.)
EVAL_GAMES="${EVAL_GAMES:-40}"         # candidate vs champion
PROMOTE_THRESHOLD="${PROMOTE_THRESHOLD:-0.52}"
MOHEX_GAMES="${MOHEX_GAMES:-8}"        # absolute yardstick, promoted nets only
EVAL_SIMS="${EVAL_SIMS:-400}"
# 100, not 1000: at 1000 MoHex wins every game against a young net and the
# absolute column is 0.000 forever. Scores are comparable only within one
# setting, so changing this restarts that series.
MOHEX_PLAYOUTS="${MOHEX_PLAYOUTS:-100}"
# Resolve MoHex to a path: the cycle runs detached, without a login shell, so
# ~/.local/bin isn't necessarily on PATH and a bare "mohex" fails to spawn.
MOHEX_BIN="${MOHEX_BIN:-$(command -v mohex || echo "$HOME/.local/bin/mohex")}"

case "${1:-run}" in
  stop)
    pkill -f "hex-az-cycle.sh"
    pkill -f "hex-engine selfplay-loop --live $LIVE"
    pkill -f "hex_az_learner.py --live $LIVE"
    pkill -f "hex_eval_loop.py --live $LIVE"
    echo "stopped"; exit 0 ;;
  status)
    gen=$($PYTHON -c "import json;print(json.load(open('$LIVE/config.json'))['generation'])" 2>/dev/null || echo '-')
    champ=$($PYTHON -c "import json;d=json.load(open('$LIVE/scores.json'));print('gen%s' % d['best_gen'])" 2>/dev/null || echo '-')
    phase=$(tail -1 "$LOGS/cycle.log" 2>/dev/null)
    echo "generation $gen, champion $champ, $(ls "$LIVE/buffer"/chunk_*.bin 2>/dev/null | wc -l) chunks"
    echo "last: $phase"
    pgrep -f "hex-az-cycle.sh" >/dev/null && echo "cycle: running" || echo "cycle: not running"
    exit 0 ;;
esac

mkdir -p "$LOGS" "$LIVE/buffer"
# The engine needs these to even load its shared libraries, so source before any
# check that runs it.
# shellcheck disable=SC1091
source /tmp/cuda_env.sh
# shellcheck disable=SC1091
source "$repo/scripts/ort-env.sh" >/dev/null
export PYTHONPATH="$repo/python/src"

version="$("$ENGINE" --version 2>&1)"
if [[ $? -ne 0 ]]; then
  echo "$ENGINE won't run: $version" >&2
  echo "(usually a missing LD_LIBRARY_PATH — see /tmp/cuda_env.sh and scripts/ort-env.sh)" >&2
  exit 1
fi
if ! grep -q '+cuda' <<<"$version"; then
  echo "$ENGINE was built without cuda ($version)." >&2
  echo "Rebuild:  source /tmp/cuda_env.sh && cargo build --release -p hex-engine --features cuda" >&2
  exit 1
fi

log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOGS/cycle.log"; }

# The learner writes the first weights; the actor needs them to play at all.
if [[ ! -f "$LIVE/latest.safetensors" ]]; then
  log "bootstrap: writing a random gen-0 net"
  $PYTHON "$repo/python/hex_az_learner.py" --live "$LIVE" --channels "$CHANNELS" \
    --blocks "$BLOCKS" --exit-after-gens 1 --gen-examples 0 --min-samples 1000000000 \
    >> "$LOGS/learner.log" 2>&1 &
  boot=$!
  # It writes gen-0 immediately, then waits for data it will never get here.
  while [[ ! -f "$LIVE/model.onnx" ]]; do sleep 2; done
  kill $boot 2>/dev/null; wait $boot 2>/dev/null
fi

cycle=0
while true; do
  cycle=$((cycle + 1))

  log "cycle $cycle: GENERATE ($TARGET_EXAMPLES examples)"
  "$ENGINE" selfplay-loop --live "$LIVE" --backend "$BACKEND" --ep "$EP" \
    --concurrency "$CONCURRENCY" --max-batch "$MAX_BATCH" --node-budget "$NODE_BUDGET" \
    --fast-node-budget "$FAST_NODE_BUDGET" --full-search-prob "$FULL_SEARCH_PROB" \
    --c-puct "$C_PUCT" --chunk-secs "$CHUNK_SECS" --max-chunks "$MAX_CHUNKS" \
    --max-examples "$MAX_EXAMPLES" \
    --target-examples "$TARGET_EXAMPLES" --seed "$cycle" >> "$LOGS/actor.log" 2>&1
  # The status matters: the actor writes its chunks before exiting, so a failure
  # here is usually a teardown race (ORT/TRT static destructors) rather than lost
  # work. Log the code so the two can be told apart -- 134 is abort().
  rc=$?
  (( rc == 0 )) || log "cycle $cycle: actor exited $rc (continuing)"

  log "cycle $cycle: TRAIN ($STEPS steps)"
  $PYTHON "$repo/python/hex_az_learner.py" --live "$LIVE" --channels "$CHANNELS" \
    --blocks "$BLOCKS" --steps-per-snapshot "$STEPS" --batch-size "$BATCH_SIZE" \
    --gen-examples 0 --exit-after-gens 1 >> "$LOGS/learner.log" 2>&1
  rc=$?
  (( rc == 0 )) || log "cycle $cycle: learner exited $rc (continuing)"

  if (( cycle % EVAL_EVERY == 0 )); then
    log "cycle $cycle: EVAL (gate: $EVAL_GAMES games vs champion)"
    $PYTHON "$repo/python/hex_eval_loop.py" --live "$LIVE" --engine "$ENGINE" \
      --mohex-bin "$MOHEX_BIN" --cuda --once --every 0 --pause-timeout 0 \
      --games "$EVAL_GAMES" --promote-threshold "$PROMOTE_THRESHOLD" \
      --mohex-games "$MOHEX_GAMES" --sims "$EVAL_SIMS" \
      --mohex-playouts "$MOHEX_PLAYOUTS" --channels "$CHANNELS" --blocks "$BLOCKS" \
      >> "$LOGS/evaluator.log" 2>&1
    rc=$?
    (( rc == 0 )) || log "cycle $cycle: evaluator exited $rc (continuing)"
  fi
done
