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
MAX_CHUNKS="${MAX_CHUNKS:-40}"
BACKEND="${BACKEND:-ort}"
EP="${EP:-trt}"

# Train.
CHANNELS="${CHANNELS:-64}"
BLOCKS="${BLOCKS:-6}"
STEPS="${STEPS:-600}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# Eval / gate.
EVAL_EVERY="${EVAL_EVERY:-3}"          # cycles between gate matches
EVAL_GAMES="${EVAL_GAMES:-24}"         # candidate vs champion
MOHEX_GAMES="${MOHEX_GAMES:-8}"        # absolute yardstick, promoted nets only
EVAL_SIMS="${EVAL_SIMS:-400}"
MOHEX_PLAYOUTS="${MOHEX_PLAYOUTS:-1000}"
MOHEX_BIN="${MOHEX_BIN:-mohex}"

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
    --target-examples "$TARGET_EXAMPLES" --seed "$cycle" >> "$LOGS/actor.log" 2>&1 \
    || log "cycle $cycle: actor exited non-zero (continuing)"

  log "cycle $cycle: TRAIN ($STEPS steps)"
  $PYTHON "$repo/python/hex_az_learner.py" --live "$LIVE" --channels "$CHANNELS" \
    --blocks "$BLOCKS" --steps-per-snapshot "$STEPS" --batch-size "$BATCH_SIZE" \
    --gen-examples 0 --exit-after-gens 1 >> "$LOGS/learner.log" 2>&1 \
    || log "cycle $cycle: learner exited non-zero (continuing)"

  if (( cycle % EVAL_EVERY == 0 )); then
    log "cycle $cycle: EVAL (gate: $EVAL_GAMES games vs champion)"
    $PYTHON "$repo/python/hex_eval_loop.py" --live "$LIVE" --engine "$ENGINE" \
      --mohex-bin "$MOHEX_BIN" --cuda --once --every 0 \
      --games "$EVAL_GAMES" --mohex-games "$MOHEX_GAMES" --sims "$EVAL_SIMS" \
      --mohex-playouts "$MOHEX_PLAYOUTS" --channels "$CHANNELS" --blocks "$BLOCKS" \
      >> "$LOGS/evaluator.log" 2>&1 \
      || log "cycle $cycle: evaluator exited non-zero (continuing)"
  fi
done
