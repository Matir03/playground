#!/usr/bin/env bash
# Supervise a Hex AlphaZero training run: actor + learner, restarted if either
# dies, logging to <live>/logs.
#
# The actor self-plays continuously into <live>/buffer and hot-reloads weights;
# the learner trains a generation per burst of fresh data and publishes
# latest.safetensors + model.onnx. Neither owns the other, so each is supervised
# on its own — a learner crash must not stop self-play, and vice versa.
#
#   scripts/hex-az-run.sh start   # launch (idempotent: refuses if already up)
#   scripts/hex-az-run.sh stop
#   scripts/hex-az-run.sh status
#
# Needs /tmp/cuda_env.sh and scripts/ort-env.sh (sourced here). Everything runs
# detached, so it survives the shell that starts it.
set -o pipefail  # not -u: the CUDA/ORT env scripts append to possibly-unset paths

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVE="${LIVE:-$repo/data/models/hex_az/live}"
LOGS="$LIVE/logs"
ENGINE="$repo/target/release/hex-engine"
PYTHON="${PYTHON:-$repo/python/.venv/bin/python}"

# Self-play knobs. Concurrency ~3x the batch keeps batches full; node-budget is
# the per-move search size.
CONCURRENCY="${CONCURRENCY:-768}"
MAX_BATCH="${MAX_BATCH:-256}"
NODE_BUDGET="${NODE_BUDGET:-400}"
# PUCT exploration. 0.6 was tried on 2026-08-10 to concentrate the search: it
# worked mechanically (target entropy 3.97 -> 1.21 nats) and bought no strength,
# because sharper targets that disagree with each other are still noise. Target
# quality now comes from policy target pruning and the ownership signal instead,
# so this is back to the AZ-standard value.
C_PUCT="${C_PUCT:-1.5}"
CHUNK_SECS="${CHUNK_SECS:-60}"
MAX_CHUNKS="${MAX_CHUNKS:-40}"
BACKEND="${BACKEND:-ort}"
EP="${EP:-trt}"

# Learner knobs. GEN_EXAMPLES is how much *fresh* self-play accumulates before a
# generation is trained, so it sets the cadence with the actor's throughput.
CHANNELS="${CHANNELS:-64}"
BLOCKS="${BLOCKS:-6}"
GEN_EXAMPLES="${GEN_EXAMPLES:-100000}"

# Gate. MoHex at 1000 playouts is a modest but real yardstick; 8 games keeps the
# actor paused only briefly, and every 3rd generation keeps that rare.
EVAL_GAMES="${EVAL_GAMES:-8}"
EVAL_SIMS="${EVAL_SIMS:-400}"
MOHEX_PLAYOUTS="${MOHEX_PLAYOUTS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-3}"
STEPS="${STEPS:-600}"
BATCH_SIZE="${BATCH_SIZE:-256}"

start() {
  if pgrep -f "hex-engine selfplay-loop --live $LIVE" >/dev/null \
    || pgrep -f "hex_az_learner.py --live $LIVE" >/dev/null; then
    echo "already running (stop first):"; status; exit 1
  fi
  mkdir -p "$LOGS" "$LIVE/buffer"
  # shellcheck disable=SC1091
  source /tmp/cuda_env.sh
  # shellcheck disable=SC1091
  source "$repo/scripts/ort-env.sh" >/dev/null

  # The learner publishes the gen-0 weights the actor waits for, so start it first.
  supervise learner "$PYTHON $repo/python/hex_az_learner.py --live $LIVE \
      --channels $CHANNELS --blocks $BLOCKS --gen-examples $GEN_EXAMPLES \
      --steps-per-snapshot $STEPS --batch-size $BATCH_SIZE" PYTHONPATH="$repo/python/src"
  supervise actor "$ENGINE selfplay-loop --live $LIVE --backend $BACKEND --ep $EP \
      --concurrency $CONCURRENCY --max-batch $MAX_BATCH --node-budget $NODE_BUDGET \
      --c-puct $C_PUCT --chunk-secs $CHUNK_SECS --max-chunks $MAX_CHUNKS"

  # The gate: scores snapshots against MoHex and promotes the best to best.*,
  # which is what the actor plays from. Without it every snapshot becomes the
  # generator and a net that gets worse feeds itself worse data. GATE=0 to skip.
  if [[ "${GATE:-1}" == "1" ]]; then
    supervise evaluator "$PYTHON $repo/python/hex_eval_loop.py --live $LIVE \
        --engine $ENGINE --mohex-bin ${MOHEX_BIN:-mohex} --cuda \
        --games $EVAL_GAMES --sims $EVAL_SIMS --mohex-playouts $MOHEX_PLAYOUTS \
        --every $EVAL_EVERY --channels $CHANNELS --blocks $BLOCKS" \
        PYTHONPATH="$repo/python/src"
  fi

  echo "started. logs in $LOGS"
}

# Run `cmd` forever, restarting it if it exits; one line per restart in the log.
supervise() {
  local name="$1" cmd="$2"; shift 2
  local env_prefix=("$@")
  (
    while true; do
      echo "=== $(date -Is) starting $name" >> "$LOGS/$name.log"
      env "${env_prefix[@]}" $cmd >> "$LOGS/$name.log" 2>&1
      echo "=== $(date -Is) $name exited ($?), restarting in 10s" >> "$LOGS/$name.log"
      sleep 10
    done
  ) </dev/null >/dev/null 2>&1 &
  disown
  echo "$name supervised (pid $!)"
}

stop() {
  pkill -f "hex-az-run.sh" 2>/dev/null   # the supervisor loops
  pkill -f "hex-engine selfplay-loop --live $LIVE"
  pkill -f "hex_az_learner.py --live $LIVE"
  pkill -f "hex_eval_loop.py --live $LIVE"
  echo "stopped"
}

status() {
  local gen ex
  gen=$(python3 -c "import json;print(json.load(open('$LIVE/config.json'))['generation'])" 2>/dev/null || echo "-")
  ex=$(ls "$LIVE/buffer"/chunk_*.bin 2>/dev/null | wc -l)
  echo "generation $gen, $ex chunks buffered"
  pgrep -af "hex-engine selfplay-loop --live $LIVE" | head -1 || echo "  actor:   DOWN"
  pgrep -af "hex_az_learner.py --live $LIVE" | head -1 || echo "  learner: DOWN"
  if [[ "${GATE:-1}" == "1" ]]; then
    pgrep -af "hex_eval_loop.py --live $LIVE" | head -1 || echo "  evaluator: DOWN"
  fi
  tail -2 "$LOGS/learner.log" 2>/dev/null
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
