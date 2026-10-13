#!/bin/bash
# Log this WSL VM's vital signs every INTERVAL seconds, so the minutes before
# a VM death survive it (the journal only has what services chose to log).
# One line per sample in $DIR/YYYY-MM-DD.log; read them with
# scripts/wsl-crashcheck.sh. Start detached:
#   setsid nohup scripts/wsl-sysmon.sh > /dev/null 2>&1 < /dev/null &
DIR=${DIR:-$HOME/.wsl-sysmon}
INTERVAL=${INTERVAL:-15}
KEEP_DAYS=${KEEP_DAYS:-14}
SMI=/usr/lib/wsl/lib/nvidia-smi
# WSL's own nvidia-smi segfaults with driver 617.42 (2026-10-12); the Windows
# one sees the whole GPU (WSL's memory included) and answers in ~1.5 s.
timeout 20 $SMI -L > /dev/null 2>&1 || SMI=/mnt/c/Windows/System32/nvidia-smi.exe
mkdir -p "$DIR"
# One monitor at a time.
exec 9> "$DIR/.lock"
flock -n 9 || { echo "wsl-sysmon already running"; exit 0; }
find "$DIR" -name '*.log' -mtime +"$KEEP_DAYS" -delete
while true; do
  m=$(awk '/^(MemAvailable|Cached|Dirty|Writeback|SwapTotal|SwapFree):/ {v[$1]=int($2/1024)} END {
        printf "avail=%dM cached=%dM dirty=%dM wb=%dM swap=%dM",
          v["MemAvailable:"], v["Cached:"], v["Dirty:"], v["Writeback:"], v["SwapTotal:"]-v["SwapFree:"] }' /proc/meminfo)
  load=$(cut -d' ' -f1-3 /proc/loadavg)
  gpu=$($SMI --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r' || echo "?")
  disk=$(df --output=avail -BG / | tail -1 | tr -d ' ')
  top=$(ps -eo rss=,comm= --sort=-rss | head -3 | awk '{printf "%s:%dM ", $2, $1/1024}')
  echo "$(date '+%F %T') up=$(cut -d' ' -f1 /proc/uptime) $m load=${load// /,} gpu=$gpu disk=$disk top=[${top% }]" \
    >> "$DIR/$(date +%F).log"
  sleep "$INTERVAL"
done
