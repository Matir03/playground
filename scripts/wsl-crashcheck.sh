#!/bin/bash
# Why did the WSL VM's previous boot end? Lines up what the VM saw (the end of
# its journal, the last scripts/wsl-sysmon.sh samples) with what Windows logged
# over the downtime (power states, Store updates of WSL, crashes, driver
# resets, low memory), and guesses a verdict from the known signatures.
#   scripts/wsl-crashcheck.sh      # the boot before this one
#   scripts/wsl-crashcheck.sh -2   # the one before that
B=${1:--1}
DIR=${DIR:-$HOME/.wsl-sysmon}
end_unix=$(journalctl -b "$B" -o short-unix -n 1 --no-pager 2>/dev/null | tail -1 | cut -d' ' -f1 | cut -d. -f1)
[ -n "$end_unix" ] || { echo "no journal for boot $B"; exit 1; }
if [ "$B" = -1 ]; then
  next_unix=$(date -d "$(uptime -s)" +%s)
else
  next_unix=$(journalctl -b "$((B + 1))" -o short-unix --no-pager 2>/dev/null | head -1 | cut -d' ' -f1 | cut -d. -f1)
fi
fmt() { date -d "@$1" '+%Y-%m-%d %H:%M:%S'; }
from=$(fmt $((end_unix - 1800))); end=$(fmt "$end_unix"); next=$(fmt "$next_unix")
echo "boot $B ended $end; next boot $next (down $(( (next_unix - end_unix) / 60 )) min)"

echo; echo "== the VM's last journal lines"
journalctl -b "$B" --no-pager -o short-precise -n 400 2>/dev/null \
  | grep -v -E "Clock change detected|wsl-pro-service" | tail -12 | cut -c1-180

echo; echo "== the last sysmon samples"
if [ -d "$DIR" ]; then
  cat "$DIR"/*.log 2>/dev/null | awk -v e="$end" '$1" "$2 <= e' | tail -5 | cut -c1-220
else
  echo "(no $DIR: start scripts/wsl-sysmon.sh)"
fi

echo; echo "== Windows, $from to $next"
cd /mnt/c || exit 1
timeout 120 powershell.exe -NoProfile -Command "
\$s=[datetime]'$from'; \$e=[datetime]'$next';
\$ev = @()
\$ev += Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=\$s; EndTime=\$e} -ErrorAction SilentlyContinue | Where-Object {
  (\$_.ProviderName -eq 'Microsoft-Windows-Kernel-Power' -and \$_.Id -in 41,42,105,107,109,506,507) -or
  (\$_.ProviderName -eq 'Microsoft-Windows-WindowsUpdateClient' -and \$_.Message -match 'Linux') -or
  (\$_.ProviderName -eq 'EventLog' -and \$_.Id -in 6006,6008) -or
  (\$_.ProviderName -eq 'User32' -and \$_.Id -eq 1074) -or
  (\$_.ProviderName -match 'Resource-Exhaustion') -or
  (\$_.ProviderName -match 'nvlddmkm|Display') -or
  (\$_.ProviderName -match 'Hyper-V' -and \$_.LevelDisplayName -in 'Error','Critical') -or
  (\$_.ProviderName -eq 'Microsoft-Windows-Hyper-V-VmSwitch' -and \$_.Id -eq 69) }
\$ev += Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=\$s; EndTime=\$e} -ErrorAction SilentlyContinue | Where-Object {
  (\$_.ProviderName -eq 'Microsoft-Windows-RestartManager' -and \$_.Message -match 'vmwp|WSL') -or
  (\$_.ProviderName -match 'Application Error|Windows Error Reporting' -and \$_.Message -match 'vmwp|vmmem|wsl') }
\$ev | Sort-Object TimeCreated | Group-Object { \$_.ProviderName + ' ' + \$_.Id + ' ' + ((\$_.Message -split \"\`n\")[0]) } |
  ForEach-Object { \$f=\$_.Group[0]; '{0:MM-dd HH:mm:ss}  x{1,-3} {2} {3}: {4}' -f \$f.TimeCreated, \$_.Count, (\$f.ProviderName -replace 'Microsoft-Windows-',''), \$f.Id, (((\$f.Message -split \"\`n\")[0]).Trim()) }
" 2>/dev/null | tr -d '\r' | cut -c1-200 | tee "$HOME/.wsl-crashcheck.last" | head -40

echo; echo "== verdict (heuristic)"
W="$HOME/.wsl-crashcheck.last"
if journalctl -b "$B" --no-pager -k 2>/dev/null | grep -q "Write-error on swap-device\|Device offlined"; then
  echo "the VM's swap disk failed (hv_storvsc write errors, device offlined) under memory pressure; check the sysmon samples for what was growing"
elif grep -q "WindowsUpdateClient 43.*Linux" "$W"; then
  echo "Store auto-update of WSL restarted the VM (turn off automatic app updates in Microsoft Store, or update WSL by hand)"
elif grep -q "Kernel-Power 506" "$W"; then
  echo "the laptop entered Modern Standby; the VM froze and did not survive it (keep the host awake while jobs run)"
elif grep -q "Kernel-Power 41\|EventLog 6008" "$W"; then
  echo "Windows itself crashed or lost power"
elif grep -q "Resource-Exhaustion" "$W"; then
  echo "Windows ran low on memory (lower memory= in .wslconfig; the VM may take 30 of 31.6 GB)"
elif grep -q -i "nvlddmkm\|Display" "$W"; then
  echo "a GPU driver reset (TDR) around the time the VM died"
elif journalctl -b "$B" --no-pager -n 30 2>/dev/null | grep -q "system will power off"; then
  echo "an orderly poweroff from the host (wsl --shutdown, an update, or a Windows restart)"
else
  echo "the VM stopped abruptly with no Windows explanation found: check the sysmon samples for memory or GPU trouble"
fi
