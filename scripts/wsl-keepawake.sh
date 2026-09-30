#!/bin/bash
# Keep the Windows host out of idle sleep while a long job runs in WSL: the
# laptop's Modern Standby froze the VM for good (2026-09-29; see
# scripts/wsl-crashcheck.sh). While a process named $WATCH is alive, resets
# Windows' idle-sleep timer every minute (SetThreadExecutionState with
# ES_SYSTEM_REQUIRED, no ES_CONTINUOUS, so nothing lingers when this stops).
# No admin rights and no power-setting changes; closing the lid or choosing
# Sleep still sleeps.
WATCH=${WATCH:-cycle.sh}
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
CALL="Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);'; [void][W.P]::SetThreadExecutionState(1)"
cd /mnt/c || exit 1
while true; do
  if pgrep -x "$WATCH" > /dev/null; then
    timeout 60 "$PS" -NoProfile -NonInteractive -Command "$CALL" > /dev/null 2>&1
  fi
  sleep 60
done
