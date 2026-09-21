#!/bin/bash
# Гасит всё, что поднял start.sh.  Использование: stop.sh /путь/к/calliope
set -euo pipefail
ROOT="${1:-}"
[ -n "$ROOT" ] || { echo "укажи корень репозитория Calliope"; exit 1; }
for s in backend web shim; do
  PID="$ROOT/logs/$s.pid"
  [ -f "$PID" ] || continue
  pkill -P "$(cat "$PID")" 2>/dev/null || true
  kill "$(cat "$PID")" 2>/dev/null && echo "  остановлен: $s" || true
  rm -f "$PID"
done
