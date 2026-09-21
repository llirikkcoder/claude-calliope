#!/bin/bash
# Поднимает шим Claude, бэкенд Calliope и фронтенд.
# Использование: start.sh /путь/к/calliope
set -euo pipefail

ROOT="${1:-}"
[ -n "$ROOT" ] || { echo "укажи корень репозитория Calliope"; exit 1; }
BACKEND="$ROOT/calliope-backend"; WEB="$ROOT/calliope-web"
LOGS="$ROOT/logs"; mkdir -p "$LOGS"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -x "$BACKEND/.venv/bin/python" ] || { echo "бэкенд не установлен: нет $BACKEND/.venv"; exit 1; }
[ -d "$WEB/node_modules" ] || { echo "фронтенд не установлен: нет $WEB/node_modules"; exit 1; }

cd "$BACKEND"
nohup .venv/bin/python "$HERE/shim.py" --port 8317 > "$LOGS/shim.log" 2>&1 &
echo $! > "$LOGS/shim.pid"

nohup .venv/bin/python -m calliope.main --host 127.0.0.1 --port 8247 > "$LOGS/backend.log" 2>&1 &
echo $! > "$LOGS/backend.pid"

cd "$WEB" && nohup npm run dev > "$LOGS/web.log" 2>&1 &
echo $! > "$LOGS/web.pid"

echo "  шим:      http://127.0.0.1:8317"
echo "  бэкенд:   http://127.0.0.1:8247"
echo "  фронтенд: http://127.0.0.1:5173"
echo "  логи:     $LOGS"
