#!/bin/bash
# Накладывает патчи плагина на исходники Calliope.
# Использование: apply_patches.sh /путь/к/calliope/calliope-backend
set -euo pipefail

BACKEND="${1:-}"
[ -n "$BACKEND" ] || { echo "укажи путь к calliope-backend"; exit 1; }
[ -d "$BACKEND/src/calliope" ] || { echo "не похоже на calliope-backend: $BACKEND"; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BACKEND"

for patch in "$HERE"/patches/*.patch; do
  name="$(basename "$patch")"
  if patch -p0 --dry-run --silent < "$patch" >/dev/null 2>&1; then
    patch -p0 --backup < "$patch" >/dev/null
    echo "  наложен: $name"
  elif patch -p0 -R --dry-run --silent < "$patch" >/dev/null 2>&1; then
    echo "  уже наложен, пропуск: $name"
  else
    echo "  НЕ ЛОЖИТСЯ (апстрим изменился?): $name"
  fi
done
echo "  перезапусти бэкенд, чтобы правки вступили в силу"
