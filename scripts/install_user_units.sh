#!/usr/bin/env bash
# Install the ATR systemd USER units (no root needed) and enable them.
#
# asterAIx has no passwordless sudo and Linger=no, so we run everything as
# `systemctl --user`. Survival across logout needs linger, which is the ONE
# step that needs an admin (run once):   sudo loginctl enable-linger "$USER"
#
# Usage:  bash scripts/install_user_units.sh [--no-start]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="${ROOT}/deploy/systemd"
UNIT_DST="${HOME}/.config/systemd/user"
START=1
[ "${1:-}" = "--no-start" ] && START=0

# Engines + gateway. vLLM is NOT a unit — the ModelManager spawns it as a
# subprocess (see docs/asteraix-environment.md / IMPLEMENTATION_PLAN.md §8).
UNITS=(atr-kraken atr-trocr atr-party atr-gateway)

mkdir -p "${UNIT_DST}"
for u in "${UNITS[@]}"; do
  cp "${UNIT_SRC}/${u}.service" "${UNIT_DST}/${u}.service"
  echo "installed ${u}.service"
done

systemctl --user daemon-reload

if ! loginctl show-user "$USER" 2>/dev/null | grep -q 'Linger=yes'; then
  echo "WARNING: linger is OFF — user services stop on logout."
  echo "         Ask an admin to run once:  sudo loginctl enable-linger $USER"
fi

for u in "${UNITS[@]}"; do
  systemctl --user enable "${u}.service"
done

if [ "${START}" -eq 1 ]; then
  # Start engines first, gateway last.
  for u in atr-kraken atr-trocr atr-party atr-gateway; do
    systemctl --user start "${u}.service" || echo "  (start failed: ${u} — check the venv exists)"
  done
fi

echo
echo "Status:"
systemctl --user --no-pager --plain list-units 'atr-*' || true
echo
echo "Logs:   journalctl --user -u atr-gateway -f"
echo "Health: curl -s localhost:8200/health"
