#!/usr/bin/env bash
# Start Bone Voyage local engine from a clone / zip extract.
# Prefer ~/bone-voyage/start-engine.sh after running install-engine.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${BONE_VOYAGE_PORT:-8742}"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "No .venv yet. Run first:"
  echo "  bash $ROOT/scripts/install-engine.sh"
  echo "  # or:  cd $ROOT && python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
echo ""
echo "🦴✈️  Bone Voyage engine → http://127.0.0.1:${PORT}"
echo "    Keep this terminal open. Return to the website — it auto-connects."
echo ""
exec opengem serve --host 127.0.0.1 --port "$PORT"
