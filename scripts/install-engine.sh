#!/usr/bin/env bash
# Bone Voyage — install the local compute engine (Python).
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Julian-Baumeler/bone-voyage/main/scripts/install-engine.sh | bash
# Or from a clone:
#   bash scripts/install-engine.sh
set -euo pipefail

REPO_URL="${BONE_VOYAGE_REPO:-https://github.com/Julian-Baumeler/bone-voyage.git}"
ZIP_URL="${BONE_VOYAGE_ZIP:-https://github.com/Julian-Baumeler/bone-voyage/archive/refs/heads/main.zip}"
INSTALL_DIR="${BONE_VOYAGE_HOME:-$HOME/bone-voyage}"
PORT="${BONE_VOYAGE_PORT:-8742}"

echo ""
echo "🦴✈️  Bone Voyage — local engine installer"
echo "    Install dir: $INSTALL_DIR"
echo ""

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

need python3
PYTHON="$(command -v python3)"
PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using Python $PY_VER ($PYTHON)"

# --- fetch sources ----------------------------------------------------------
if [[ -f "$INSTALL_DIR/pyproject.toml" ]]; then
  echo "Found existing checkout at $INSTALL_DIR"
elif command -v git >/dev/null 2>&1; then
  echo "Cloning $REPO_URL …"
  git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
else
  need curl
  need unzip
  TMP="$(mktemp -d)"
  echo "Downloading source zip…"
  curl -fL "$ZIP_URL" -o "$TMP/bone-voyage.zip"
  unzip -q "$TMP/bone-voyage.zip" -d "$TMP"
  # GitHub zip extracts to bone-voyage-main/
  SRC="$(find "$TMP" -maxdepth 1 -type d -name 'bone-voyage-*' | head -1)"
  mkdir -p "$(dirname "$INSTALL_DIR")"
  mv "$SRC" "$INSTALL_DIR"
  rm -rf "$TMP"
fi

cd "$INSTALL_DIR"

# --- venv + deps ------------------------------------------------------------
if [[ ! -d .venv ]]; then
  echo "Creating virtualenv…"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
echo "Installing Bone Voyage (this pulls NumPy / SimpleITK / VTK — may take a few minutes)…"
pip install -e .

# --- launcher ---------------------------------------------------------------
START="$INSTALL_DIR/start-engine.sh"
cat > "$START" <<EOF
#!/usr/bin/env bash
# Start Bone Voyage local engine (CT processing stays on this machine).
set -euo pipefail
cd "$INSTALL_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
export BONE_VOYAGE_PORT="\${BONE_VOYAGE_PORT:-$PORT}"
echo ""
echo "🦴✈️  Bone Voyage engine → http://127.0.0.1:\${BONE_VOYAGE_PORT}"
echo "    Keep this terminal open. Return to the website and it will connect."
echo "    Research use only — not for diagnosis or treatment."
echo ""
exec opengem serve --host 127.0.0.1 --port "\${BONE_VOYAGE_PORT}"
EOF
chmod +x "$START"

# Optional convenience link
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
if [[ -d "$BIN_DIR" ]] || mkdir -p "$BIN_DIR" 2>/dev/null; then
  ln -sf "$START" "$BIN_DIR/bone-voyage-engine" 2>/dev/null || true
fi

echo ""
echo "✅  Engine installed."
echo ""
echo "Next steps:"
echo "  1. Start the engine:"
echo "       $START"
if [[ -x "$BIN_DIR/bone-voyage-engine" ]]; then
  echo "     or: bone-voyage-engine   (if $BIN_DIR is on your PATH)"
fi
echo "  2. Open / keep open the Bone Voyage website (GitHub Pages)."
echo "  3. Wait until the page says the engine is online — then drop a CT."
echo ""
echo "Data stays under ~/.opengem/  — nothing is uploaded to GitHub."
echo ""
