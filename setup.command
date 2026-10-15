#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_ROOT="$HOME/Library/Application Support/BabyX-CV-RL"
PYTHON="$ENV_ROOT/venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11 is required. Install it, then rerun this setup file."
    exit 1
  fi
  mkdir -p "$ENV_ROOT"
  python3.11 -m venv "$ENV_ROOT/venv"
fi

"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
echo "Setup complete. Open run.command to launch the GUI."
