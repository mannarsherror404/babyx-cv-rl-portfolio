#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$HOME/Library/Application Support/BabyX-CV-RL/venv"
PYTHON="$ENV_DIR/bin/python"

if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c 'import importlib.util; assert all(importlib.util.find_spec(name) for name in ("customtkinter", "PIL", "cv2", "mediapipe", "torch", "ultralytics"))' >/dev/null 2>&1; then
  echo "BabyX needs a one-time dependency setup. This may download over 1 GB."
  "$SCRIPT_DIR/setup.command"
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" "$SCRIPT_DIR/baby_gui.py"
