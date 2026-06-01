#!/usr/bin/env bash
# To Run: `source ./launch_env.sh`

VENV_DIR=".LeafflictionVenv"

if [ ! -d "$VENV_DIR" ]; then
	echo "🐍 Creation du venv ($VENV_DIR)..."
	python3 -m venv "$VENV_DIR"
fi

echo "📦 Installation des dependances..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip --no-cache-dir
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt --no-cache-dir

source "$VENV_DIR/bin/activate"
echo "🚀 Ready to Go..."
