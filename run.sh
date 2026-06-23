#!/usr/bin/env bash
# One-command startup: creates the venv on first run, installs deps, launches the server.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.12}"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv) with $PYTHON ..."
  "$PYTHON" -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

if [ ! -f ".env" ]; then
  echo "WARNING: no .env file found. Copy .env.example to .env and add your OPENAI_API_KEY."
fi

echo "Starting server on http://localhost:8000 ..."
exec ./.venv/bin/uvicorn app.main:app --reload --port 8000
