#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Dependencies are not installed. Run: bash install.sh" >&2
  exit 1
fi

[[ -f .env ]] || cp .env.example .env
echo "Starting FinRisk at http://127.0.0.1:8000"
exec .venv/bin/python scripts/run_web.py
