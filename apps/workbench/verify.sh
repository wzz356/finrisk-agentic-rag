#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Test dependencies are not installed. Run: bash install.sh" >&2
  exit 1
fi

.venv/bin/python scripts/verify_delivery.py
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
mkdir -p .tmp
export TMPDIR="$PROJECT_ROOT/.tmp"
.venv/bin/python -m pytest -q --basetemp=.tmp/pytest-verify

if command -v node >/dev/null 2>&1; then
  node --check src/web/static/app.js
  node --check src/web/static/source-map.js
  node --test tests/web/*.test.mjs
  node --check agent_runtime/worker.mjs
  node --check agent_runtime/runtime.mjs
  node --test agent_runtime/test/*.test.mjs
else
  echo "Node.js is required to validate Pi Agent." >&2
  exit 1
fi

echo "All FinRisk delivery checks passed."
