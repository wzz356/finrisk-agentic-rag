#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "FinRisk requires Python 3.11 or newer." >&2
  exit 1
}

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip

if [[ "${1:-}" == "--runtime-only" ]]; then
  REQUIREMENTS="requirements.lock.txt"
else
  REQUIREMENTS="requirements-dev.lock.txt"
fi

.venv/bin/python -m pip install --requirement "$REQUIREMENTS"
command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 || {
  echo "Pi Agent requires Node.js 22.19 or newer and npm." >&2
  exit 1
}
node -e 'const [major,minor]=process.versions.node.split(".").map(Number);process.exit(major>22||(major===22&&minor>=19)?0:1)'
(cd agent_runtime && npm ci --ignore-scripts --no-audit --no-fund --cache .cache/npm)
[[ -f .env ]] || cp .env.example .env
.venv/bin/python scripts/verify_delivery.py --allow-missing-knowledge

echo "FinRisk dependencies installed. Once the knowledge base is restored, run: bash start.sh"
