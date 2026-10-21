#!/usr/bin/env bash
# Start SARIMA forecasting, rule-based classification, and LP recommendations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${ODIN_TRUSTED_HISTORY_SHARED_SECRET:-}" ]]; then
  printf '%s\n' \
    'Missing ODIN_TRUSTED_HISTORY_SHARED_SECRET.' \
    'Copy .env.example to .env and set the same signing secret used by Odin API.' >&2
  exit 1
fi

if [[ ! -x .venv/bin/python || ! -x .venv/bin/uvicorn ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' \
      'Runtime dependencies are missing.' \
      'Install uv and run: uv sync --locked --no-dev' \
      'Or follow the virtual environment setup in README.md.' >&2
    exit 1
  fi
  uv sync --locked --no-dev
fi

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf '%s\n' \
  'Starting Budgie ML Service: SARIMA, Rule Based, Linear Programming.' \
  "Listening on ${HOST:-127.0.0.1}:${PORT:-8000} (one worker)."

exec .venv/bin/python -m uvicorn app.main:app \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --workers 1 --limit-concurrency 16 --timeout-keep-alive 5
