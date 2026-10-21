#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${ODIN_TRUSTED_HISTORY_SHARED_SECRET:?Set the gateway shared secret in .env or the environment}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec .venv/bin/uvicorn app.main:app \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --workers 1 --limit-concurrency 16 --timeout-keep-alive 5
