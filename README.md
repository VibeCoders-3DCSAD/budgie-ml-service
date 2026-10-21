# Budgie ML Service

Inference-only service extracted from [Odin-ML](https://github.com/VibeCoders-3DCSAD/Odin-ml), intended for a dedicated 2 GB host.

| Endpoint | Implementation |
| --- | --- |
| `POST /api/v1/classification/v2` | Financial classification rules (`financial_rules_v1`) |
| `POST /api/v1/forecast/predict` | 12 released category SARIMA models plus personal spending baselines |
| `POST /api/v1/budget/recommend` | Hierarchical SciPy linear programming |
| `POST /api/v1/budget/recommend/batch` | Up to 10 recommendations for the same signed user |
| `GET /health` | Process liveness |
| `GET /ready` | Model release and signing configuration readiness; HTTP 503 when unavailable |
| `GET /metrics` | Service and active model/rule versions as JSON |

Training scripts, datasets, PFP/SVM, anomaly detection, old forecasters, PyTorch, and scikit-learn are excluded. PyArrow is required because the released SARIMA objects contain Arrow-backed pandas data.

## Local setup

Use Python 3.14.4, matching the artifact release, and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --no-dev
cp .env.example .env
# Set ODIN_TRUSTED_HISTORY_SHARED_SECRET to the secret used by Odin API.
./start.sh
```

The launcher binds to `127.0.0.1:8000` by default. Set `HOST` and `PORT` in `.env` if needed. It runs one worker, caps accepted concurrent connections at 16, and sets numerical thread counts to one. It does not enable reload. Configure a process supervisor to restart it on failure.

Without uv, install the locked runtime dependencies into a virtual environment:

```bash
python3.14 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.txt
```

Artifacts are resolved relative to the source tree, not the current directory. Run from a checkout or the supplied container; the Python wheel alone does not bundle models.

## Container

```bash
docker build -t budgie-ml-service .
docker run --rm --name budgie-ml-service \
  --memory=1536m --memory-swap=1536m --cpus=1 --pids-limit=128 \
  --env-file .env -p 127.0.0.1:8000:8000 budgie-ml-service
```

The image runs as an unprivileged user and checks `/ready`. It listens on port 8000 inside the container; `HOST` and `PORT` apply to the local launcher only. The 1536 MiB container cap reserves host memory on a 2 GB machine; validate against your actual traffic before deployment.

## Authentication and integration

Point Odin's existing ML base URL configuration at this service. Endpoint paths, response contracts, model identifiers, and signing inputs remain compatible. The ML service receives history and financial inputs from Odin API; it does not connect to a database.

Every inference request requires:

- `X-Odin-User-Id`: the request body's `user_id`.
- `X-Odin-History-Source`: `odin_api`.
- `X-Odin-History-Signature`: hexadecimal HMAC-SHA256 using `ODIN_TRUSTED_HISTORY_SHARED_SECRET` over `source + ":" + user_id + ":" + raw_body`.

Sign and send the exact same body bytes. Every request in a batch must belong to that signed user. Missing or invalid signatures return 401; missing signing configuration returns 503. Health and metadata endpoints are unsigned. The inherited signature scheme has no timestamp or replay window; it relies on a trusted transport between Odin API and this service.

## Resource limits

- Request bodies: 4 MiB, checked before JSON parsing and signature verification.
- Forecast history: 10,000 transactions; at most 12 category entries.
- Forecast target: current month through 12 months ahead; category/history availability rules still apply.
- Recommendation: 100 categories, 10,000 optional historical transactions, 10 requests per batch.
- Classification: 50 credit card accounts, 120 cycles per account.
- One numerical computation at a time across forecasting and recommendations, offloaded to a thread so health requests remain responsive.

These upper bounds constrain work; they are not latency guarantees. Oversized bodies return 413, invalid schemas or unsupported forecast horizons return 422, and Uvicorn rejects excess concurrency with 503.

## Artifact releases

`models/hfce_sarima_v4/` contains the 12 serving artifacts and their original metadata/evaluation. `models/budget/` preserves the budget release metadata/configuration and response model identifier; actual optimization rules are implemented in `app/services/budget_service.py`.

1. Train and evaluate in Odin-ML. Export a complete trusted release with metadata and evaluation reports.
2. Replace the relevant files in `models/`; update `models/manifest.json` with the source commit and SHA-256 of every serving file except the manifest itself.
3. Preserve the training runtime versions or explicitly validate a dependency migration. Joblib deserialization requires compatible libraries and can execute code; only ship trusted artifacts. Checksums detect corruption, not malicious releases.
4. Run the full test suite, including the real 12-category forecast test, then build and exercise the container before releasing the updated commit.

Startup verifies all file checksums before deserialization and fails on an incomplete, corrupt, or incompatible release. Models load once per process; no training or fitting runs at request time.

The initial extraction uses Odin-ML commit `1685d2ec1006e5a9c6421cd4bc2718d57d41c6c5` plus the current working-tree classification and budget updates from 2026-10-21. The source checkout was left untouched.

## Development

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy app
```

After changing dependencies, commit the updated lock file and regenerate the hashed runtime requirements:

```bash
uv lock
uv export --locked --no-dev --no-emit-project --format requirements-txt --output-file requirements.txt
```

Tests use synthetic request inputs and the bundled serving artifacts; no training datasets are needed. See `docs/validation.md` for the extraction's validation and memory measurements.
