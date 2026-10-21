# Budgie ML Service

This repository serves classification v2, HFCE SARIMA forecasts, and budget recommendations. Model training and experiments belong in `VibeCoders-3DCSAD/Odin-ml`.

- Preserve the existing `/api/v1` request/response shapes and Odin HMAC signing headers; scope every inference payload to the signed user.
- Treat `pyproject.toml` and `uv.lock` as dependency sources. After dependency changes, regenerate `requirements.txt` with the command in README.md.
- Model releases include metadata and checksums. Read README.md's artifact release instructions before replacing artifacts; verify real forecasts after replacement.
- Keep numerical computation behind the shared limiter and off the event loop. Deployment uses one Uvicorn worker with reload disabled.
- Log operational diagnostics without transaction values, financial assessments, signing secrets, or request bodies.
- Run `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy app` before completing a code change.
