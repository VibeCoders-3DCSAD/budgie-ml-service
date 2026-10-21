# Budgie ML Service Implementation Plan

> Execute inline in this session. The user requires direct implementation without subagents.

**Goal:** Extract Odin's active inference endpoints into an independent repository for a 2 GB host.

**Architecture:** Preserve the existing classification, SARIMA forecast, budget recommendation, and HMAC contracts. Load only released HFCE artifacts and budget metadata; keep all training code in Odin-ML.

**Tech Stack:** Python 3.14.4, FastAPI, Uvicorn, Pydantic, statsmodels, pandas, NumPy, SciPy, joblib.

## Global Constraints

- Folder and repository: `budgie-ml-service`; organization: `VibeCoders-3DCSAD`.
- Preserve the current working files from `odin-ml`, including its uncommitted behavior, without modifying the source repository.
- Exclude PyTorch, scikit-learn, training data, training scripts, PFP, and unused forecasters.
- One production worker, no reload, bounded request sizes and serial numerical computation.
- Existing global git instructions require review findings and the user's `reviewed` keyword before pushing.

## Task 1: Extract the serving implementation

- [x] Copy the active routes, schemas, and business logic into `app/` and their behavioral tests into `tests/`.
- [x] Replace `app/services/features.py` with the transaction-to-dataframe conversion used by forecasting.
- [x] Replace `app/models/registry.py` with HFCE and budget loading; add checksums in `models/manifest.json` and validate before deserialization.
- [x] Fix readiness to check the active HFCE category artifacts, budget metadata, and signing configuration, returning HTTP 503 when unavailable.

## Task 2: Bound the runtime and package deployment

- [x] Add shared computation limiting in `app/api/deps.py`, offload forecast and LP work to a thread, and constrain lists, body size, and forecast horizon.
- [x] Add `pyproject.toml`, a lock file, `.python-version`, `.env.example`, `start.sh`, a container definition, and setup/release documentation.
- [x] Create a clean virtual environment using `uv sync`, verifying no training-only packages are installed.

## Task 3: Verify and prepare the repository

- [x] Run inherited endpoint/rule/optimization tests and add tests for model integrity, readiness, authentication, and resource limits.
- [x] Verify all 12 real category artifacts can forecast and the original and extracted forecast results match.
- [x] Run lint, formatting, and type checks; measure startup and warmed inference memory.
- [x] Initialize a local git repository and create the GitHub repository in the source organization, matching its public visibility.
- [x] Review and commit the complete implementation; report the review and request `reviewed` before pushing.

## Publication status

Local implementation and validation complete. GitHub repository created; source upload awaits the user's `reviewed` keyword under `/home/charles/Documents/AGENTS.md`.
