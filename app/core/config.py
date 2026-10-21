from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
SERVICE_NAME = "budgie-ml-service"
SERVICE_VERSION = "0.1.0"
# Preserve the gateway's existing request-signing contract.
TRUSTED_HISTORY_SHARED_SECRET_ENV = "ODIN_TRUSTED_HISTORY_SHARED_SECRET"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_FORECAST_MONTHS = 12
