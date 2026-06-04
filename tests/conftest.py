import os

# MLflow 3.x puts the local filesystem tracking backend (``file://``/``./mlruns``)
# into "maintenance mode" and raises unless this opt-out is set. Several unit
# tests intentionally log to a local file store (no DB/UC available in CI), so
# allow it suite-wide. Set before any test imports mlflow tracking.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
