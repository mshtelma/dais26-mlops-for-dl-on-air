#!/usr/bin/env bash
# Build the wheel and upload it to the Databricks workspace so that the
# `@distributed` notebook entrypoint can `%pip install` it from /Workspace.
#
# Usage:
#   ./scripts/upload_wheel.sh [profile]
#
# Requires: `databricks` CLI configured, `uv` installed.

set -euo pipefail

PROFILE="${1:-DEFAULT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Building wheel"
rm -rf dist/
uv build

WHEEL="$(ls dist/*.whl | head -1)"
if [ -z "$WHEEL" ]; then
  echo "ERROR: no wheel produced in dist/"
  exit 1
fi
echo "==> Built $WHEEL"

CURRENT_USER="$(databricks --profile "$PROFILE" current-user me --output json | python3 -c 'import json,sys; print(json.load(sys.stdin)["userName"])')"
DEST="/Workspace/Users/${CURRENT_USER}/dais26/dist/$(basename "$WHEEL")"

echo "==> Uploading to $DEST"
databricks --profile "$PROFILE" workspace mkdirs "/Workspace/Users/${CURRENT_USER}/dais26/dist" || true
databricks --profile "$PROFILE" workspace import \
  --format AUTO --overwrite \
  --file "$WHEEL" \
  "$DEST"

echo "==> Done. In your notebook, %pip install:"
echo "    $DEST"
