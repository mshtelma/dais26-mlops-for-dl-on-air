#!/usr/bin/env bash
# Interim local deploy: deploy the DAB and wire the MLflow deployment-job linkage
# in one shot. This is the manual equivalent of .github/workflows/deploy.yml,
# used while CI cannot reach the workspace (GitHub-hosted runner IPs are blocked
# by the workspace IP access list).
#
# DABs cannot declare `deployment_job_id` on a registered model, so after every
# `bundle deploy` we must run `connect_deployment_job` (notebooks/13), which calls
# update_registered_model(deployment_job_id=...) on the detector models.
#
# Usage:
#   ./scripts/deploy_bundle.sh [-t dev|prod] [-p PROFILE] [-y]
#
# Options:
#   -t TARGET    Bundle target to deploy (default: dev)
#   -p PROFILE   Databricks CLI profile to use (default: $DATABRICKS_CONFIG_PROFILE or DEFAULT)
#   -y           Skip the confirmation prompt (non-interactive)
#
# Env overrides:
#   SP_APP_ID    Service principal application ID (UUID). When set, passed as
#                --var sp_app_id=... (required for prod run_as if you don't rely
#                on the default in databricks.yml).
#
# Requires: `databricks` CLI configured, `uv` installed (the wheel artifact builds
# during deploy).

set -euo pipefail

TARGET="dev"
PROFILE="${DATABRICKS_CONFIG_PROFILE:-DEFAULT}"
ASSUME_YES=0

while getopts ":t:p:yh" opt; do
  case "$opt" in
    t) TARGET="$OPTARG" ;;
    p) PROFILE="$OPTARG" ;;
    y) ASSUME_YES=1 ;;
    h)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    \?) echo "ERROR: unknown option -$OPTARG" >&2; exit 2 ;;
    :)  echo "ERROR: option -$OPTARG requires an argument" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "dev" && "$TARGET" != "prod" ]]; then
  echo "ERROR: -t must be 'dev' or 'prod' (got '$TARGET')" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional sp_app_id override (mirrors the workflow's --var passing).
VAR_ARGS=()
if [[ -n "${SP_APP_ID:-}" ]]; then
  VAR_ARGS+=("--var=sp_app_id=${SP_APP_ID}")
fi

echo "==> Target:  $TARGET"
echo "==> Profile: $PROFILE"

echo "==> Verifying auth"
WHOAMI="$(databricks --profile "$PROFILE" current-user me --output json \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("userName") or d.get("displayName") or d.get("id"))')"
echo "    authenticated as: $WHOAMI"

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Deploy bundle to '$TARGET' as '$WHOAMI'? [y/N] " reply
  case "$reply" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

# Ensure the UC schemas the bundle's registered models live in exist.
#   * dais26_vfm (CATALOG.SCHEMA) is shared, data-laden dev infra and is NEVER
#     bundle-managed — always ensured here.
#   * dais26_vfm_prod (CHAMPION_SCHEMA) is Terraform-managed ONLY in the prod
#     target (see databricks.yml targets.prod.resources.schemas). So we
#     pre-create it for the dev target (dev's champion models point at the literal
#     schema but the bundle doesn't manage it there), and DO NOT touch it for prod
#     (Terraform owns it; bind it once if it pre-exists). Keep in sync with
#     notebooks/00_config.py.
CATALOG="mlops_pj"
SCHEMAS=("dais26_vfm")
if [[ "$TARGET" == "dev" ]]; then
  SCHEMAS+=("dais26_vfm_prod")
fi

echo "==> Ensuring UC schemas exist in '$CATALOG'"
for sch in "${SCHEMAS[@]}"; do
  if databricks --profile "$PROFILE" schemas get "${CATALOG}.${sch}" >/dev/null 2>&1; then
    echo "    exists:  ${CATALOG}.${sch}"
  else
    echo "    creating: ${CATALOG}.${sch}"
    databricks --profile "$PROFILE" schemas create "$sch" "$CATALOG" \
      --comment "Managed by scripts/deploy_bundle.sh (see notebooks/00_config.py)." >/dev/null
  fi
done

echo "==> Deploying bundle (-t $TARGET)"
databricks --profile "$PROFILE" bundle deploy -t "$TARGET" "${VAR_ARGS[@]+"${VAR_ARGS[@]}"}"

echo "==> Wiring deployment-job linkage (connect_deployment_job)"
databricks --profile "$PROFILE" bundle run connect_deployment_job -t "$TARGET" "${VAR_ARGS[@]+"${VAR_ARGS[@]}"}"

echo "==> Done. Bundle deployed and deployment-job linkage wired for '$TARGET'."
