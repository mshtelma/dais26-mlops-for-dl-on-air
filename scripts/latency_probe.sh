#!/usr/bin/env bash
# latency_probe.sh — probe a Mosaic AI Model Serving endpoint every 60s.
# Outputs a single line per probe: ISO-timestamp, status, latency_ms.
# Fails (exit 1) after 2 consecutive failures (per RUNBOOK switch-to-video threshold).
#
# Usage:
#     DATABRICKS_HOST=<host> DATABRICKS_TOKEN=<token> \
#     bash scripts/latency_probe.sh dais26-cradio-detector-prod 60
set -uo pipefail

ENDPOINT="${1:-dais26-cradio-detector-prod}"
INTERVAL="${2:-60}"
FAIL_THRESHOLD=2
fail_count=0

# 1x1 black PNG, base64-encoded
B64_PNG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

if [[ -z "${DATABRICKS_HOST:-}" || -z "${DATABRICKS_TOKEN:-}" ]]; then
  echo "DATABRICKS_HOST and DATABRICKS_TOKEN must be set." >&2
  exit 2
fi

URL="${DATABRICKS_HOST%/}/serving-endpoints/${ENDPOINT}/invocations"

while true; do
  ts=$(date -u +%FT%TZ)
  start=$(date +%s%3N)
  status=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"dataframe_split\":{\"columns\":[\"image\"],\"data\":[[\"${B64_PNG}\"]]}}" \
    "$URL")
  end=$(date +%s%3N)
  latency=$((end - start))
  if [[ "$status" == "200" ]]; then
    echo "${ts} OK ${latency}ms"
    fail_count=0
  else
    echo "${ts} FAIL status=${status} latency=${latency}ms"
    fail_count=$((fail_count + 1))
    if (( fail_count >= FAIL_THRESHOLD )); then
      echo "${ts} ALERT: ${fail_count} consecutive failures -- switch to backup video" >&2
      exit 1
    fi
  fi
  sleep "$INTERVAL"
done
