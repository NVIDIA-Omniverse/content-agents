#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build and smoke-test the OVRTX rendering API image as its non-root user.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

IMAGE_TAG="${OVRTX_SMOKE_IMAGE_TAG:-ovrtx-rendering-api:non-root-smoke}"
CONTAINER_NAME="${OVRTX_SMOKE_CONTAINER_NAME:-ovrtx-non-root-smoke}"
HOST_PORT="${OVRTX_SMOKE_HOST_PORT:-8011}"
OUTPUT_DIR="${OVRTX_SMOKE_OUTPUT_DIR:-${REPO_ROOT}/_ovrtx_non_root_smoke}"
KEEP_CONTAINER="${OVRTX_SMOKE_KEEP_CONTAINER:-0}"
LOG_PID=""
CURL_CONNECT_TIMEOUT="${OVRTX_SMOKE_CURL_CONNECT_TIMEOUT:-5}"
CURL_HEALTH_MAX_TIME="${OVRTX_SMOKE_CURL_HEALTH_MAX_TIME:-30}"
CURL_RENDER_MAX_TIME="${OVRTX_SMOKE_CURL_RENDER_MAX_TIME:-900}"

mkdir -p "${OUTPUT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERROR: python3 or python is required for smoke response validation" >&2
    exit 1
  fi
fi

cleanup() {
  local status=$?
  set +e
  if [[ -n "${LOG_PID}" ]]; then
    kill "${LOG_PID}" >/dev/null 2>&1
  fi
  docker logs "${CONTAINER_NAME}" > "${OUTPUT_DIR}/container-final.log" 2>&1
  if [[ "${KEEP_CONTAINER}" != "1" ]]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
  fi
  return "${status}"
}
trap cleanup EXIT

if [[ "${OVRTX_SMOKE_SKIP_BUILD:-0}" != "1" ]]; then
  docker build \
    -f "${REPO_ROOT}/apps/ovrtx_rendering_api/Dockerfile" \
    -t "${IMAGE_TAG}" \
    "${REPO_ROOT}"
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  -e OVRTX_LOG_LEVEL="${OVRTX_LOG_LEVEL:-warn}" \
  -e OVRTX_RENDER_MODE="${OVRTX_RENDER_MODE:-pt}" \
  -e OVRTX_NUM_SENSOR_UPDATES="${OVRTX_NUM_SENSOR_UPDATES:-1}" \
  -e OVRTX_DAEMON_START_TIMEOUT="${OVRTX_DAEMON_START_TIMEOUT:-900}" \
  -e OVRTX_DAEMON_RENDER_TIMEOUT="${OVRTX_DAEMON_RENDER_TIMEOUT:-900}" \
  -p "127.0.0.1:${HOST_PORT}:8000" \
  "${IMAGE_TAG}" > "${OUTPUT_DIR}/container-id.txt"

docker logs -f "${CONTAINER_NAME}" > "${OUTPUT_DIR}/container.log" 2>&1 &
LOG_PID="$!"

container_uid="$(docker exec "${CONTAINER_NAME}" id -u)"
container_gid="$(docker exec "${CONTAINER_NAME}" id -g)"
if [[ "${container_uid}:${container_gid}" != "10001:10001" ]]; then
  echo "ERROR: expected container user 10001:10001, got ${container_uid}:${container_gid}" >&2
  exit 1
fi

health_url="http://127.0.0.1:${HOST_PORT}/health"
health_file="${OUTPUT_DIR}/health.json"

for attempt in $(seq 1 "${OVRTX_SMOKE_HEALTH_ATTEMPTS:-90}"); do
  if curl -fsS \
    --connect-timeout "${CURL_CONNECT_TIMEOUT}" \
    --max-time "${CURL_HEALTH_MAX_TIME}" \
    "${health_url}" > "${health_file}"; then
    cat "${health_file}"
    echo
    if "${PYTHON_BIN}" - "${health_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    payload = json.load(f)
raise SystemExit(0 if payload.get("gpu_initialized") is True else 1)
PY
    then
      break
    fi
  fi
  if [[ "${attempt}" == "${OVRTX_SMOKE_HEALTH_ATTEMPTS:-90}" ]]; then
    echo "ERROR: OVRTX did not report gpu_initialized=true" >&2
    exit 1
  fi
  sleep 10
done

request_file="${OUTPUT_DIR}/render-request.json"
response_file="${OUTPUT_DIR}/render-response.json"
"${PYTHON_BIN}" - "${REPO_ROOT}/apps/ovrtx_rendering_api/tests/renders/smoke_cube.usda" > "${request_file}" <<'PY'
import base64
import json
import sys
from pathlib import Path

usd_bytes = Path(sys.argv[1]).read_bytes()
payload = {
    "url": "data:application/octet-stream;base64,"
    + base64.b64encode(usd_bytes).decode("ascii"),
    "render_settings": {
        "camera_paths": ["/World/Camera"],
        "frame_range": {"start": 0, "end": 0},
        "camera_parameters": {"width": 64, "height": 64},
        "num_sensor_updates": 1,
        "render_mode": "pt",
    },
}
print(json.dumps(payload))
PY

curl -fsS \
  --connect-timeout "${CURL_CONNECT_TIMEOUT}" \
  --max-time "${CURL_RENDER_MAX_TIME}" \
  -H "Content-Type: application/json" \
  --data-binary @"${request_file}" \
  "http://127.0.0.1:${HOST_PORT}/render" > "${response_file}"

"${PYTHON_BIN}" - "${response_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    payload = json.load(f)
if payload.get("status") != "success":
    raise SystemExit(f"render failed: {payload}")
try:
    image_b64 = payload["images"]["0"]["/World/Camera"]["images"]
except KeyError as exc:
    raise SystemExit(f"render response missing image payload: {payload}") from exc
if not image_b64:
    raise SystemExit("render response image payload is empty")
print("OVRTX non-root container smoke passed")
PY

kill "${LOG_PID}" >/dev/null 2>&1 || true
