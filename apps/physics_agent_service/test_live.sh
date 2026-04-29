#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Live smoke test for Physics Agent Service
# Usage: ./test_live.sh [BASE_URL] [RENDER_BACKEND]
#
# Requires the service running at BASE_URL (default: http://localhost:8000)
#
# Rendering backend (default: remote):
#   remote - HTTP renderer backend (the bundled compose points this at OVRTX)
#   warp   - In-process CUDA GPU raytracer (fast, no Vulkan needed)
#   ovrtx  - Local RTX path tracer (PBR quality, requires Vulkan)
#
# Examples:
#   ./test_live.sh                                # default URL + remote
#   ./test_live.sh http://localhost:8000 remote   # bundled OVRTX sidecar
#   ./test_live.sh http://localhost:8000 ovrtx  # local RTX

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
RENDER_BACKEND="${2:-remote}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$REPO_ROOT/apps/physics_agent/configs/lightbulb.yaml"
USD="$REPO_ROOT/apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda"
MAX_WAIT=300
POLL_INTERVAL=5

readonly SEPARATOR="========================================"

echo "$SEPARATOR"
echo "Physics Agent Service - Live Smoke Test"
echo "$SEPARATOR"
echo "Base URL:         $BASE_URL"
echo "Render backend:   $RENDER_BACKEND"
echo

# 1. Health check
echo "[1/6] Health check..."
HEALTH=$(curl -sf "$BASE_URL/health")
echo "  $HEALTH" | python3 -m json.tool
echo

# 2. Start pipeline
echo "[2/6] Starting pipeline..."
echo "  Config: $CONFIG"
echo "  USD:    $USD"
RESULT=$(curl -sf -X POST "$BASE_URL/pipeline" \
  -F "config_file=@$CONFIG" \
  -F "usd_file=@$USD" \
  -F "render_backend=$RENDER_BACKEND")
SESSION_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "  Session ID: $SESSION_ID"
echo

# 3. Poll until completion
echo "[3/6] Polling status..."
ELAPSED=0
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
  STATUS_JSON=$(curl -sf "$BASE_URL/pipeline/$SESSION_ID/status")
  STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  PERCENT=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['overall_progress']['percent'])")

  echo "  [${ELAPSED}s] status=$STATUS progress=$PERCENT%"

  if [[ "$STATUS" = "completed" ]]; then
    echo "  Pipeline completed!"
    break
  elif [[ "$STATUS" = "failed" || "$STATUS" = "cancelled" ]]; then
    echo "  Pipeline $STATUS!"
    echo "$STATUS_JSON" | python3 -m json.tool
    exit 1
  fi

  sleep $POLL_INTERVAL
  ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
  echo "  Timed out after ${MAX_WAIT}s"
  exit 1
fi
echo

# 4. Check results
echo "[4/6] Fetching results..."
curl -sf "$BASE_URL/pipeline/$SESSION_ID/results" | python3 -m json.tool
echo

# 5. Check artifacts
echo "[5/6] Checking artifacts..."
PRED_STATUS=$(curl -so /dev/null -w "%{http_code}" "$BASE_URL/artifacts/$SESSION_ID/predictions")
DS_STATUS=$(curl -so /dev/null -w "%{http_code}" "$BASE_URL/artifacts/$SESSION_ID/dataset")
echo "  Predictions: $PRED_STATUS"
echo "  Dataset:     $DS_STATUS"
echo

# 6. List sessions
echo "[6/6] Listing sessions..."
curl -sf "$BASE_URL/sessions" | python3 -m json.tool
echo

echo "$SEPARATOR"
echo "SMOKE TEST PASSED"
echo "$SEPARATOR"
echo
echo "Artifacts:"
echo "  Predictions: $BASE_URL/artifacts/$SESSION_ID/predictions"
echo "  Report:      $BASE_URL/artifacts/$SESSION_ID/report"
echo "  Dataset:     $BASE_URL/artifacts/$SESSION_ID/dataset"
