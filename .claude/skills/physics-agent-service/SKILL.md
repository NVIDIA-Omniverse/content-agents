---
name: physics-agent-service
description: Make requests to the Physics Agent REST API service for VLM-based physics/component classification of 3D USD assets. Use when user wants to call the physics agent API, upload a USD file to the service, start a classification pipeline via REST, check pipeline status, download predictions, or write a client script. Trigger phrases include "physics agent service", "physics agent API", "call physics agent", "classify asset via service", "upload USD to physics service", "physics agent REST", "physics pipeline API".
---

# Physics Agent Service API

The Physics Agent Service is a REST API (FastAPI) for VLM-based physics / component classification of 3D USD assets. It provides endpoints to upload USD files, run the classification pipeline, monitor progress via SSE, and download predictions.

## Prerequisites

- Service running at a known base URL (default: `http://localhost:8000`)
- Optional: Bearer token for authentication (set `PHYSICS_AGENT_TOKEN` env var or pass in `Authorization: Bearer` header)

## Quick Start with Python Client

The repo ships a Python client at `apps/physics_agent_service/client/client.py`. Supports both local file upload and S3 URI input modes.

```python
from apps.physics_agent_service.client.client import PhysicsAgentClient

client = PhysicsAgentClient(base_url="http://localhost:8000")

# Local file upload
session_id, status = client.run_and_monitor(
    usd_path="/path/to/scene.usdz",
    user_prompt="Focus on identifying furniture parts",
    render_backend="remote",  # or "warp", "ovrtx"
)

# OR S3 URI (service downloads server-side, better for large files)
session_id, status = client.run_and_monitor(
    s3_uri="s3://your-bucket/path/to/scene.usdz",
)

print(session_id, status)
```

See `apps/physics_agent_service/client/README.md` for the full client guide.

## Quick Start with curl

```bash
BASE_URL="http://localhost:8000"

# Upload USD and start pipeline
SESSION=$(curl -s -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" | jq -r .session_id)

# Poll status
curl -s "$BASE_URL/pipeline/$SESSION/status" | jq .status

# Stream events (SSE)
curl -N "$BASE_URL/pipeline/$SESSION/events"

# Download predictions
curl -o predictions.jsonl "$BASE_URL/artifacts/$SESSION/predictions"

# Download HTML report
curl -o report.html "$BASE_URL/artifacts/$SESSION/report"

# Download simulation-ready USD (physics schemas applied)
curl -o scene_physics.usda "$BASE_URL/artifacts/$SESSION/output-usd"
```

## Core Workflow

1. **Upload USD** — `POST /pipeline/upload-usd` (two-step) or directly to `POST /pipeline` (one-step)
2. **Start pipeline** — `POST /pipeline` (with `session_id` or `usd_file` / `s3_uri`)
3. **Monitor** — `GET /pipeline/{session_id}/events` (SSE) or `/status` (polling)
4. **Download results** — `GET /artifacts/{session_id}/predictions`, `/report`, `/dataset`, `/output-usd`

## API Endpoints

### Pipeline

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/upload-usd` | Upload USD file, create session (201) |
| POST | `/pipeline` | Start pipeline (202) |
| GET | `/pipeline/{id}/status` | Get execution status |
| GET | `/pipeline/{id}/results` | Get results (only when completed) |
| GET | `/pipeline/{id}/events` | Stream progress via SSE |
| POST | `/pipeline/{id}/cancel` | Cancel running pipeline |
| POST | `/pipeline/{id}/regenerate` | Re-run specific steps |

### Artifacts (downloads)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/artifacts/{id}/predictions` | Download predictions JSONL |
| GET | `/artifacts/{id}/report` | View prediction HTML report |
| GET | `/artifacts/{id}/dataset` | Download dataset JSONL |
| GET | `/artifacts/{id}/output-usd` | Download simulation-ready USD (`scene_physics.usda`) with `UsdPhysics` schemas applied |

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/sessions` | List all sessions |
| GET | `/sessions/{id}` | Get session details |
| DELETE | `/sessions/{id}` | Delete session and artifacts |

### Utility

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |

## Key `POST /pipeline` Parameters

The main pipeline endpoint accepts multipart form-data:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `usd_file` | conditional | | USD file (unless `session_id` or `s3_uri` provided) |
| `session_id` | conditional | | Existing session from `/pipeline/upload-usd` |
| `s3_uri` | conditional | | S3 URI for server-side download |
| `user_prompt` | No | | Custom VLM prompt |
| `render_backend` | No | `remote` | One of `remote`, `warp`, `ovrtx` |

Exactly one of `usd_file`, `session_id`, or `s3_uri` must be provided.

## Pipeline Status Values

`pending` → `running` → `completed` | `failed` | `cancelled`

## Predictions Format

`predictions.jsonl` — one JSON object per line. Each entry has:

- `id` — prim path
- `classification` — `{material, component_type, physical_properties}`
- `reasoning` — VLM's explanation of the classification

## OpenAPI Specification

See `apps/physics_agent_service/openapi.yaml` for the full OpenAPI 3.1 spec.
Swagger UI lives at `GET /docs` once the service is running.

## Common Issues

### Connection refused
Cause: Service not running.
Solution: Start via docker compose (`docker compose -f apps/physics_agent_service/docker-compose.yml up --build`) or uvicorn (`uvicorn service.main:app --port 8000` from `apps/physics_agent_service/`).

### 413 File too large
Cause: USD file exceeds upload limit (default 500 MB).
Solution: Set `PA_MAX_UPLOAD_SIZE_MB` env var on the server, or use S3 URI input mode for large files.

### 202 when fetching results
Cause: Pipeline still running.
Solution: Wait for completion. Use SSE events or poll `/status` until `status` is `completed`.

### OVRTX sidecar cold start
Cause: Bundled rendering container needs ~5 min to warm up on the GPU on first start.
Solution: Wait. `GET /health` on port 8001 reports `gpu_initialized=false` during warm-up.
