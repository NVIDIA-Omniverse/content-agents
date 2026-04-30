---
name: texture-agent-client
description: Make requests to the Texture Agent REST API service for AI-driven texture generation on materialized USD assets. Use when user wants to use the Texture Agent client, call the texture agent service API, upload a materialized USD to the service, start a texture pipeline via REST, check pipeline status, download textures or textured output, or write a client script. Trigger phrases include "texture agent client", "texture agent service", "texture agent API", "call texture agent", "generate textures via service", "upload to texture service", "texture agent REST", "texture pipeline API".
---

# Texture Agent Client

The Texture Agent Service is a REST API (FastAPI) for AI-driven texture generation on materialized USD assets. It provides endpoints to upload a materialized USD file (typically the output of the Material Agent), run the texture pipeline (discover materials → generate textures → apply textures), monitor progress via SSE, and download generated textures plus the textured USDZ output.

## Prerequisites

- Service running at a known base URL (default: `http://localhost:8001`)
- At least one image-generation API key configured on the server (typically `NVIDIA_API_KEY` or `GOOGLE_API_KEY`)

## Quick Start with Python Client

The repo ships a Python client at `apps/texture_agent_service/client/client.py`.

```python
from apps.texture_agent_service.client.client import TextureAgentClient

client = TextureAgentClient("http://localhost:8001")

# Run pipeline and monitor progress
session_id, status = client.run_and_monitor(
    usd_path="materialized_scene.usd",
    material_textures={
        "Steel_Carbon": {"prompt": "rusted steel", "opacity": 0.85},
        "Wood_Oak":     {"prompt": "weathered oak planks", "opacity": 0.9},
    },
)

# Download artifacts
client.download_output(session_id, "output.usdz")     # self-contained USDZ
client.download_textures(session_id, "./textures/")   # generated texture maps
```

## Quick Start with curl

```bash
BASE_URL="http://localhost:8001"

# Upload materialized USD and start texture pipeline
SESSION=$(curl -s -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"rusted steel"}}' | jq -r .session_id)

# Poll status
curl -s "$BASE_URL/pipeline/$SESSION/status" | jq .status

# Stream events (SSE)
curl -N "$BASE_URL/pipeline/$SESSION/events"

# Download textured USDZ output
curl -o output.usdz "$BASE_URL/artifacts/$SESSION/output"

# Download all generated textures as a ZIP
curl -o textures.zip "$BASE_URL/artifacts/$SESSION/textures"
```

## Core Workflow

1. **Upload materialized USD** — `POST /pipeline/upload-usd` (two-step) or directly to `POST /pipeline` (one-step)
2. **Start pipeline** — `POST /pipeline` (multipart form-data: `usd_file` or `session_id` or `s3_uri`, plus optional `material_textures_json` prompt map)
3. **Monitor** — `GET /pipeline/{id}/events` (SSE) or `/status` (polling)
4. **Download results** — `GET /artifacts/{id}/output` (USDZ) and `/textures` (ZIP)

## API Endpoints

### Pipeline

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/upload-usd` | Upload USD file, create session |
| POST | `/pipeline` | Start texture pipeline |
| GET | `/pipeline/{id}/status` | Pipeline status with progress |
| GET | `/pipeline/{id}/results` | Final results + download URLs |
| GET | `/pipeline/{id}/events` | SSE progress stream |
| POST | `/pipeline/{id}/cancel` | Cancel running pipeline |
| POST | `/pipeline/{id}/regenerate` | Re-run specific steps |

### Artifacts (downloads)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/artifacts/{id}/materials` | Discovered materials JSON |
| GET | `/artifacts/{id}/textures` | All generated textures (ZIP) |
| GET | `/artifacts/{id}/textures/{name}` | Single texture file |
| GET | `/artifacts/{id}/output` | Textured output USDZ (self-contained) |
| GET | `/artifacts/{id}/renders` | Rendered preview images (ZIP) |

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/sessions` | List all sessions |
| GET | `/sessions/{id}` | Session details |
| DELETE | `/sessions/{id}` | Delete session |

### Utility

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |

## Key `POST /pipeline` Parameters

`POST /pipeline` is a **multipart form-data** endpoint. Exactly one of `usd_file`, `session_id`, or `s3_uri` must be provided.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `usd_file` | file | conditional | Materialized USD file (optional if `session_id` or `s3_uri` is provided). |
| `session_id` | string | conditional | Existing session from `/pipeline/upload-usd`. |
| `s3_uri` | string | conditional | S3 URI to a USD file (e.g. `s3://bucket/path/scene.usdz`). Service downloads server-side. |
| `material_textures_json` | string | no | **JSON-encoded string** of per-material texture config. Example: `{"Steel":{"prompt":"rusted steel","opacity":0.85}}`. Omit to auto-prompt every discovered material. |
| `user_prompt` | string | no | Aesthetic direction for auto-prompt generation (e.g. `"old and weathered"`). |

> ⚠️  The wire parameter is **`material_textures_json`** — a JSON-encoded string, not a JSON object. The Python client exposes it as a Python `dict` named `material_textures` and handles the serialization internally.

## Pipeline Status Values

`pending` → `running` → `completed` | `failed` | `cancelled`

## Material Textures Map (shape)

The value of `material_textures_json` (once decoded by the server) is a JSON object keyed by material name (as discovered by the pipeline). Each value is an object with:

- `prompt` (string, required): text prompt describing the desired texture
- `opacity` (float, optional, default ~0.85): blend opacity for compositing

Use `GET /artifacts/{id}/materials` (from a prior run, or `GET /pipeline/{id}/results`) to discover the available material names first.

## OpenAPI Specification

See `apps/texture_agent_service/openapi.yaml` for the full OpenAPI 3.1 spec.
Swagger UI lives at `GET /docs` once the service is running.

## Common Issues

### Connection refused
Cause: Service not running.
Solution: Start via docker compose (`docker compose -f apps/texture_agent_service/docker-compose.yml up --build`) or `texture-agent-service` CLI, or `uvicorn service.main:app --port 8001` from `apps/texture_agent_service/`.

### 413 File too large
Cause: USD file exceeds upload limit (default 500 MB).
Solution: Set `TA_MAX_UPLOAD_SIZE_MB` env var on the server.

### 202 when fetching results
Cause: Pipeline still running.
Solution: Wait for completion. Use SSE events or poll `/status` until `status` is `completed`.

### No textures generated for a material
Cause: the submitted map (`material_textures_json` over REST, or the `material_textures` kwarg via the Python client) did not include that material name, or the key didn't match what the pipeline discovered.
Solution: First run the pipeline without any per-material map (or check `/artifacts/{id}/materials`) to see the exact material names, then resubmit with matching keys.
