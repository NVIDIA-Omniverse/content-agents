---
name: material-agent-client
description: Make requests to the Material Agent REST API service for VLM-based material assignment. Use when user wants to use the Material Agent client, call the material agent service API, upload a USD file to the service, start a pipeline via REST, check pipeline status, download results, or write a client script. Trigger phrases include "material agent client", "material agent service", "material agent API", "call material agent", "upload USD to service", "start pipeline API", "check pipeline status", "material agent REST".
---

# Material Agent Service API

The Material Agent Service is a REST API (FastAPI) for VLM-based material assignment to 3D USD objects. It provides endpoints to upload USD files, run the material assignment pipeline, monitor progress via SSE, and download results.

## Prerequisites

- Service running at a known base URL (default: `http://localhost:8000`)
- Optional: Bearer token for authentication (set `MATERIAL_AGENT_TOKEN` env var or pass in header)
- Optional generated reference images use server-side `MA_IMAGE_GEN_*` settings. The public default is `MA_IMAGE_GEN_BACKEND=gemini`, which requires `GOOGLE_API_KEY`.
- For one local OVRTX rendering service instance, pass `render_num_workers=1` (or client CLI `--render-num-workers 1`) so the service sets both `num_workers` and `max_concurrent_requests` to `1`. Increase it only when the deployment has multiple independent rendering service instances behind the endpoint.

## Quick Start with Python Client

The service has a Python client class. Copy the client from references/client.py into your project, then use it:

```python
from client import MaterialAgentClient

client = MaterialAgentClient(base_url="http://localhost:8000")

# Run pipeline and monitor progress
session_id, status = client.run_and_monitor(
    usd_path="scene.usd",
    reference_images=["reference.jpg"],
    render_num_workers=1,
    user_email="user@example.com",
)

# Download results
results = client.get_results(session_id)
print(results["download_urls"])
```

See references/client.py for the full client source.

To generate an AI reference image before the pipeline starts, pass
`generated_reference_prompt`. The client uploads the USD, waits for the input
preview, calls `generate-reference-image`, then starts the pipeline with the
returned `reference_id`.

```python
session_id, status = client.run_and_monitor(
    usd_path="scene.usd",
    generated_reference_prompt="Satin red painted metal with black rubber tires",
    user_email="user@example.com",
)
```

## Quick Start with curl

```bash
BASE_URL="http://localhost:8000"

# Upload USD and start pipeline
SESSION=$(curl -s -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "user_email=user@example.com" \
  -F "render_num_workers=1" \
  -F "reference_images=@reference.jpg" | jq -r .session_id)

# Poll status
curl -s "$BASE_URL/pipeline/$SESSION/status" | jq .status

# Stream events (SSE)
curl -N "$BASE_URL/pipeline/$SESSION/events"

# Download output USD
curl -o output.usd "$BASE_URL/artifacts/$SESSION/output"

# Download predictions
curl -o predictions.jsonl "$BASE_URL/artifacts/$SESSION/predictions"
```

Generate an AI reference image from the input preview:

```bash
SESSION=$(curl -s -X POST "$BASE_URL/pipeline/upload-usd" \
  -F "usd_file=@scene.usd" | jq -r .session_id)

# Wait until this returns 200.
curl -I "$BASE_URL/assets/$SESSION/input-render"

REF_ID=$(curl -s -X POST "$BASE_URL/pipeline/$SESSION/generate-reference-image" \
  -F "prompt=Satin red painted metal with black rubber tires" | jq -r .reference_id)

curl -s -X POST "$BASE_URL/pipeline" \
  -F "session_id=$SESSION" \
  -F "generated_reference_id=$REF_ID" \
  -F "user_email=user@example.com" | jq .
```

## Core Workflow

1. **Upload USD** -- `POST /pipeline/upload-usd` or directly to `POST /pipeline`
2. **Optional generated reference** -- wait for `/assets/{session_id}/input-render`, then `POST /pipeline/{session_id}/generate-reference-image`
3. **Start pipeline** -- `POST /pipeline` (with session_id or usd_file, plus `generated_reference_id` if using a generated reference)
4. **Monitor** -- `GET /pipeline/{session_id}/events` (SSE) or `/status` (polling)
5. **Download results** -- `GET /artifacts/{session_id}/output`, `/predictions`, `/report`

## API Endpoints

### Pipeline

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/upload-usd` | Upload USD file, create session (201) |
| POST | `/pipeline` | Start pipeline (202) |
| POST | `/pipeline/{id}/generate-reference-image` | Generate AI reference image from input preview |
| DELETE | `/pipeline/{id}/generated-reference-image/{ref_id}` | Delete generated reference before pipeline start |
| GET | `/pipeline/{id}/status` | Get execution status |
| GET | `/pipeline/{id}/results` | Get results (only when completed) |
| GET | `/pipeline/{id}/events` | Stream progress via SSE |
| POST | `/pipeline/{id}/cancel` | Cancel running pipeline |

### Artifacts (downloads)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/artifacts/{id}/output` | Download output USD with materials |
| GET | `/artifacts/{id}/final-render` | Download final render PNG |
| GET | `/artifacts/{id}/predictions` | Download predictions JSONL |
| GET | `/artifacts/{id}/report` | View prediction HTML report |
| GET | `/artifacts/{id}/optimization-report` | View optimization report JSON |

### Assets (images/previews)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/assets/{id}/input-render` | Input USD preview (before materials) |
| GET | `/assets/{id}/generated-ref/{ref_id}` | AI-generated reference image |
| GET | `/assets/{id}/previews` | List preview images |
| GET | `/assets/{id}/preview/{name}` | Get a preview image |
| GET | `/assets/{id}/references` | List reference images |
| GET | `/assets/{id}/reference/{name}` | Get a reference image |
| GET | `/assets/{id}/reference-pdfs` | List reference PDFs |
| GET | `/assets/{id}/reference-pdf/{name}` | Get a reference PDF |

### Sessions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/sessions` | List all sessions |
| GET | `/sessions/{id}` | Get session details |
| DELETE | `/sessions/{id}` | Delete session and artifacts |
| GET | `/sessions/usage` | Usage stats (optional: from_date, to_date, user_email) |
| POST | `/sessions/admin/cleanup` | Trigger manual cleanup |

### Materials

| Method | Path | Description |
|--------|------|-------------|
| GET | `/materials` | List available materials |
| GET | `/materials/icon/{name}` | Get material icon PNG |
| GET | `/materials/template` | Download default materials ZIP |

### Utility

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/config/vlm-models` | List available VLM models |

## Key `POST /pipeline` Parameters

The main pipeline endpoint accepts multipart form-data:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `usd_file` | Yes* | | USD file (unless session_id provided) |
| `session_id` | Yes* | | Existing session from /upload-usd |
| `user_email` | Yes | | User email for tracking |
| `reference_images` | No | | Reference images (multiple) |
| `reference_pdfs` | No | | Reference PDFs (multiple) |
| `materials_zip` | No | | ZIP with custom materials |
| `user_prompt` | No | | Custom VLM prompt |
| `camera_views` | No | +x+y+z,-x-y-z | Camera view directions |
| `vlm_model` | No | | VLM model override |
| `optimize_usd` | No | true | Enable USD optimization |
| `vlm_max_workers` | No | 64 | Max parallel VLM workers |
| `render_num_workers` | No | Material Agent default | Max parallel render workers for `build_dataset_usd`; use `1` for a single rendering service instance |

*Either `usd_file` or `session_id` must be provided.

## Pipeline Status Values

`pending` -> `running` -> `completed` | `failed` | `cancelled`

## OpenAPI Specification

See `apps/material_agent_service/openapi.yaml` for the full OpenAPI 3.1 spec.

## Common Issues

### Connection refused
Cause: Service not running.
Solution: Start the service (`uvicorn service.main:app --port 8000` from `apps/material_agent_service/`).

### 413 File too large
Cause: USD file exceeds upload limit (default 500MB).
Solution: Set `MA_MAX_UPLOAD_SIZE_MB` env var on the server.

### 202 when fetching results
Cause: Pipeline still running.
Solution: Wait for completion. Use SSE events or poll `/status` until `status` is `completed`.

### Custom materials
Upload a ZIP file with `materials.yaml` and the USD library file via the `materials_zip` form parameter.
