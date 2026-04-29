# Physics Agent Service API Reference

REST API for VLM-based asset classification of USD files. The service accepts USD scene uploads, runs an async pipeline (optimize, render, build dataset, predict), and streams real-time progress via SSE.

When launched with the bundled Docker Compose stack, the CPU-only main service
waits for the `ovrtx-rendering-api` sidecar to finish GPU warm-up. Expect cold
startup to take roughly 5 minutes before `GET /health` on port `8000` becomes
reachable.

**Base URL:** `http://localhost:8000`
**Interactive docs:** `GET /docs` (Swagger UI)

---

## Table of Contents

- [Authentication](#authentication)
- [Root Endpoints](#root-endpoints)
- [Pipeline](#pipeline)
- [Artifacts](#artifacts)
- [Sessions](#sessions)
- [Server-Sent Events (SSE)](#server-sent-events-sse)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Configuration](#configuration)

---

## Authentication

No authentication is required. The service accepts all origins via permissive CORS.

Optional: set `PHYSICS_AGENT_TOKEN` and pass it as `Authorization: Bearer <token>` from clients. The service does not currently enforce this.

---

## Root Endpoints

### `GET /`

Redirects to `/api`.

### `GET /api`

Returns service info and a map of all available endpoints.

**Response** `200`
```json
{
  "service": "Physics Agent Service",
  "version": "0.2.23",
  "docs": "/docs",
  "health": "/health",
  "api": {
    "pipeline": {
      "create": "POST /pipeline",
      "status": "GET /pipeline/{session_id}/status",
      "results": "GET /pipeline/{session_id}/results",
      "cancel": "POST /pipeline/{session_id}/cancel",
      "events": "GET /pipeline/{session_id}/events",
      "regenerate": "POST /pipeline/{session_id}/regenerate"
    },
    "artifacts": {
      "predictions": "GET /artifacts/{session_id}/predictions",
      "report": "GET /artifacts/{session_id}/report",
      "dataset": "GET /artifacts/{session_id}/dataset"
    },
    "sessions": {
      "list": "GET /sessions",
      "get": "GET /sessions/{session_id}",
      "delete": "DELETE /sessions/{session_id}"
    }
  }
}
```

### `GET /health`

Health check.

**Response** `200`
```json
{
  "status": "healthy",
  "service": "Physics Agent Service",
  "version": "0.2.23",
  "api_keys_configured": true,
  "max_active_sessions": 8
}
```

---

## Pipeline

### Upload USD

```
POST /pipeline/upload-usd
```

Upload a USD file and create a session without starting the pipeline. Use the returned `session_id` with `POST /pipeline` to start processing later.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `usd_file` | file | yes | USD file (`.usd`, `.usda`, `.usdc`, `.usdz`) |

**Response** `201`
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "ready",
  "message": "USD uploaded successfully",
  "estimated_duration_minutes": 0
}
```

**Errors:**
- `400` Invalid file extension
- `413` File exceeds `PA_MAX_UPLOAD_SIZE_MB` (default 500 MB)

---

### Create Pipeline

```
POST /pipeline
```

Create and execute an asset classification pipeline. Supports three modes:
1. **New upload:** provide `usd_file` to create a new session and start processing.
2. **Existing session:** provide `session_id` (from `/pipeline/upload-usd`) to start processing a previously uploaded file.
3. **S3 source:** provide `s3_uri` to let the service download the USD and start processing.

Pipeline execution is async -- the endpoint returns `202` immediately.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `usd_file` | file | conditional | USD file. Required if `session_id` is not provided. |
| `session_id` | string | conditional | Existing session ID. Required if `usd_file` is not provided. |
| `s3_uri` | string | conditional | S3 URI to a USD file. Required if neither `usd_file` nor `session_id` is provided. |
| `user_prompt` | string | no | Custom prompt for the VLM prediction step. |
| `render_backend` | string | no | Rendering backend: `remote` (default, HTTP render service; the bundled compose points this at the OVRTX sidecar), `warp` (local CUDA), or `ovrtx` (local Vulkan subprocess). |
| `optimize_usd` | boolean | no | Enable Scene Optimizer before rendering and prediction. Default: `false`. |
| `enable_deinstance` | boolean | no | Enable deinstance when `optimize_usd=true`. Default: `true`; required for instanced assets that use shared prototypes. |
| `enable_split` | boolean | no | Enable split meshes when `optimize_usd=true`. Default: `false`. |
| `enable_deduplicate` | boolean | no | Enable deduplicate when `optimize_usd=true`. Default: `false`. |

FastAPI accepts common boolean form values for the optimizer flags, including
`true`/`false`, `1`/`0`, `yes`/`no`, and `on`/`off`.

**Response** `202`
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Pipeline queued for execution",
  "estimated_duration_minutes": 15
}
```

**Errors:**
- `400` Neither `usd_file`, `session_id`, nor `s3_uri` provided; invalid file extension; input USD not found for session; `optimize_usd=true` with all optimizer operation flags disabled
- `404` Session not found (when using `session_id`)
- `413` File too large

---

### Get Pipeline Status

```
GET /pipeline/{session_id}/status
```

Real-time pipeline status with step-level progress. Reads from the in-memory event bus for active sessions, falls back to disk for completed ones.

**Response** `200` -- [PipelineStatus](#pipelinestatus)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "running",
  "current_step": {
    "name": "predict",
    "display_name": "Predicting materials",
    "started_at": "2026-02-24T10:05:00Z",
    "progress": {
      "current": 47,
      "total": 95,
      "percent": 49,
      "message": "Predicted /World/Part_47"
    },
    "elapsed_seconds": 120
  },
  "completed_steps": [
    {
      "name": "optimize_usd",
      "display_name": "Optimizing USD",
      "started_at": "2026-02-24T10:00:00Z",
      "completed_at": "2026-02-24T10:01:30Z",
      "duration_seconds": 90,
      "stats": {}
    }
  ],
  "overall_progress": {
    "current_step": 3,
    "total_steps": 4,
    "percent": 62,
    "estimated_remaining_seconds": 180
  },
  "preview_images": ["/artifacts/a1b2c3d4-.../preview/abc123.png"],
  "can_cancel": true,
  "elapsed_seconds": 300,
  "created_at": "2026-02-24T10:00:00Z",
  "updated_at": "2026-02-24T10:05:00Z"
}
```

**Errors:**
- `404` Session not found

---

### Get Pipeline Results

```
GET /pipeline/{session_id}/results
```

Returns final results when the pipeline has completed, or error details if it failed.

**Response** `200` (completed) -- [PipelineResults](#pipelineresults)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "completed",
  "stats": {
    "prims_processed": 142,
    "images_generated": 284,
    "predictions_made": 142
  },
  "download_urls": {
    "predictions": "/artifacts/a1b2c3d4-.../predictions",
    "report": "/artifacts/a1b2c3d4-.../report",
    "dataset": "/artifacts/a1b2c3d4-.../dataset"
  },
  "duration_seconds": 600,
  "completed_at": "2026-02-24T10:10:00Z"
}
```

**Response** `200` (failed) -- [PipelineError](#pipelineerror)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "failed",
  "error_message": "VLM inference timeout",
  "failed_step": "predict",
  "completed_steps": ["optimize_usd", "build_dataset_usd"],
  "partial_results": null
}
```

**Errors:**
- `202` Pipeline still running (check `/status` for progress)
- `404` Session not found

---

### Cancel Pipeline

```
POST /pipeline/{session_id}/cancel
```

Cancel a running or pending pipeline.

**Response** `200`
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "cancelling",
  "message": "Pipeline cancellation requested"
}
```

**Errors:**
- `400` Pipeline already completed/failed/cancelled
- `404` Session not found
- `500` Session not in job registry

---

### Stream Events (SSE)

```
GET /pipeline/{session_id}/events
```

Server-Sent Events stream for real-time progress. See [SSE section](#server-sent-events-sse) for details.

**Errors:**
- `404` Session not found

---

### Regenerate Pipeline

```
POST /pipeline/{session_id}/regenerate
```

Re-run specific pipeline steps using cached data from a previous run. Useful for re-running the `predict` step with a different prompt without re-rendering.

**Request:** `application/json` -- [RegenerateRequest](#regeneraterequest)
```json
{
  "steps": ["predict"],
  "user_prompt": "Classify each material as metal, plastic, or fabric"
}
```

**Response** `202` -- [SessionCreated](#sessioncreated)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Regenerating steps: predict"
}
```

**Errors:**
- `400` Pipeline still running/pending/cancelling; original config not found
- `404` Session not found

---

### Get Event Log

```
GET /pipeline/{session_id}/event-log
```

Get the full persisted event history for a session. Useful for replaying progress after a session completes.

**Response** `200`
```json
{
  "events": [
    {
      "session_id": "a1b2c3d4-...",
      "step": "optimize_usd",
      "state": "running",
      "percent": 0,
      "message": "Starting USD optimization",
      "timestamp": "2026-02-24T10:00:00Z"
    }
  ],
  "total": 42
}
```

**Errors:**
- `404` Session not found

---

## Artifacts

### Download Predictions

```
GET /artifacts/{session_id}/predictions
```

Download the predictions file (JSONL).

**Response** `200`
Content-Type: `application/x-ndjson`
Filename: `predictions.jsonl`

**Errors:**
- `404` Session or predictions not found

---

### View Report

```
GET /artifacts/{session_id}/report
```

View the HTML prediction report in the browser. The report is generated on-demand if it doesn't already exist.

**Response** `200`
Content-Type: `text/html`

**Errors:**
- `404` Session not found; predictions not available yet; dataset not available
- `500` Report generation failed

---

### Download Dataset

```
GET /artifacts/{session_id}/dataset
```

Download the dataset file (JSONL).

**Response** `200`
Content-Type: `application/x-ndjson`
Filename: `dataset.jsonl`

**Errors:**
- `404` Session or dataset not found

---

## Sessions

### List Sessions

```
GET /sessions
```

List all sessions sorted by creation time (newest first).

**Response** `200`
```json
{
  "sessions": [
    {
      "session_id": "a1b2c3d4-...",
      "status": "completed",
      "created_at": "2026-02-24T10:00:00Z",
      "updated_at": "2026-02-24T10:10:00Z",
      "elapsed_seconds": 600,
      "config": {
        "project_name": "my_scene",
        "usd_path": "/var/physics-agent/sessions/a1b2c3d4-.../input/scene.usd",
        "has_usd_upload": true,
        "user_prompt": null
      }
    }
  ],
  "total": 1
}
```

---

### Get Session

```
GET /sessions/{session_id}
```

Get full session metadata.

**Response** `200` -- Full `session.json` contents (see [Session Metadata](#session-metadata)).

**Errors:**
- `404` Session not found

---

### Delete Session

```
DELETE /sessions/{session_id}
```

Delete a session and all its artifacts. Cancels any running pipeline first.

**Response** `204` No Content

**Errors:**
- `404` Session not found
- `500` Deletion failed after retries

---

## Server-Sent Events (SSE)

Connect to `GET /pipeline/{session_id}/events` to receive real-time progress updates.

### Event Types

| Event | Description |
|-------|-------------|
| `progress` | Pipeline step progress update ([ProgressEvent](#progressevent)) |
| `ping` | Keepalive sent every 30 seconds |
| `done` | Pipeline completed, failed, or cancelled -- stream closes after this |

### JavaScript Example

```javascript
const events = new EventSource("/pipeline/a1b2c3d4-.../events");

events.addEventListener("progress", (e) => {
  const data = JSON.parse(e.data);
  console.log(`[${data.step}] ${data.state} ${data.percent}%: ${data.message}`);
});

events.addEventListener("done", (e) => {
  const data = JSON.parse(e.data);
  console.log(`Pipeline ${data.final_state}`);
  events.close();
});
```

### ProgressEvent Payload

```json
{
  "session_id": "a1b2c3d4-...",
  "step": "predict",
  "state": "running",
  "current": 47,
  "total": 95,
  "percent": 49,
  "message": "Predicted /World/Part_47",
  "timestamp": "2026-02-24T10:05:00Z",
  "extra": { "prim_id": "/World/Part_47" },
  "overall_percent": 62
}
```

---

## Data Models

### SessionCreated

Returned when a session is created or a pipeline is queued.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | UUID session identifier |
| `status` | string | `"pending"`, `"ready"` |
| `message` | string | Human-readable message |
| `estimated_duration_minutes` | int or null | Rough time estimate |

### PipelineStatus

Detailed pipeline execution status.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | `pending`, `running`, `completed`, `failed`, `cancelled`, `cancelling` |
| `current_step` | [CurrentStepInfo](#currentstepinfo) or null | Currently executing step |
| `completed_steps` | [CompletedStepInfo](#completedstepinfo)[] | Steps that have finished |
| `overall_progress` | [OverallProgress](#overallprogress) | Aggregate progress |
| `preview_images` | string[] | URLs to rendered preview thumbnails |
| `can_cancel` | bool | Whether the pipeline can be cancelled |
| `elapsed_seconds` | int | Total elapsed time |
| `created_at` | string | ISO 8601 timestamp |
| `updated_at` | string | ISO 8601 timestamp |

### CurrentStepInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Internal step name |
| `display_name` | string | Human-readable name |
| `started_at` | string | ISO 8601 timestamp |
| `progress` | [StepProgress](#stepprogress) | Step-level progress |
| `elapsed_seconds` | int | Time since step started |

### CompletedStepInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Internal step name |
| `display_name` | string | Human-readable name |
| `started_at` | string | ISO 8601 timestamp |
| `completed_at` | string | ISO 8601 timestamp |
| `duration_seconds` | int | Step duration |
| `stats` | object | Step-specific statistics |

### StepProgress

| Field | Type | Description |
|-------|------|-------------|
| `current` | int | Items processed so far |
| `total` | int | Total items to process |
| `percent` | int | Percentage complete (0-100) |
| `message` | string | Human-readable progress message |

### OverallProgress

| Field | Type | Description |
|-------|------|-------------|
| `current_step` | int | Current step number (1-indexed) |
| `total_steps` | int | Total pipeline steps |
| `percent` | int | Overall percentage (0-100) |
| `estimated_remaining_seconds` | int or null | Estimated time remaining |

### PipelineResults

Returned when the pipeline completes successfully.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | `"completed"` |
| `stats` | object | `{prims_processed, images_generated, predictions_made}` |
| `download_urls` | object | `{predictions, report, dataset}` -- relative URL paths |
| `duration_seconds` | int | Total pipeline duration |
| `completed_at` | string | ISO 8601 timestamp |

### PipelineError

Returned when the pipeline fails.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | `"failed"` |
| `error_message` | string | Error description |
| `failed_step` | string | Step that failed |
| `completed_steps` | string[] | Steps completed before failure |
| `partial_results` | object or null | Any partial results available |

### RegenerateRequest

Request body for regenerating specific pipeline steps.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `steps` | [PipelineStep](#pipelinestep)[] | yes | Steps to re-run |
| `user_prompt` | string or null | no | Override prompt for the prediction step |

### PipelineStep

Enum of available pipeline steps:

| Value | Description |
|-------|-------------|
| `optimize_usd` | Optimize USD scene structure |
| `build_dataset_usd` | Render images from USD prims |
| `build_dataset_prepare_dataset` | Prepare dataset with prompts |
| `predict` | Run VLM predictions |

### ProgressEvent

SSE event payload for real-time progress.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `step` | string | Step name |
| `state` | string | `queued`, `running`, `completed`, `failed`, `cancelled` |
| `current` | int or null | Items processed |
| `total` | int or null | Total items |
| `percent` | int or null | Step percentage (0-100) |
| `message` | string or null | Progress message |
| `timestamp` | string | ISO 8601 timestamp |
| `extra` | object or null | Step-specific data |
| `overall_percent` | int or null | Pipeline-wide percentage (0-100) |

### Session Metadata

Full session metadata stored on disk (`session.json`).

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | UUID identifier |
| `created_at` | string | ISO 8601 creation time |
| `updated_at` | string | ISO 8601 last update time |
| `status` | string | `pending`, `running`, `completed`, `failed`, `cancelled` |
| `current_step` | object or null | Current step info |
| `completed_steps` | object[] | Completed steps |
| `overall_progress` | object | Aggregate progress |
| `preview_images` | string[] | Preview image filenames |
| `can_cancel` | bool | Cancellation availability |
| `elapsed_seconds` | int | Elapsed time |
| `config` | object | `{project_name, usd_path, has_usd_upload, user_prompt}` |
| `ttl_expires_at` | string | Expiration timestamp |
| `results` | object | Final stats |
| `duration_seconds` | int | Total duration |
| `completed_at` | string | Completion timestamp |
| `timings` | object | Per-step durations |

---

## Error Handling

All errors return JSON with a `detail` field:

```json
{
  "detail": "Session not found"
}
```

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `201` | Resource created (upload-usd) |
| `202` | Accepted -- pipeline queued or still running |
| `204` | Deleted successfully (no body) |
| `400` | Bad request (missing params, invalid state) |
| `404` | Session or artifact not found |
| `413` | File too large |
| `500` | Internal server error |

---

## Configuration

All settings use the `PA_` environment variable prefix.

### Service Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PA_SESSION_STORAGE_PATH` | `/var/physics-agent/sessions` | Session storage directory (falls back to `./sessions` in dev) |
| `PA_SESSION_TTL_HOURS` | `24` | Hours before sessions are auto-cleaned |
| `PA_MAX_ACTIVE_SESSIONS` | `1` | Max concurrent pipeline executions (semaphore) |
| `PA_MAX_UPLOAD_SIZE_MB` | `500` | Max upload file size in MB |

### VLM/LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PA_VLM_BACKEND` | `nim` | VLM inference backend |
| `PA_VLM_MODEL` | `qwen/qwen3.5-397b-a17b` | VLM model identifier |
| `PA_VLM_TEMPERATURE` | `1.0` | VLM sampling temperature |
| `PA_RENDER_BACKEND` | `remote` | Rendering backend: `remote`, `warp`, or `ovrtx` |
| `RENDER_ENDPOINT` | `http://ovrtx-rendering-api:8000` in bundled Docker Compose | Base URL for the remote HTTP renderer used when `PA_RENDER_BACKEND=remote` |

### API Keys

| Variable | Fallback | Description |
|----------|----------|-------------|
| `PA_NVIDIA_API_KEY` | `NVIDIA_API_KEY` | NVIDIA API key for public NIM or other NVIDIA-hosted model backends |
| `NGC_API_KEY` | `NGC_API_KEY` | Only required when `RENDER_ENDPOINT` or `NVCF_RENDER_FUNCTION_ID` points to an authenticated NVCF render function |

### AWS (Optional)

| Variable | Description |
|----------|-------------|
| `AWS_CONFIG_FILE` | Path to `.env` file with AWS credentials |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_DEFAULT_REGION` | AWS region |

---

## Typical Workflow

```
1. POST /pipeline/upload-usd        Upload USD file
       ↓
2. POST /pipeline                    Start pipeline (with session_id)
       ↓
3. GET /pipeline/{id}/events         Stream SSE for real-time progress
       ↓                             (or poll GET /pipeline/{id}/status)
4. GET /pipeline/{id}/results        Get final stats and download URLs
       ↓
5. GET /artifacts/{id}/report        View HTML report
   GET /artifacts/{id}/predictions   Download predictions JSONL
   GET /artifacts/{id}/dataset       Download dataset JSONL
       ↓
6. POST /pipeline/{id}/regenerate    (Optional) Re-run steps with new prompt
       ↓
7. DELETE /sessions/{id}             Clean up when done
```

Or use the single-step shortcut:

```
1. POST /pipeline (with usd_file)   Upload + start in one call
       ↓
2. ... (same as steps 3-7 above)
```
