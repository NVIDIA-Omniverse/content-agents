# Material Agent Service API Reference

REST API for VLM-based material assignment to 3D USD files. The service accepts USD scene uploads, runs an async pipeline (optimize, render, build dataset, predict, apply, render final), and streams real-time progress via SSE.

**Base URL:** `http://localhost:8000`
**Interactive docs:** `GET /docs` (Swagger UI)
**OpenAPI:** [`../openapi.yaml`](../openapi.yaml)

---

## Table of Contents

- [Authentication](#authentication)
- [Root Endpoints](#root-endpoints)
- [Config](#config)
- [Pipeline](#pipeline)
- [Artifacts](#artifacts)
- [Assets](#assets)
- [Sessions](#sessions)
- [Materials](#materials)
- [Server-Sent Events (SSE)](#server-sent-events-sse)
- [Data Models](#data-models)
- [Configuration](#configuration)

---

## Authentication

No authentication is required by default. The service accepts all origins via permissive CORS.

---

## Root Endpoints

### `GET /health`

Health check. Includes backend credential readiness, image-generation readiness, and max active sessions.

**Response** `200`

```json
{
  "status": "healthy",
  "service": "Material Agent Service",
  "version": "0.3.7",
  "api_keys_configured": true,
  "image_gen_configured": true,
  "max_active_sessions": 4
}
```

---

## Config

### `GET /config/vlm-models`

List the VLM models exposed by the service. The list depends on which API keys are configured at startup.

**Response** `200` — list of `{backend, model, display_name}` entries.

---

## Pipeline

### `POST /pipeline/upload-usd`

Upload a USD file and create a new session. Use this when the USD lives on the client — the service stores it and returns a session you can then start a pipeline on.

**Request** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `usd_file` | file | USD asset to apply materials to. |

**Response** `201` — [`SessionCreated`](#sessioncreated)

### `POST /pipeline`

Start a material assignment pipeline on an existing session.

**Request body**

```json
{
  "session_id": "abc123",
  "materials_manifest": "default",
  "vlm": {
    "backend": "nim",
    "model": "qwen/qwen3.5-397b-a17b"
  },
  "reference_images": ["ref1.jpg"],
  "generated_reference_id": "optional-generated-reference-id",
  "reference_pdfs": []
}
```

**Response** `202` — [`SessionCreated`](#sessioncreated)

### `POST /pipeline/{session_id}/generate-reference-image`

Generate an AI reference image from the uploaded input preview and a text prompt.
The response includes a `reference_id`; pass it as `generated_reference_id` to
`POST /pipeline` to use that generated image for the run.

**Response** `200`

```json
{
  "status": "ok",
  "reference_id": "generated-reference-id",
  "image_url": "/assets/session-id/generated-ref/generated-reference-id"
}
```

### `GET /pipeline/{session_id}/status`

Get the current pipeline status — step progress, active step, completion fraction.

**Response** `200` — [`PipelineStatus`](#pipelinestatus)

### `GET /pipeline/{session_id}/results`

Get the final pipeline results. Returns `409 Conflict` until the pipeline completes.

**Response** `200` — [`PipelineResults`](#pipelineresults)

### `POST /pipeline/{session_id}/cancel`

Request cancellation of a running pipeline.

**Response** `200`

### `GET /pipeline/{session_id}/events`

Subscribe to real-time pipeline events via Server-Sent Events. See [Server-Sent Events](#server-sent-events-sse) below.

**Response** `200` — `text/event-stream`

---

## Artifacts

All artifact endpoints require the corresponding pipeline step to have completed.

### `GET /artifacts/{session_id}/output`

Download the output USD file with materials applied.

**Response** `200` — `application/octet-stream`

### `GET /artifacts/{session_id}/final-render`

Download the final render image of the materialized asset.

**Response** `200` — `image/png`

### `GET /artifacts/{session_id}/predictions`

Download the predictions JSONL file (one prediction per line).

**Response** `200` — `application/x-ndjson`

### `GET /artifacts/{session_id}/report`

View the prediction HTML report (rendered side-by-side of predictions and ground truth, if available).

**Response** `200` — `text/html`

### `GET /artifacts/{session_id}/optimization-report`

View the USD optimization JSON report (output of the `optimize_usd` step).

**Response** `200` — `application/json`

---

## Assets

Input assets are distinguished from output artifacts — assets are what the user uploaded, artifacts are what the pipeline produced.

### `GET /assets/{session_id}/input-render`

Get a render of the input USD (before any material assignment).

### `GET /assets/{session_id}/previews`

List all preview images rendered during the pipeline.

**Response** `200` — list of preview filenames

### `GET /assets/{session_id}/preview/{image_name}`

Download a specific preview image.

### `GET /assets/{session_id}/references`

List reference images for the session.

### `GET /assets/{session_id}/reference/{image_name}`

Download a specific reference image.

### `GET /assets/{session_id}/reference-pdfs`

List reference PDFs for the session.

### `GET /assets/{session_id}/reference-pdf/{pdf_name}`

Download a specific reference PDF.

---

## Sessions

### `GET /sessions`

List all sessions on the server.

**Response** `200` — `list[SessionSummary]`

### `GET /sessions/usage`

Get aggregate usage statistics (counts by status, total compute time, etc.).

### `GET /sessions/{session_id}`

Get full session details including pipeline configuration and current status.

### `DELETE /sessions/{session_id}`

Delete a session and all its artifacts.

**Response** `204`

### `POST /sessions/admin/cleanup`

Trigger manual cleanup of expired sessions (normally runs on a schedule).

---

## Materials

### `GET /materials`

List available materials from the default material library (or the one configured via `MA_MATERIAL_LIBRARY`).

**Response** `200` — list of `{name, description, binding, icon_url, icon_path}` entries.
`icon_url` / `icon_path` are `null` when the library does not ship thumbnails.

### `GET /materials/icon/{material_name}`

Get the icon image (thumbnail) for a named material.

**Response** `200` — `image/png`

### `GET /materials/template`

Download the default materials template ZIP. Use this as a starting point for building your own material library.

The bundled default template ships `materials.yaml` plus the USD library.
Thumbnail icons are optional and are not included by default.

**Response** `200` — `application/zip`

---

## Server-Sent Events (SSE)

### `GET /pipeline/{session_id}/events`

Subscribe to real-time pipeline events. Events include step start/end, progress updates, and errors.

**Response** `200` — `text/event-stream`

Example events:

```
event: step_started
data: {"step": "build_dataset_usd", "started_at": "..."}

event: step_progress
data: {"step": "predict", "current": 7, "total": 20}

event: step_completed
data: {"step": "apply", "ended_at": "...", "success": true}

event: pipeline_completed
data: {"status": "completed"}
```

Clients should reconnect on disconnect; the server resends the event history for the session (up to the retention window).

---

## Data Models

### `SessionCreated`

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique session identifier. |
| `status` | string | Initial status (usually `queued`). |
| `created_at` | datetime | ISO timestamp. |

### `PipelineStatus`

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session ID. |
| `status` | `queued \| running \| completed \| failed \| cancelled` | Overall pipeline state. |
| `current_step` | `CurrentStepInfo \| null` | The step currently executing (if any). |
| `completed_steps` | `list[CompletedStepInfo]` | Steps that have finished. |
| `progress` | `OverallProgress` | Fraction of total steps completed. |

### `PipelineResults`

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session ID. |
| `output_usd_url` | string | Path to the materialized USD (served by `/artifacts/.../output`). |
| `final_render_url` | string | Path to the final render image. |
| `predictions_url` | string | Path to the predictions JSONL. |
| `report_url` | string | Path to the HTML report. |
| `metrics` | object | Aggregate metrics (material coverage, prediction confidence, etc.). |

### Pipeline Steps

The material-agent pipeline runs the following steps (some opt-in):

1. `validate_input` — Sanity-check the USD and configuration
2. `optimize_usd` — Flatten and deinstance via scene optimizer
3. `render_preview` — Lightweight whole-scene preview rendering
4. `generate_reference_image` — Generate photorealistic reference images (opt-in)
5. `build_dataset_usd` — Render prim views for VLM input
6. `build_dataset_prepare_dataset` — Prepare dataset entries with material specs
7. `predict` — VLM inference for material assignment
8. `validate_predictions` — Validate/repair predicted material names against the library
9. `harmonize_predictions` — Resolve conflicts for instanced parts
10. `apply` — Apply predicted materials to USD
11. `restore_usd` — Restore the original USD hierarchy (reverses optimize_usd changes)
12. `validate_output` — Sanity-check the output USD
13. `render` — Final render of the materialized asset

---

## Configuration

The service reads its configuration from environment variables at startup. See `.env_example` for the full list. Key settings:

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | Required if using `nim` VLM backend |
| `OPENAI_API_KEY` | Required if using `openai` backend |
| `ANTHROPIC_API_KEY` | Required if using `anthropic` backend |
| `GOOGLE_API_KEY` | Required if using `gemini` backend |
| `MA_SESSION_STORAGE_PATH` | Where session directories are written |
| `MA_MAX_UPLOAD_SIZE_MB` | Max USD file size for `/pipeline/upload-usd` |
| `MA_MAX_WORKERS` | Concurrency for VLM inference |
| `MA_VLM_BACKEND`, `MA_VLM_MODEL` | Default VLM backend + model |
| `MA_LLM_BACKEND`, `MA_LLM_MODEL` | Default LLM backend + model for validate/harmonize |
| `MA_IMAGE_GEN_BACKEND` | Image generation backend for generated reference images (default `gemini`) |
| `MA_IMAGE_GEN_MODEL` | Optional image generation model override |
| `MA_IMAGE_GEN_BASE_URL` | Optional image generation API base URL override |
| `MA_RENDERER_BACKEND` | `ovrtx` or `remote` |
