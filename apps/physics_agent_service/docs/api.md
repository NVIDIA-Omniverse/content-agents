# Physics Agent Service API Reference

REST API for VLM-based asset classification and physics auto-tuning of USD files. The service accepts USD scene uploads, runs async pipeline and predict workflows, and exposes single-shot tuning over physics-authored USDs.

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
- [Predict](#predict)
- [Tune](#tune)
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
    "predict": {
      "create": "POST /predict",
      "status": "GET /predict/{session_id}/status",
      "results": "GET /predict/{session_id}/results",
      "cancel": "POST /predict/{session_id}/cancel",
      "events": "GET /predict/{session_id}/events"
    },
    "tune": {
      "create": "POST /tune",
      "status": "GET /tune/{session_id}/status",
      "results": "GET /tune/{session_id}/results",
      "events": "GET /tune/{session_id}/events",
      "cancel": "POST /tune/{session_id}/cancel",
      "artifact": "GET /tune/{session_id}/artifacts/{name}"
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

## Predict

`/predict` is a first-class route group for prediction-only workflows. It is
**not** a thin alias for `/pipeline`: it runs prediction (and the minimum
upstream prep, when needed), but exposes a separate session lifecycle and
result endpoints. Use `/pipeline` when you need the full classify/apply flow
(rendering, dataset prep, predict, **and** apply_physics with UsdPhysics
schemas baked into a simulation-ready USD).

### When to use `/predict` vs `/pipeline`

| Use case | Endpoint |
|----------|----------|
| Just want predictions for a USD scene; no physics-augmented USD output | `POST /predict` |
| Already have a prepared `dataset.jsonl`; only need VLM inference | `POST /predict` (Mode A) |
| Want predictions **and** the simulation-ready USD with `UsdPhysics*` schemas | `POST /pipeline` |
| Re-run only `predict` from a previously-built pipeline session | `POST /pipeline/{id}/regenerate` with `steps=["predict"]` |

The full classify/apply flow (`/pipeline`) is **unchanged** by the `/predict`
route — old callers using `POST /pipeline` and `POST /pipeline/{id}/regenerate`
with `steps=["predict"]` keep working exactly as before.

### Mode A vs Mode B (auto-detected)

`POST /predict` auto-picks one of two modes at job start, based on what's
already on disk for the session:

* **Mode A — `dataset_only`.** A prepared `dataset.jsonl` is already
  available (either at the session's `cache/dataset/dataset.jsonl`, or
  supplied via the `dataset_path` form field). Only the `predict` step runs.
* **Mode B — `full_predict`.** No prepared dataset is present. The minimum
  upstream steps run before predicting:
  `optimize_usd` (optional) → `identify_asset` → `build_dataset_usd` →
  `build_dataset_prepare_dataset` → `predict`. **`apply_physics` is
  intentionally not part of `/predict`.**

The detected mode is persisted to session metadata under `predict_mode` and
returned by `GET /predict/{id}/results` as `mode`. Required upstream steps
are never silently skipped — Mode B always runs the prep listed above before
prediction.

### Create Predict

```http
POST /predict
```

Create and execute a prediction job.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `usd_file` | file | conditional | USD file (Mode B). Required if no other input source is provided. |
| `session_id` | string | conditional | Existing session ID (e.g. from `/pipeline/upload-usd`). If the session already has a prepared dataset, runs Mode A. |
| `s3_uri` | string | conditional | S3 URI to a USD file. |
| `dataset_path` | string | conditional | Absolute path to a prepared `dataset.jsonl` on the server. When set and readable, forces Mode A. The path must resolve inside the session-storage root or one of the colon-separated locations listed in the `PA_DATASET_ALLOWED_ROOTS` env var; anything else is rejected with `403`. |
| `user_prompt` | string | no | Custom prompt for the VLM (Mode B). |
| `render_backend` | string | no | Mode B only: `remote` (default), `warp`, or `ovrtx`. |
| `optimize_usd` | boolean | no | Mode B only: enable Scene Optimizer. Default `false`. |
| `enable_deinstance` | boolean | no | Mode B only: enable deinstance op. Default `true`. |
| `enable_split` | boolean | no | Mode B only: enable split-meshes op. Default `false`. |
| `enable_deduplicate` | boolean | no | Mode B only: enable deduplicate op. Default `false`. |

At least one of `usd_file`, `session_id`, `s3_uri`, or `dataset_path` is
required.

**Supported input combinations:**

- Exactly **one** of `usd_file`, `session_id`, or `s3_uri` may be provided
  as the primary source. Sending more than one is rejected with `400`.
- `dataset_path` may be sent on its own (pure Mode A) or together with
  `session_id` (override the session's prepared dataset). It is **not**
  compatible with `usd_file` or `s3_uri` (Mode B inputs) — those
  combinations are rejected with `400`.

**Response** `202`
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Predict job queued for execution",
  "estimated_duration_minutes": 10
}
```

**Errors:**
- `400` No input source provided; ambiguous combination (more than one of `usd_file`/`session_id`/`s3_uri`, or `dataset_path` paired with `usd_file`/`s3_uri`); invalid USD extension; missing `dataset_path`; invalid optimizer config; no input USD or prepared dataset for this session
- `403` S3 access denied (when `s3_uri` is provided)
- `404` `session_id` provided but session does not exist; S3 object not found
- `409` Predict already `pending`/`running`/`cancelling` for the supplied `session_id`, or the same-pod `JobRegistry` race-guard fired
- `413` File too large; S3 file too large
- `500` Internal failure copying the upload to disk
- `502` Failed to download from S3

### Cancel Predict response body

```json
{
  "session_id": "a1b2c3d4-...",
  "status": "cancelling",
  "message": "Predict cancellation requested"
}
```

### Predict Events

`/predict/{id}/events` is an SSE stream (Content-Type: `text/event-stream`).
Event names match `/pipeline`:
- `progress` (data: `ProgressEvent` JSON)
- `done` (data: `{session_id, final_state}` — `final_state` is `completed`, `failed`, or `cancelled`)
- `ping` (data: `keepalive`, every ~30s when no progress events have arrived)

### Get Predict Status

```http
GET /predict/{session_id}/status
```

Same response schema as `/pipeline/{id}/status` (`PipelineStatus`), so
existing progress UIs work unchanged.

**Errors:**
- `404` Session not found

### Get Predict Results

```http
GET /predict/{session_id}/results
```

Returns predict results once the job is complete. Surfaces the
source-of-truth `PredictOutput` fields directly:

**Response** `200` -- [PredictResults](#predictresults)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "completed",
  "mode": "full_predict",
  "steps_run": ["identify_asset", "build_dataset_usd", "build_dataset_prepare_dataset", "predict"],
  "stats": {
    "predictions_made": 142,
    "failed_count": 0,
    "predictions_path": "/var/lib/physics-agent/.../predictions.jsonl",
    "token_stats": {"prompt_tokens": 12345, "completion_tokens": 6789}
  },
  "predictions_count": 142,
  "failed_count": 0,
  "predictions_path": "/var/lib/physics-agent/.../predictions.jsonl",
  "token_stats": {"prompt_tokens": 12345, "completion_tokens": 6789},
  "download_urls": {
    "predictions": "/artifacts/a1b2c3d4-.../predictions",
    "report": "/artifacts/a1b2c3d4-.../report",
    "dataset": "/artifacts/a1b2c3d4-.../dataset"
  },
  "duration_seconds": 312,
  "completed_at": "2026-02-24T10:10:12Z"
}
```

`download_urls.dataset` is only present when the session has a
`cache/dataset/dataset.jsonl` on disk. When Mode A is triggered by an
external `dataset_path`, the route copies that file into the session's
cache so all three URLs (`predictions`, `report`, `dataset`) are
available; if Mode A picked up an existing session that already had a
prepared dataset, the same file is reused.

> [!NOTE]
> When `dataset_path` points at an external dataset, the **JSONL file** is
> staged into the session's cache, but the **per-prim render images** the
> JSONL references are not copied. Inference resolves the original images
> next to the original `dataset_path`, so that location must remain
> accessible to the service worker until the predict job finishes. The
> dataset artifact returned via `/artifacts/{id}/dataset` is therefore not
> self-contained: it lists image filenames whose absolute resolution
> depends on the original directory layout.

When the predict job fails (`status="failed"`), `GET /predict/{id}/results`
returns HTTP 200 with a `PipelineError` body (`error_message`,
`failed_step`, `completed_steps`, `partial_results`) — same shape as
`/pipeline/{id}/results`.

**Errors:**
- `202` Predict still pending or running
- `404` Session not found

### Cancel Predict

```http
POST /predict/{session_id}/cancel
```

Cancel a running predict job. Mirrors `/pipeline/{id}/cancel` semantics
but refuses sessions that were created via `/pipeline` or
`/pipeline/upload-usd` — call `POST /pipeline/{session_id}/cancel` for
those instead.

**Errors:**
- `400` Job not in a cancellable state
- `404` Session not found
- `409` Session is not a predict session (it was created via `/pipeline` or `/pipeline/upload-usd`)

### Stream Predict Events

```http
GET /predict/{session_id}/events
```

SSE stream of progress events. Same semantics as `/pipeline/{id}/events`:
only works on the executing instance; for cross-instance progress, poll
`GET /predict/{session_id}/status` instead.

**Errors:**
- `404` Session not found
- `503` Predict is running on a different instance; poll status instead

---

## Tune

`/tune` is the service entry point for single-shot Physics Agent auto-tuning.
It runs the same tuning API as `physics-agent tune`: patch tunable physics
parameters, evaluate each trial with a simulation backend, write tune artifacts,
and optionally run the VLM judge over scenario/history/best parameters and
reference media.

Tune expects a physics-authored USD, not a raw asset USD. Supply one of:

| Source | Field | Notes |
|--------|-------|-------|
| Local upload | `physics_usd` | Upload an `apply_physics` output USD/USDA/USDC/USDZ |
| S3 object | `s3_uri` | Service downloads the physics USD server-side |
| Completed pipeline | `source_session_id` | Service copies that session's `output_usd` artifact |

Exactly one source field is required. For raw USD classification and physics
authoring, run `/pipeline` first and then pass its completed session id as
`source_session_id`.

`/tune` is single-shot. The Physics Agent CLI/Python API also provide iterative
`refine`, but the service does not currently expose a first-class `/refine`
route. `judge_max_iterations` is accepted for compatibility and audit metadata;
single-shot `/tune` does not re-run tuning when the judge returns `continue`.

### Create Tune

```http
POST /tune
```

Create and queue a tune session. Execution is async and returns `202` once the
job is registered.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `physics_usd` | file | conditional | Physics-authored USD. Required if `s3_uri` and `source_session_id` are absent. |
| `s3_uri` | string | conditional | S3 URI to a physics-authored USD. |
| `source_session_id` | string | conditional | Completed pipeline session whose `output_usd` should be tuned. |
| `scenario_yaml` | string | conditional | Tuning scenario YAML body. Optional when `user_prompt` is supplied. |
| `user_prompt` | string | conditional | Natural-language description such as `make this object bouncy`. Optional when full `scenario_yaml` is supplied. |
| `reference_images` | file[] | no | Optional reference images for the visual/VLM judge. |
| `reference_videos` | file[] | no | Optional reference videos for the visual/VLM judge. |
| `reference_descriptions` | JSON string[] | no | Descriptions parallel to `reference_images`. |
| `reference_video_descriptions` | JSON string[] | no | Descriptions parallel to `reference_videos`. |
| `optimizer` | string | no | `auto` (default, resolves to BoTorch), `botorch`, `random`, or `cma-es`. |
| `engine` | string | no | `ovphysx` (default) or `fake` for tests. |
| `max_trials` | int | no | Trial budget. Must be between 1 and 1000. |
| `seed` | int | no | Seed for optimizer and backend. |
| `enable_judge` | bool | no | Run the VLM judge at the end of tune. Default `true`. |
| `judge_max_iterations` | int | no | Compatibility/audit field. Must be 1-10; single-shot `/tune` does not iterate. |
| `judge_max_tokens` | int or null | no | Optional judge response token cap. |
| `judge_temperature` | float or null | no | Optional judge temperature. |

Either `scenario_yaml` or `user_prompt` must be non-empty. When both are
provided, explicit YAML fields win and the prompt interpreter fills gaps.

Reference media uploads are capped by the same upload budget as USD uploads.
`reference_descriptions` and `reference_video_descriptions` must be JSON arrays
with one string per corresponding file.

`optimizer` and `engine` are submitted as form strings. Some invalid values are
not rejected before the `202` response and can instead fail the queued tune job;
poll status/results to inspect those asynchronous failures.

**Response** `202` -- [SessionCreated](#sessioncreated)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Tune queued for execution",
  "estimated_duration_minutes": 5
}
```

**Errors:**
- `400` Missing or ambiguous source; missing scenario/prompt; invalid scenario YAML; unsupported engine/scenario pair when explicit `scenario_yaml` names a known scenario; invalid trial, judge, reference-description, source-session-id, or USD-extension value
- `403` S3 object access denied
- `404` `source_session_id` not found, or S3 object not found
- `413` Upload, scenario YAML, prompt, or reference media too large
- `422` Multipart/form parsing or FastAPI validation failed before route logic, for example a non-coercible integer, boolean, or float form value
- `502` S3 download failed

### Get Tune Status

```http
GET /tune/{session_id}/status
```

Returns a flat trial-progress view. Tune is one iterative loop, so it does not
use the multi-step `PipelineStatus` shape.

**Response** `200` -- [TuneStatus](#tunestatus)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "running",
  "n_trials": 7,
  "max_trials": 30,
  "best_score": 0.093,
  "best_params": {
    "mass_scale": 1.2,
    "static_friction": 0.4,
    "dynamic_friction": 0.3,
    "restitution": 0.8
  },
  "elapsed_seconds": 84,
  "can_cancel": true,
  "created_at": "2026-02-24T10:00:00Z",
  "updated_at": "2026-02-24T10:01:24Z"
}
```

**Errors:**
- `404` Session not found

### Get Tune Results

```http
GET /tune/{session_id}/results
```

Returns final tune results when the job is terminal. Completed, failed-with-
partial-results, and cancelled-after-trials sessions use the same `TuneResults`
shape so callers can discover artifact URLs.

**Response** `200` -- [TuneResults](#tuneresults)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "completed",
  "best_params": {
    "mass_scale": 1.2,
    "static_friction": 0.4,
    "dynamic_friction": 0.3,
    "restitution": 0.8
  },
  "best_score": 0.093,
  "n_trials": 30,
  "optimizer_used": "botorch",
  "engine_used": "ovphysx",
  "download_urls": {
    "best_params": "/tune/a1b2c3d4-.../artifacts/best_params.json",
    "tune_results": "/tune/a1b2c3d4-.../artifacts/tune_results.json",
    "history": "/tune/a1b2c3d4-.../artifacts/history.jsonl",
    "report": "/tune/a1b2c3d4-.../artifacts/report.md",
    "tuned_usd": "/tune/a1b2c3d4-.../artifacts/tuned_physics.usda",
    "visual_comparison": "/tune/a1b2c3d4-.../artifacts/comparison.png"
  },
  "duration_seconds": 420,
  "completed_at": "2026-02-24T10:07:00Z",
  "error_message": null
}
```

**Errors:**
- `202` Tune still pending or running
- `404` Session not found

### Stream Tune Events

```http
GET /tune/{session_id}/events
```

SSE stream of tune progress events. Same-instance clients receive `progress`
frames for `tune.started`, each completed trial, terminal failure/cancel, and
artifact-ready completion. In multi-instance deployments, use
`GET /tune/{session_id}/status` polling when the SSE endpoint returns `503`.

**Errors:**
- `404` Session not found
- `503` Tune is running on a different instance; poll status instead

### Cancel Tune

```http
POST /tune/{session_id}/cancel
```

Cooperatively cancels a pending or running tune. Tune jobs run optimizer loops in
a worker thread, so cancellation writes the shared cancellation marker and the
runner exits between trials. Cancelling a non-tune session is rejected.

**Response** `200`
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "cancelling",
  "message": "Tune cancellation requested"
}
```

**Errors:**
- `400` Tune already completed, failed, or cancelled
- `404` Session not found
- `409` Session is not a tune session

### Download Tune Artifact

```http
GET /tune/{session_id}/artifacts/{name}
```

Downloads one canonical tune artifact. Unknown names are rejected to avoid path
traversal. Artifact reads pull from the shared store when the local instance
does not already have the file.

| Name | Content |
|------|---------|
| `best_params.json` | Best parameter set |
| `tune_results.json` | Full result payload, including judge data when enabled |
| `history.jsonl` | One JSON object per optimizer trial |
| `report.md` | Human-readable Markdown report |
| `tuned_physics.usda` | Physics USD patched with best parameters |
| `comparison.png` | Optional VLM judge contact sheet when visual comparison ran |

**Errors:**
- `404` Session not found, artifact unavailable, or unknown artifact name

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

### Download Output USD

```http
GET /artifacts/{session_id}/output-usd
```

Download the simulation-ready USD written by `apply_physics`. `.usd`, `.usda`,
and `.usdc` inputs are returned as `scene_physics.usd`, `scene_physics.usda`,
or `scene_physics.usdc`. `.usdz` inputs default to `scene_physics.usda` so
Omniverse MDL shader references can remain as runtime-resolved asset paths
instead of being bundled into a new USDZ package. Package-local asset
dependencies from the source USDZ are copied beside the USDA and rewritten to
relative paths when the output references them. When those sidecar assets are
present, the endpoint returns a ZIP bundle containing `scene_physics.usda` and
the `scene_physics_assets/` directory; otherwise it returns the single USD file.

**Response** `200`
Content-Type: `text/plain` for USDA, `model/vnd.usdz+zip` for USDZ artifacts,
`application/zip` for USDA plus sidecar assets, or `application/octet-stream`
for USD/USDC.

**Errors:**
- `404` Session or output USD not found

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

Connect to `GET /pipeline/{session_id}/events`, `GET /predict/{session_id}/events`, or `GET /tune/{session_id}/events` to receive real-time progress updates.

### Event Types

| Event | Description |
|-------|-------------|
| `progress` | Pipeline, predict, or tune progress update ([ProgressEvent](#progressevent)) |
| `ping` | Keepalive sent every 30 seconds |
| `done` | Job completed, failed, or cancelled -- stream closes after this |

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

### PredictResults

Returned by `GET /predict/{id}/results` when the predict job completes.
Mirrors the prediction-relevant subset of `PipelineResults` and additionally
hoists the source-of-truth `PredictOutput` fields out of `stats`.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | `"completed"` |
| `mode` | string | Detected predict mode: `dataset_only` (Mode A) or `full_predict` (Mode B) |
| `steps_run` | string[] | Pipeline steps the predict route actually drove |
| `stats` | object | Execution statistics (mirrors `predictions_made`, `failed_count`, `predictions_path`, `token_stats`) |
| `predictions_count` | int | Number of predictions produced (sticky field from `PredictOutput`) |
| `failed_count` | int | Number of failed predictions (sticky field from `PredictOutput`) |
| `predictions_path` | string or null | Server-side path to `predictions.jsonl` |
| `token_stats` | object | VLM token usage statistics when available |
| `download_urls` | object | `{predictions, report, dataset?}` — `dataset` only present when `cache/dataset/dataset.jsonl` exists |
| `duration_seconds` | int | Total predict duration |
| `completed_at` | string | ISO 8601 timestamp |

### TuneStatus

Returned by `GET /tune/{id}/status`.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | `pending`, `running`, `completed`, `failed`, `cancelled`, or `cancelling` |
| `n_trials` | int | Trials completed so far |
| `max_trials` | int | Configured trial budget |
| `best_score` | float or null | Best score so far; lower is better |
| `best_params` | object or null | Best parameter set so far |
| `elapsed_seconds` | int | Total elapsed time |
| `can_cancel` | bool | Whether tune can be cancelled |
| `created_at` | string | ISO 8601 timestamp |
| `updated_at` | string | ISO 8601 timestamp |

### TuneResults

Returned by `GET /tune/{id}/results` for completed tune sessions, and for
failed or cancelled sessions when partial tune artifacts are available.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | Terminal status: `completed`, `failed`, or `cancelled` |
| `best_params` | object | Best parameter set |
| `best_score` | float or null | Best score; null when no successful trial completed |
| `n_trials` | int | Number of trials evaluated |
| `optimizer_used` | string | Resolved optimizer name, e.g. `botorch` |
| `engine_used` | string | Engine used, e.g. `ovphysx` |
| `download_urls` | object | Tune artifact URLs |
| `duration_seconds` | int | Total tune duration |
| `completed_at` | string | ISO 8601 timestamp |
| `error_message` | string or null | Failure reason for failed sessions with partial artifacts |

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
| `identify_asset` | Identify the whole asset before per-component classification |
| `build_dataset_usd` | Render images from USD prims |
| `build_dataset_prepare_dataset` | Prepare dataset with prompts |
| `predict` | Run VLM predictions |
| `restore_usd` | Map predictions on optimized/deinstanced prims back to original-scene prim paths |

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
| `kind` | string or null | `tune` for `/tune` sessions; absent for older or non-tune sessions |
| `config` | object | Route-specific request metadata. Pipeline/predict sessions include `{project_name, usd_path, has_usd_upload, user_prompt, predict_route?}`. Tune sessions include `{kind: "tune", engine, optimizer, max_trials, seed, physics_usd, scenario_path?, user_prompt?, reference_images, reference_videos, enable_judge, judge_*}`. |
| `ttl_expires_at` | string | Expiration timestamp |
| `results` | object | Final stats |
| `duration_seconds` | int | Total duration |
| `completed_at` | string | Completion timestamp |
| `timings` | object | Per-step durations |
| `predict_mode` | string | `dataset_only` or `full_predict` — present on `/predict` sessions; mirrors what `GET /predict/{id}/results.mode` reports |
| `predict_steps_run` | string[] | Pipeline steps `/predict` actually drove for this job (Mode A is `["predict"]`; Mode B is the configured upstream prep + `predict`) |

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
| `409` | Route/session kind conflict or active job race |
| `413` | File too large |
| `500` | Internal server error |
| `502` | Upstream S3 or service dependency failure |

---

## Configuration

All settings use the `PA_` environment variable prefix.

### Service Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PA_SESSION_STORAGE_PATH` | `/var/physics-agent/sessions` | Session storage directory (falls back to `./sessions` in dev) |
| `PA_SESSION_TTL_HOURS` | `24` | Hours before sessions are auto-cleaned |
| `PA_MAX_ACTIVE_SESSIONS` | `1` | Max concurrent pipeline executions (semaphore) |
| `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` | `1` | Process-wide render request cap for local OVRTX compose |
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

Tune a completed pipeline output:

```text
1. POST /pipeline                    Produce apply_physics output_usd
       ↓
2. GET /pipeline/{id}/results        Confirm completed output_usd exists
       ↓
3. POST /tune                        Send source_session_id=<pipeline id>
       ↓
4. GET /tune/{id}/events             Stream trial progress
       ↓                             (or poll GET /tune/{id}/status)
5. GET /tune/{id}/results            Get best params and artifact URLs
       ↓
6. GET /tune/{id}/artifacts/report.md
   GET /tune/{id}/artifacts/tuned_physics.usda
```
