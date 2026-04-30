# Texture Agent Service API Reference

REST API for AI-driven texture generation on USD materials. Upload a materialized USD file (or reference one by S3 URI), run the texture pipeline end-to-end, stream real-time progress via Server-Sent Events, and download the generated PBR texture set and textured USDZ output.

**Base URL:** `http://localhost:8001`
**Interactive docs:** `GET /docs` (Swagger UI), `GET /redoc`
**OpenAPI:** [`../openapi.yaml`](../openapi.yaml) — snapshot of the live spec produced by FastAPI from the route definitions. When in doubt, the runtime spec at `/openapi.json` is authoritative.

---

## Table of Contents

- [Authentication](#authentication)
- [Root endpoints](#root-endpoints)
- [Pipeline](#pipeline)
- [Sessions](#sessions)
- [Artifacts](#artifacts)
- [Server-Sent Events (SSE)](#server-sent-events-sse)
- [Pipeline steps](#pipeline-steps)
- [Configuration](#configuration)

---

## Authentication

No authentication is required. The service accepts all origins via permissive CORS.

---

## Root endpoints

### `GET /health`

Health check.

**Response** `200`

```json
{
  "status": "healthy",
  "service": "Texture Agent Service",
  "version": "0.0.1-dev",
  "image_gen_backend": "nim",
  "active_backend_key_configured": true,
  "nvidia_api_key_configured": true,
  "max_active_sessions": 4
}
```

### `GET /api`

Returns service info and a map of available endpoints (same catalog as in this document).

---

## Pipeline

### `POST /pipeline/upload-usd`

Upload a USD asset and create a new session without starting a pipeline. Use this when you want to stage a file first, inspect the generated `session_id`, and then call `POST /pipeline` against the same session.

**Request** `multipart/form-data` — supply exactly one of:

| Field | Type | Description |
|-------|------|-------------|
| `usd_file` | file | USD asset (`.usd`, `.usda`, `.usdc`, `.usdz`) uploaded directly. |
| `s3_uri` | string | `s3://bucket/key/path.usd` — the service fetches the file server-side. |

**Response** `201` — `SessionCreated`

```json
{
  "session_id": "11ea5cb5-35aa-491d-9440-dabae87a8f0c",
  "status": "ready",
  "message": "USD uploaded successfully",
  "estimated_duration_minutes": 0
}
```

For S3 uploads, `message` is `USD downloaded from S3 successfully (<size>MB)`.

### `POST /pipeline`

Create a session and kick off the texture pipeline in one call.

**Request** `multipart/form-data` — supply the USD via one of `usd_file` / `s3_uri` / `session_id`:

| Field | Type | Description |
|-------|------|-------------|
| `usd_file` | file | USD uploaded directly (same rules as `/pipeline/upload-usd`). |
| `s3_uri` | string | `s3://...` reference; service fetches it. |
| `session_id` | string | Reuse a session previously created via `/pipeline/upload-usd`. |
| `material_textures_json` | string | JSON map `{material_name: {prompt, opacity, per_prim}}`. `prompt` is required and must be non-empty, `opacity` is optional and must be between `0.0` and `1.0`, and unknown fields are rejected. Materials not listed get auto-generated prompts via the configured LLM (see `TA_LLM_BACKEND`). |
| `user_prompt` | string | Optional aesthetic direction, e.g. `"weathered mossy patina"`. Used by the LLM auto-prompt step. |

**Example (curl)**

```bash
curl -X POST http://localhost:8001/pipeline \
  -F "usd_file=@/path/to/ladder.usd" \
  -F "user_prompt=rusty look" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"heavy patchy rust","opacity":0.85}}'
```

`per_prim` may override individual prim paths under a material. Each per-prim
entry must include `prompt`, `opacity`, or both, and uses the same opacity bounds.
Providing any `per_prim` override automatically runs the texture pipeline in
per-prim mode for that request:

```json
{
  "Steel_Carbon": {
    "prompt": "aged steel",
    "opacity": 0.85,
    "per_prim": {
      "/World/Ladder/Rung_01": {"prompt": "fresh scrape marks"},
      "/World/Ladder/Rung_02": {"opacity": 0.65}
    }
  }
}
```

**Response** `202` — `SessionCreated`

```json
{
  "session_id": "11ea5cb5-35aa-491d-9440-dabae87a8f0c",
  "status": "pending",
  "message": "Pipeline queued for execution",
  "estimated_duration_minutes": 10
}
```

### `GET /pipeline/{session_id}/status`

Pipeline state with per-step progress.

**Response** `200` — `PipelineStatus`

```json
{
  "session_id": "11ea5cb5-...",
  "status": "running",
  "current_step": {
    "name": "generate_textures",
    "display_name": "Generating PBR Textures",
    "started_at": "2026-04-21T19:14:23Z",
    "progress": {"current": 3, "total": 8, "percent": 38, "message": "Aluminum_Brushed"},
    "elapsed_seconds": 42
  },
  "completed_steps": [
    {
      "name": "prepare_uvs",
      "display_name": "Preparing UV Coordinates",
      "started_at": "2026-04-21T19:13:00Z",
      "completed_at": "2026-04-21T19:13:01Z",
      "duration_seconds": 1,
      "stats": {}
    },
    {
      "name": "discover_materials",
      "display_name": "Discovering Materials",
      "started_at": "2026-04-21T19:13:01Z",
      "completed_at": "2026-04-21T19:13:01Z",
      "duration_seconds": 0,
      "stats": {}
    },
    {
      "name": "generate_prompts",
      "display_name": "Generating Texture Prompts",
      "started_at": "2026-04-21T19:13:01Z",
      "completed_at": "2026-04-21T19:13:19Z",
      "duration_seconds": 18,
      "stats": {}
    }
  ],
  "overall_progress": {
    "current_step": 4,
    "total_steps": 8,
    "percent": 45,
    "estimated_remaining_seconds": 95
  },
  "preview_images": [],
  "can_cancel": true,
  "elapsed_seconds": 100,
  "created_at": "2026-04-21T19:13:00Z",
  "updated_at": "2026-04-21T19:14:40Z"
}
```

Overall `status` values: `pending | running | completed | failed | cancelled | cancelling`. The `cancelling` state is returned by `POST /pipeline/{session_id}/cancel` and held until the worker reaches the next cancellation checkpoint, after which the status flips to `cancelled`. If a synchronous worker step does not stop within `TA_CANCEL_DRAIN_TIMEOUT_SECONDS`, the session flips to `failed` and a stalled-worker guard blocks deletion until the worker thread finishes.

### `GET /pipeline/{session_id}/results`

Final results and download URLs. Returns `202` while the pipeline is still pending, running, or cancelling; call `/status` first or subscribe to `/events`.

**Response** `200` — `PipelineResults`

```json
{
  "session_id": "11ea5cb5-...",
  "status": "completed",
  "stats": {
    "materials_found": 12,
    "textures_generated": 12,
    "output_usd_count": 1,
    "renders_count": 2
  },
  "download_urls": {
    "materials": "/artifacts/11ea5cb5-.../materials",
    "textures": "/artifacts/11ea5cb5-.../textures",
    "output": "/artifacts/11ea5cb5-.../output",
    "renders": "/artifacts/11ea5cb5-.../renders"
  },
  "duration_seconds": 142,
  "completed_at": "2026-04-21T19:18:45Z"
}
```

Each artifact is fetched via the matching `/artifacts/{session_id}/{key}` endpoint — see [Artifacts](#artifacts) for the response media types.

### `GET /pipeline/{session_id}/events`

Server-Sent Events stream of pipeline progress. See [Server-Sent Events](#server-sent-events-sse).

### `GET /pipeline/{session_id}/event-log`

Full buffered event log (for replay / debugging). Useful on a completed or failed pipeline.

**Response** `200` — list of SSE-shaped events.

### `POST /pipeline/{session_id}/cancel`

Request cancellation. The worker stops at the next cancellation checkpoint, or — if a step is mid-flight — when asyncio cancellation propagates to the next `await` point. Poll `GET /status` to observe the eventual terminal state: normally `cancelled`, or `failed` if the in-flight synchronous worker step exceeds `TA_CANCEL_DRAIN_TIMEOUT_SECONDS`. In the timeout case, DELETE may continue returning `409` until the stalled worker marker clears after the thread exits.

**Response** `200`

```json
{
  "session_id": "11ea5cb5-35aa-491d-9440-dabae87a8f0c",
  "status": "cancelling",
  "message": "Pipeline cancellation requested"
}
```

### `POST /pipeline/{session_id}/regenerate`

Re-run a subset of steps on an existing session — useful when tweaking prompts without re-uploading.

**Request** `application/json`

```json
{
  "steps": ["generate_textures", "blend_textures", "apply_textures"],
  "material_textures": {
    "Steel_Carbon": {"prompt": "fresh polished steel", "opacity": 0.75}
  }
}
```

`material_textures` follows the same validated shape as `material_textures_json`
on `POST /pipeline`: material keys must be non-empty, material prompts are
required, opacity is bounded to `0.0` through `1.0`, `per_prim` entries may
override prompt and/or opacity, and unknown fields are rejected. A nested
`per_prim` override promotes the regenerated run to per-prim texture mode;
material-only overrides preserve the session's existing texture mode.

**Response** `202` — `SessionCreated` (same session_id; new pipeline run).

---

## Sessions

### `GET /sessions`

List known sessions. Includes state, creation time, and basic metadata.

### `GET /sessions/{session_id}`

Session details (status, timestamps, artifact availability).

### `DELETE /sessions/{session_id}`

Remove a session and all associated artifacts from storage.

**Response** `204` — session and stored artifacts removed.

**Error** `404` — JSON response when the session does not exist.

**Error** `409` — JSON response when a live pipeline job is still active or a worker lock shows artifact writes are still in progress. Cancel the pipeline and wait for the worker to stop before deleting the session. Persisted `cancelling` metadata without a live worker lock can still be deleted, which lets restarted services clean up stale session artifacts.

---

## Artifacts

Artifact endpoints are scoped to a `session_id` and only succeed once the corresponding pipeline step has completed.

Unlike the stale pre-0.3.6 contract, these endpoints **return downloadable payloads**, not list-style metadata. Use `GET /pipeline/{session_id}/results` to enumerate available artifact URLs.

Artifact routes intentionally use per-kind media types:

| Endpoint | Success media type | Payload |
|----------|--------------------|---------|
| `GET /artifacts/{session_id}/materials` | `application/json` | Discovered material metadata |
| `GET /artifacts/{session_id}/textures` | `application/zip` | ZIP containing generated textures under `textures/` |
| `GET /artifacts/{session_id}/textures/{filename}` | `image/png` | Single texture image |
| `GET /artifacts/{session_id}/output` | `model/vnd.usdz+zip` | Self-contained textured USDZ |
| `GET /artifacts/{session_id}/renders` | `application/zip` | ZIP containing final rendered images under `renders/` |
| `GET /artifacts/{session_id}/renders/{filename}` | `image/png` | Single render image |
| `GET /artifacts/{session_id}/preview/{filename}` | `image/png` | Single material preview image |

Error responses, including missing sessions or unavailable artifacts, are JSON.

### `GET /artifacts/{session_id}/materials`

Discovered-material metadata from the `discover_materials` step.

**Response** `200` — `application/json`, list of `MaterialInfo` records.

### `GET /artifacts/{session_id}/textures`

All generated texture files bundled as a ZIP.

**Response** `200` — `application/zip` with a top-level `textures/` folder.

### `GET /artifacts/{session_id}/textures/{filename}`

Single texture image (PNG).

**Response** `200` — `image/png`.

### `GET /artifacts/{session_id}/output`

Textured output asset as a **self-contained USDZ** (USD + embedded textures). Clients should save with a `.usdz` extension regardless of `Content-Disposition`.

**Response** `200` — `model/vnd.usdz+zip`.

### `GET /artifacts/{session_id}/renders`

Rendered preview images (final textured asset) as a ZIP.

**Response** `200` — `application/zip` with a top-level `renders/` folder.

### `GET /artifacts/{session_id}/renders/{filename}`

Single render image.

**Response** `200` — `image/png`.

### `GET /artifacts/{session_id}/preview/{filename}`

Material preview image from the optional `render_previews` step.

**Response** `200` — `image/png`.

---

## Server-Sent Events (SSE)

### `GET /pipeline/{session_id}/events`

Stream real-time events for the pipeline. Standard SSE format:

```
event: step_started
data: {"step": "generate_textures", "started_at": "2026-04-21T19:14:23Z"}

event: step_progress
data: {"step": "generate_textures", "current": 2, "total": 8, "message": "Aluminum_Matte"}

event: step_completed
data: {"step": "apply_textures", "duration_seconds": 1.2, "stats": {"units": 8}}

event: pipeline_completed
data: {"status": "completed", "output_usd_url": "/artifacts/.../output"}
```

Clients should reconnect on disconnect; the `event-log` endpoint lets you replay missed events.

---

## Pipeline steps

The texture pipeline runs these steps, in order:

1. `prepare_uvs` — Prepare UV coordinates for geometry.
2. `discover_materials` — Discover and catalog materials in the scene.
3. `generate_prompts` — Auto-generate per-material texture prompts via the configured LLM for any material not covered in `material_textures_json`. Falls back to a templated prompt (`"{user_prompt}, applied to {material_name}"`) when the LLM is unavailable.
4. `render_previews` — Render preview images of the current scene (opt-in; disabled by default).
5. `generate_textures` — Generate albedo + normal + roughness per material via the configured image-gen backend. Normal/roughness passes condition on the albedo for coherence, except on backends where conditioning is not supported (see note below).
6. `blend_textures` — Composite generated maps at the per-material opacity.
7. `apply_textures` — Attach the textures to the USD materials and write the output USD(Z).
8. `render` — Render the final textured asset (opt-in; disabled by default).

**Image-gen conditioning note.** The cloud `nim` image-gen backend is text-only and drops reference images, so `generate_textures` produces text-conditioned normal/roughness without albedo guidance when that backend is active. For tightly-coupled PBR sets switch `TA_IMAGE_GEN_BACKEND` to `gemini` or `openai`, or run `--profile image-gen` in docker compose to route through the local FLUX.2 NIM sidecar (which does expose `images.edit`). The pipeline logs a one-line warning at the start of `generate_textures` whenever the active backend lacks conditioning support.

### Texture modes

- `per_material` (default) — one texture set per material, shared across every geometry referencing it.
- `per_prim` — clones materials per geometry prim so each mesh gets unique textures.

---

## Configuration

Environment variables read at startup (prefix `TA_`). Place them in either `<repo-root>/.env` (shared with other agent services) or `apps/texture_agent_service/.env` (service-local; wins on duplicate keys).

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | Auth for `nim` image-gen and `nim` chat (build.nvidia.com). |
| `OPENAI_API_KEY` | — | Auth for `openai` image-gen. Any value accepted when routing at a local NIM sidecar via `TA_IMAGE_GEN_BASE_URL`. |
| `GOOGLE_API_KEY` | — | Auth for `gemini` image-gen. |
| `TA_IMAGE_GEN_BACKEND` | `nim` | `nim` / `gemini` / `openai`. |
| `TA_IMAGE_GEN_MODEL` | (backend default) | Override the image-gen model. |
| `TA_IMAGE_GEN_BASE_URL` | — | Override image-gen base URL; used by the multi-gpu overlay to route at the local FLUX sidecar. |
| `TA_LLM_BACKEND` | `nim` | LLM backend for auto-prompt generation. |
| `TA_LLM_MODEL` | `qwen/qwen3.5-397b-a17b` | LLM model. |
| `TA_LLM_BASE_URL` | — | Override LLM base URL; set by the overlay when running `--profile llm`. |
| `TA_TEXTURE_SIZE` | `1024` | Output texture resolution. |
| `TA_TEXTURE_WORKERS` | `4` | Parallel texture generation workers. |
| `TA_BLEND_OPACITY` | `0.85` | Default per-material blend opacity. |
| `TA_SESSION_STORAGE_PATH` | `/var/texture-agent/sessions` | Session storage root. |
| `TA_SESSION_TTL_HOURS` | `24` | Session expiry. |
| `TA_MAX_ACTIVE_SESSIONS` | `4` | Max concurrent pipelines. |
| `TA_CANCEL_DRAIN_TIMEOUT_SECONDS` | `30.0` | Seconds cancellation waits for a synchronous worker thread to stop before marking the session failed with a stalled-worker deletion guard. |
| `TA_MAX_UPLOAD_SIZE_MB` | `500` | Max upload size for `/pipeline/upload-usd`. |

See [`../../.claude/skills/deploy-texture-agent-docker/SKILL.md`](../../.claude/skills/deploy-texture-agent-docker/SKILL.md) for the full docker-compose deployment recipe.
