# Texture Agent Service

FastAPI service for AI-driven texture generation on materialized USD assets. Wraps the [Texture Agent](../texture_agent/) pipeline behind a REST API with session management, async progress streaming via Server-Sent Events (SSE), and Docker-ready deployment.

## Default backends (product-default change)

The service ships with an NVIDIA-first default stack:

| Role | Default | Env var | Model |
|---|---|---|---|
| Image generation | `nim` | `TA_IMAGE_GEN_BACKEND` | `black-forest-labs/flux_2-klein-4b` (build.nvidia.com) |
| Auto-prompt LLM  | `nim` | `TA_LLM_BACKEND`       | `qwen/qwen3.5-397b-a17b` (build.nvidia.com) |

Both honor `NVIDIA_API_KEY`. One key unlocks the whole default path.

> **PBR coherence trade-off — read before deploying.**
>
> The cloud `nim` image-gen endpoint does not accept reference images, so
> the normal- and roughness-map passes (which otherwise condition on the
> generated albedo) run text-only on the default path. The pipeline still
> produces a full PBR set, but the normal/roughness maps are less coherent
> with the albedo than on a conditioning-capable backend. The pipeline
> logs one warning per run to the service stdout (visible via
> `docker logs texture-agent-service`) so operators can tell at a glance
> whether a given run went through the text-only path.
>
> To keep full PBR coherence:
> - run `docker compose --profile image-gen` (local FLUX.2 NIM sidecar —
>   same model, but exposes `images.edit` and supports conditioning), **or**
> - set `TA_IMAGE_GEN_BACKEND=gemini` or `TA_IMAGE_GEN_BACKEND=openai`
>   (both support img2img, but leave the NVIDIA-only stack).
>
> See `.claude/skills/deploy-texture-agent-docker/SKILL.md` for the full
> deployment matrix.

## Quick Start (Docker)

Requires **Docker Compose v2.24+** (for `env_file: required: false` support).

```bash
# From the repo root -- set your image-gen provider key
# (NIM or Gemini). The compose file reads .env at the repo root
# via env_file.
echo 'NVIDIA_API_KEY=your_key' > .env

# Build and run. `--env-file .env` is required so that any `${VAR}`
# overrides in compose (e.g. `TA_IMAGE_GEN_BACKEND=gemini`) read from
# the repo-root `.env`. Without it, Compose's variable substitution
# looks for `.env` next to the compose file
# (`apps/texture_agent_service/.env`) and silently falls back to the
# built-in defaults.
docker compose --env-file .env \
  -f apps/texture_agent_service/docker-compose.yml up --build

# Service available at http://localhost:8001
```

Unlike the material and physics services, the texture service does not bundle a GPU rendering sidecar — texture generation runs against the configured image-gen backend. Cold start is fast (no GPU warm-up step).

## Quick Start (Local Dev)

```bash
# From repo root
source .venv/bin/activate

# Install
uv pip install -e ".[dev]"
uv pip install -e apps/texture_agent -e apps/texture_agent_service

# Configure
cp .env_example .env
# Edit .env to set NVIDIA_API_KEY or GOOGLE_API_KEY

# Run
texture-agent-service
# or: uvicorn service.main:app --host 0.0.0.0 --port 8001
```

## API

- **Interactive docs:** http://localhost:8001/docs (Swagger UI) once the service is running.
- **Full reference:** [`docs/api.md`](docs/api.md).
- **OpenAPI spec:** [`openapi.yaml`](openapi.yaml).

The pipeline endpoints (`POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, etc.) accept a materialized USD file (typically the output of the Material Agent) and a per-material texture prompt map, then run the texture discovery / generation / apply pipeline. Stream real-time progress over SSE at `GET /pipeline/{id}/events`. Download textured output USDZ and textures via `/artifacts/{id}/output` and `/artifacts/{id}/textures`.

### Session Cleanup

Long-lived deployments should delete sessions after downloading required artifacts so session storage does not grow indefinitely:

```bash
curl -X DELETE http://localhost:8001/sessions/$SESSION_ID
```

`DELETE /sessions/{session_id}` returns `204 No Content` when the session, stored artifacts, and in-memory progress state are removed. It returns JSON `404 Not Found` when the session does not exist, and JSON `409 Conflict` when a live pipeline job is still active or a worker lock shows artifact writes are still in progress; cancel the pipeline and wait for the worker to stop before deleting it. If a service restart leaves a persisted `cancelling` status with no live worker lock, deletion is allowed so stale artifacts can be cleaned up.

### Artifact Response Types

The `/artifacts/{session_id}/...` routes use per-kind response media types:

| Endpoint | Success media type | Payload |
|----------|--------------------|---------|
| `GET /artifacts/{session_id}/materials` | `application/json` | Discovered material metadata |
| `GET /artifacts/{session_id}/textures` | `application/zip` | ZIP containing generated textures under `textures/` |
| `GET /artifacts/{session_id}/textures/{filename}` | `image/png` | Single texture image |
| `GET /artifacts/{session_id}/output` | `model/vnd.usdz+zip` | Self-contained textured USDZ |
| `GET /artifacts/{session_id}/renders` | `application/zip` | ZIP containing final rendered images under `renders/` |
| `GET /artifacts/{session_id}/renders/{filename}` | `image/png` | Single render image |
| `GET /artifacts/{session_id}/preview/{filename}` | `image/png` | Single material preview image |

Error responses, including missing artifacts, are JSON.

## Python Client

```python
from client.client import TextureAgentClient

client = TextureAgentClient("http://localhost:8001")

# Upload and run
session_id, status = client.run_and_monitor(
    usd_path="scene.usd",
    material_textures={
        "Steel_Carbon": {"prompt": "rusted steel", "opacity": 0.85},
    },
)

# Download artifacts
client.download_output(session_id, "output.usdz")
client.download_textures(session_id, "./textures/")

# Delete the session after required artifacts are downloaded
client.delete_session(session_id)
```

## Configuration

Service configuration is loaded from environment variables at startup. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | - | API key for NIM image generation |
| `GOOGLE_API_KEY` | - | API key for Gemini image generation |
| `TA_TEXTURE_BACKEND` | `simple_image_gen` | Texture gen backend |
| `TA_IMAGE_GEN_BACKEND` | `nim` | Image gen backend (`nim`, `gemini`, `openai`) |
| `TA_TEXTURE_SIZE` | `1024` | Texture resolution |
| `TA_TEXTURE_WORKERS` | `4` | Parallel gen workers |
| `TA_BLEND_OPACITY` | `0.85` | Default blend opacity |
| `TA_SESSION_STORAGE_PATH` | `/var/texture-agent/sessions` | Session storage |
| `TA_SESSION_TTL_HOURS` | `24` | Session expiry |
| `TA_MAX_ACTIVE_SESSIONS` | `4` | Max concurrent pipelines |
| `TA_CANCEL_DRAIN_TIMEOUT_SECONDS` | `30.0` | Seconds a cancelled request waits for a synchronous worker thread to stop before marking the session failed with a stalled-worker deletion guard |
| `TA_MAX_UPLOAD_SIZE_MB` | `500` | Max USD upload size |

## Project Structure

```
texture_agent_service/
├── client/                     # Python client (client.py)
├── docs/                       # Documentation (api.md REST reference)
├── service/                    # FastAPI app, routers, runtime, storage
├── tests/                      # Test suite
├── docker-compose.yml          # Docker Compose
├── Dockerfile                  # Service image
├── openapi.yaml                # API specification
└── pyproject.toml              # Install metadata
```
