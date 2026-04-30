# Physics Agent Service

FastAPI service for VLM-based physics property classification of 3D USD assets. Wraps the [Physics Agent](../physics_agent/) pipeline behind a REST API with session management, async progress streaming, and Docker-ready deployment.

## Quick Start (Docker)

Requires **Docker Compose v2.24+** (for `env_file: required: false` support).

```bash
# From the repo root -- set your VLM provider key
# (NIM, OpenAI, Anthropic, or Gemini). The compose file reads .env at
# the repo root via env_file.
echo 'NVIDIA_API_KEY=your_key' > .env

# Build and run (pulls in OVRTX rendering as a sidecar). `--env-file .env`
# is required so that any `${VAR}` overrides in compose (e.g.
# `PA_VLM_BACKEND=openai`) read from the repo-root `.env`. Without it,
# Compose's variable substitution looks for `.env` next to the compose
# file (`apps/physics_agent_service/.env`) and silently falls back to
# the built-in defaults.
docker compose --env-file .env \
  -f apps/physics_agent_service/docker-compose.yml up --build

# Service available at http://localhost:8000
```

The bundled `ovrtx-rendering-api` sidecar has a cold-start GPU warm-up phase.
Expect `physics-agent-service` to stay blocked for roughly 5 minutes until the
sidecar health check flips to `gpu_initialized=true`.

## Quick Start (Local Dev)

```bash
# From repo root
source .venv/bin/activate

# Install
uv pip install -e ".[dev]"
uv pip install -e apps/physics_agent -e apps/physics_agent_service

# Configure
cp .env_example .env
# Edit .env to set NVIDIA_API_KEY (or another VLM provider key)

# Run
cd apps/physics_agent_service
uvicorn service.main:app --reload --port 8000
```

## API

- **Interactive docs:** http://localhost:8000/docs (Swagger UI) once the service is running.
- **Full reference:** [`docs/api.md`](docs/api.md).
- **OpenAPI spec:** [`openapi.yaml`](openapi.yaml).

The pipeline endpoints (`POST /pipeline`, `GET /pipeline/{id}/status`, etc.) accept a USD file — either uploaded directly or referenced by S3 URI — then run the multi-step classification pipeline (optimize, identify asset, render, build dataset, predict, apply physics). Stream real-time progress over SSE at `GET /pipeline/{id}/events`.

## Python Client

See [`client/README.md`](client/README.md) for the bundled Python client, which supports both local file upload and S3 URI input modes. Example:

```bash
# Local file upload
python apps/physics_agent_service/client/client.py /path/to/scene.usdz

# S3 URI (service downloads server-side)
python apps/physics_agent_service/client/client.py \
  --s3-uri s3://your-bucket/path/to/scene.usdz
```

## Configuration

Service configuration is loaded from environment variables at startup. Key settings:

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | Required if using `nim` VLM backend |
| `OPENAI_API_KEY` | Required if using `openai` backend |
| `ANTHROPIC_API_KEY` | Required if using `anthropic` backend |
| `GOOGLE_API_KEY` | Required if using `gemini` backend |
| `PA_VLM_BACKEND` | Default: `nim` |
| `PA_VLM_MODEL` | Default: `qwen/qwen3.5-397b-a17b` |
| `PA_RENDER_BACKEND` | Default: `remote` (resolves via `RENDER_ENDPOINT`) |
| `RENDER_ENDPOINT` | URL of OVRTX rendering API or compatible service |
| `PA_SESSION_STORAGE_PATH` | Where session directories are written |
| `PA_MAX_UPLOAD_SIZE_MB` | Max USD upload size (default: 500) |

## Architecture

```
Upload USD → Session Created → Pipeline Runs → Download Output
                                    ↓
                            (SSE progress events)
                                    ↓
                            Per-component classification
```

Pipeline steps run in order:

1. `optimize_usd` — Flatten/deinstance via scene optimizer when enabled (`optimize_usd`, `enable_deinstance`, `enable_split`, `enable_deduplicate` form flags)
2. `identify_asset` — Preview-render whole asset, VLM identifies asset type
3. `build_dataset_usd` — Render per-prim views for VLM input
4. `build_dataset_prepare_dataset` — Compose dataset with classification specs
5. `predict` — VLM inference for per-component classification (type, material, physics)
6. `restore_usd` — Map optimized prediction paths back to original paths when `optimize_usd` is enabled.
7. `apply_physics` — Flatten target USD and author `UsdPhysics.RigidBodyAPI` / `CollisionAPI` / `MassAPI` / `MaterialAPI` on each predicted prim plus a `PhysicsScene`. When optimization ran, physics is authored on the optimized/deinstanced USD so instance-proxy descendants are writable. Downloadable via `GET /artifacts/{id}/output-usd` as `scene_physics.usda`.

## Project Structure

```
physics_agent_service/
├── client/                     # Python client (file-upload + S3 modes)
├── docs/                       # Documentation (api.md REST reference)
├── service/                    # FastAPI app, routers, runtime, storage
├── tests/                      # Test suite
├── docker-compose.yml          # Docker Compose (service + OVRTX sidecar)
├── Dockerfile                  # Service image
├── openapi.yaml                # API specification
└── pyproject.toml              # Install metadata
```
