---
name: deploy-material-agent-docker
description: Deploy the material-agent-service locally using Docker Compose with OVRTX rendering. Use when user wants to run material agent with docker, docker compose, set up local deployment, run the service locally with GPU rendering, start material agent containers, or configure VLM provider for docker deployment. Trigger phrases include "docker compose", "docker deploy", "run locally with docker", "start material agent docker", "docker up", "local deployment", "compose up".
---

# Deploy Material Agent Service with Docker Compose

Deploy the material-agent-service and OVRTX rendering API locally using Docker Compose. No NGC login or Kubernetes required for the default setup.

## Prerequisites

Check before deploying:

1. **Docker Compose v2.24+**: `docker compose version` -- required for `env_file: required: false` long-form syntax
2. **NVIDIA GPU** with 48GB+ VRAM (L40, L40S, A100, etc.): `nvidia-smi`
3. **NVIDIA Container Toolkit** installed: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
4. **VLM provider API key** (at least one): OpenAI, Anthropic, Gemini, or NVIDIA
5. **Generated reference image key** (optional): public default is Gemini, so set `GOOGLE_API_KEY` if the user plans to use generated reference images. Otherwise set `MA_IMAGE_GEN_BACKEND` to another configured provider.

## Quick Start

### Step 1: Set VLM API Key

Create `.env` at the **repo root** (the compose file reads it via `env_file: ../../.env`):

```bash
# Pick ONE provider:
echo 'OPENAI_API_KEY=sk-...' > .env
# OR
echo 'NVIDIA_API_KEY=nvapi-...' > .env
# OR
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
# OR
echo 'GOOGLE_API_KEY=...' > .env
```

### Step 2: Start Services

```bash
docker compose -f apps/material_agent_service/docker-compose.yml up --build
```

This starts:
- **material-agent-service** on port 8000 (REST API)
- **ovrtx-rendering-api** on port 8001 (GPU rendering, built from source)

First build takes ~10 minutes. First render takes ~5 minutes (shader compilation, cached after).

### Step 3: Access

- **Health**: http://localhost:8000/health
- **API Docs** (Swagger UI): http://localhost:8000/docs
- **OpenAPI spec**: http://localhost:8000/openapi.json

## Services

| Service | Port | GPU | Builds From | Always Starts |
|---|---|---|---|---|
| material-agent-service | 8000 | No | Source | Yes |
| ovrtx-rendering-api | 8001 | 1x (48GB) | Source | Yes |
| vlm-nim | 8003 | 1x (48GB) | NGC image | No (profile) |

## Adding Local VLM NIM

To run Cosmos Reason2 8B locally instead of using a cloud VLM API:

```bash
# NGC login required for VLM NIM image
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY

# Add NGC_API_KEY to .env for model weight download
echo 'NGC_API_KEY=...' >> .env

# Start with vlm profile + multi-gpu overlay
# The overlay pins ovrtx-rendering-api to GPU 0 and vlm-nim to GPU 1.
# It is REQUIRED here: NIM's profile selector reports "Free GPUs: <None>"
# and refuses to launch when ovrtx is also holding a GPU, even with tens
# of GB free. See docker-compose.multi-gpu.yml for the pinning details.
docker compose \
  -f apps/material_agent_service/docker-compose.yml \
  -f apps/material_agent_service/docker-compose.multi-gpu.yml \
  --profile vlm up --build
```

Requires 2 GPUs (48GB each). VLM NIM takes ~15 minutes to start (model compilation).

The multi-gpu overlay also routes `MA_VLM_*` and `MA_LLM_*` to the local
vlm-nim sidecar (`nvidia/cosmos-reason2-8b`), so no extra `.env` edits
are needed for `--profile vlm`.

## Operations

### View Logs

```bash
# All services
docker compose -f apps/material_agent_service/docker-compose.yml logs -f

# Specific service
docker logs material-agent-service
docker logs ovrtx-rendering-api
```

### Stop

```bash
# Stop all services
docker compose -f apps/material_agent_service/docker-compose.yml down

# Stop and remove session data
docker compose -f apps/material_agent_service/docker-compose.yml down -v
```

### Rebuild After Code Changes

```bash
docker compose -f apps/material_agent_service/docker-compose.yml up --build

# Force full rebuild (no cache)
docker compose -f apps/material_agent_service/docker-compose.yml build --no-cache
docker compose -f apps/material_agent_service/docker-compose.yml up
```

### Check Health

```bash
curl http://localhost:8000/health   # main service
curl http://localhost:8001/health   # rendering API
```

## Resource Requirements

| Configuration | GPUs | CPU | Memory |
|---|---|---|---|
| Default (main + rendering) | 1 | 10 | 20G |
| + VLM NIM | 2 | 16 | 56G |

## GPU Configuration

To assign more GPUs to the rendering API, edit `docker-compose.yml`:

```yaml
ovrtx-rendering-api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 2          # number of GPUs
            capabilities: [gpu]
```

Or assign specific GPU IDs:

```yaml
            device_ids: ['0', '1']
```

## Environment Variables

All configurable via `.env` file at the repo root:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | | OpenAI VLM provider |
| `ANTHROPIC_API_KEY` | | Anthropic VLM provider |
| `GOOGLE_API_KEY` | | Google Gemini VLM provider |
| `NVIDIA_API_KEY` | | NVIDIA (build.nvidia.com) VLM provider |
| `NGC_API_KEY` | | NGC auth (only for VLM NIM profile) |
| `MA_IMAGE_GEN_BACKEND` | `gemini` | Generated reference image backend (`gemini`, `openai`, `nim`) |
| `MA_IMAGE_GEN_MODEL` | | Optional generated reference image model override |
| `MA_IMAGE_GEN_BASE_URL` | | Optional image-gen API base URL override |
| `MA_MAX_WORKERS` | 16 | Parallel pipeline workers |
| `MA_SESSION_TTL_HOURS` | 24 | Session expiry time |
| `MA_MAX_UPLOAD_SIZE_MB` | 500 | Max USD upload size |
| `OVRTX_NUM_SAMPLES` | 1 | Path tracer samples per frame |
| `NIM_MAX_MODEL_LEN` | 131072 | VLM NIM context length (for 48GB GPUs) |

Generated reference images are optional. If enabled from the client or UI, the
service uses `MA_IMAGE_GEN_BACKEND=gemini` by default.

## Common Issues

### OVRTX rendering API not starting

Check GPU access:
```bash
docker logs ovrtx-rendering-api
docker exec ovrtx-rendering-api nvidia-smi
```

If shader compilation messages appear, wait ~5 minutes (first-time only).

### Main service unhealthy before rendering API ready

The main service `depends_on` the rendering API health check. If rendering takes long to start (shader compilation), increase `start_period` in `docker-compose.yml`.

### VLM NIM KV cache error on 48GB GPUs

Already configured with `NIM_MAX_MODEL_LEN=131072` by default. To reduce further:

```bash
NIM_MAX_MODEL_LEN=65536 docker compose --profile vlm -f apps/material_agent_service/docker-compose.yml up
```
