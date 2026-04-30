---
name: deploy-physics-agent-docker
description: Deploy the physics-agent-service locally using Docker Compose with the bundled OVRTX GPU rendering sidecar. Use when user wants to run physics agent with docker, docker compose, set up local deployment of the physics service, run it on a GPU box, start physics agent containers, or configure the VLM provider for physics docker deployment. Trigger phrases include "deploy physics agent", "docker compose physics", "run physics agent locally", "start physics service docker", "physics compose up", "physics agent docker".
---

# Deploy Physics Agent Service with Docker Compose

Deploy the `physics-agent-service` and the bundled OVRTX rendering API locally using Docker Compose. The physics service is CPU-only; the rendering sidecar uses the GPU.

## Prerequisites

Check before deploying:

1. **Docker Compose v2.24+**: `docker compose version` -- required for `env_file: required: false` long-form syntax
2. **NVIDIA GPU** with ~16 GB+ VRAM: `nvidia-smi`
3. **NVIDIA Container Toolkit** installed: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
4. **VLM provider API key** (at least one): NVIDIA NIM, OpenAI, Anthropic, or Gemini

## Quick Start

### Step 1: Set VLM API Key

Create `.env` at the **repo root** (the compose file reads it via `env_file: ../../.env`):

```bash
# Pick ONE provider:
echo 'NVIDIA_API_KEY=nvapi-...' > .env
# OR
echo 'OPENAI_API_KEY=sk-...' > .env
# OR
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
# OR
echo 'GOOGLE_API_KEY=...' > .env
```

### Step 2: Start Services

```bash
docker compose -f apps/physics_agent_service/docker-compose.yml up --build
```

This starts:
- **physics-agent-service** on port 8000 (REST API)
- **ovrtx-rendering-api** on port 8001 (GPU rendering, built from source)

First build takes ~10 minutes. First render takes ~5 minutes (shader compilation; cached after).

### Step 3: Access

- **Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **OpenAPI spec**: `apps/physics_agent_service/openapi.yaml`

## Services

| Service | Port | GPU | Builds From | Always Starts |
|---|---|---|---|---|
| physics-agent-service | 8000 | No | Source | Yes |
| ovrtx-rendering-api | 8001 | 1x | Source | Yes |

The main service `depends_on` the rendering API's health check passing (which flips `gpu_initialized` to `true`). On cold start expect the physics-agent container to sit in "waiting" state for ~5 minutes before it comes up.

## Operations

### View Logs

```bash
# All services
docker compose -f apps/physics_agent_service/docker-compose.yml logs -f

# Specific service
docker logs physics-agent-service
docker logs physics-ovrtx-rendering-api
```

### Stop

```bash
# Stop all services
docker compose -f apps/physics_agent_service/docker-compose.yml down

# Stop and remove session data
docker compose -f apps/physics_agent_service/docker-compose.yml down -v
```

### Rebuild After Code Changes

```bash
docker compose -f apps/physics_agent_service/docker-compose.yml up --build

# Force full rebuild (no cache)
docker compose -f apps/physics_agent_service/docker-compose.yml build --no-cache
docker compose -f apps/physics_agent_service/docker-compose.yml up
```

### Check Health

```bash
curl http://localhost:8000/health   # main service
curl http://localhost:8001/health   # rendering API (reports gpu_initialized)
```

## Resource Requirements

| Configuration | GPUs | CPU | Memory |
|---|---|---|---|
| Default (main + rendering) | 1 | 10 | 20 G |

## Environment Variables

Configurable via `.env` at the repo root. Key settings:

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | | NVIDIA (build.nvidia.com) VLM provider |
| `OPENAI_API_KEY` | | OpenAI VLM provider |
| `ANTHROPIC_API_KEY` | | Anthropic VLM provider |
| `GOOGLE_API_KEY` | | Google Gemini VLM provider |
| `PA_VLM_BACKEND` | `nim` | Which VLM backend to use |
| `PA_VLM_MODEL` | `qwen/qwen3.5-397b-a17b` | Model id for the selected backend |
| `PA_VLM_TEMPERATURE` | `1.0` | Sampling temperature |
| `PA_MAX_ACTIVE_SESSIONS` | `1` | Max concurrent pipelines |
| `PA_SESSION_TTL_HOURS` | `24` | Session expiry time |
| `PA_MAX_UPLOAD_SIZE_MB` | `500` | Max USD upload size |
| `OVRTX_NUM_SAMPLES` | `1` | Path tracer samples per frame (rendering sidecar) |

## GPU Configuration

To assign specific GPUs to the rendering API, edit
`apps/physics_agent_service/docker-compose.yml`:

```yaml
ovrtx-rendering-api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1          # number of GPUs
            capabilities: [gpu]
```

Or pin a specific GPU ID:

```yaml
            device_ids: ['0']
```

## Common Issues

### OVRTX rendering API not starting

Check GPU access:

```bash
docker logs physics-ovrtx-rendering-api
docker exec physics-ovrtx-rendering-api nvidia-smi
```

Shader compilation on first boot takes ~5 minutes; wait it out.

### Main service unhealthy before rendering API ready

The main service `depends_on` the rendering API's health check. If rendering takes long to start, the physics-agent-service container will stay in "waiting" state. Check `docker compose ps` to see which container is blocking.

### 503 / VLM failures under load

`PA_MAX_ACTIVE_SESSIONS` defaults to 1 because rendering plus a VLM call per prim is the main throughput bottleneck. Raising this requires headroom on both CPU memory and VLM provider quota.
