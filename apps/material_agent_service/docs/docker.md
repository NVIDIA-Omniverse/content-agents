# Material Agent Service - Docker Compose Deployment

## Quick Start

### Prerequisites

- Docker Compose **v2.24+** (required for the `env_file: required: false` long-form syntax used by the compose file)
- NVIDIA GPU with 48GB+ VRAM (e.g., L40, L40S, A100)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- VLM provider API key (OpenAI, Anthropic, Gemini, or NVIDIA)

### Setup

```bash
# From the repo root -- create .env with your VLM provider key.
# The compose file reads this .env via env_file: ../../.env.
echo 'OPENAI_API_KEY=sk-...' > .env

# Build and start (main service + OVRTX rendering)
docker compose -f apps/material_agent_service/docker-compose.yml up --build
```

That's it. No NGC login needed -- the OVRTX rendering API builds from source.
Expect the bundled `ovrtx-rendering-api` sidecar to spend roughly 5 minutes in
GPU warm-up on a cold start. During that time `/health` on port `8001` returns
`gpu_initialized=false`, and `material-agent-service` will wait to start.

### Access

- **Health**: http://localhost:8000/health
- **API Docs** (Swagger UI): http://localhost:8000/docs
- **Rendering API Health**: http://localhost:8001/health

## Services

| Service | Port | GPU | Build | Description |
|---|---|---|---|---|
| material-agent-service | 8000 | No | From source | Main service (pipeline + REST API) |
| ovrtx-rendering-api | 8001 | 1x | From source | OVRTX-based USD rendering |
| vlm-nim (optional) | 8003 | 1x | NGC image | Local Cosmos Reason2 8B VLM |

## Usage

```bash
# Main + rendering (default, no NGC needed)
docker compose -f apps/material_agent_service/docker-compose.yml up --build

# Add local VLM NIM (requires NGC login)
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
docker compose -f apps/material_agent_service/docker-compose.yml --profile vlm up --build

# Detached mode
docker compose -f apps/material_agent_service/docker-compose.yml up --build -d

# View logs
docker compose -f apps/material_agent_service/docker-compose.yml logs -f

# Stop
docker compose -f apps/material_agent_service/docker-compose.yml down

# Stop and remove session data
docker compose -f apps/material_agent_service/docker-compose.yml down -v
```

## VLM Provider Configuration

Set one of these in `.env` (or export before running):

| Provider | Environment Variable |
|---|---|
| NVIDIA (build.nvidia.com) | `NVIDIA_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

Generated reference images use `MA_IMAGE_GEN_BACKEND`, which defaults to
`gemini` and requires `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Set
`MA_IMAGE_GEN_BACKEND=openai` with `OPENAI_API_KEY`, or set
`MA_IMAGE_GEN_BACKEND=nim` with `NVIDIA_API_KEY`. For a no-auth local
OpenAI-compatible image endpoint, set `MA_IMAGE_GEN_BASE_URL` and explicitly set
`MA_IMAGE_GEN_API_KEY=not-used`.

When using the `vlm` profile (local Cosmos VLM NIM), set `NGC_API_KEY` for model weight download. The VLM NIM takes ~15 minutes to start on first run (model compilation).

## Resource Requirements

| Configuration | GPUs | CPU | Memory |
|---|---|---|---|
| Main + rendering (default) | 1 | 10 | 20G |
| + VLM NIM | 2 | 16 | 56G |

### GPU Notes

- OVRTX rendering needs 1 GPU with 48GB VRAM
- VLM NIM needs 1 additional GPU with 48GB VRAM
- On 48GB GPUs (L40/L40S), VLM NIM uses `NIM_MAX_MODEL_LEN=131072` by default

## Startup Timeline

| Service | Ready |
|---|---|
| material-agent-service | ~1 min |
| ovrtx-rendering-api | ~5 min on cold start before `gpu_initialized=true` |
| vlm-nim | ~15 min (model compilation) |

## Troubleshooting

### OVRTX rendering fails

```bash
# Check rendering API logs
docker logs ovrtx-rendering-api

# Verify GPU is visible
docker exec ovrtx-rendering-api nvidia-smi
```

If you see shader compilation messages, wait -- first-run compilation takes ~5 minutes.
The main service will not become reachable until the OVRTX sidecar health check
passes.

### VLM NIM KV cache error

If VLM NIM crashes with KV cache memory error on 48GB GPUs:

```bash
# Already set to 131072 by default in docker-compose.yml
# To override:
NIM_MAX_MODEL_LEN=65536 docker compose --profile vlm up
```

### Out of memory

Reduce resource limits in `docker-compose.yml` or allocate more GPUs:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # more GPUs
          capabilities: [gpu]
```

### Rebuild after code changes

```bash
docker compose -f apps/material_agent_service/docker-compose.yml up --build

# Force full rebuild (no cache)
docker compose -f apps/material_agent_service/docker-compose.yml build --no-cache
docker compose -f apps/material_agent_service/docker-compose.yml up
```
