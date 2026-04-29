---
name: deploy-texture-agent-docker
description: Deploy the texture-agent-service locally using Docker Compose, with an optional FLUX.2 NIM image-gen sidecar. Use when user wants to run texture agent with docker, docker compose, set up local deployment, run the service locally, start texture agent containers, or route image generation to a locally-hosted NIM. Trigger phrases include "docker compose texture", "docker deploy texture", "run texture agent locally", "start texture agent docker", "texture agent up", "local texture deployment", "image-gen sidecar".
---

# Deploy Texture Agent Service with Docker Compose

Deploy the texture-agent-service locally using Docker Compose. The default setup is CPU-only and talks to NVIDIA's hosted FLUX.2 image-gen at build.nvidia.com (backend `nim`, requires `NVIDIA_API_KEY`). Opt into `--profile image-gen` to launch a local FLUX.2 NIM sidecar instead and route image generation through it — no image-gen API key required.

> **PBR coherence note.** The cloud `nim` endpoint is text-only and cannot accept reference images, so normal/roughness passes are generated from text alone (less coherent with the albedo). For tightly-matched PBR sets, switch `TA_IMAGE_GEN_BACKEND` to `gemini` or `openai` — or use `--profile image-gen` which routes through the local FLUX.2 NIM sidecar's `images.edit` endpoint (img2img-capable). The pipeline logs a warning when the active backend can't accept conditioning images.

## Prerequisites

Check before deploying:

1. **Docker 20.10+** with Compose v2: `docker compose version`
2. For the default (cloud) setup: `NVIDIA_API_KEY` for the hosted FLUX.2 NIM at build.nvidia.com (or switch `TA_IMAGE_GEN_BACKEND` to `gemini`/`openai` and supply the matching key)
3. For the `--profile image-gen` setup:
   - **NVIDIA GPU** with ~24GB+ VRAM free (FLUX.2 Klein 4B fits in ~13GB; leave headroom): `nvidia-smi`
   - **NVIDIA Container Toolkit**: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
   - **NGC API key** (for pulling `nvcr.io/nim/*` images and fetching weights)
   - **Hugging Face token** (FLUX weights download)

## Quick Start — Cloud Image-Gen (default)

### Step 1: Set an image-gen provider key

The compose file picks up either the repo-root `.env` (`<repo>/.env`, shared across agent services) or a service-specific one (`apps/texture_agent_service/.env`, wins on duplicate keys). Either works:

```bash
# Default backend `nim` targets NVIDIA's hosted FLUX.2 Klein 4B at build.nvidia.com:
echo 'NVIDIA_API_KEY=nvapi-...' > apps/texture_agent_service/.env

# To switch providers, also set TA_IMAGE_GEN_BACKEND:
echo 'GOOGLE_API_KEY=...' >> apps/texture_agent_service/.env
echo 'TA_IMAGE_GEN_BACKEND=gemini' >> apps/texture_agent_service/.env
# OR
echo 'OPENAI_API_KEY=sk-...' >> apps/texture_agent_service/.env
echo 'TA_IMAGE_GEN_BACKEND=openai' >> apps/texture_agent_service/.env
```

The same `NVIDIA_API_KEY` is also used by the default LLM backend (`nim` + `qwen/qwen3.5-397b-a17b`) for auto-prompt generation.

### Step 2: Start Services

```bash
docker compose -f apps/texture_agent_service/docker-compose.yml up --build
```

This starts:
- **texture-agent-service** on port 8001 (REST API + web client)

### Step 3: Access

- **Health**: http://localhost:8001/health
- **API docs**: http://localhost:8001/docs
- **Web client**: serve `apps/texture_agent_service/client/` against the API

## Adding Local NIM Sidecars

Two optional sidecars can replace the cloud calls for image generation and/or LLM auto-prompt generation:

- **`image-gen-nim`** — FLUX.2 Klein 4B (`nvcr.io/nim/black-forest-labs/flux.2-klein-4b:1.0.0-variant`). Profile: `image-gen`. Warmup ~90 s.
- **`llm-nim`** — Llama 3.1 Nemotron Nano 8B v1 (`nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-8b-v1:1.8.4`). Profile: `llm`. Text-only instruct model, much lighter than a VLM; warmup ~3-5 min on first start.

Both require NGC login. The multi-gpu overlay pins each sidecar to its own GPU (`image-gen-nim` → GPU 0, `llm-nim` → GPU 1 by default) and routes the main service's clients at the in-network sidecar endpoints.

```bash
# NGC login required for NIM image pulls
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"

# Add NGC_API_KEY + HF_TOKEN to .env for the sidecars
cat >> apps/texture_agent_service/.env <<'EOF'
NGC_API_KEY=...
HF_TOKEN=hf_...
EOF

# Enable image-gen sidecar only
docker compose \
  -f apps/texture_agent_service/docker-compose.yml \
  -f apps/texture_agent_service/docker-compose.multi-gpu.yml \
  --profile image-gen up --build

# Enable LLM sidecar only
docker compose \
  -f apps/texture_agent_service/docker-compose.yml \
  -f apps/texture_agent_service/docker-compose.multi-gpu.yml \
  --profile llm up --build

# Enable both (requires 2 free GPUs)
docker compose \
  -f apps/texture_agent_service/docker-compose.yml \
  -f apps/texture_agent_service/docker-compose.multi-gpu.yml \
  --profile image-gen --profile llm up --build
```

Edit `device_ids` in `docker-compose.multi-gpu.yml` if GPU 0 or GPU 1 isn't free on your host (e.g. when co-running material-agent). The `llm-nim` sidecar honors `NIM_MAX_MODEL_LEN` (default 131072 for 48 GB GPUs) — drop it for smaller cards.

## Services

| Service | Port | GPU | Builds From | Always Starts |
|---|---|---|---|---|
| texture-agent-service | 8001 | No | Source | Yes |
| image-gen-nim | 8005 | 1× (24 GB+) | NGC image | No (profile `image-gen`) |
| llm-nim | 8006 | 1× (48 GB) | NGC image | No (profile `llm`) |

## Operations

### View logs

```bash
# All services
docker compose -f apps/texture_agent_service/docker-compose.yml logs -f

# Specific service
docker logs texture-agent-service
docker logs image-gen-nim
docker logs llm-nim
```

### Stop

```bash
# Stop all services (keeps session data)
docker compose -f apps/texture_agent_service/docker-compose.yml down

# Stop and remove session data
docker compose -f apps/texture_agent_service/docker-compose.yml down -v
```

### Rebuild after code changes

```bash
docker compose -f apps/texture_agent_service/docker-compose.yml up --build

# Force full rebuild (no cache)
docker compose -f apps/texture_agent_service/docker-compose.yml build --no-cache
docker compose -f apps/texture_agent_service/docker-compose.yml up
```

### Check health

```bash
curl http://localhost:8001/health                   # main service
curl http://localhost:8005/v1/health/ready          # image-gen NIM (if running)
curl http://localhost:8006/v1/health/ready          # llm NIM (if running)
```

## Resource Requirements

| Configuration | GPUs | CPU | Memory |
|---|---|---|---|
| Default (cloud image-gen, cloud LLM) | 0 | 4 | 8 GB |
| + `--profile image-gen` (FLUX.2 sidecar) | 1 (24 GB+) | 8 | 24 GB |
| + `--profile llm` (Nemotron Nano 8B sidecar) | 1 (48 GB) | 8 | 24 GB |
| + both profiles | 2 (24 GB+ and 48 GB) | 12 | 40 GB |

## Environment Variables

All configurable via `.env` in `apps/texture_agent_service/`:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini provider (used when `TA_IMAGE_GEN_BACKEND=gemini`) |
| `OPENAI_API_KEY` | — | OpenAI backend (also used by the sidecar client when `TA_IMAGE_GEN_BACKEND=openai` — any value ok for local NIM) |
| `NVIDIA_API_KEY` | — | NVIDIA inference / cloud NIM backend |
| `NGC_API_KEY` | — | NGC auth (required for `--profile image-gen`) |
| `HF_TOKEN` | — | Hugging Face token (required for `--profile image-gen` weight download) |
| `TA_IMAGE_GEN_BACKEND` | `nim` | `nim`, `gemini`, or `openai` |
| `TA_IMAGE_GEN_MODEL` | (backend default) | Override the model name |
| `TA_IMAGE_GEN_BASE_URL` | (backend default) | Override base URL (set by multi-gpu overlay to point at the sidecar) |
| `TA_TEXTURE_SIZE` | 1024 | Output texture resolution |
| `TA_TEXTURE_WORKERS` | 4 | Parallel texture generation workers |
| `TA_BLEND_OPACITY` | 0.85 | Default blend opacity (0–1) |
| `TA_SESSION_TTL_HOURS` | 24 | Session expiry time |
| `TA_MAX_UPLOAD_SIZE_MB` | 500 | Max USD upload size |
| `TA_MAX_ACTIVE_SESSIONS` | 4 | Concurrent pipeline sessions |

## Common Issues

### `image-gen-nim` reports `Free GPUs: <None>`

NIM's profile selector considers a GPU "occupied" if any process holds VRAM on it, even if there's plenty free. Fix by pointing the sidecar at a truly-idle GPU:

```yaml
# docker-compose.multi-gpu.yml
image-gen-nim:
  environment:
    - NVIDIA_VISIBLE_DEVICES=2
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['2']
            capabilities: [gpu]
```

### `Application timeout caused pair closure` during warmup

FLUX.2 NIM running in a multi-GPU configuration can deadlock on the inter-GPU broadcast during warmup (A40 GPUs with no NVLink hit the 30-min torch.distributed timeout). Pin to a single GPU in the multi-gpu overlay (the default `device_ids: ['0']` already does this; just ensure `count: 1` in the base compose). Confirm by grepping logs for `num_gpus: 1, enable_cfg_parallel: False`.

### `texture-agent-service` ignores the sidecar

Check the `/health` endpoint and verify the env reached the container:

```bash
docker exec texture-agent-service env | grep TA_IMAGE_GEN
# Expect: TA_IMAGE_GEN_BACKEND=openai, TA_IMAGE_GEN_BASE_URL=http://image-gen-nim:8000/v1
```

If the main service was started without the multi-gpu overlay, `TA_IMAGE_GEN_BASE_URL` will be empty and the cloud backend is used. Re-run with both `-f` flags.

### Slow first request after restart

The sidecar does not mount a cache volume, so FLUX weights re-download on every fresh container. First `/v1/health/ready` takes ~90 s from cold start. If you need to restart frequently during development, add a bind-mount to the sidecar — `~/.cache/nim:/opt/nim/.cache/` with the host directory `chmod 1777` — to persist weights across container lifecycles.
