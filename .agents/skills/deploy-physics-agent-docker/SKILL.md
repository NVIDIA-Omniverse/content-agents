---
name: deploy-physics-agent-docker
description: Deploy the physics-agent-service locally using Docker Compose with the bundled OVRTX GPU rendering sidecar. Use when user wants to run physics agent with docker, docker compose, set up local deployment of the physics service, run it on a GPU box, start physics agent containers, or configure the VLM provider for physics docker deployment. Trigger phrases include "deploy physics agent", "docker compose physics", "run physics agent locally", "start physics service docker", "physics compose up", "physics agent docker".
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - physics-agent
  - docker
  - ovrtx
  - deployment
tools:
  - Shell
  - Docker
  - curl
  - Python
  - Filesystem
compatibility: Requires Docker daemon, Docker Compose v2.24+, NVIDIA Container Toolkit, an NVIDIA GPU with about 16GB+ VRAM for the OVRTX sidecar, repo-root .env provider credentials, and free host ports 8000/8001.
---

# Deploy Physics Agent Service with Docker Compose

Deploy the `physics-agent-service` and the bundled OVRTX rendering API locally using Docker Compose. The physics service is CPU-only; the rendering sidecar uses the GPU.

## When to Use

- Use when the user wants to run `physics-agent-service` locally with Docker Compose.
- Use when the user needs the bundled OVRTX rendering sidecar for physics classification.
- Use when the user wants to configure VLM provider credentials or run local smoke requests with optimizer flags.
- Use `quickstart` for a shorter first local POC, and use `deploy-collection` when running multiple Content Agents together.

## Limitations

- The default stack owns host ports 8000 and 8001. Stop overlapping Material, Physics, Texture, or standalone OVRTX stacks before startup.
- The main service waits on OVRTX readiness; OVRTX is ready only when `/health` reports `gpu_initialized: true`.
- First build and first render are long-running operations. Return logs and health commands rather than holding an agent session open indefinitely.
- Keep secrets out of chat and commits. Tell the user to edit `.env`; do not ask them to paste keys.

## Prerequisites

Check before deploying:

1. **Docker Compose v2.24+**: `docker compose version` -- required for `env_file: required: false` long-form syntax
2. **NVIDIA GPU** with ~16 GB+ VRAM: `nvidia-smi`
3. **NVIDIA Container Toolkit** installed: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
4. **VLM provider API key** (at least one): NVIDIA NIM, OpenAI, Anthropic, or Gemini

## Instructions

1. Confirm Docker, Compose, GPU, NVIDIA Container Toolkit, and port availability before starting the stack.
2. Create or update the repo-root `.env` with exactly the VLM provider credentials the selected backend needs.
3. Start the Physics Agent compose stack from the repo root.
4. Wait for both the main service and OVRTX readiness checks before reporting the service ready.
5. For optimizer-sensitive smoke assets, use the optimizer form fields below.
6. Return service URLs, health state, log commands, and stop commands using the output format below.

### Set VLM API Key

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

### Start Services

```bash
docker compose -f apps/physics_agent_service/docker-compose.yml up --build
```

This starts:
- **physics-agent-service** on port 8000 (REST API)
- **ovrtx-rendering-api** on port 8001 (GPU rendering, built from source)

First build takes ~10 minutes. First render takes ~5 minutes (shader compilation; cached after).

### Access

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
python - <<'PY'
import json
from urllib.request import urlopen

try:
    with urlopen("http://localhost:8001/health", timeout=10) as response:
        health = json.load(response)
except Exception as exc:
    print(f"rendering API unreachable: {exc}")
    raise SystemExit(1)

print(json.dumps(health))
if health.get("status") == "unhealthy":
    print("rendering API unhealthy")
    raise SystemExit(1)
if health.get("gpu_initialized") is True:
    print("rendering API ready")
else:
    print("rendering API warming")
PY
```

### REST Smoke with Optimizer Flags

For ordinary smoke assets, run the pipeline without optimizer flags. For
instanced USDs, assets that fail `apply_physics` with an instance-proxy authoring
error, or one combined mesh that needs split-by-component predictions, pass the
optimizer form fields to `POST /pipeline`:

```bash
# Instance-proxy authoring fix.
curl -X POST "http://localhost:8000/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "optimize_usd=true" \
  -F "enable_deinstance=true"

# Also split one combined disjoint mesh into separate component predictions.
curl -X POST "http://localhost:8000/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "optimize_usd=true" \
  -F "enable_deinstance=true" \
  -F "enable_split=true"
```

Use `enable_deduplicate=true` only when repeated identical geometry should be
collapsed. At least one optimizer operation must be enabled when
`optimize_usd=true`.

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
| `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` | `1` | Process-wide outbound render request cap for the local OVRTX sidecar |
| `OVRTX_NUM_SENSOR_UPDATES` | `500` | Sensor update count before capture (rendering sidecar) |

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

## Output Format

When handing control back to the user, report:

- `SERVICE_URL`: `http://localhost:8000`
- `DOCS_URL`: `http://localhost:8000/docs`
- `SERVICE_HEALTH`: `healthy`, `starting`, or `unhealthy`
- `OVRTX_HEALTH`: `healthy` only when `/health` contains `"gpu_initialized":true`; otherwise `warming` or `unhealthy`
- `LOGS`: `docker compose -f apps/physics_agent_service/docker-compose.yml logs -f`
- `STOP`: `docker compose -f apps/physics_agent_service/docker-compose.yml down`
- Any missing credentials, port conflicts, GPU/toolkit blockers, or optimizer flags used for smoke validation.

## Troubleshooting

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
