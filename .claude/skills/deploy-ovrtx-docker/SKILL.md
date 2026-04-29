---
name: deploy-ovrtx-docker
description: Deploy just the OVRTX rendering API standalone via Docker Compose so one or more agent CLIs / services can point at it as a shared rendering endpoint. Use when user wants to run OVRTX standalone, deploy a shared rendering service, separate the rendering endpoint from an agent service, expose OVRTX to multiple callers, or set RENDER_ENDPOINT to a dedicated render host. Trigger phrases include "deploy ovrtx", "run ovrtx standalone", "shared rendering service", "ovrtx rendering docker", "render endpoint deploy", "ovrtx docker compose", "standalone rendering".
---

# Deploy OVRTX Rendering API Standalone

Bring up just the OVRTX rendering API without the rest of an agent service stack. Useful when:

- Multiple agent CLIs (or services) need to share a single rendering host.
- The rendering GPU box is separate from the machine running the CLI / agent service.
- You want to iterate on the agent pipeline without rebooting the rendering container each time.

`apps/ovrtx_rendering_api/` does not ship a dedicated `docker-compose.yml` — OVRTX is defined as a sidecar inside each agent service's compose. This skill takes advantage of Docker Compose's ability to bring up a named service selectively.

## Prerequisites

1. **Docker Compose v2.24+**: `docker compose version` -- required for `env_file: required: false` long-form syntax used in the compose files.
2. **NVIDIA GPU** with ~16 GB+ VRAM: `nvidia-smi`
3. **NVIDIA Container Toolkit**: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

No VLM provider key is required — OVRTX itself does not call a VLM.

## Quick Start

Pick **one** agent service compose file as the source for the OVRTX definition (they all define the same sidecar; physics' compose is the simplest). Then run `docker compose up` with the service name so only OVRTX starts:

```bash
# Using physics_agent_service's compose (recommended -- simplest OVRTX definition)
docker compose -f apps/physics_agent_service/docker-compose.yml up ovrtx-rendering-api --build
```

Alternatives:

```bash
# From material_agent_service (identical OVRTX definition, pulls in multi-gpu profile support if you later want it)
docker compose -f apps/material_agent_service/docker-compose.yml up ovrtx-rendering-api --build
```

First build takes ~10 minutes (OVRTX builds from source). First render takes ~5 minutes (shader compilation, cached after).

By default the container binds to host port **8001**. Health check succeeds once the GPU has warmed up (`gpu_initialized: true`).

## Point agent CLIs / services at it

Once the health check passes, export `RENDER_ENDPOINT` in the environment of the machine running the agent CLI or service:

```bash
export RENDER_ENDPOINT=http://localhost:8001            # same host
export RENDER_ENDPOINT=http://render-host.lan:8001      # remote host
```

Then invoke agent CLIs / services that use `backend: remote` rendering:

```bash
material-agent run apps/material_agent/configs/unified_example.yaml
physics-agent run apps/physics_agent/configs/lightbulb_remote.yaml   # config that sets render.backend: remote
texture-agent run apps/texture_agent/configs/texture_example.yaml
```

For an agent service deployed via Docker Compose alongside the rendering host, set `RENDER_ENDPOINT=http://render-host.lan:8001` in the agent service's `.env`.

## Check health

```bash
curl http://localhost:8001/health
# {"status": "healthy", "gpu_initialized": true, ...}
```

Until GPU warm-up completes, `gpu_initialized` is `false`. Agent CLIs / services that depend on the endpoint will block on their own retry logic until warm-up finishes.

## Operations

### Logs

```bash
docker compose -f apps/physics_agent_service/docker-compose.yml logs -f ovrtx-rendering-api
# or by container name (physics compose names it "physics-ovrtx-rendering-api"):
docker logs physics-ovrtx-rendering-api
```

### Stop

```bash
docker compose -f apps/physics_agent_service/docker-compose.yml stop ovrtx-rendering-api
docker compose -f apps/physics_agent_service/docker-compose.yml rm -f ovrtx-rendering-api
```

### Rebuild

```bash
docker compose -f apps/physics_agent_service/docker-compose.yml up ovrtx-rendering-api --build --force-recreate
```

## GPU Assignment

To pin OVRTX to a specific GPU, edit the relevant agent service's `docker-compose.yml`:

```yaml
ovrtx-rendering-api:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['0']      # specific GPU by ID
            capabilities: [gpu]
```

Or raise `count` to reserve multiple GPUs (OVRTX uses the first one in the assigned set).

## Environment Variables

Configurable via `.env` at the repo root (the compose files all read it via `env_file: ../../.env`):

| Variable | Default | Description |
|---|---|---|
| `OVRTX_LOG_LEVEL` | `warn` | Log verbosity (`trace`, `debug`, `info`, `warn`, `error`) |
| `OVRTX_NUM_SAMPLES` | `1` | Path tracer samples per frame |

## Common Issues

### `gpu_initialized: false` after several minutes

Cause: Shader compilation is slow on first boot (cold cache) or the GPU is being shared with another process. Check `nvidia-smi` for VRAM pressure.
Solution: Wait up to ~5 minutes on cold start. If it persists, make sure no other CUDA process is holding VRAM on the same GPU.

### Port 8001 already in use

Cause: Another process (e.g., another agent service's OVRTX sidecar, or a prior run) is using 8001.
Solution: Stop the other container, or bring OVRTX up via a compose overlay that remaps the port:

```yaml
# apps/ovrtx_rendering_api/docker-compose.override.yml
services:
  ovrtx-rendering-api:
    ports:
      - "8010:8000"
```

Then `docker compose -f apps/physics_agent_service/docker-compose.yml -f apps/ovrtx_rendering_api/docker-compose.override.yml up ovrtx-rendering-api --build`.

### Agent CLI fails with connection refused

Cause: OVRTX container still warming up, or `RENDER_ENDPOINT` is unset / pointing at the wrong host.
Solution: Poll `/health` first and confirm `gpu_initialized: true` before invoking the CLI; verify `RENDER_ENDPOINT` with `echo $RENDER_ENDPOINT`.
