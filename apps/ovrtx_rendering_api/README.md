# OVRTX Rendering API

USD rendering service using the [OVRTX](https://github.com/NVIDIA-Omniverse/ovrtx) local RTX renderer. Provides a drop-in REST API compatible with the Kit-based rendering service, so upstream agents (material, physics, texture) can use either backend interchangeably.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health status including GPU initialization state |
| `POST` | `/render` | Render a USD file and return base64-encoded images |

The render endpoint accepts the same request body as the Kit-based `rendering-api` and returns the same V1 response format (`images[frame][camera][sensor] = base64`). See [`openapi.yaml`](openapi.yaml) for the full schema.

## Quick Start

```bash
# From repository root
uv pip install -e apps/ovrtx_rendering_api

# Run the service
uvicorn service.main:app --host 0.0.0.0 --port 8001
```

The service listens on port 8001 by default. Visit `/docs` for the Swagger UI.
On a cold start, GPU warm-up commonly takes around 5 minutes; `/health`
continues to return `gpu_initialized=false` until that warm-up completes.

## Docker

The service is packaged as a GPU-enabled container. It is typically launched as a sidecar via the main `material-agent-service` Docker Compose setup (see `apps/material_agent_service/docker-compose.yml`).

```bash
# Build the image
docker build -f apps/ovrtx_rendering_api/Dockerfile -t ovrtx-rendering-api .

# Run with GPU access
docker run --rm --gpus all -p 8001:8001 ovrtx-rendering-api
```

The container health check now waits for `/health` to report
`gpu_initialized=true`, so a fresh `docker run` may stay in `starting` state
for several minutes before it becomes healthy.

## Example Request

```bash
curl -X POST http://localhost:8001/render \
  -H "Content-Type: application/json" \
  -d '{
    "url": "file:///data/scene.usd",
    "render_settings": {
      "camera_paths": ["/Camera"],
      "frame_range": {"start": 0, "end": 0},
      "camera_parameters": {"width": 1024, "height": 1024}
    }
  }'
```

## Requirements

- NVIDIA GPU with driver 535+
- CUDA runtime (provided by the container base image)
- USD asset accessible via URL (file, http, or s3)

## Project Structure

```
ovrtx_rendering_api/
├── service/        # FastAPI app, render pipeline, models
├── client/         # Optional Python client for programmatic use
├── tests/          # Unit and integration tests
├── Dockerfile      # Container build
├── openapi.yaml    # API specification
└── pyproject.toml  # Install metadata
```
