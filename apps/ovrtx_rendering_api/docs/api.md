# OVRTX Rendering API Reference

USD rendering service using the OVRTX local RTX renderer. Drop-in replacement for the Kit-based rendering API: identical request body and V1 image response format.

**Base URL:** `http://localhost:8001`
**Interactive docs:** `GET /docs` (Swagger UI)
**OpenAPI:** [`../openapi.yaml`](../openapi.yaml)

---

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [`GET /health`](#get-health)
  - [`POST /render`](#post-render)
- [Data Models](#data-models)
- [Error Handling](#error-handling)

---

## Authentication

No authentication. The service is intended to run on a trusted internal network (e.g., as a sidecar to a material/physics/texture agent service).

---

## Endpoints

### `GET /health`

Health check including GPU initialization state.

**Response** `200`

```json
{
  "status": "healthy",
  "service": "ovrtx-rendering-api",
  "version": "0.1.0",
  "renderer": "ovrtx",
  "gpu_initialized": true
}
```

The `gpu_initialized` flag is `false` until the renderer finishes its cold-start
GPU warm-up. In practice this commonly takes around 5 minutes, so readiness
checks should tolerate `false` during that window.

### `POST /render`

Render a USD file and return base64-encoded images for each (frame, camera, sensor) tuple.

**Request body** -- `application/json` -- [`RenderRequest`](#renderrequest)

```json
{
  "url": "file:///data/scene.usd",
  "force_render": true,
  "render_settings": {
    "camera_paths": ["/Camera"],
    "frame_range": {"start": 0, "end": 0},
    "camera_parameters": {"width": 1024, "height": 1024},
    "sensors": ["rgb"],
    "apply_background_mask": false
  }
}
```

**Response** `200` -- [`RenderResponse`](#renderresponse)

The response structure is `images[frame_number][camera_path][sensor_name] = base64_string`. A successful render always returns `status: "success"`; failures return `status: "exception"` with an `error` message and an empty `images` map.

```json
{
  "status": "success",
  "error": null,
  "images": {
    "0": {
      "/Camera": {
        "rgb": "iVBORw0KGgoAAAANSUhEUg..."
      }
    }
  }
}
```

**Notes**

- The endpoint is a synchronous Python function (not `async def`). OVRTX serializes renders internally, so the event loop is intentionally blocked rather than starved.
- The `url` field accepts `file://`, `http://`/`https://`, and `s3://` schemes. S3 URLs require the container to have AWS credentials available.
- Large `frame_range` requests are run sequentially. For parallelism, run multiple replicas behind a load balancer (see `docker-compose.multi-gpu.yml` in the material-agent-service for an example).

---

## Data Models

### `RenderRequest`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | string | -- | USD asset URL (`file://`, `http://`, `s3://`). |
| `force_render` | bool | `true` | If `true`, re-render even when cached. |
| `render_settings` | [`RenderSettings`](#rendersettings) | -- | Render parameters. |

### `RenderSettings`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `camera_paths` | `list[str]` | `["/Camera"]` | USD prim paths of cameras to render. |
| `frame_range` | [`FrameRange`](#framerange) | `{start: 0, end: 0}` | Inclusive frame range (both ends). |
| `camera_parameters` | [`CameraParameters`](#cameraparameters) | `{width: 1024, height: 1024}` | Per-camera image resolution. |
| `sensors` | `list[str] \| null` | `null` (= `rgb` only) | Sensor outputs, e.g. `["rgb", "depth", "instance_id"]`. |
| `apply_background_mask` | bool | `false` | If `true`, apply dome-light background masking. |

### `FrameRange`

| Field | Type | Default |
|-------|------|---------|
| `start` | int | `0` |
| `end` | int | `0` |

### `CameraParameters`

| Field | Type | Default |
|-------|------|---------|
| `width` | int | `1024` |
| `height` | int | `1024` |

### `RenderResponse`

| Field | Type | Description |
|-------|------|-------------|
| `status` | `"success" \| "exception"` | Overall result. |
| `error` | `string \| null` | Error message if `status == "exception"`. |
| `images` | nested map | `images[frame][camera][sensor] = base64 string` (PNG). |

---

## Error Handling

Errors are returned in the response body with `status: "exception"`. The HTTP status is still `200` to match the Kit rendering-api contract. Clients should inspect the JSON `status` field.

Common error cases:

- USD file not found or not a valid USD stage
- Camera path does not exist on the stage
- GPU initialization failure (check `/health` — `gpu_initialized` will be `false`)
- OVRTX daemon crash (container will self-restart; client should retry)
