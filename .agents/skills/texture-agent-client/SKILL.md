---
name: texture-agent-client
description: Make requests to the Texture Agent REST API service for AI-driven texture generation on materialized USD assets. Use when the user wants to call the Texture Agent service API, upload a materialized USD, start or monitor a texture pipeline, control per-material texture prompts, download generated textures or USDZ output, or write a client script.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - texture-agent
  - rest-api
  - client
  - image-generation
tools:
  - Shell
  - Filesystem
  - Python
  - curl
  - jq
compatibility: Requires a running Texture Agent REST service, curl or the bundled Python client, a materialized USD input, and service-side image-generation and optional LLM credentials configured for the selected backend.
---

# Texture Agent Client

Use the Texture Agent REST API to upload a materialized USD, discover materials,
generate image textures, apply them, and download the textured USDZ output.

## When to Use

- Use when the user asks for the Texture Agent service, API, REST workflow,
  Python client, or curl examples.
- Use when the user wants to upload a materialized USD, submit per-material
  prompts, keep texture generation scoped to specific materials, monitor
  pipeline status, or download generated textures and renders.
- Use the Docker deploy or quickstart skills first when no Texture Agent
  service is running yet.
- Use `texture-agent-cli` instead when the user wants the local CLI.

## Limitations

- Keep provider credentials out of chat and commits. They must be configured
  on the service; never ask the user to paste secrets.
- This skill calls an already running service. It does not build or start the
  service container.
- Input should already have material bindings, typically from Material Agent.
- The REST wire parameter is `material_textures_json`, a JSON-encoded string.
  The Python client exposes the same data as a `dict` named
  `material_textures`.
- Set `auto_prompt_enabled=false` when the user wants strict scope and only
  listed materials should be processed.

## Prerequisites

- Texture Agent service base URL, usually `http://localhost:8001`.
- `curl` and `jq` for shell examples, or Python plus
  `apps/texture_agent_service/client/client.py`.
- A materialized `.usd`, `.usda`, `.usdc`, or `.usdz` file.
- Service-side image-generation credentials such as `NVIDIA_API_KEY`,
  `GOOGLE_API_KEY`, or an endpoint-specific key.
- Optional LLM credentials when auto-prompt generation is enabled.

## Instructions

1. Confirm the service is reachable with `GET /health`.
2. Choose direct `POST /pipeline`, two-step upload, or S3 URI input.
3. Use exactly one input source: `usd_file`, `session_id`, or `s3_uri`.
4. Discover material names first when the user needs exact targeting. For a
   fresh USD with no completed session yet, run an auto-prompt discovery pass
   first, then resubmit the strict material map.
5. Submit `material_textures_json` as a JSON string for per-material prompts.
6. Use `auto_prompt_enabled=false` for strict listed-material scope.
7. Monitor with `GET /pipeline/{id}/events` for SSE or
   `GET /pipeline/{id}/status` for polling.
8. Download output USDZ, textures ZIP, manifest, materials JSON, and renders
   after status is `completed`.

## Python Client

```python
from apps.texture_agent_service.client.client import TextureAgentClient

client = TextureAgentClient("http://localhost:8001")

session_id, status = client.run_and_monitor(
    usd_path="materialized_scene.usd",
    material_textures={
        "Steel_Carbon": {"prompt": "rusted steel", "opacity": 0.85},
        "Wood_Oak": {"prompt": "weathered oak planks", "opacity": 0.9},
    },
    auto_prompt_enabled=False,
)

client.download_output(session_id, "output.usdz")
client.download_textures(session_id, "./textures/")
```

## curl Workflow

```bash
BASE_URL="http://localhost:8001"

curl -fsS "$BASE_URL/health" | jq .

SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F "auto_prompt_enabled=false" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"rusted steel"}}' \
  | jq -r .session_id)

curl -fsS "$BASE_URL/pipeline/$SESSION/status" | jq .
curl -N "$BASE_URL/pipeline/$SESSION/events"

curl -fL -o output.usdz "$BASE_URL/artifacts/$SESSION/output"
curl -fL -o textures.zip "$BASE_URL/artifacts/$SESSION/textures"
curl -fL -o manifest.json "$BASE_URL/artifacts/$SESSION/manifest"
```

Fresh USD strict targeting:

```bash
# First pass: let the service discover material names and complete a session.
DISCOVERY_SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F "auto_prompt_enabled=true" \
  | jq -r .session_id)

# Wait for the discovery pass to reach completed status before fetching materials.
curl -fsS "$BASE_URL/pipeline/$DISCOVERY_SESSION/status" | jq .
curl -N "$BASE_URL/pipeline/$DISCOVERY_SESSION/events"
curl -fsS "$BASE_URL/artifacts/$DISCOVERY_SESSION/materials" | jq . > materials.json

# Second pass: use exact names from materials.json and strict scope.
STRICT_SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F "auto_prompt_enabled=false" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"rusted steel"}}' \
  | jq -r .session_id)
```

Nested per-prim prompt map:

```bash
curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F "auto_prompt_enabled=false" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"aged steel","per_prim":{"/World/Bolt":{"prompt":"scratched bolt head"}}}}' \
  | jq .
```

## Endpoint Reference

| Area | Endpoints |
|---|---|
| Health/API | `GET /health`, `GET /api`, `GET /` |
| Pipeline | `POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, `GET /pipeline/{id}/results`, `GET /pipeline/{id}/events`, `POST /pipeline/{id}/cancel`, `POST /pipeline/{id}/regenerate`, `GET /pipeline/{id}/event-log` |
| Artifacts | `GET /artifacts/{id}/materials`, `/manifest`, `/textures`, `/textures/{name}`, `/output`, `/renders`, `/renders/{name}`, `/preview/{name}` |
| Sessions | `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}` |

## Key Pipeline Parameters

`POST /pipeline` accepts multipart form data. Exactly one of `usd_file`,
`session_id`, or `s3_uri` must be provided.

| Parameter | Required | Description |
|---|---|---|
| `usd_file` | Conditional | Materialized USD file. |
| `session_id` | Conditional | Existing uploaded session. |
| `s3_uri` | Conditional | Service-side S3 input. |
| `material_textures_json` | No | JSON-encoded string of per-material texture configuration. |
| `user_prompt` | No | Aesthetic direction for auto-prompt generation. |
| `auto_prompt_enabled` | No | Defaults to service behavior. Set `false` to process only listed materials. |

Decoded `material_textures_json` is keyed by discovered material name. Each
value can include:

- `prompt`: text prompt describing the desired texture.
- `opacity`: optional blend opacity.
- `per_prim`: optional nested map keyed by prim path for prim-specific
  overrides.

Use `GET /artifacts/{id}/materials` to discover exact material names from a
completed session before submitting a strict map. `GET /pipeline/{id}/results`
is also completed-only and can confirm artifact URLs, but it is not available
before the first run. For a fresh USD, first submit a discovery pass with
`auto_prompt_enabled=true`, wait for it to complete, fetch
`GET /artifacts/{id}/materials`, then submit the exact
`material_textures_json` keys in a second run with `auto_prompt_enabled=false`.

Status values are `pending`, `running`, `completed`, `failed`, `cancelled`,
and `cancelling`.

## Output Format

Return a concise summary with:

- Service base URL and input mode.
- Session ID, status, and progress source.
- Whether auto-prompting was enabled or strict listed-material scope was used.
- Submitted material names and any `per_prim` overrides.
- Downloaded artifact paths or URLs for output USDZ, textures ZIP, manifest,
  materials JSON, event log, and renders.
- Any blocker such as missing image-generation credentials, unmatched material
  names, upload size, or non-terminal status.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Connection refused | Service is not running or base URL is wrong. | Start the Texture Agent service or correct `BASE_URL`. |
| `413` upload response | Input exceeds service upload limit. | Increase `TA_MAX_UPLOAD_SIZE_MB`, use a smaller file, or submit an S3 URI. |
| Results return `202` | Pipeline is still running. | Poll `/status` until `completed` or stream `/events`. |
| No textures generated for a material | The material name did not match discovered names, or strict mode skipped it. | Fetch `/artifacts/{id}/materials`, then resubmit exact keys. |
| Unexpected extra textures | Auto-prompting processed unlisted materials. | Set `auto_prompt_enabled=false`. |
| `material_textures_json` parse error | The REST form value was not valid JSON text. | Quote it as one JSON string, or use the Python client dict abstraction. |
| Image generation fails | Service-side backend credentials or endpoint are missing. | Check service logs and the configured image-generation backend. |
