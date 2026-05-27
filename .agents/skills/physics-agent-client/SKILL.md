---
name: physics-agent-client
description: Make requests to the Physics Agent REST API service for VLM-based physics and component classification of 3D USD assets. Use when the user wants to call the Physics Agent service API, upload or reference a USD file, run the REST classification pipeline, use prediction-only or tuning routes, monitor status, download predictions or simulation-ready USD, or write a client script.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - physics-agent
  - rest-api
  - client
  - usd
tools:
  - Shell
  - Filesystem
  - Python
  - curl
  - jq
compatibility: Requires a running Physics Agent REST service, optional bearer-token authentication, curl or the bundled Python client, and service-side VLM, rendering, optimizer, S3, and optional tuning dependencies configured for the requested route.
---

# Physics Agent Client

Use the Physics Agent REST API to classify USD components, generate
`predictions.jsonl`, and download a simulation-ready USD with physics schemas
applied.

## When to Use

- Use when the user asks for the Physics Agent service, API, REST workflow,
  Python client, or curl examples.
- Use when the user wants to upload a USD file, submit an S3 URI, run the full
  classification pipeline, check status, stream events, or download artifacts.
- Use `/predict` when the user explicitly wants prediction-only behavior.
- Use `/tune` only when the user has a physics-authored USD and wants a
  service-side tuning run.
- Use `physics-agent-cli` instead when the user wants to run the local CLI.

## Limitations

- Keep credentials out of chat and commits. Use service-side configuration or
  `PHYSICS_AGENT_TOKEN`; never ask the user to paste secrets.
- This skill calls an already running service. It does not build or start the
  service container.
- The main guidance is for `/pipeline`; `/predict` and `/tune` are related API
  families with separate session kinds and cancel endpoints.
- Use optimizer flags only when the asset needs preprocessing. Deinstance fixes
  instance-proxy authoring failures; split is for one mesh that contains
  multiple disjoint components.
- OVRTX can return `gpu_initialized=false` while warming up even when the HTTP
  service is reachable.

## Prerequisites

- Physics Agent service base URL, usually `http://localhost:8000`.
- `curl` and `jq` for shell examples, or Python plus
  `apps/physics_agent_service/client/client.py`.
- Optional bearer token supplied as `Authorization: Bearer <token>` or through
  `PHYSICS_AGENT_TOKEN`.
- A local USD file, an uploaded session, or an S3 URI readable by the service.
- Service-side provider credentials and render/optimizer endpoints configured
  for the requested run.

## Instructions

1. Confirm the service is reachable with `GET /health`.
2. Choose direct `POST /pipeline`, two-step upload, or S3 URI input.
3. Use exactly one input source: `usd_file`, `session_id`, or `s3_uri`.
4. Add optimizer flags only when required by instancing, instance-proxy
   failures, disjoint mesh splitting, or deduplication.
5. Monitor with `GET /pipeline/{id}/events` for SSE or
   `GET /pipeline/{id}/status` for polling.
6. Download predictions, report, dataset, and output USD after status is
   `completed`.
7. For prediction-only or tuning flows, use the matching `/predict` or `/tune`
   status, events, results, and cancel endpoints.

## Python Client

```python
from apps.physics_agent_service.client.client import PhysicsAgentClient

client = PhysicsAgentClient(base_url="http://localhost:8000")

session_id, status = client.run_and_monitor(
    usd_path="/path/to/scene.usdz",
    user_prompt="Focus on identifying furniture parts",
    render_backend="remote",
)

print(session_id, status)
```

Use S3 input mode for large files when the service can download the asset:

```python
session_id, status = client.run_and_monitor(
    s3_uri="s3://your-bucket/path/to/scene.usdz",
)
```

## curl Workflow

```bash
BASE_URL="http://localhost:8000"

curl -fsS "$BASE_URL/health" | jq .

SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" | jq -r .session_id)

curl -fsS "$BASE_URL/pipeline/$SESSION/status" | jq .
curl -N "$BASE_URL/pipeline/$SESSION/events"

curl -fL -o predictions.jsonl "$BASE_URL/artifacts/$SESSION/predictions"
curl -fL -o report.html "$BASE_URL/artifacts/$SESSION/report"
curl -fLOJ "$BASE_URL/artifacts/$SESSION/output-usd"
```

Optimizer examples:

```bash
# Deinstance to fix instance-proxy apply_physics failures.
curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "optimize_usd=true" \
  -F "enable_deinstance=true" | jq .

# Deinstance and split when one mesh must become separate components.
curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "optimize_usd=true" \
  -F "enable_deinstance=true" \
  -F "enable_split=true" | jq .
```

## Endpoint Reference

| Area | Endpoints |
|---|---|
| Health | `GET /health` |
| Pipeline | `POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, `GET /pipeline/{id}/results`, `GET /pipeline/{id}/events`, `POST /pipeline/{id}/cancel`, `POST /pipeline/{id}/regenerate`, `GET /pipeline/{id}/event-log` |
| Predict-only | `POST /predict`, `GET /predict/{id}/status`, `GET /predict/{id}/results`, `GET /predict/{id}/events`, `POST /predict/{id}/cancel` |
| Artifacts | `GET /artifacts/{id}/predictions`, `/report`, `/dataset`, `/output-usd` |
| Tune | `POST /tune`, `GET /tune/{id}/status`, `GET /tune/{id}/results`, `GET /tune/{id}/events`, `POST /tune/{id}/cancel`, `GET /tune/{id}/artifacts/{name}` |
| Sessions | `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}` |

## Key Pipeline Parameters

`POST /pipeline` accepts multipart form data.

| Parameter | Required | Default | Description |
|---|---|---|---|
| `usd_file` | Conditional |  | USD input file. Required unless `session_id` or `s3_uri` is provided. |
| `session_id` | Conditional |  | Existing session from `/pipeline/upload-usd`. |
| `s3_uri` | Conditional |  | Service-side S3 input. |
| `user_prompt` | No |  | Custom classification guidance. |
| `render_backend` | No | `remote` | `remote`, `warp`, or `ovrtx`. |
| `optimize_usd` | No | `false` | Run Scene Optimizer before rendering and prediction. |
| `enable_deinstance` | No | `true` when optimizing | Deinstance optimized USDs; required for instance-proxy authoring failures. |
| `enable_split` | No | `false` | Split disjoint pieces in one mesh into separate components. |
| `enable_deduplicate` | No | `false` | Collapse repeated identical geometry and restore by correspondence. |

Exactly one of `usd_file`, `session_id`, or `s3_uri` must be provided. When
`optimize_usd=true`, enable at least one optimizer operation.

Status values are `pending`, `running`, `completed`, `failed`, `cancelled`,
and `cancelling`.

## Output Format

Return a concise summary with:

- Service base URL and authentication mode, without printing tokens.
- Endpoint family used: `/pipeline`, `/predict`, or `/tune`.
- Session ID, input mode, current status, and progress source.
- Optimizer flags used and why they were needed.
- Downloaded artifact paths or URLs for predictions, report, dataset, output
  USD, and tune artifacts when applicable.
- Any blocker such as missing service credentials, upload size, S3 access,
  renderer warm-up, or non-terminal status.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Connection refused | Service is not running or base URL is wrong. | Start the Physics Agent service or correct `BASE_URL`. |
| Unauthorized | Bearer token is missing or wrong. | Set `PHYSICS_AGENT_TOKEN` locally or pass the correct header. |
| `413` upload response | Input exceeds the service upload limit. | Increase `PA_MAX_UPLOAD_SIZE_MB`, use a smaller file, or submit an S3 URI. |
| Results return `202` | Pipeline is still running. | Poll `/status` until `completed` or stream `/events`. |
| Instance-proxy authoring failure | Physics schemas cannot be authored on instance proxies. | Re-run with `optimize_usd=true` and `enable_deinstance=true`. |
| Only one prediction from a combined mesh | Deinstance made the prim writable but did not split geometry. | Add `enable_split=true` when separate component entries are needed. |
| OVRTX health shows `gpu_initialized=false` | Renderer is still warming. | Wait and re-check the render sidecar health before treating it as failed. |
